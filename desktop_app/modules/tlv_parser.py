import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
FRAME_HEADER_LEN = 40

TLV_TYPE_POINT_CLOUD = 1
TLV_TYPE_TRACKED_OBJECTS = 6
TLV_TYPE_TRACKER_TARGET_LIST = 1010
TLV_TYPE_TRACKER_TARGET_INDEX = 1011
TLV_TYPE_TRACKER_TARGET_HEIGHT = 1012
TLV_TYPE_COMPRESSED_POINTS = 1020
TLV_TYPE_PRESENCE_INDICATION = 1021
TLV_TYPE_VITAL_SIGNS = 0x410

VITAL_TRACK_ID_MAX = 255
VITAL_FALLBACK_ID_MIN = 0x8000
VITAL_FALLBACK_ID_MAX = 0x80FF
VITAL_RANGEBIN_MAX = 255
VITAL_HEART_RATE_MAX = 220.0
VITAL_BREATHING_RATE_MAX = 60.0
VITAL_BREATHING_DEVIATION_MAX = 100.0

LEGACY_TRACK_STRUCT = struct.Struct("<I10f")
TRACKER_TARGET_BASE_STRUCT = struct.Struct("<I9f")
TRACKER_TARGET_FULL_STRUCT = struct.Struct("<I27f")
TRACKER_TARGET_HEIGHT_STRUCT = struct.Struct("<Iff")
VITAL_SIGNS_OUTPUT_STRUCT = struct.Struct("<HH" + ("f" * 33))
COMPRESSED_POINT_UNIT_STRUCT = struct.Struct("<5f")
COMPRESSED_POINT_DTYPE = np.dtype(
    [
        ("elevation", "i1"),
        ("azimuth", "i1"),
        ("doppler", "<i2"),
        ("range", "<u2"),
        ("snr", "<u2"),
    ]
)


@dataclass
class ParsedFrame:
    ts: float
    points: Optional[np.ndarray]
    tracks: List[dict]
    raw: bytes = b""
    frame_number: int = 0
    num_detected_obj: int = 0
    num_tlvs: int = 0
    tlv_types: List[int] = field(default_factory=list)
    vitals: List[dict] = field(default_factory=list)
    point_track_ids: Optional[np.ndarray] = None
    point_track_ids_lag: int = 0
    presence: Optional[int] = None
    tracks_are_world: bool = False
    tracks_lag: int = 0
    skeleton_lines: Optional[np.ndarray] = None
    invalid_vital_records: int = 0
    malformed_tlvs: int = 0

    def to_jsonable(self, include_points: bool = False, max_points: int = 300):
        payload = {
            "ts": self.ts,
            "frameNumber": int(self.frame_number),
            "numDetectedObj": int(self.num_detected_obj),
            "numTlvs": int(self.num_tlvs),
            "tlvTypes": [int(item) for item in self.tlv_types],
            "tracks": self.tracks,
            "invalidVitalRecords": int(self.invalid_vital_records),
            "malformedTlvs": int(self.malformed_tlvs),
        }
        if self.vitals:
            payload["vitals"] = self.vitals
        if self.presence is not None:
            payload["presence"] = int(self.presence)
        if include_points and self.points is not None and len(self.points) > 0:
            pts = self.points
            if len(pts) > max_points:
                idx = np.linspace(0, len(pts) - 1, num=max_points, dtype=int)
                pts = pts[idx]
            payload["points"] = pts.tolist()
        if self.skeleton_lines is not None and len(self.skeleton_lines) > 0:
            payload["skeletons"] = np.asarray(self.skeleton_lines, dtype=np.float32).tolist()
        return payload


class TlvFrameParser:
    def __init__(self):
        self._buffer = b""
        self._tlv_overrun_count = 0
        self._bad_frame_count = 0

    def push_and_extract(self, chunk: bytes) -> List[bytes]:
        if not chunk:
            return []

        self._buffer += chunk
        frames: List[bytes] = []

        while True:
            start = self._buffer.find(MAGIC_WORD)
            if start < 0:
                self._buffer = b""
                break

            if start > 0:
                self._buffer = self._buffer[start:]
                start = 0

            if len(self._buffer) < start + FRAME_HEADER_LEN:
                break

            total_len = struct.unpack_from("<I", self._buffer, start + 12)[0]
            if total_len < FRAME_HEADER_LEN or total_len > 65536:
                self._bad_frame_count += 1
                if self._bad_frame_count <= 3 or self._bad_frame_count % 50 == 0:
                    print(f"[tlv_parser] invalid frame length skipped: totalLen={total_len}")
                self._buffer = self._buffer[start + 1 :]
                continue

            if len(self._buffer) < start + total_len:
                break

            frame = self._buffer[start : start + total_len]
            frames.append(frame)
            self._buffer = self._buffer[start + total_len :]

        return frames

    def parse_frame(self, frame_bytes: bytes, ts: Optional[float] = None) -> Optional[ParsedFrame]:
        if len(frame_bytes) < FRAME_HEADER_LEN or not frame_bytes.startswith(MAGIC_WORD):
            return None

        if ts is None:
            ts = time.time()

        frame_number = struct.unpack_from("<I", frame_bytes, 20)[0]
        num_detected_obj = struct.unpack_from("<I", frame_bytes, 28)[0]
        num_tlvs = struct.unpack_from("<I", frame_bytes, 32)[0]
        cursor = FRAME_HEADER_LEN

        points: Optional[np.ndarray] = None
        tracks: List[dict] = []
        vitals: List[dict] = []
        tlv_types: List[int] = []
        heights_by_id: Dict[int, Dict[str, float]] = {}
        point_track_ids: Optional[np.ndarray] = None
        presence: Optional[int] = None
        tracks_are_world = False
        tracks_lag = 0
        invalid_vital_records = 0
        malformed_tlvs = 0

        for _ in range(num_tlvs):
            if cursor + 8 > len(frame_bytes):
                malformed_tlvs += 1
                break

            tlv_type, tlv_len = struct.unpack_from("<II", frame_bytes, cursor)
            tlv_types.append(int(tlv_type))
            cursor += 8
            if cursor + tlv_len > len(frame_bytes):
                malformed_tlvs += 1
                self._tlv_overrun_count += 1
                if self._tlv_overrun_count <= 3 or self._tlv_overrun_count % 50 == 0:
                    print(
                        f"[tlv_parser] TLV overrun detected: type={tlv_type} len={tlv_len} "
                        f"remaining={len(frame_bytes) - cursor}"
                    )
                break
            payload = frame_bytes[cursor : cursor + tlv_len]

            if tlv_type == TLV_TYPE_POINT_CLOUD:
                parsed_points = self._parse_legacy_points(payload)
                if parsed_points is not None:
                    points = parsed_points

            elif tlv_type == TLV_TYPE_TRACKED_OBJECTS:
                tracks = self._parse_legacy_tracks(payload)

            elif tlv_type == TLV_TYPE_COMPRESSED_POINTS:
                parsed_points = self._parse_compressed_points(payload)
                if parsed_points is not None:
                    points = parsed_points

            elif tlv_type == TLV_TYPE_TRACKER_TARGET_LIST:
                tracks = self._parse_tracker_targets(payload)
                tracks_are_world = True
                tracks_lag = 1

            elif tlv_type == TLV_TYPE_TRACKER_TARGET_INDEX:
                point_track_ids = np.frombuffer(payload[:tlv_len], dtype=np.uint8).copy()
                tracks_lag = 1

            elif tlv_type == TLV_TYPE_TRACKER_TARGET_HEIGHT:
                heights_by_id = self._parse_target_heights(payload)

            elif tlv_type == TLV_TYPE_PRESENCE_INDICATION and len(payload) >= 4:
                presence = int(struct.unpack_from("<I", payload, 0)[0])

            elif tlv_type == TLV_TYPE_VITAL_SIGNS:
                parsed_vitals, invalid_count = self._parse_vital_signs_checked(payload)
                vitals.extend(parsed_vitals)
                invalid_vital_records += invalid_count

            cursor += tlv_len

        if heights_by_id:
            for track in tracks:
                height_info = heights_by_id.get(int(track.get("id", -1)))
                if not height_info:
                    continue
                track.update(height_info)

        return ParsedFrame(
            ts=ts,
            points=points,
            tracks=tracks,
            raw=frame_bytes,
            frame_number=int(frame_number),
            num_detected_obj=int(num_detected_obj),
            num_tlvs=int(num_tlvs),
            tlv_types=tlv_types,
            vitals=vitals,
            point_track_ids=point_track_ids,
            point_track_ids_lag=1 if point_track_ids is not None else 0,
            presence=presence,
            tracks_are_world=tracks_are_world,
            tracks_lag=tracks_lag,
            invalid_vital_records=invalid_vital_records,
            malformed_tlvs=malformed_tlvs,
        )

    @staticmethod
    def _parse_legacy_points(payload: bytes) -> Optional[np.ndarray]:
        if len(payload) == 0 or len(payload) % 16 != 0:
            return None
        return np.frombuffer(payload, dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def _parse_legacy_tracks(payload: bytes) -> List[dict]:
        stride = LEGACY_TRACK_STRUCT.size
        if len(payload) == 0 or len(payload) % stride != 0:
            return []

        tracks: List[dict] = []
        for offset in range(0, len(payload), stride):
            unpacked = LEGACY_TRACK_STRUCT.unpack_from(payload, offset)
            tid = int(unpacked[0])
            x, y, z, vx, vy, vz, ax, ay, az, g = unpacked[1:]
            tracks.append(
                {
                    "id": tid,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "vx": float(vx),
                    "vy": float(vy),
                    "vz": float(vz),
                    "ax": float(ax),
                    "ay": float(ay),
                    "az": float(az),
                    "g": float(g),
                }
            )
        return tracks

    @staticmethod
    def _parse_compressed_points(payload: bytes) -> Optional[np.ndarray]:
        if len(payload) < COMPRESSED_POINT_UNIT_STRUCT.size:
            return None

        elevation_unit, azimuth_unit, doppler_unit, range_unit, _snr_unit = COMPRESSED_POINT_UNIT_STRUCT.unpack_from(
            payload, 0
        )
        points_payload = payload[COMPRESSED_POINT_UNIT_STRUCT.size :]
        count = len(points_payload) // COMPRESSED_POINT_DTYPE.itemsize
        if count <= 0:
            return np.empty((0, 4), dtype=np.float32)

        points_payload = points_payload[: count * COMPRESSED_POINT_DTYPE.itemsize]
        compressed = np.frombuffer(points_payload, dtype=COMPRESSED_POINT_DTYPE, count=count)

        elevation = compressed["elevation"].astype(np.float32) * elevation_unit
        azimuth = compressed["azimuth"].astype(np.float32) * azimuth_unit
        doppler = compressed["doppler"].astype(np.float32) * doppler_unit
        ranges = compressed["range"].astype(np.float32) * range_unit

        cos_elev = np.cos(elevation)
        x = ranges * np.sin(azimuth) * cos_elev
        y = ranges * np.cos(azimuth) * cos_elev
        z = ranges * np.sin(elevation)

        return np.column_stack((x, y, z, doppler)).astype(np.float32)

    @staticmethod
    def _parse_tracker_targets(payload: bytes) -> List[dict]:
        if len(payload) == 0:
            return []

        if len(payload) % TRACKER_TARGET_FULL_STRUCT.size == 0:
            stride = TRACKER_TARGET_FULL_STRUCT.size
            full = True
        elif len(payload) % TRACKER_TARGET_BASE_STRUCT.size == 0:
            stride = TRACKER_TARGET_BASE_STRUCT.size
            full = False
        else:
            return []

        tracks: List[dict] = []
        for offset in range(0, len(payload), stride):
            if full:
                unpacked = TRACKER_TARGET_FULL_STRUCT.unpack_from(payload, offset)
                tid = int(unpacked[0])
                pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z = unpacked[1:10]
                gating_gain = float(unpacked[-2])
                confidence = float(unpacked[-1])
            else:
                unpacked = TRACKER_TARGET_BASE_STRUCT.unpack_from(payload, offset)
                tid = int(unpacked[0])
                pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z = unpacked[1:10]
                gating_gain = 0.0
                confidence = 0.0

            tracks.append(
                {
                    "id": tid,
                    "x": float(pos_x),
                    "y": float(pos_y),
                    "z": float(pos_z),
                    "vx": float(vel_x),
                    "vy": float(vel_y),
                    "vz": float(vel_z),
                    "ax": float(acc_x),
                    "ay": float(acc_y),
                    "az": float(acc_z),
                    "g": gating_gain,
                    "confidence": float(np.clip(confidence, 0.0, 1.0)),
                }
            )

        return tracks

    @staticmethod
    def _parse_target_heights(payload: bytes) -> Dict[int, Dict[str, float]]:
        stride = TRACKER_TARGET_HEIGHT_STRUCT.size
        if len(payload) == 0 or len(payload) % stride != 0:
            return {}

        heights: Dict[int, Dict[str, float]] = {}
        for offset in range(0, len(payload), stride):
            tid, max_z, min_z = TRACKER_TARGET_HEIGHT_STRUCT.unpack_from(payload, offset)
            heights[int(tid)] = {
                "z_max": float(max_z),
                "z_min": float(min_z),
                "height": max(0.15, float(max_z - min_z)),
            }
        return heights

    @staticmethod
    def _parse_vital_signs(payload: bytes) -> List[dict]:
        vitals, _invalid_count = TlvFrameParser._parse_vital_signs_checked(payload)
        return vitals

    @staticmethod
    def _parse_vital_signs_checked(payload: bytes) -> tuple[List[dict], int]:
        stride = VITAL_SIGNS_OUTPUT_STRUCT.size
        if len(payload) == 0 or len(payload) % stride != 0:
            return [], 0

        vitals: List[dict] = []
        invalid_count = 0
        for offset in range(0, len(payload), stride):
            raw = VITAL_SIGNS_OUTPUT_STRUCT.unpack_from(payload, offset)
            signal_deviation = float(raw[2])
            heart_bpm = float(raw[3])
            breath_bpm = float(raw[4])
            vital = {
                "id": int(raw[0]),
                "range_bin": int(raw[1]),
                "signal_deviation": signal_deviation,
                "heart_bpm": heart_bpm,
                "breath_bpm": breath_bpm,
                "heart_history": [float(value) for value in raw[5:20]],
                "breath_history": [float(value) for value in raw[20:35]],
                # Keep firmware/validation naming available for downstream callers.
                "rangebin": int(raw[1]),
                "breathingDeviation": signal_deviation,
                "heartRate": heart_bpm,
                "breathingRate": breath_bpm,
            }
            if TlvFrameParser._is_valid_vital_record(vital):
                vitals.append(vital)
            else:
                invalid_count += 1
        return vitals, invalid_count

    @staticmethod
    def _is_valid_vital_record(vital: dict) -> bool:
        try:
            target_id = int(vital["id"])
            rangebin = int(vital["range_bin"])
            deviation = float(vital["signal_deviation"])
            heart = float(vital["heart_bpm"])
            breath = float(vital["breath_bpm"])
            heart_history = [float(value) for value in vital.get("heart_history", [])]
            breath_history = [float(value) for value in vital.get("breath_history", [])]
        except (TypeError, ValueError, KeyError):
            return False

        valid_target_id = (0 <= target_id <= VITAL_TRACK_ID_MAX) or (
            VITAL_FALLBACK_ID_MIN <= target_id <= VITAL_FALLBACK_ID_MAX
        )
        if not valid_target_id:
            return False
        if not (0 <= rangebin <= VITAL_RANGEBIN_MAX):
            return False
        if not (np.isfinite(deviation) and 0.0 <= deviation <= VITAL_BREATHING_DEVIATION_MAX):
            return False
        if not (np.isfinite(heart) and 0.0 <= heart <= VITAL_HEART_RATE_MAX):
            return False
        if not (np.isfinite(breath) and 0.0 <= breath <= VITAL_BREATHING_RATE_MAX):
            return False
        if any((not np.isfinite(value)) or value < 0.0 or value > VITAL_HEART_RATE_MAX for value in heart_history):
            return False
        if any((not np.isfinite(value)) or value < 0.0 or value > VITAL_BREATHING_RATE_MAX for value in breath_history):
            return False
        return True


def decode_single_frame(frame_bytes: bytes, ts: Optional[float] = None) -> Optional[ParsedFrame]:
    parser = TlvFrameParser()
    return parser.parse_frame(frame_bytes, ts=ts)
