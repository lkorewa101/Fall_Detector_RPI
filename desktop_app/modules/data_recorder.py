import json
import os
import time
from collections import deque
import numpy as np
from datetime import datetime

class DataRecorder:
    """
    Handles recording of raw radar point cloud data.
    States: IDLE, RECORDING, PAUSED
    """
    STATE_IDLE = 0
    STATE_RECORDING = 1
    STATE_PAUSED = 2

    def __init__(
        self,
        output_dir=None,
        blackbox_pre_seconds=8.0,
        blackbox_post_seconds=4.0,
        blackbox_cooldown_seconds=30.0,
        blackbox_max_bytes=None,
        blackbox_max_events=30,
    ):
        if output_dir is None:
            app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(app_root, "records")
        self.output_dir = os.path.abspath(output_dir)
        self.state = self.STATE_IDLE
        self.buffer = []  # List of tuples: (frame_num, ts, x, y, z, doppler)
        self.raw_buffer = [] # Store raw binary TLV frames
        self.recorded_frames = 0
        self.start_time = 0

        self.blackbox_pre_seconds = float(blackbox_pre_seconds)
        self.blackbox_post_seconds = float(blackbox_post_seconds)
        self.blackbox_cooldown_seconds = float(blackbox_cooldown_seconds)
        self.blackbox_max_bytes = self._resolve_blackbox_max_bytes(blackbox_max_bytes)
        self.blackbox_max_events = int(blackbox_max_events)
        self.pre_event_frames = deque()
        self.active_blackbox_event = None
        self.last_blackbox_trigger_time = 0.0

    def start(self):
        """Starts or resumes recording."""
        if self.state == self.STATE_IDLE:
            self.buffer = []
            self.raw_buffer = []
            self.recorded_frames = 0
            self.start_time = time.time()
        
        self.state = self.STATE_RECORDING

    def pause(self):
        """Pauses the recording."""
        if self.state == self.STATE_RECORDING:
            self.state = self.STATE_PAUSED

    def stop_and_save(self) -> str:
        """Stops recording and saves the buffer to a CSV file."""
        if self.state == self.STATE_IDLE:
            return ""

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_array = np.array(self.buffer, dtype=np.float64) if self.buffer else np.empty((0, 6))
        filepath = self._save_artifacts(f"record_{timestamp_str}", data_array, self.raw_buffer)

        self._reset()
        return filepath

    def cancel(self):
        """Discards the buffer and goes back to IDLE."""
        self._reset()

    def _reset(self):
        self.state = self.STATE_IDLE
        self.buffer = []
        self.raw_buffer = []
        self.recorded_frames = 0
        self.start_time = 0

    def add_points(self, frame_num: int, ts: float, points: np.ndarray, raw_bytes: bytes = b""):
        """
        Adds points from a single frame to the manual recorder and blackbox ring buffer.
        points shape: (N, 4) -> x, y, z, doppler
        raw_bytes: the original TLV binary frame from radar (if any)
        """
        frame = self._make_frame(frame_num, ts, points, raw_bytes)
        self._remember_pre_event_frame(frame)
        self._append_active_blackbox_frame(frame)

        if self.state != self.STATE_RECORDING:
            return

        self.recorded_frames += 1
        data_rows = self._frame_to_rows(frame)
        if data_rows.size > 0:
            self.buffer.extend(data_rows.tolist())

        if frame["raw"]:
            self.raw_buffer.append(frame["raw"])

    def trigger_blackbox_event(
        self,
        track_id=None,
        score=None,
        location=None,
        source=None,
        frame_num=None,
        ts=None,
    ) -> str:
        """Starts an automatic fall-event blackbox capture."""
        now = time.time()
        if self.active_blackbox_event is not None:
            return ""
        if now - self.last_blackbox_trigger_time < self.blackbox_cooldown_seconds:
            return ""

        event_ts = self._safe_timestamp(ts if ts is not None else now)
        safe_track = str(track_id if track_id is not None else "unknown").replace(os.sep, "_")
        filename_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"fall_{filename_ts}_track{safe_track}"
        self.active_blackbox_event = {
            "stem": stem,
            "trigger_ts": event_ts,
            "deadline_ts": event_ts + self.blackbox_post_seconds,
            "frames": list(self.pre_event_frames),
            "metadata": {
                "type": "fall_blackbox",
                "createdAt": datetime.now().isoformat(timespec="seconds"),
                "preSeconds": self.blackbox_pre_seconds,
                "postSeconds": self.blackbox_post_seconds,
                "triggerFrame": frame_num,
                "trackId": track_id,
                "score": float(score) if score is not None else None,
                "location": location,
                "source": source,
            },
        }
        self.last_blackbox_trigger_time = now

        if self.blackbox_post_seconds <= 0:
            self._finalize_active_blackbox_event()

        return stem

    def get_status_str(self) -> str:
        """Returns a human-readable string of the current status."""
        if self.state == self.STATE_IDLE:
            return "대기 중 (IDLE)"
        elif self.state == self.STATE_PAUSED:
            return f"일시정지 (PAUSED) - {self.recorded_frames} 프레임"
        elif self.state == self.STATE_RECORDING:
            elapsed = time.time() - self.start_time
            return f"녹화 중 ({elapsed:.1f}s) - {self.recorded_frames} 프레임"
        return "알 수 없음"

    def _make_frame(self, frame_num: int, ts: float, points: np.ndarray, raw_bytes: bytes):
        copied_points = None
        if points is not None and len(points) > 0:
            copied_points = np.asarray(points, dtype=np.float64).copy()

        return {
            "frame_num": int(frame_num),
            "ts": self._safe_timestamp(ts),
            "points": copied_points,
            "raw": bytes(raw_bytes) if raw_bytes else b"",
        }

    def _remember_pre_event_frame(self, frame):
        self.pre_event_frames.append(frame)
        cutoff = frame["ts"] - self.blackbox_pre_seconds
        while self.pre_event_frames and self.pre_event_frames[0]["ts"] < cutoff:
            self.pre_event_frames.popleft()

    def _append_active_blackbox_frame(self, frame):
        if self.active_blackbox_event is None:
            return

        self.active_blackbox_event["frames"].append(frame)
        if frame["ts"] >= self.active_blackbox_event["deadline_ts"]:
            self._finalize_active_blackbox_event()

    def _finalize_active_blackbox_event(self):
        event = self.active_blackbox_event
        if event is None:
            return ""

        self.active_blackbox_event = None
        frames = event["frames"]
        data_array = self._frames_to_rows(frames)
        raw_buffers = [frame["raw"] for frame in frames if frame["raw"]]
        metadata = {
            **event["metadata"],
            "savedAt": datetime.now().isoformat(timespec="seconds"),
            "frameCount": len(frames),
            "pointRows": int(data_array.shape[0]) if data_array.ndim == 2 else 0,
        }
        filepath = self._save_artifacts(event["stem"], data_array, raw_buffers, metadata=metadata)
        self._enforce_blackbox_retention()
        if filepath:
            print(f"[DataRecorder] Saved fall blackbox: {filepath}")
        return filepath

    def _frame_to_rows(self, frame):
        points = frame["points"]
        if points is None or len(points) == 0:
            return np.empty((0, 6), dtype=np.float64)

        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 4:
            return np.empty((0, 6), dtype=np.float64)

        point_values = points[:, :4]
        frame_col = np.full((point_values.shape[0], 1), frame["frame_num"], dtype=np.float64)
        ts_col = np.full((point_values.shape[0], 1), frame["ts"], dtype=np.float64)
        return np.hstack((frame_col, ts_col, point_values))

    def _frames_to_rows(self, frames):
        rows = [self._frame_to_rows(frame) for frame in frames]
        rows = [row for row in rows if row.size > 0]
        if not rows:
            return np.empty((0, 6), dtype=np.float64)
        return np.vstack(rows)

    def _save_artifacts(self, stem: str, data_array, raw_buffers, metadata=None) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        csv_filepath = os.path.join(self.output_dir, f"{stem}.csv")
        saved_path = ""

        if data_array is not None and getattr(data_array, "size", 0) > 0:
            try:
                header = "Frame,Timestamp,X,Y,Z,Doppler"
                np.savetxt(
                    csv_filepath,
                    data_array,
                    delimiter=",",
                    header=header,
                    comments="",
                    fmt="%d,%.3f,%.4f,%.4f,%.4f,%.4f",
                )
                npz_filepath = os.path.join(self.output_dir, f"{stem}.npz")
                np.savez_compressed(npz_filepath, data=data_array)
                saved_path = csv_filepath
                print(f"[DataRecorder] Saved CSV: {csv_filepath}")
                print(f"[DataRecorder] Saved NPZ: {npz_filepath}")
            except Exception as e:
                print(f"[DataRecorder] Error saving data: {e}")

        if raw_buffers:
            bin_filepath = os.path.join(self.output_dir, f"{stem}.bin")
            try:
                with open(bin_filepath, "wb") as f:
                    for raw in raw_buffers:
                        if raw:
                            f.write(raw)
                saved_path = saved_path or bin_filepath
                print(f"[DataRecorder] Saved BIN: {bin_filepath}")
            except Exception as e:
                print(f"[DataRecorder] Error saving BIN: {e}")

        if metadata:
            json_filepath = os.path.join(self.output_dir, f"{stem}.json")
            try:
                with open(json_filepath, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                saved_path = saved_path or json_filepath
                print(f"[DataRecorder] Saved JSON: {json_filepath}")
            except Exception as e:
                print(f"[DataRecorder] Error saving JSON: {e}")

        return saved_path

    def _enforce_blackbox_retention(self):
        groups = {}
        for name in os.listdir(self.output_dir):
            if not name.startswith("fall_"):
                continue
            stem, ext = os.path.splitext(name)
            if ext.lower() not in {".csv", ".npz", ".bin", ".json"}:
                continue
            path = os.path.join(self.output_dir, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            group = groups.setdefault(stem, {"paths": [], "size": 0, "mtime": stat.st_mtime})
            group["paths"].append(path)
            group["size"] += stat.st_size
            group["mtime"] = min(group["mtime"], stat.st_mtime)

        ordered = sorted(groups.values(), key=lambda item: item["mtime"])
        total_size = sum(item["size"] for item in ordered)

        while ordered and (
            total_size > self.blackbox_max_bytes
            or len(ordered) > self.blackbox_max_events
        ):
            item = ordered.pop(0)
            for path in item["paths"]:
                try:
                    os.remove(path)
                    print(f"[DataRecorder] Removed old fall blackbox: {path}")
                except OSError as e:
                    print(f"[DataRecorder] Error removing old blackbox file: {e}")
            total_size -= item["size"]

    @staticmethod
    def _resolve_blackbox_max_bytes(value) -> int:
        if value is None:
            value = os.getenv("FALL_BLACKBOX_MAX_GB", "3")
            multiplier = 1024 ** 3
        else:
            multiplier = 1

        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 3.0
            multiplier = 1024 ** 3

        if parsed <= 0:
            parsed = 3.0
            multiplier = 1024 ** 3
        return int(parsed * multiplier)

    @staticmethod
    def _safe_timestamp(value) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return time.time()
        if not np.isfinite(parsed):
            return time.time()
        return parsed
