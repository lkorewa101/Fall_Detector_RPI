import pickle
import struct
import tempfile
import unittest
import zipfile
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from main import apply_point_cloud_alignment, select_synced_processing_points
from modules import ai_model as ai_model_module
from modules import fall_detector as fall_detector_module
from modules import pose_estimator as pose_estimator_module
from modules.config_manager import ConfigManager
from modules.csv_stream import CsvStream
from modules.processing import PointCloudProcessor, Track
from modules.tlv_parser import MAGIC_WORD, ParsedFrame, TlvFrameParser
from modules.visualization_manager import VisualizationManager
from tools import evaluate_fall_dataset as fall_eval
from tools import train_fall_frame_model as fall_frame_train
from tools import train_fall_sequence_model as fall_train
from tools import train_pose_pseudo_model as pose_train
from tools.validate_mmwave_dataset import validate_dataset
from tools.validate_vital_datasets import validate_mendeley_vital_dataset


def _build_frame(tlvs):
    payload = bytearray()
    for tlv_type, tlv_payload in tlvs:
        payload.extend(struct.pack("<II", tlv_type, len(tlv_payload)))
        payload.extend(tlv_payload)

    total_len = 40 + len(payload)
    header = bytearray(40)
    header[:8] = MAGIC_WORD
    struct.pack_into("<I", header, 12, total_len)
    struct.pack_into("<I", header, 20, 1234)
    struct.pack_into("<I", header, 28, 7)
    struct.pack_into("<I", header, 32, len(tlvs))
    return bytes(header + payload)


def _build_vital_record(track_id, range_bin, heart_bpm, breath_bpm, deviation=0.01):
    history = [float(heart_bpm)] * 15 + [float(breath_bpm)] * 15
    return struct.pack(
        "<HH" + ("f" * 33),
        int(track_id),
        int(range_bin),
        float(deviation),
        float(heart_bpm),
        float(breath_bpm),
        *history,
    )


class _DummyAI:
    def __init__(self):
        self.enabled = False

    def predict_all(self, track):
        return 0.0, "Unknown", 0.0


class _PickleGoodModel:
    def predict_proba(self, features):
        return np.array([[0.20, 0.80]], dtype=np.float32)


class _PicklePoseModel:
    def predict(self, features):
        rows = np.asarray(features).shape[0]
        return np.zeros((rows, len(pose_estimator_module.JOINT_NAMES) * 3), dtype=np.float32)


class _PickleSequenceModel:
    def predict_proba(self, features):
        rows = np.asarray(features).shape[0]
        return np.tile(np.array([[0.30, 0.70]], dtype=np.float32), (rows, 1))


class _DummyTrack:
    def __init__(self, track_id=1):
        self.id = track_id
        self.position = np.array([0.0, 0.0, 1.25], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.dims = np.array([0.6, 0.5, 1.7], dtype=np.float32)
        self.point_count = 24
        self.points = np.zeros((24, 4), dtype=np.float32)


def _synthetic_replay_frame(frame, z_low, z_high, x_half_span):
    z_vals = np.linspace(z_low, z_high, 16, dtype=np.float32)
    x_vals = np.linspace(-x_half_span, x_half_span, 16, dtype=np.float32)
    return pd.DataFrame(
        {
            "frame": np.full(16, frame, dtype=np.int32),
            "x": x_vals,
            "y": np.full(16, 2.0, dtype=np.float32),
            "z": z_vals,
            "v": np.zeros(16, dtype=np.float32),
        }
    )


class TiAlgorithmTests(unittest.TestCase):
    def test_point_cloud_alignment_applies_to_points_only(self):
        points = np.array([[0.0, 2.0, 0.0, 0.5]], dtype=np.float32)

        aligned = apply_point_cloud_alignment(points, tilt_degrees=0.0, height_offset=0.25)

        np.testing.assert_allclose(aligned[:, :3], [[0.0, 2.0, 0.25]], atol=1e-6)
        np.testing.assert_allclose(points[:, :3], [[0.0, 2.0, 0.0]], atol=1e-6)

    def test_point_cloud_alignment_uses_downward_tilt_sign(self):
        points = np.array([[0.0, 2.0, -0.5, 0.0]], dtype=np.float32)

        aligned = apply_point_cloud_alignment(points, tilt_degrees=15.0, height_offset=1.8)

        expected_y = 2.0 * np.cos(np.radians(15.0)) + (-0.5) * np.sin(np.radians(15.0))
        expected_z = -2.0 * np.sin(np.radians(15.0)) + (-0.5) * np.cos(np.radians(15.0)) + 1.8
        np.testing.assert_allclose(aligned[:, :3], [[0.0, expected_y, expected_z]], atol=1e-6)
        self.assertLess(float(aligned[0, 2]), 1.8)

    def test_motion_track_color_uses_doppler_when_enabled(self):
        class DummyTrack:
            state_label = "Normal"
            missing = 0
            doppler_mean = -0.6
            velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        color = VisualizationManager._track_color(DummyTrack(), motion_color=True)

        self.assertGreater(color[0], color[2])

    def test_parser_marks_tracker_tlvs_as_lagged(self):
        parser = TlvFrameParser()
        track_payload = struct.pack("<I9f", 7, 1.0, 2.0, 3.0, 0.1, 0.2, -0.3, 0.0, 0.0, 0.0)
        index_payload = bytes([7, 255, 255])
        frame = _build_frame(
            [
                (1010, track_payload),
                (1011, index_payload),
            ]
        )

        parsed = parser.parse_frame(frame, ts=123.0)

        self.assertIsInstance(parsed, ParsedFrame)
        self.assertEqual(parsed.tracks_lag, 1)
        self.assertEqual(parsed.point_track_ids_lag, 1)
        self.assertTrue(parsed.tracks_are_world)
        self.assertEqual(int(parsed.tracks[0]["id"]), 7)
        self.assertEqual(parsed.frame_number, 1234)
        self.assertEqual(parsed.num_detected_obj, 7)
        self.assertEqual(parsed.num_tlvs, 2)
        self.assertEqual(parsed.tlv_types, [1010, 1011])

    def test_parser_reads_multi_subject_vital_tlv(self):
        parser = TlvFrameParser()
        vital_payload = b"".join(
            [
                _build_vital_record(1, 8, 82.5, 18.0, 0.010),
                _build_vital_record(2, 13, 91.0, 20.5, 0.020),
                _build_vital_record(3, 18, 0.0, 0.0, 0.000),
                _build_vital_record(4, 22, 74.25, 15.5, 0.015),
            ]
        )
        frame = _build_frame([(0x410, vital_payload)])

        parsed = parser.parse_frame(frame, ts=123.0)

        self.assertIsInstance(parsed, ParsedFrame)
        self.assertEqual(len(parsed.vitals), 4)
        self.assertEqual(parsed.vitals[0]["id"], 1)
        self.assertEqual(parsed.vitals[1]["range_bin"], 13)
        self.assertAlmostEqual(parsed.vitals[1]["heart_bpm"], 91.0, places=3)
        self.assertAlmostEqual(parsed.vitals[1]["breath_bpm"], 20.5, places=3)
        self.assertAlmostEqual(parsed.vitals[1]["signal_deviation"], 0.020, places=3)

    def test_parser_accepts_vital_tlv_before_points_and_tracks(self):
        parser = TlvFrameParser()
        vital_payload = _build_vital_record(1, 8, 82.5, 18.0, 0.010)
        point_payload = struct.pack("<4f", 0.25, 2.0, 1.1, 0.05)
        track_payload = struct.pack("<I9f", 1, 0.25, 2.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        frame = _build_frame(
            [
                (0x410, vital_payload),
                (1, point_payload),
                (1010, track_payload),
            ]
        )

        parsed = parser.parse_frame(frame, ts=123.0)

        self.assertIsInstance(parsed, ParsedFrame)
        self.assertEqual(parsed.tlv_types, [0x410, 1, 1010])
        self.assertEqual(len(parsed.vitals), 1)
        self.assertEqual(len(parsed.tracks), 1)
        np.testing.assert_allclose(parsed.points, [[0.25, 2.0, 1.1, 0.05]], atol=1e-6)

    def test_parser_filters_invalid_vital_records(self):
        parser = TlvFrameParser()
        vital_payload = b"".join(
            [
                _build_vital_record(1, 8, 82.5, 18.0, 0.010),
                _build_vital_record(19525, 17054, 0.0, 0.0, 10.0),
            ]
        )
        frame = _build_frame([(0x410, vital_payload)])

        parsed = parser.parse_frame(frame, ts=123.0)

        self.assertIsInstance(parsed, ParsedFrame)
        self.assertEqual(len(parsed.vitals), 1)
        self.assertEqual(parsed.vitals[0]["id"], 1)
        self.assertEqual(parsed.invalid_vital_records, 1)
        self.assertEqual(parsed.to_jsonable()["invalidVitalRecords"], 1)

    def test_config_requires_six_people_tracking_for_current_firmware(self):
        one_track_cfg = "sensorPosition 1.8 0 15\ntrackingCfg 1 2 800 1 46 96 90\nvitalsign 15 300\n"
        six_track_cfg = "sensorPosition 1.8 0 15\ntrackingCfg 1 2 800 6 46 96 90\nvitalsign 15 300\n"

        self.assertFalse(ConfigManager._is_current_firmware_content(one_track_cfg))
        self.assertTrue(ConfigManager._is_current_firmware_content(six_track_cfg))
        self.assertEqual(ConfigManager._tracking_max_tracks(six_track_cfg), 6)

    def test_fall_ai_falls_back_when_first_model_candidate_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = f"{tmpdir}/bad.pkl"
            good_path = f"{tmpdir}/good.pkl"
            with open(bad_path, "wb") as handle:
                handle.write(b"not a pickle")
            with open(good_path, "wb") as handle:
                pickle.dump(_PickleGoodModel(), handle)

            with mock.patch.object(
                ai_model_module.FallAI,
                "_candidate_model_paths",
                return_value=[ai_model_module.Path(bad_path), ai_model_module.Path(good_path)],
            ):
                ai = ai_model_module.FallAI()

        self.assertTrue(ai.enabled)
        self.assertTrue(str(ai.model_path).endswith("good.pkl"))

    def test_csv_stream_reads_cartesian_point_columns(self):
        frame_data = pd.DataFrame(
            {
                "x": [0.25, -0.5],
                "y": [2.0, 1.5],
                "z": [1.1, 0.7],
                "doppler": [0.05, -0.2],
            }
        )

        points = CsvStream._points_from_frame(frame_data)

        self.assertEqual(points.shape, (2, 4))
        np.testing.assert_allclose(points, [[0.25, 2.0, 1.1, 0.05], [-0.5, 1.5, 0.7, -0.2]], atol=1e-6)

    def test_csv_stream_converts_ti_spherical_point_columns(self):
        frame_data = pd.DataFrame(
            {
                "range": [2.0, 2.0],
                "azimuth": [0.0, 90.0],
                "elevation": [0.0, 0.0],
                "doppler": [0.25, -0.1],
            }
        )

        points = CsvStream._points_from_frame(frame_data)

        self.assertEqual(points.shape, (2, 4))
        np.testing.assert_allclose(points[0], [0.0, 2.0, 0.0, 0.25], atol=1e-6)
        np.testing.assert_allclose(points[1], [2.0, 0.0, 0.0, -0.1], atol=1e-5)

    def test_csv_stream_converts_zenodo_wide_list_point_columns(self):
        frame_data = pd.DataFrame(
            {
                "frameno": [2704],
                "elevation_0": ["[0.0, 0.0]"],
                "azimuth_0": ["[0.0, 90.0]"],
                "doppler_0": ["[0.25, -0.1]"],
                "range_0": ["[2.0, 2.0]"],
                "snr_0": ["[10.0, 11.0]"],
            }
        )

        points = CsvStream._points_from_frame(frame_data)

        self.assertEqual(points.shape, (2, 4))
        np.testing.assert_allclose(points[0], [0.0, 2.0, 0.0, 0.25], atol=1e-6)
        np.testing.assert_allclose(points[1], [2.0, 0.0, 0.0, -0.1], atol=1e-5)

    def test_csv_stream_accepts_ti_frame_column_alias(self):
        df = pd.DataFrame({"framenumber": [10, 10], "range": [1.0, 1.2], "azimuth": [0.0, 0.0]})

        self.assertEqual(
            CsvStream._first_column(df, ("frame", "framenum", "framenumber", "frame_number", "frameid", "frame_id")),
            "framenumber",
        )
        self.assertEqual(
            CsvStream._first_column(df.rename(columns={"framenumber": "frameno"}), ("frame", "frameno", "framenum", "framenumber")),
            "frameno",
        )

    def test_csv_stream_reads_pointcloud_csv_from_zip(self):
        header = "frameNo,elevation_0,azimuth_0,doppler_0,range_0,snr_0\n"
        valid = header + '2704,"[0.0]","[0.0]","[0.25]","[2.0]","[10.0]"\n'
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = f"{tmpdir}/dataset.zip"
            member = "DATASET/IWR6843_Data/session_IWR6843_TLV_1020_pointcloud.csv"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("DATASET/IWR6843_Data/empty_IWR6843_TLV_1020_pointcloud.csv", header)
                archive.writestr(member, valid)

            df, source = CsvStream._read_csv_source(zip_path)
            self.assertEqual(len(df), 1)
            self.assertTrue(source.endswith(member))
            df.columns = df.columns.str.strip().str.lower()
            points = CsvStream._points_from_frame(df)
            np.testing.assert_allclose(points, [[0.0, 2.0, 0.0, 0.25]], atol=1e-6)

            df_specific, source_specific = CsvStream._read_csv_source(f"{zip_path}!{member}")
            self.assertEqual(len(df_specific), 1)
            self.assertTrue(source_specific.endswith(member))

    def test_csv_stream_reads_generic_cartesian_csv_from_zip(self):
        valid = "frame,DetObj#,x,y,z,v,snr,noise\n0,0,0.25,2.0,1.1,0.05,20,30\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = f"{tmpdir}/sareebali_style.zip"
            member = "GatheredData/Fall/sample.csv"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("README.csv", "not,a,point,file\n1,2,3,4\n")
                archive.writestr(member, valid)

            df, source = CsvStream._read_csv_source(zip_path)

        self.assertEqual(len(df), 1)
        self.assertTrue(source.endswith(member))
        df.columns = df.columns.str.strip().str.lower()
        np.testing.assert_allclose(CsvStream._points_from_frame(df), [[0.25, 2.0, 1.1, 0.05]], atol=1e-6)

    def test_dataset_validator_recurses_zip_archives_in_directory(self):
        valid = "frame,DetObj#,x,y,z,v,snr,noise\n0,0,0.25,2.0,1.1,0.05,20,30\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = f"{tmpdir}/sareebali_style.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("GatheredData/Fall/sample.csv", valid)

            report = validate_dataset(Path(tmpdir), max_files=5, max_rows=20, max_frames=10)

        self.assertEqual(report["sources_found"], 1)
        self.assertEqual(report["sources_ok"], 1)
        self.assertEqual(report["formats"], {"cartesian": 1})

    def test_vital_dataset_validator_pairs_adc_with_two_reference_targets(self):
        rows = ["Column1,Column2\n"]
        for idx in range(800):
            ecg = -30.0 if idx % 80 == 0 else -1.0
            pcg = 20.0 * np.sin(idx / 30.0)
            rows.append(f"{ecg:.3f},{pcg:.3f}\n")
        csv_text = "".join(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adc_dir = root / "1_AsymmetricalPosition" / "1_Radar_Raw_Data" / "position_ (1)"
            t1_dir = root / "1_AsymmetricalPosition" / "2_Log_data" / "Target1" / "position_ (1)"
            t2_dir = root / "1_AsymmetricalPosition" / "2_Log_data" / "Target2" / "position_ (1)"
            adc_dir.mkdir(parents=True)
            t1_dir.mkdir(parents=True)
            t2_dir.mkdir(parents=True)
            (adc_dir / "adc_2GHZ_positon1_ (1).bin").write_bytes(b"\x00\x01\x02\x03" * 16)
            (t1_dir / "log_Target1_2GHZ_position1_ (1).csv").write_text(csv_text, encoding="utf-8")
            (t2_dir / "log_Target2_2GHZ_position1_ (1).csv").write_text(csv_text, encoding="utf-8")

            report = validate_mendeley_vital_dataset(root)

        self.assertEqual(report["cases_found"], 1)
        self.assertEqual(report["complete_cases"], 1)
        self.assertEqual(report["adc_files"], 1)
        self.assertEqual(report["reference_log_files"], 2)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["reference_rate_summary"]["heart_bpm"]["count"], 2)

    def test_parser_ignores_malformed_vital_payload_only(self):
        parser = TlvFrameParser()
        track_payload = struct.pack("<I9f", 7, 1.0, 2.0, 3.0, 0.1, 0.2, -0.3, 0.0, 0.0, 0.0)
        frame = _build_frame([(1010, track_payload), (0x410, b"\x00" * 16)])

        parsed = parser.parse_frame(frame, ts=123.0)

        self.assertIsInstance(parsed, ParsedFrame)
        self.assertEqual(len(parsed.tracks), 1)
        self.assertEqual(parsed.vitals, [])

    def test_sync_uses_previous_points_for_lagged_tracks(self):
        previous = np.array([[1.0, 2.0, 3.0, 0.1]], dtype=np.float32)
        current = np.array([[9.0, 8.0, 7.0, 0.2]], dtype=np.float32)

        display_points, association_points = select_synced_processing_points(
            incoming_points=current,
            prev_points=previous,
            sensor_tracks=[{"id": 1}],
            tracks_lag=1,
            point_track_ids=np.array([1], dtype=np.uint8),
            point_track_ids_lag=1,
            prev_association_points=previous,
        )

        np.testing.assert_allclose(display_points, previous)
        np.testing.assert_allclose(association_points, previous)

    def test_processor_prefers_exact_points_and_keeps_gap_fill_cluster(self):
        processor = PointCloudProcessor(eps=0.45, min_samples=3)
        cluster_a = np.array(
            [
                [0.00, 2.00, 1.00, 0.10],
                [0.10, 2.05, 1.02, 0.10],
                [-0.08, 1.95, 0.98, 0.09],
                [0.04, 2.10, 1.01, 0.08],
            ],
            dtype=np.float32,
        )
        cluster_b = np.array(
            [
                [3.00, 2.00, 1.00, 0.00],
                [3.10, 2.05, 1.02, 0.01],
                [2.92, 1.95, 0.99, 0.00],
                [3.05, 2.10, 1.01, -0.01],
            ],
            dtype=np.float32,
        )
        points = np.vstack([cluster_a, cluster_b])
        point_track_ids = np.array([1, 1, 1, 1, 255, 255, 255, 255], dtype=np.uint8)
        sensor_tracks = [{"id": 1, "x": 0.02, "y": 2.02, "z": 1.00, "vx": 0.0, "vy": 0.0, "vz": 0.0}]

        result = processor.process(
            points,
            sensor_tracks=sensor_tracks,
            max_tracks=4,
            point_track_ids=point_track_ids,
            association_points=points,
        )

        sensor_detection = next(det for det in result.detections if det.sensor_track_id == 1)
        gap_fill_detection = next(det for det in result.detections if det.source == "cluster_gap_fill")

        self.assertEqual(sensor_detection.stats["point_assignment_mode"], 3.0)
        self.assertEqual(sensor_detection.points.shape[0], 4)
        self.assertEqual(gap_fill_detection.points.shape[0], 4)

    def test_processor_falls_back_to_matched_cluster_without_point_ids(self):
        processor = PointCloudProcessor(eps=0.45, min_samples=3)
        cluster = np.array(
            [
                [0.00, 2.00, 1.05, 0.10],
                [0.08, 2.02, 1.00, 0.08],
                [-0.05, 1.96, 0.98, 0.07],
                [0.04, 2.08, 1.03, 0.09],
            ],
            dtype=np.float32,
        )
        sensor_tracks = [{"id": 9, "x": 0.01, "y": 2.01, "z": 1.01, "vx": 0.0, "vy": 0.0, "vz": 0.0}]

        result = processor.process(cluster, sensor_tracks=sensor_tracks, max_tracks=2)

        sensor_detection = next(det for det in result.detections if det.sensor_track_id == 9)
        self.assertEqual(sensor_detection.stats["point_assignment_mode"], 2.0)
        self.assertGreaterEqual(sensor_detection.points.shape[0], 4)

    def test_track_skeleton_uses_seven_joint_line_layout(self):
        z_vals = np.linspace(0.0, 1.7, 18, dtype=np.float32)
        points = np.column_stack(
            [
                np.sin(z_vals * 4.0) * 0.12,
                np.full_like(z_vals, 2.0),
                z_vals,
                np.zeros_like(z_vals),
            ]
        ).astype(np.float32)
        track = Track(
            np.array([0.0, 2.0, 0.85], dtype=np.float32),
            np.array([0.55, 0.45, 1.7], dtype=np.float32),
            points,
        )

        skeleton = track.get_skeleton()

        self.assertEqual(skeleton.shape, (12, 3))
        self.assertGreater(float(skeleton[0, 2]), float(skeleton[4, 2]))
        self.assertGreater(float(skeleton[2, 2]), float(skeleton[8, 2]))

    def test_track_skeleton_lies_down_for_flat_point_cloud(self):
        x_vals = np.linspace(-0.9, 0.9, 18, dtype=np.float32)
        points = np.column_stack(
            [
                x_vals,
                np.full_like(x_vals, 2.0),
                np.full_like(x_vals, 0.22),
                np.zeros_like(x_vals),
            ]
        ).astype(np.float32)
        track = Track(
            np.array([0.0, 2.0, 0.25], dtype=np.float32),
            np.array([1.8, 0.45, 0.35], dtype=np.float32),
            points,
        )

        skeleton = track.get_skeleton()

        self.assertEqual(skeleton.shape, (12, 3))
        xy_span = float(np.ptp(skeleton[:, 0]))
        z_span = float(np.ptp(skeleton[:, 2]))
        self.assertGreater(xy_span, z_span)
        self.assertLess(z_span, 0.5)

    def test_track_skeleton_stays_compact_for_sparse_low_cloud(self):
        x_vals = np.linspace(-0.2, 0.2, 8, dtype=np.float32)
        points = np.column_stack(
            [
                x_vals,
                np.full_like(x_vals, 2.0),
                np.full_like(x_vals, 0.35),
                np.zeros_like(x_vals),
            ]
        ).astype(np.float32)
        track = Track(
            np.array([0.0, 2.0, 0.35], dtype=np.float32),
            np.array([0.4, 0.2, 0.2], dtype=np.float32),
            points,
        )

        skeleton = track.get_skeleton()

        self.assertLessEqual(float(np.ptp(skeleton[:, 2])), 1.10)

    def test_pose_skeleton_is_fitted_to_point_extent(self):
        lines = np.array(
            [
                [-10.0, 0.0, 0.0],
                [10.0, 0.0, 2.0],
                [0.0, -10.0, 0.0],
                [0.0, 10.0, 2.0],
            ],
            dtype=np.float32,
        )
        points = np.array(
            [
                [-0.2, 2.0, 0.30, 0.0],
                [0.2, 2.1, 0.35, 0.0],
                [0.0, 1.9, 0.40, 0.0],
            ],
            dtype=np.float32,
        )

        fitted = pose_estimator_module.fit_skeleton_lines_to_points(lines, points)

        self.assertLessEqual(float(np.ptp(fitted[:, 0])), 0.65)
        self.assertLessEqual(float(np.ptp(fitted[:, 1])), 0.45)
        self.assertLessEqual(float(np.ptp(fitted[:, 2])), 0.30)

    def test_track_skeleton_reuses_previous_when_points_are_sparse(self):
        z_vals = np.linspace(0.0, 1.7, 18, dtype=np.float32)
        points = np.column_stack(
            [
                np.zeros_like(z_vals),
                np.full_like(z_vals, 2.0),
                z_vals,
                np.zeros_like(z_vals),
            ]
        ).astype(np.float32)
        track = Track(
            np.array([0.0, 2.0, 0.85], dtype=np.float32),
            np.array([0.55, 0.45, 1.7], dtype=np.float32),
            points,
        )
        first = track.get_skeleton()
        track.points = np.zeros((0, 4), dtype=np.float32)
        track.point_count = 0
        track.display_points_local.clear()

        second = track.get_skeleton()

        np.testing.assert_allclose(second, first, atol=1e-6)

    def test_track_skeleton_limits_large_extremity_jumps(self):
        z_vals = np.linspace(0.0, 1.7, 18, dtype=np.float32)
        points = np.column_stack(
            [
                np.sin(z_vals * 4.0) * 0.12,
                np.full_like(z_vals, 2.0),
                z_vals,
                np.zeros_like(z_vals),
            ]
        ).astype(np.float32)
        track = Track(
            np.array([0.0, 2.0, 0.85], dtype=np.float32),
            np.array([0.55, 0.45, 1.7], dtype=np.float32),
            points,
        )
        first = track.get_skeleton()

        shifted = points.copy()
        shifted[:, 0] += 3.0
        track.points = shifted
        track.point_count = len(shifted)
        track.display_points_local.clear()
        second = track.get_skeleton()

        max_step = float(np.max(np.linalg.norm(second - first, axis=1)))
        self.assertLessEqual(max_step, Track.SKELETON_MAX_JOINT_STEP + 1e-5)

    def test_fall_detector_does_not_trigger_without_upright_baseline(self):
        with mock.patch.object(fall_detector_module, "FallAI", _DummyAI):
            detector = fall_detector_module.HybridFallDetector()
        track = _DummyTrack()
        track.position = np.array([0.0, 0.0, 0.22], dtype=np.float32)
        track.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        track.dims = np.array([1.2, 0.6, 0.45], dtype=np.float32)

        fall_results = []
        for frame_index in range(10):
            is_fall, _, details = detector.detect(track, frame_index=frame_index)
            fall_results.append(is_fall)
            self.assertFalse(details["should_alert"])

        self.assertFalse(any(fall_results))

    def test_fall_detector_latches_once_after_upright_drop(self):
        with mock.patch.object(fall_detector_module, "FallAI", _DummyAI):
            detector = fall_detector_module.HybridFallDetector()
        track = _DummyTrack()

        for frame_index in range(6):
            track.position = np.array([0.0, 0.0, 1.25], dtype=np.float32)
            track.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            track.dims = np.array([0.6, 0.5, 1.7], dtype=np.float32)
            detector.detect(track, frame_index=frame_index)

        track.position = np.array([0.0, 0.0, 0.80], dtype=np.float32)
        track.velocity = np.array([0.0, 0.0, -1.1], dtype=np.float32)
        track.dims = np.array([0.7, 0.5, 1.05], dtype=np.float32)
        detector.detect(track, frame_index=6)

        alerts = []
        falls = []
        for frame_index in range(7, 15):
            track.position = np.array([0.0, 0.0, 0.22], dtype=np.float32)
            track.velocity = np.array([0.0, 0.0, -0.05], dtype=np.float32)
            track.dims = np.array([1.2, 0.6, 0.45], dtype=np.float32)
            is_fall, _, details = detector.detect(track, frame_index=frame_index)
            alerts.append(bool(details["should_alert"]))
            falls.append(bool(is_fall))

        self.assertTrue(any(falls))
        self.assertEqual(sum(alerts), 1)

    def test_fall_detector_latches_sustained_low_posture_after_baseline_drop(self):
        with mock.patch.object(fall_detector_module, "FallAI", _DummyAI):
            detector = fall_detector_module.HybridFallDetector()
        track = _DummyTrack()

        for frame_index in range(6):
            track.position = np.array([0.0, 0.0, 1.25], dtype=np.float32)
            track.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            track.dims = np.array([0.6, 0.5, 1.7], dtype=np.float32)
            detector.detect(track, frame_index=frame_index)

        falls = []
        strong_flags = []
        for frame_index in range(6, 48):
            progress = min(1.0, (frame_index - 6) / 32.0)
            center_z = 1.25 + ((0.24 - 1.25) * progress)
            height = 1.70 + ((0.42 - 1.70) * progress)
            width = 0.60 + ((1.25 - 0.60) * progress)
            track.position = np.array([0.0, 0.0, center_z], dtype=np.float32)
            track.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            track.dims = np.array([width, 0.65, height], dtype=np.float32)
            is_fall, _, details = detector.detect(track, frame_index=frame_index)
            falls.append(bool(is_fall))
            strong_flags.append(bool(details["strong_posture_fall"]))

        self.assertTrue(any(falls))
        self.assertTrue(any(strong_flags))

    def test_fall_dataset_evaluator_detects_synthetic_fall_replay(self):
        frames = []
        for frame in range(8):
            frames.append(_synthetic_replay_frame(frame, 0.35, 1.95, 0.20))
        frames.append(_synthetic_replay_frame(8, 0.20, 1.20, 0.35))
        for frame in range(9, 18):
            frames.append(_synthetic_replay_frame(frame, 0.05, 0.35, 0.95))
        df = pd.concat(frames, ignore_index=True)

        with mock.patch.object(fall_detector_module, "FallAI", _DummyAI):
            result = fall_eval.evaluate_dataframe(
                df,
                source_label="dataset/GatheredData/Fall/sample.csv",
                max_frames=30,
                min_points=3,
                frame_delta_seconds=0.1,
            )

        self.assertEqual(result["label"], "fall")
        self.assertEqual(result["predicted"], "fall")
        self.assertGreater(result["fall_frame_count"], 0)

    def test_fall_dataset_evaluator_keeps_synthetic_normal_replay_negative(self):
        df = pd.concat(
            [_synthetic_replay_frame(frame, 0.35, 1.95, 0.20) for frame in range(18)],
            ignore_index=True,
        )

        with mock.patch.object(fall_detector_module, "FallAI", _DummyAI):
            result = fall_eval.evaluate_dataframe(
                df,
                source_label="dataset/GatheredData/NonFall/sample.csv",
                max_frames=30,
                min_points=3,
                frame_delta_seconds=0.1,
            )

        self.assertEqual(result["label"], "non_fall")
        self.assertEqual(result["predicted"], "non_fall")
        self.assertEqual(result["fall_frame_count"], 0)

    def test_fall_dataset_label_inference_uses_zip_member_before_root_folder(self):
        self.assertEqual(
            fall_eval.infer_label(r"sample_data\fall\dataset.zip!GatheredData\Not\sample.csv"),
            "non_fall",
        )
        self.assertEqual(
            fall_eval.infer_label(r"sample_data\fall\dataset.zip!GatheredData\Fall\sample.csv"),
            "fall",
        )

    def test_fall_sequence_training_feature_shape(self):
        df = pd.concat(
            [_synthetic_replay_frame(frame, 0.35, 1.95, 0.20) for frame in range(12)],
            ignore_index=True,
        )

        feature = fall_train.extract_sequence_feature_from_dataframe(
            df,
            max_frames=30,
            min_points=3,
            frame_delta_seconds=0.1,
        )

        self.assertIsNotNone(feature)
        self.assertEqual(len(feature), len(fall_train.sequence_feature_names()))
        self.assertTrue(np.all(np.isfinite(feature)))

    def test_fall_sequence_ai_loads_pickle_and_scores_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "fall_sequence.pkl"
            with model_path.open("wb") as handle:
                pickle.dump(
                    {
                        "model": _PickleSequenceModel(),
                        "feature_names": ai_model_module.sequence_feature_names(),
                        "threshold": 0.6,
                    },
                    handle,
                )

            with mock.patch.dict(os.environ, {"FALL_SEQUENCE_MODEL_PATH": str(model_path)}):
                sequence_ai = ai_model_module.FallSequenceAI()

            history = [
                np.full(len(ai_model_module.SEQUENCE_FRAME_FEATURE_NAMES), frame, dtype=np.float32)
                for frame in range(sequence_ai.min_frames)
            ]
            score = sequence_ai.predict_from_history(history, frame_delta_seconds=0.1)

        self.assertTrue(sequence_ai.enabled)
        self.assertEqual(sequence_ai.threshold, 0.6)
        self.assertAlmostEqual(score, 0.70, places=5)

    def test_fall_frame_training_feature_shape(self):
        points = _synthetic_replay_frame(0, 0.35, 1.95, 0.20)[["x", "y", "z", "v"]].to_numpy(dtype=np.float32)

        feature = fall_frame_train.frame_feature(points)

        self.assertIsNotNone(feature)
        self.assertEqual(len(feature), len(fall_frame_train.FEATURE_NAMES))
        self.assertTrue(np.all(np.isfinite(feature)))

    def test_pose_pseudo_training_target_shape(self):
        points = _synthetic_replay_frame(0, 0.35, 1.95, 0.20)[["x", "y", "z", "v"]].to_numpy(dtype=np.float32)

        joints = pose_train.pseudo_joints_from_points(points)

        self.assertIsNotNone(joints)
        target = pose_train.target_from_joints(joints)
        self.assertEqual(len(target), len(pose_estimator_module.JOINT_NAMES) * 3)
        self.assertTrue(np.all(np.isfinite(target)))

    def test_pose_estimator_loads_sklearn_pickle_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "pose_model.pkl"
            with model_path.open("wb") as handle:
                pickle.dump(
                    {
                        "model": _PicklePoseModel(),
                        "target_scale": pose_estimator_module.TARGET_SCALE,
                    },
                    handle,
                )

            estimator = pose_estimator_module.PoseEstimator(model_path=model_path)
            points = _synthetic_replay_frame(0, 0.35, 1.95, 0.20)[["x", "y", "z", "v"]].to_numpy(dtype=np.float32)
            skeleton = estimator.predict_skeleton(points)

        self.assertTrue(estimator.available)
        self.assertEqual(estimator.backend, "sklearn")
        self.assertIsNotNone(skeleton)
        self.assertEqual(skeleton.shape, (len(pose_estimator_module.BONE_PAIRS) * 2, 3))


if __name__ == "__main__":
    unittest.main()
