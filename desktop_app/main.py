import os
import sys
import time
import socket
import shutil
import tempfile
from pathlib import Path
from collections import deque

# Fix for Qt platform plugin error when project path contains non-ASCII characters
try:
    import PyQt5
    _qt_base_path = Path(PyQt5.__file__).parent / "Qt5"
    _qt_platforms_src = _qt_base_path / "plugins" / "platforms"
    _qt_bin_path = _qt_base_path / "bin"
    
    if _qt_platforms_src.exists():
        _qt_tmp = Path(tempfile.gettempdir()) / "pyqt5_plugins"
        _qt_tmp_platforms = _qt_tmp / "platforms"
        
        if not _qt_tmp_platforms.exists() or not (_qt_tmp_platforms / "qwindows.dll").exists():
            if _qt_tmp.exists():
                shutil.rmtree(_qt_tmp, ignore_errors=True)
            _qt_tmp.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(_qt_platforms_src), str(_qt_tmp_platforms))
            
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_qt_tmp)
        os.environ["QT_PLUGIN_PATH"] = str(_qt_tmp)
        
        if hasattr(os, "add_dll_directory") and _qt_bin_path.exists():
            os.add_dll_directory(str(_qt_bin_path))
        os.environ["PATH"] = str(_qt_bin_path) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import numpy as np
from PyQt5.QtCore import QSettings, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow

from modules.config_logic import RadarConfigLogic
from modules.config_manager import ConfigManager
from modules.connection_manager import ConnectionManager
from modules.data_recorder import DataRecorder
from modules.fall_detector import HybridFallDetector
from modules.gui_manager import GUIManager
from modules.pose_estimator import PoseEstimator
from modules.processing import PointCloudProcessor, Tracker
from modules.visualization_manager import VisualizationManager
from modules.web_client import WebClient


def dashboard_urls(port=8000):
    addresses = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = item[4][0]
            if address and not address.startswith("127."):
                addresses.add(address)
    except OSError:
        pass

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            address = probe.getsockname()[0]
            if address and not address.startswith("127."):
                addresses.add(address)
        finally:
            probe.close()
    except OSError:
        pass

    return [f"http://{address}:{port}" for address in sorted(addresses)]


def _normalize_point_cloud(points):
    if points is None:
        return np.empty((0, 4), dtype=np.float32)

    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    if arr.shape[1] == 3:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)])
    elif arr.shape[1] > 4:
        arr = arr[:, :4]

    return np.ascontiguousarray(arr, dtype=np.float32)


def select_synced_processing_points(
    incoming_points,
    prev_points=None,
    sensor_tracks=None,
    tracks_lag: int = 0,
    point_track_ids=None,
    point_track_ids_lag: int = 0,
    prev_association_points=None,
):
    incoming_points = _normalize_point_cloud(incoming_points)
    prev_points = _normalize_point_cloud(prev_points)
    prev_association_points = _normalize_point_cloud(prev_association_points)

    has_sensor_tracks = bool(sensor_tracks)
    display_points = incoming_points
    if has_sensor_tracks and int(tracks_lag or 0) == 1 and len(prev_points) > 0:
        display_points = prev_points.copy()

    association_points = None
    if point_track_ids is not None:
        if int(point_track_ids_lag or 0) == 1 and len(prev_association_points) > 0:
            association_points = prev_association_points.copy()
        else:
            association_points = display_points.copy()

    return display_points.copy(), association_points


def apply_point_cloud_alignment(points, tilt_degrees: float, height_offset: float):
    """Convert sensor-frame points to room/world coordinates.

    TI People Tracking configs use a positive sensorPosition tilt for a sensor
    pitched downward toward the floor. To render those points in the room frame,
    rotate by the opposite pitch sign and then add the sensor mounting height.
    """
    aligned = _normalize_point_cloud(points).copy()
    if aligned.shape[0] == 0:
        return aligned

    if abs(float(tilt_degrees)) < 1e-6 and abs(float(height_offset)) < 1e-6:
        return aligned

    pitch_rad = np.radians(float(tilt_degrees))
    cos_p = np.cos(pitch_rad)
    sin_p = np.sin(pitch_rad)
    y = aligned[:, 1].copy()
    z = aligned[:, 2].copy()
    aligned[:, 1] = y * cos_p + z * sin_p
    aligned[:, 2] = -y * sin_p + z * cos_p + float(height_offset)
    return np.ascontiguousarray(aligned, dtype=np.float32)


class MainWindow(QMainWindow):
    data_signal = pyqtSignal(object)
    log_signal = pyqtSignal(object)
    command_signal = pyqtSignal(object)
    stream_state_signal = pyqtSignal(object)
    playback_progress_signal = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fall Detection System v1.9")
        self.resize(1400, 900)

        self.tracker = Tracker(max_missing=8)
        self.processor = PointCloudProcessor()
        self.detector = HybridFallDetector()
        self.web_client = WebClient()
        self.dashboard_url_text = "\n".join(dashboard_urls()) or "http://<device-ip>:8000"
        self.config_logic = RadarConfigLogic()
        self.calculated_params = None
        self.stream = None
        self.runtime_log_path = Path(__file__).with_name("desktop_runtime.log")
        self.recorder = DataRecorder()

        self.gui_manager = GUIManager(self)
        self.viz_manager = VisualizationManager(self)
        self.conn_manager = ConnectionManager(self)
        self.cfg_manager = ConfigManager(self)

        self.is_connected = False
        self.is_connecting = False
        self.frame_count = 0
        self.last_alert_time = 0.0
        self.current_cfg_content = ""
        self.last_data_time = time.time()
        self.stream_started_at = 0.0
        self.stream_stall_seconds = 30.0
        self.stream_restart_cooldown = 60.0
        self.last_stream_restart_time = 0.0
        self.stream_grace_until = 0.0
        self.stream_restart_in_progress = False
        self.stream_auto_reconnect = False
        self.stream_restart_attempts = 0
        self.max_stream_restart_attempts = 0
        self.last_stream_event = {}
        self.last_stale_log_time = 0.0
        self.last_board_status = ""
        self.last_board_status_at = 0.0

        self.latest_points = np.empty((0, 4), dtype=np.float32)
        self.latest_tracks = []
        self.latest_sensor_tracks = []
        self.latest_vitals = []
        self.latest_vitals_by_id = {}
        self.latest_frame_meta = {}
        self.latest_details = {}
        self.latest_detections = []
        self.latest_pose_skeleton_lines = None
        self.latest_pose_skeleton_source = ""
        self.point_history = deque(maxlen=100)
        self.prev_association_points = None
        self.prev_frame_points = None
        self.viz_trail_length = 0
        self.viz_point_size = 5.0
        self.viz_color_by_speed = False
        self.viz_doppler_color_scale = 4.3
        self.viz_cluster_dist_filter = False
        self.viz_cluster_dist_threshold = 0.5
        self.pt_min_height = -0.3
        self.pt_max_height = 3.0
        self.point_height_filter_enabled = False
        self.body_ratios = {"head": 0.15, "legs": 0.15}
        self.gravity_bias = True
        self.max_tracks = 6
        self.sensor_tilt = 0.0
        self.sensor_height = 0.0
        self.inference_skip_count = 0 # Counter for AI throttling
        self.last_status_ui_update = 0.0

        self.gui_manager.init_ui()
        self.cfg_manager.load_cfg_file()
        self.conn_manager.scan_ports()

        self.web_client.register_command_callback(self.handle_web_command)
        self.web_client.connect_websocket()

        self.data_signal.connect(self.update_data)
        self.log_signal.connect(self._append_gui_log)
        self.command_signal.connect(self.process_command_signal)
        self.stream_state_signal.connect(self.handle_stream_state)
        self.playback_progress_signal.connect(self.update_playback_progress)
        self.pose_estimator = PoseEstimator(log_callback=self.log_signal.emit)

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.send_heartbeat)
        self.status_timer.start(1000)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_view)
        self.update_timer.start(33)

        self.track_items = {}

        self.update_viz_params()
        self.update_ai_params()
        self._update_rec_ui()
        self.gui_manager.switch_to_clock()

    def load_cfg_file(self):
        self.cfg_manager.load_cfg_file()

    def browse_cfg_file(self):
        self.cfg_manager.browse_cfg_file()

    def browse_csv_file(self):
        self.conn_manager.browse_csv_file()

    def scan_ports(self):
        self.conn_manager.scan_ports()

    def toggle_connection(self):
        now = time.time()
        starting = not self.is_connected and not self.is_connecting
        self.last_data_time = now
        if starting:
            self.stream_restart_attempts = 0
            self.stream_started_at = 0.0
            self.stream_grace_until = now + 45.0
            self.stream_auto_reconnect = True
        else:
            self.stream_grace_until = 0.0
            if not self.stream_restart_in_progress:
                self.stream_auto_reconnect = False
        self.prev_association_points = None
        self.prev_frame_points = None
        self.latest_points = np.empty((0, 4), dtype=np.float32)
        self.point_history.clear()
        self.conn_manager.toggle_connection()

    def refresh_view(self):
        self.viz_manager.refresh_view()

    def switch_to_monitor(self):
        self.gui_manager.switch_to_monitor()

    def switch_to_clock(self):
        self.gui_manager.switch_to_clock()

    def toggle_web_client(self, state):
        self.gui_manager.toggle_web_client(state)

    def update_sensitivity(self):
        self.gui_manager.update_sensitivity()

    def closeEvent(self, event):
        clock_page = getattr(self, "clock_page", None)
        env_sensor = getattr(clock_page, "env_sensor", None)
        if env_sensor is not None:
            env_sensor.close()
        self.gui_manager.close_event(event)

    def log(self, message):
        self.log_signal.emit(str(message))

    def _append_gui_log(self, message):
        text = str(message)
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.runtime_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {text}\n")
        except Exception:
            pass
        self.gui_manager.log(text)

    def apply_settings(self, speed, height):
        self.gui_manager.apply_settings(speed, height)

    def update_viz_params(self):
        if hasattr(self, "enable_clustering_check"):
            self.enable_clustering = self.enable_clustering_check.isChecked()
        else:
            self.enable_clustering = True

        self.viz_point_size = float(self.pt_size_slider.value())
        self.viz_trail_length = self.trail_slider.value()
        self.viz_color_by_speed = True
        self.pt_size_label.setText(f"점 크기: {int(self.viz_point_size)}")
        self.trail_label.setText(f"화면 잔상 (Trace 프레임): {self.viz_trail_length}")

        if hasattr(self, "doppler_color_scale_spin"):
            self.viz_doppler_color_scale = float(self._clamp_spin_value(self.doppler_color_scale_spin, 0.05, 5.0))

        if hasattr(self, "cluster_dist_filter_check"):
            self.viz_cluster_dist_filter = self.cluster_dist_filter_check.isChecked()
            self.viz_cluster_dist_threshold = float(self._clamp_spin_value(self.cluster_dist_spin, 0.1, 5.0))

        if hasattr(self, "pt_min_height_spin") and hasattr(self, "pt_max_height_spin"):
            if hasattr(self, "point_height_filter_check"):
                self.point_height_filter_enabled = self.point_height_filter_check.isChecked()
            new_min_height = float(self._clamp_spin_value(self.pt_min_height_spin, -2.0, 10.0))
            new_max_height = float(self._clamp_spin_value(self.pt_max_height_spin, 0.0, 10.0))
            if new_min_height >= new_max_height:
                self.log(
                    f"[height] Invalid Z range ignored: min={new_min_height:.2f}m, "
                    f"max={new_max_height:.2f}m"
                )
                self.pt_min_height_spin.blockSignals(True)
                self.pt_max_height_spin.blockSignals(True)
                self.pt_min_height_spin.setValue(self.pt_min_height)
                self.pt_max_height_spin.setValue(self.pt_max_height)
                self.pt_min_height_spin.blockSignals(False)
                self.pt_max_height_spin.blockSignals(False)
            else:
                self.pt_min_height = new_min_height
                self.pt_max_height = new_max_height
            for widget_name in ("pt_min_height_spin", "pt_max_height_spin"):
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    widget.setEnabled(self.point_height_filter_enabled)

        if hasattr(self, "eps_spin") and hasattr(self, "min_samples_spin"):
            self.processor.update_params(
                self._clamp_spin_value(self.eps_spin, 0.1, 2.0),
                int(self._clamp_spin_value(self.min_samples_spin, 2, 9999)),
                min_point_height=self.pt_min_height,
                max_point_height=self.pt_max_height,
                enable_height_filter=self.point_height_filter_enabled,
            )

        if hasattr(self, "head_ratio_slider") and hasattr(self, "leg_ratio_slider"):
            head_ratio = self.head_ratio_slider.value() / 100.0
            leg_ratio = self.leg_ratio_slider.value() / 100.0
            self.body_ratios = {"head": head_ratio, "legs": leg_ratio}
            self.head_ratio_label.setText(f"머리: {int(head_ratio * 100)}%")
            self.leg_ratio_label.setText(f"다리: {int(leg_ratio * 100)}%")
            if hasattr(self, "arm_ratio_label") and hasattr(self, "arm_ratio_slider"):
                arm_ratio = self.arm_ratio_slider.value() / 100.0
                self.arm_ratio_label.setText(f"팔: {int(arm_ratio * 100)}%")

        if hasattr(self, "gravity_check"):
            self.gravity_bias = self.gravity_check.isChecked()

        if hasattr(self, "max_tracks_spin"):
            self.max_tracks = int(self._clamp_spin_value(self.max_tracks_spin, 1, 9999))
            if hasattr(self, "tracker"):
                self.tracker.max_tracks = self.max_tracks

        self.viz_manager.refresh_view()

    def update_tracker_params(self):
        if hasattr(self, "max_missing_spin"):
            self.tracker.max_missing = int(self._clamp_spin_value(self.max_missing_spin, 1, 150))
        if hasattr(self, "predict_missing_motion_check"):
            self.tracker.predict_missing_motion_enabled = self.predict_missing_motion_check.isChecked()
        if hasattr(self, "lost_descent_fall_check"):
            self.detector.enable_lost_descent_fall = self.lost_descent_fall_check.isChecked()
        if hasattr(self, "lost_descent_missing_spin"):
            self.detector.lost_descent_missing_frames = int(
                self._clamp_spin_value(self.lost_descent_missing_spin, 1, 150)
            )
        if hasattr(self, "lost_descent_window_spin"):
            self.detector.lost_descent_window_frames = int(
                self._clamp_spin_value(self.lost_descent_window_spin, 1, 300)
            )
        if hasattr(self, "lost_descent_drop_spin"):
            self.detector.lost_descent_min_center_drop = float(
                self._clamp_spin_value(self.lost_descent_drop_spin, 0.01, 10.0)
            )

    def update_ai_params(self):
        if hasattr(self, "ai_interval_spin"):
            self.detector.ai_min_interval_frames = int(self._clamp_spin_value(self.ai_interval_spin, 1, 30))
        if hasattr(self, "ai_strict_mode_check"):
            self.detector.use_rule_candidate_filter = self.ai_strict_mode_check.isChecked()
        if hasattr(self, "falling_prob_spin"):
            falling_prob = self._clamp_spin_value(self.falling_prob_spin, 1, 100)
            if hasattr(self, "falling_prob_slider") and self.falling_prob_slider.value() != int(falling_prob):
                self.falling_prob_slider.blockSignals(True)
                self.falling_prob_slider.setValue(int(falling_prob))
                self.falling_prob_slider.blockSignals(False)
            self.detector.ai_falling_threshold = falling_prob / 100.0
        if hasattr(self, "ai_model_combo") and self.ai_model_combo.isEnabled():
            selected_model = self.ai_model_combo.currentData()
            if selected_model:
                selected_abs = os.path.abspath(str(selected_model))
                failed_paths = getattr(self, "_failed_ai_model_paths", set())
                if selected_abs in failed_paths:
                    return
                current_path = getattr(self.detector.ai_model, "model_path", None)
                current_text = str(current_path) if current_path else ""
                if str(selected_model) != current_text:
                    loaded = self.detector.ai_model.set_model_path(str(selected_model))
                    settings = QSettings("FallDetector", "RadarApp")
                    if loaded:
                        self.detector.state_by_track.clear()
                        settings.setValue("selected_ai_model_path", str(selected_model))
                    else:
                        active_path = getattr(self.detector.ai_model, "model_path", None)
                        active_text = str(active_path) if active_path else ""
                        if active_text:
                            settings.setValue("selected_ai_model_path", active_text)
                            for index in range(self.ai_model_combo.count()):
                                item_path = self.ai_model_combo.itemData(index)
                                if item_path and os.path.abspath(str(item_path)) == os.path.abspath(active_text):
                                    self.ai_model_combo.blockSignals(True)
                                    self.ai_model_combo.setCurrentIndex(index)
                                    self.ai_model_combo.blockSignals(False)
                                    break
                        failed_paths.add(selected_abs)
                        self._failed_ai_model_paths = failed_paths
                    status = "loaded" if loaded else "failed; keeping previous"
                    if hasattr(self, "log"):
                        self.log(f"AI model {status}: {selected_model}")

    def update_calibration_params(self):
        if hasattr(self, "sensor_tilt_spin") and hasattr(self, "sensor_height_spin"):
            self.sensor_tilt = float(self._clamp_spin_value(self.sensor_tilt_spin, -90.0, 90.0))
            self.sensor_height = float(self._clamp_spin_value(self.sensor_height_spin, -10.0, 10.0))
        if hasattr(self, "enable_firmware_box_height_check"):
            self.enable_firmware_box_height = self.enable_firmware_box_height_check.isChecked()
        else:
            self.enable_firmware_box_height = False

    def _clamp_spin_value(self, widget, minimum, maximum):
        value = widget.value()
        clamped = max(minimum, min(maximum, value))
        if clamped != value:
            widget.blockSignals(True)
            widget.setValue(clamped)
            widget.blockSignals(False)
        return clamped

    def update_playback_progress(self, current, total):
        if not hasattr(self, "timeline_label") or not hasattr(self, "timeline_slider"):
            return

        self.timeline_label.setText(f"{current} / {total}")
        self.timeline_slider.blockSignals(True)
        percent = int((current / total) * 100) if total > 0 else 0
        self.timeline_slider.setValue(percent)
        self.timeline_slider.blockSignals(False)

        if current >= total and total > 0 and hasattr(self, "play_pause_btn"):
            self.play_pause_btn.setText("▶ 재생")
            self.play_pause_btn.setStyleSheet("background-color: #d32f2f; color: white;")

    def toggle_playback(self):
        if not self.stream or not hasattr(self.stream, "toggle_pause"):
            return

        if self.stream.current_idx >= self.stream.total_frames and self.stream.total_frames > 0:
            self.stream.seek(0)

        self.stream.toggle_pause()
        if self.stream.paused:
            self.play_pause_btn.setText("▶ 재생")
            self.play_pause_btn.setStyleSheet("background-color: #d32f2f; color: white;")
        else:
            self.play_pause_btn.setText("⏸ 일시정지")
            self.play_pause_btn.setStyleSheet("")

    def seek_playback(self, value):
        if not self.stream or not hasattr(self.stream, "seek"):
            return

        self.stream.seek(value)
        if value < 100 and self.stream.paused:
            self.play_pause_btn.setText("▶ 재생")
            self.play_pause_btn.setStyleSheet("background-color: #d32f2f; color: white;")

    def replay_playback(self):
        if not self.stream or not hasattr(self.stream, "seek"):
            return

        self.stream.seek(0)
        if self.stream.paused:
            self.toggle_playback()

    def change_playback_speed(self, index):
        if not self.stream or not hasattr(self.stream, "set_speed"):
            return

        speeds = [0.1, 0.5, 1.0, 1.5, 2.0]
        if 0 <= index < len(speeds):
            self.stream.set_speed(speeds[index])

    def on_rec_start(self):
        self.recorder.start()
        self._update_rec_ui()
        self.log("[DataRecorder] 녹화(버퍼링)를 시작합니다.")

    def on_rec_pause(self):
        self.recorder.pause()
        self._update_rec_ui()
        self.log("[DataRecorder] 녹화를 일시정지합니다.")

    def on_rec_save(self):
        filepath = self.recorder.stop_and_save()
        self._update_rec_ui()
        if filepath:
            self.log(f"[DataRecorder] 데이터가 저장되었습니다: {filepath}")
        else:
            self.log("[DataRecorder] 저장할 데이터가 없습니다.")

    def on_rec_cancel(self):
        self.recorder.cancel()
        self._update_rec_ui()
        self.log("[DataRecorder] 녹화가 취소되었습니다. (데이터 폐기)")

    def _update_rec_ui(self):
        if hasattr(self, "record_status_label"):
            self.record_status_label.setText(f"상태: {self.recorder.get_status_str()}")

    def process_command_signal(self, action):
        self.log(f"[main] command: {action}")
        if action == "START" and not self.is_connected and not self.is_connecting:
            self.toggle_connection()
        elif action == "STOP" and (self.is_connected or self.is_connecting):
            self.toggle_connection()

    def handle_web_command(self, data):
        command_type = data.get("type")
        action = data.get("action")
        self.log(f"[web] {command_type}: {action}")
        if command_type == "CONTROL":
            self.command_signal.emit(action)
        elif command_type == "SETTINGS":
            payload = data.get("data", {})
            speed = payload.get("speed")
            height = payload.get("height")
            if speed is not None and height is not None:
                QTimer.singleShot(0, lambda: self.apply_settings(speed, height))

    def _status_track_payloads(self):
        payloads = []
        for track in list(self.latest_tracks or [])[:8]:
            position = getattr(track, "position", [])
            dims = getattr(track, "dims", [])
            if hasattr(position, "tolist"):
                position = position.tolist()
            if hasattr(dims, "tolist"):
                dims = dims.tolist()
            payloads.append(
                {
                    "id": int(getattr(track, "id", 0) or 0),
                    "pos": list(position or [])[:3],
                    "dims": list(dims or [])[:3],
                    "state": str(getattr(track, "state_label", "")),
                    "source": str(getattr(track, "source", "")),
                    "pointCount": int(getattr(track, "point_count", 0) or 0),
                }
            )
        return payloads

    def send_heartbeat(self):
        frame_meta = dict(self.latest_frame_meta or {})
        self._handle_dead_radar_stream()
        data_stale = self._radar_data_stale()
        stream_alive = False
        stream_bytes = 0
        stream_chunks = 0
        stream_parsed_frames = 0
        if self.stream is not None:
            try:
                stream_alive = self.stream.is_alive()
            except Exception:
                stream_alive = False
            stream_bytes = int(getattr(self.stream, "bytes_received", 0) or 0)
            stream_chunks = int(getattr(self.stream, "chunks_received", 0) or 0)
            stream_parsed_frames = int(getattr(self.stream, "frames_parsed", 0) or 0)
        self.web_client.send_status(
            {
                "running": self.is_connected,
                "connecting": self.is_connecting,
                "streamAlive": stream_alive,
                "streamState": self.last_stream_event.get("event", ""),
                "streamRestartAttempts": self.stream_restart_attempts,
                "streamBytes": stream_bytes,
                "streamChunks": stream_chunks,
                "streamParsedFrames": stream_parsed_frames,
                "personCount": len(self.latest_tracks),
                "tracks": self._status_track_payloads(),
                "frame": self.frame_count,
                "frameNumber": frame_meta.get("frameNumber", 0),
                "pointCount": int(len(self.latest_points)),
                "firmwareObjects": int(frame_meta.get("numDetectedObj", 0) or 0),
                "sensorTracks": int(len(self.latest_sensor_tracks)),
                "vitalRecords": int(len(self.latest_vitals)),
                "invalidVitalRecords": int(frame_meta.get("invalidVitalRecords", 0) or 0),
                "fallbackVitalRecords": int(
                    sum(1 for vital in self.latest_vitals if self._is_fallback_vital(vital))
                ),
                "dataStale": data_stale,
                "alertCount": 0,
            }
        )
        self._watch_radar_stream(data_stale)

    def handle_stream_state(self, state):
        state = dict(state or {})
        event = state.get("event", "")

        current_stream_id = id(self.stream) if self.stream is not None else None
        state_stream_id = state.get("stream_id")
        if state_stream_id is not None and current_stream_id is None:
            return
        if current_stream_id is not None and state_stream_id not in (None, current_stream_id):
            return

        self.last_stream_event = state

        if event in {"connecting", "configuring", "waiting_data", "data_received"}:
            self.is_connecting = True
            self.is_connected = False
            self._show_stream_status(event, state.get("message", ""))
            if hasattr(self, "connect_btn"):
                label_map = {
                    "connecting": "Starting...",
                    "configuring": "Configuring...",
                    "waiting_data": "Waiting Data...",
                    "data_received": "Parsing Data...",
                }
                label = label_map.get(event, "Starting...")
                self.connect_btn.setText(label)
                self.connect_btn.setStyleSheet("background-color: #fd7e14; font-weight: bold;")
                self.connect_btn.repaint()
            return

        if event == "started":
            now = time.time()
            self.stream_started_at = now
            self.last_data_time = now
            self.stream_grace_until = now + 15.0
            self.last_board_status = ""
            self._show_stream_status(event, "First valid TLV frame parsed.")
            self.is_connecting = False
            self.is_connected = True
            if hasattr(self, "connect_btn"):
                self.connect_btn.setText("Stop")
                self.connect_btn.setStyleSheet("background-color: #dc3545; font-weight: bold;")
                self.connect_btn.repaint()
            return

        if event in {"connection_failed", "config_failed", "no_data", "parse_stalled"}:
            message = state.get("message") or event
            self.stream_auto_reconnect = False
            self._show_stream_status(event, message)
            self._set_disconnected_ui(f"[RadarStream] {event}: {message}", stop_stream=False)
            self.log("[watchdog] automatic reconnect disabled until the board is reset or ports are checked")
            return

        if event == "error":
            message = state.get("message") or event
            self._show_stream_status(event, message)
            self._set_disconnected_ui(f"[RadarStream] {event}: {message}", stop_stream=False)
            self._schedule_stream_reconnect(reason=event)
            return

        if event == "stopped":
            if self.is_connected or self.is_connecting:
                self._set_disconnected_ui(stop_stream=False)
                if not state.get("requested"):
                    self._schedule_stream_reconnect(reason="stopped")

    def _show_stream_status(self, event, message=""):
        if not hasattr(self, "status_text"):
            return

        event_label = {
            "connecting": "opening serial ports",
            "configuring": "sending radar config",
            "waiting_data": "waiting for Data UART bytes",
            "data_received": "bytes received, parsing TLV frames",
            "started": "live TLV frames parsed",
            "connection_failed": "serial connection failed",
            "config_failed": "config send failed",
            "no_data": "Data UART is 0 bytes",
            "parse_stalled": "bytes received but no TLV frames",
            "error": "stream error",
            "stopped": "stream stopped",
        }.get(event, event or "unknown")

        state = dict(self.last_stream_event or {})
        cli_port = state.get("cli_port") or (
            self.cli_combo.currentText() if hasattr(self, "cli_combo") else "--"
        )
        data_port = state.get("data_port") or (
            self.data_combo.currentText() if hasattr(self, "data_combo") else "--"
        )
        cfg_path = self.cfg_input.text().strip() if hasattr(self, "cfg_input") else ""
        cfg_name = os.path.basename(cfg_path) if cfg_path else "--"
        bytes_received = int(state.get("bytes_received") or 0)
        chunks_received = int(state.get("chunks_received") or 0)
        frames_parsed = int(state.get("frames_parsed") or 0)

        lines = [
            "WEB DASHBOARD:",
            getattr(self, "dashboard_url_text", "http://<device-ip>:8000"),
            "-" * 30,
            f"Radar state: {event_label}",
            f"CLI: {cli_port} | Data: {data_port}",
            f"Config: {cfg_name}",
            f"Data UART bytes: {bytes_received}",
            f"Data chunks: {chunks_received}",
            f"Parsed TLV frames: {frames_parsed}",
            "-" * 30,
        ]

        if message:
            lines.append(str(message))
            lines.append("-" * 30)

        if event == "no_data":
            lines.extend(
                [
                    "Board is connected but not streaming on the Data UART.",
                    "Do this on the board: set S1 to functional mode, press/release NRST, then try again.",
                    "If it still shows 0 bytes, power-cycle USB and verify CLI=COM4, Data=COM3.",
                ]
            )
        elif event == "parse_stalled":
            lines.extend(
                [
                    "Data bytes are arriving, but they are not valid mmWave TLV frames.",
                    "Check Data baud rate 921600, CLI/Data port order, and firmware/output protocol.",
                ]
            )
        elif event in {"waiting_data", "data_received"}:
            lines.extend(
                [
                    "This screen will change to live frame counts only after a valid TLV frame is parsed.",
                    "Until then, old point/vital values are not reused.",
                ]
            )
        elif event in {"connection_failed", "config_failed", "error"}:
            lines.extend(
                [
                    "Fix the serial/board state first, then press Start again.",
                    "For a full check, run run_board_diagnostics.bat from the project folder.",
                ]
            )

        self.last_board_status = "\n".join(lines)
        self.last_board_status_at = time.time()
        self.status_text.setPlainText(self.last_board_status)

    def _set_disconnected_ui(self, reason="", stop_stream=True):
        self.is_connected = False
        self.is_connecting = False
        self.last_stream_event = {"event": "stopped"}
        if self.stream:
            try:
                if stop_stream:
                    self.stream.stop()
            except Exception:
                pass
            self.stream = None
        if hasattr(self, "connect_btn"):
            self.connect_btn.setText("Start")
            self.connect_btn.setStyleSheet("background-color: #198754; font-weight: bold;")
            self.connect_btn.repaint()
        if reason:
            self.log(reason)

    def _schedule_stream_reconnect(self, reason=""):
        if not self.stream_auto_reconnect or self.stream_restart_in_progress:
            return
        if self.max_stream_restart_attempts <= 0:
            self.stream_auto_reconnect = False
            self.log(
                "[watchdog] automatic restart disabled; "
                "keep the app stopped and reset the board/ports before retrying"
            )
            return
        if self.stream_restart_attempts >= self.max_stream_restart_attempts:
            self.stream_auto_reconnect = False
            self.log(
                "[watchdog] radar data did not recover after automatic retries; "
                "board reset or port reconnect is required"
            )
            return
        now = time.time()
        if now - self.last_stream_restart_time < self.stream_restart_cooldown:
            return
        self.stream_restart_in_progress = True
        self.last_stream_restart_time = now
        self.stream_restart_attempts += 1
        if reason:
            self.log(
                f"[watchdog] scheduling radar restart after {reason} "
                f"({self.stream_restart_attempts}/{self.max_stream_restart_attempts})"
            )
        QTimer.singleShot(2000, self._restart_stream_after_watchdog)

    def _handle_dead_radar_stream(self):
        if (not self.is_connected and not self.is_connecting) or self.stream is None:
            return
        try:
            alive = self.stream.is_alive()
        except Exception:
            alive = False
        if alive:
            return

        should_reconnect = self.stream_auto_reconnect
        self._set_disconnected_ui("[watchdog] radar stream thread stopped")
        if should_reconnect and not self.stream_restart_in_progress:
            self._schedule_stream_reconnect(reason="thread stopped")

    def _radar_data_stale(self):
        if not self.is_connected or self.is_connecting:
            return False
        is_file_source = hasattr(self, "source_combo") and self.source_combo.currentIndex() == 1
        if is_file_source:
            return False
        if time.time() < self.stream_grace_until:
            return False
        return (time.time() - self.last_data_time) >= self.stream_stall_seconds

    def _watch_radar_stream(self, data_stale=None):
        data_stale = self._radar_data_stale() if data_stale is None else bool(data_stale)
        if not data_stale or self.stream_restart_in_progress:
            return

        now = time.time()
        if now - self.last_stream_restart_time < self.stream_restart_cooldown:
            return
        if self.max_stream_restart_attempts <= 0:
            if now - self.last_stale_log_time >= 30.0:
                stream_bytes = int(getattr(self.stream, "bytes_received", 0) or 0)
                frames = int(getattr(self.stream, "frames_parsed", 0) or 0)
                last_chunk = float(getattr(self.stream, "last_chunk_time", 0.0) or 0.0)
                raw_age = (now - last_chunk) if last_chunk > 0 else -1.0
                raw_text = f"{raw_age:.1f}s" if raw_age >= 0 else "never"
                self.log(
                    "[watchdog] parsed radar frame is stale; keeping serial stream open "
                    f"(bytes={stream_bytes}, frames={frames}, lastRaw={raw_text})"
                )
                self.last_stale_log_time = now
            return
        if self.stream_restart_attempts >= self.max_stream_restart_attempts:
            self.stream_auto_reconnect = False
            self.log(
                "[watchdog] radar data did not recover after automatic retries; "
                "board reset or port reconnect is required"
            )
            self._set_disconnected_ui("[watchdog] automatic radar restart stopped")
            return

        self.stream_restart_in_progress = True
        self.last_stream_restart_time = now
        self.stream_restart_attempts += 1
        stale_for = now - self.last_data_time
        self.log(
            f"[watchdog] radar data stale for {stale_for:.1f}s; restarting stream "
            f"({self.stream_restart_attempts}/{self.max_stream_restart_attempts})"
        )
        try:
            if self.is_connected:
                self.toggle_connection()
        finally:
            QTimer.singleShot(2000, self._restart_stream_after_watchdog)

    def _restart_stream_after_watchdog(self):
        self.last_data_time = time.time()
        self.stream_grace_until = time.time() + 45.0
        try:
            if not self.is_connected and not self.is_connecting:
                self.toggle_connection()
        finally:
            QTimer.singleShot(10000, self._clear_stream_restart_watchdog)

    def _clear_stream_restart_watchdog(self):
        self.stream_restart_in_progress = False

    @staticmethod
    def _vital_id(vital):
        try:
            return int(vital.get("id", -1))
        except (AttributeError, TypeError, ValueError):
            return -1

    @staticmethod
    def _vital_bpm_text(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "측정 대기"
        if not np.isfinite(value) or value <= 0.0:
            return "측정 대기"
        return f"{value:.1f} bpm"

    @classmethod
    def _is_fallback_vital(cls, vital):
        return (cls._vital_id(vital) & 0x8000) != 0

    @staticmethod
    def _has_nonzero_vital(vital):
        if not vital:
            return False
        try:
            heart = float(vital.get("heart_bpm", vital.get("heartRate", 0.0)) or 0.0)
            breath = float(vital.get("breath_bpm", vital.get("breathingRate", 0.0)) or 0.0)
        except (AttributeError, TypeError, ValueError):
            return False
        return (np.isfinite(heart) and heart > 0.0) or (np.isfinite(breath) and breath > 0.0)

    @staticmethod
    def _vital_web_payload(vital):
        if not vital:
            return None
        vital_id = MainWindow._vital_id(vital)
        return {
            "id": vital_id,
            "rangeBin": int(vital.get("range_bin", vital.get("rangebin", 0))),
            "heartBpm": float(vital.get("heart_bpm", vital.get("heartRate", 0.0)) or 0.0),
            "breathBpm": float(vital.get("breath_bpm", vital.get("breathingRate", 0.0)) or 0.0),
            "signalDeviation": float(
                vital.get("signal_deviation", vital.get("breathingDeviation", 0.0)) or 0.0
            ),
            "fallback": (vital_id & 0x8000) != 0,
        }

    @classmethod
    def _visible_vital_payloads(cls, vitals, matched_vital_ids):
        payloads = []
        for vital in vitals:
            vital_id = cls._vital_id(vital)
            if cls._is_fallback_vital(vital) and vital_id not in matched_vital_ids:
                continue
            payload = cls._vital_web_payload(vital)
            if payload:
                payloads.append(payload)
        return payloads

    @classmethod
    def _format_vital_status(cls, vital):
        if not vital:
            return "HR: 측정 대기 | BR: 측정 대기"
        heart = cls._vital_bpm_text(vital.get("heart_bpm", vital.get("heartRate", 0.0)))
        breath = cls._vital_bpm_text(vital.get("breath_bpm", vital.get("breathingRate", 0.0)))
        range_bin = int(vital.get("range_bin", vital.get("rangebin", 0)))
        deviation = float(vital.get("signal_deviation", vital.get("breathingDeviation", 0.0)) or 0.0)
        return f"HR: {heart} | BR: {breath} | bin: {range_bin} | dev: {deviation:.4f}"

    @staticmethod
    def _vital_for_track(track, vitals_by_id):
        candidate_ids = []
        sensor_track_id = getattr(track, "sensor_track_id", None)
        if sensor_track_id is not None:
            candidate_ids.append(sensor_track_id)
        candidate_ids.append(getattr(track, "id", -1))

        for candidate_id in candidate_ids:
            try:
                vital = vitals_by_id.get(int(candidate_id))
            except (TypeError, ValueError):
                vital = None
            if vital:
                return vital
        return None

    @classmethod
    def _assign_vitals_to_tracks(cls, tracks, vitals, vitals_by_id):
        assigned = {}
        used_vital_ids = set()

        for track in tracks:
            vital = cls._vital_for_track(track, vitals_by_id)
            if not vital:
                continue
            assigned[id(track)] = vital
            used_vital_ids.add(cls._vital_id(vital))

        remaining_tracks = [track for track in tracks if id(track) not in assigned]
        remaining_vitals = [
            vital
            for vital in vitals
            if cls._vital_id(vital) not in used_vital_ids
        ]
        if not remaining_tracks or not remaining_vitals:
            return assigned

        def track_range(track):
            try:
                pos = np.asarray(track.position, dtype=np.float32).reshape(-1)
                if pos.size >= 3:
                    return float(np.linalg.norm(pos[:3]))
            except Exception:
                pass
            return 0.0

        def vital_range_bin(vital):
            try:
                return int(vital.get("range_bin", vital.get("rangebin", 0)))
            except (AttributeError, TypeError, ValueError):
                return 0

        for track, vital in zip(
            sorted(remaining_tracks, key=track_range),
            sorted([v for v in remaining_vitals if cls._has_nonzero_vital(v)], key=vital_range_bin),
        ):
            assigned[id(track)] = vital

        return assigned

    def handle_data(self, parsed):
        self.data_signal.emit(parsed)

    def update_data(self, parsed):
        self.frame_count += 1
        if self.stream_restart_in_progress:
            self.log("[watchdog] radar data recovered")
            self.stream_restart_in_progress = False
        self.stream_restart_attempts = 0
        self.last_data_time = time.time()
        self.stream_grace_until = 0.0

        points = getattr(parsed, "points", None)
        points = _normalize_point_cloud(points)

        sensor_tracks = [dict(track) for track in (getattr(parsed, "tracks", []) or [])]
        tracks_lag = int(getattr(parsed, "tracks_lag", 0) or 0)
        presence = getattr(parsed, "presence", None)
        vitals = [dict(vital) for vital in (getattr(parsed, "vitals", []) or [])]
        vitals_by_id = {}
        for vital in vitals:
            try:
                vitals_by_id[int(vital.get("id", -1))] = vital
            except (TypeError, ValueError):
                continue
        self.latest_vitals = vitals
        self.latest_vitals_by_id = vitals_by_id
        point_track_ids = getattr(parsed, "point_track_ids", None)
        point_track_ids_lag = int(getattr(parsed, "point_track_ids_lag", 0) or 0)
        frame_meta = {
            "frameNumber": int(getattr(parsed, "frame_number", 0) or 0),
            "numDetectedObj": int(getattr(parsed, "num_detected_obj", len(points)) or 0),
            "numTlvs": int(getattr(parsed, "num_tlvs", 0) or 0),
            "tlvTypes": [int(item) for item in (getattr(parsed, "tlv_types", []) or [])],
            "invalidVitalRecords": int(getattr(parsed, "invalid_vital_records", 0) or 0),
            "malformedTlvs": int(getattr(parsed, "malformed_tlvs", 0) or 0),
        }
        self.latest_frame_meta = frame_meta
        self.latest_sensor_tracks = sensor_tracks

        # People Tracking target boxes are already in the tracker/world frame.
        # Align only the point cloud, then use those same points for rendering,
        # clustering, AI, recording, and web payloads.
        is_file_source = hasattr(self, "source_combo") and self.source_combo.currentIndex() == 1
        if not is_file_source:
            points = apply_point_cloud_alignment(points, self.sensor_tilt, self.sensor_height)
            if getattr(self, "enable_firmware_box_height", False):
                pitch_rad = np.radians(float(self.sensor_tilt))
                cos_p = np.cos(pitch_rad)
                sin_p = np.sin(pitch_rad)
                for track in sensor_tracks:
                    y = float(track.get("y", 0.0))
                    z = float(track.get("z", 0.0))
                    track["y"] = y * cos_p + z * sin_p
                    track["z"] = -y * sin_p + z * cos_p + self.sensor_height

        frame_ts = getattr(parsed, "ts", time.time())
        self.recorder.add_points(self.frame_count, frame_ts, points, getattr(parsed, "raw", b""))
        if self.frame_count % 10 == 0:
            self._update_rec_ui()

        display_points, association_points = select_synced_processing_points(
            points,
            prev_points=self.prev_frame_points,
            sensor_tracks=sensor_tracks,
            tracks_lag=tracks_lag,
            point_track_ids=point_track_ids,
            point_track_ids_lag=point_track_ids_lag,
            prev_association_points=self.prev_association_points,
        )

        self.latest_points = display_points
        self.point_history.append(display_points)
        sidecar_skeleton = getattr(parsed, "skeleton_lines", None)
        if sidecar_skeleton is not None and len(sidecar_skeleton) > 0:
            self.latest_pose_skeleton_lines = np.asarray(sidecar_skeleton, dtype=np.float32)
            self.latest_pose_skeleton_source = "synthetic_gt"
        else:
            estimated = self.pose_estimator.predict_skeleton(display_points) if getattr(self, "pose_estimator", None) is not None else None
            self.latest_pose_skeleton_lines = estimated
            self.latest_pose_skeleton_source = "pose_model" if estimated is not None else ""

        process_points = display_points

        if (
            not sensor_tracks
            and hasattr(self, "history_cluster_check")
            and self.history_cluster_check.isChecked()
            and display_points.shape[0] < 32
        ):
            history_frames = [frame for frame in list(self.point_history)[-3:] if frame is not None and len(frame) > 0]
            if history_frames:
                try:
                    process_points = np.vstack(history_frames)
                except Exception:
                    process_points = display_points

        result = self.processor.process(
            process_points,
            sensor_tracks=sensor_tracks,
            max_tracks=self.max_tracks,
            point_track_ids=point_track_ids,
            association_points=association_points,
            enable_clustering=getattr(self, "enable_clustering", True),
        )
        self.prev_association_points = np.asarray(points, dtype=np.float32).copy()
        self.prev_frame_points = np.asarray(points, dtype=np.float32).copy()
        tracks = self.tracker.update(result.detections)
        self.latest_detections = result.detections
        if display_points.ndim == 2 and display_points.shape[1] >= 4 and len(display_points) > 0:
            doppler_values = display_points[:, 3]
            finite_doppler = doppler_values[np.isfinite(doppler_values)]
            if len(finite_doppler) > 0:
                negative_count = int(np.count_nonzero(finite_doppler < 0.0))
                positive_count = int(np.count_nonzero(finite_doppler > 0.0))
                zero_count = int(len(finite_doppler) - negative_count - positive_count)
                doppler_line = (
                    f"Doppler: {float(np.min(finite_doppler)):.3f}.."
                    f"{float(np.max(finite_doppler)):.3f} m/s "
                    f"(-/0/+ {negative_count}/{zero_count}/{positive_count})"
                )
            else:
                doppler_line = "Doppler: n/a"
        else:
            doppler_line = "Doppler: n/a"

        status_lines = [
            "WEB DASHBOARD:",
            getattr(self, "dashboard_url_text", "http://<device-ip>:8000"),
            "-" * 30,
            f"Frame: {self.frame_count}",
            f"Firmware frame: {frame_meta['frameNumber']}",
            f"Firmware objects: {frame_meta['numDetectedObj']}",
            f"Points: {len(display_points)}",
            doppler_line,
            f"Tracked people: {len(tracks)}",
            f"Sensor tracks: {len(sensor_tracks)}",
            f"Vital records: {len(vitals)}",
            "-" * 30,
        ]
        if frame_meta["invalidVitalRecords"] > 0 or frame_meta["malformedTlvs"] > 0:
            status_lines.insert(
                7,
                f"Parser filtered: invalid vital={frame_meta['invalidVitalRecords']} malformed TLV={frame_meta['malformedTlvs']}",
            )
        if presence is not None:
            status_lines.insert(4, f"Presence: {int(presence)}")
        if self.latest_pose_skeleton_source:
            status_lines.insert(5, f"Skeleton: {self.latest_pose_skeleton_source}")

        current_details = {}
        active_tracks = [track for track in tracks if track.missing == 0]
        lost_descent_enabled = bool(getattr(self.detector, "enable_lost_descent_fall", False))
        lost_detection_tracks = []
        if lost_descent_enabled:
            lost_detection_window = max(
                1,
                int(getattr(self.detector, "lost_descent_window_frames", 10))
                + int(getattr(self.detector, "lost_descent_missing_frames", 3)),
            )
            for track in tracks:
                if track.missing <= 0:
                    continue
                state = self.detector.state_by_track.get(track.id)
                if state is None:
                    continue
                if track.missing <= lost_detection_window or state.fall_latched:
                    lost_detection_tracks.append(track)
        detection_tracks = active_tracks + lost_detection_tracks
        simulate_ai = self.sim_ai_check.isChecked()
        matched_vital_ids = set()
        assigned_vitals = self._assign_vitals_to_tracks(active_tracks, vitals, vitals_by_id)

        for track in detection_tracks:
            vital = assigned_vitals.get(id(track))
            track.vital = vital
            if vital:
                matched_vital_ids.add(self._vital_id(vital))

            # AI Inference on EVERY frame (as requested)
            is_fall, score, details = self.detector.detect(track, simulate_ai, frame_index=self.frame_count)
            
            track.state_label = details["display_label"]
            if track.missing > 0:
                if is_fall or details.get("fall_latched"):
                    track.state_label = f"FALL LOST ({track.missing})"
                else:
                    track.state_label = f"LOST ({track.missing})"

            if details.get("should_alert"):
                blackbox_id = self.recorder.trigger_blackbox_event(
                    track_id=track.id,
                    score=score,
                    location=track.position.tolist(),
                    source=track.source,
                    frame_num=self.frame_count,
                    ts=frame_ts,
                )
                alert_payload = {
                    "track_id": track.id,
                    "score": score,
                    "location": track.position.tolist(),
                    "source": track.source,
                }
                if blackbox_id:
                    alert_payload["blackbox_id"] = blackbox_id

                self.web_client.send_alert(alert_payload)
                if blackbox_id:
                    self.log(f"[blackbox] capture started: {blackbox_id}")
                self.log(f"[alert] sent track={track.id} score={score:.2f}")

            current_details[track.id] = details
            status_lines.extend(
                [
                    f"[ID {track.id}] {track.state_label}",
                    f"  score={score:.2f} state={details['fsm_state']} points={track.point_count}",
                    (
                        f"  rule={details.get('rule_score', 0.0):.2f} "
                        f"ai={details.get('ai_score', 0.0):.2f} "
                        f"seq={details.get('sequence_score', 0.0):.2f}"
                    ),
                    (
                        f"  cand={int(bool(details.get('candidate_state')))} "
                        f"aiRun={int(bool(details.get('should_run_ai')))} "
                        f"seqRun={int(bool(details.get('should_run_sequence')))}"
                    ),
                    (
                        f"  lostDrop={int(bool(details.get('lost_descent_candidate')))} "
                        f"miss={details.get('missing_frames', 0)} "
                        f"drop={details.get('lost_descent_last_drop', 0.0):.2f}m"
                    ),
                    f"  {self._format_vital_status(vital)}",
                    "-" * 30,
                ]
            )

        unmatched_vitals = [
            vital
            for vital in vitals
            if self._vital_id(vital) not in matched_vital_ids and not self._is_fallback_vital(vital)
        ]
        fallback_vitals = [
            vital
            for vital in vitals
            if self._vital_id(vital) not in matched_vital_ids and self._is_fallback_vital(vital)
        ]
        if unmatched_vitals:
            status_lines.extend(["Sensor vitals:", "-" * 30])
            for vital in unmatched_vitals[:4]:
                status_lines.extend(
                    [
                        f"[Vital ID {self._vital_id(vital)}]",
                        f"  {self._format_vital_status(vital)}",
                        "-" * 30,
                    ]
                )
        elif fallback_vitals and not active_tracks:
            status_lines.extend(
                [
                    "Sensor vitals: tracker not locked",
                    f"  fallback candidates hidden: {len(fallback_vitals)}",
                    "  Move/stand in the radar view until points or tracks appear.",
                    "-" * 30,
                ]
            )

        now = time.time()
        if now - self.last_status_ui_update >= 0.25:
            scroll_value = self.status_text.verticalScrollBar().value()
            self.status_text.setPlainText("\n".join(status_lines))
            self.status_text.verticalScrollBar().setValue(scroll_value)
            self.last_status_ui_update = now

        visible_tracks = list(active_tracks)
        for track in lost_detection_tracks:
            details = current_details.get(track.id, {})
            if details.get("fall_latched") or details.get("lost_descent_latched"):
                visible_tracks.append(track)

        self.latest_tracks = visible_tracks
        self.latest_details = current_details

        web_tracks = []
        for track in visible_tracks:
            details = current_details.get(track.id, {})
            web_tracks.append(
                {
                    "id": int(track.id),
                    "sensorTrackId": getattr(track, "sensor_track_id", None),
                    "pos": track.position.tolist(),
                    "dims": track.dims.tolist(),
                    "state": track.state_label,
                    "source": track.source,
                    "pointCount": int(getattr(track, "point_count", 0) or 0),
                    "confidence": float(getattr(track, "confidence", 0.0) or 0.0),
                    "fallScore": float(details.get("total_score", 0.0) or 0.0),
                    "fallState": details.get("fsm_state", ""),
                    "vital": self._vital_web_payload(getattr(track, "vital", None)),
                }
            )

        if self.latest_pose_skeleton_lines is not None and len(self.latest_pose_skeleton_lines) > 0:
            web_skeletons = self.latest_pose_skeleton_lines
        else:
            skeletons = []
            for track in visible_tracks:
                if hasattr(track, "get_skeleton"):
                    lines = track.get_skeleton(self.body_ratios, self.gravity_bias)
                    if lines is not None and len(lines) > 0:
                        skeletons.append(lines)
            web_skeletons = np.vstack(skeletons) if skeletons else None
        visible_vitals = self._visible_vital_payloads(vitals, matched_vital_ids)
        realtime_meta = {
            **frame_meta,
            "personCount": int(len(visible_tracks)),
            "pointCount": int(len(display_points)),
            "sensorTracks": int(len(sensor_tracks)),
            "vitalRecords": int(len(vitals)),
            "visibleVitalRecords": int(len(visible_vitals)),
            "invalidVitalRecords": int(frame_meta.get("invalidVitalRecords", 0) or 0),
            "malformedTlvs": int(frame_meta.get("malformedTlvs", 0) or 0),
            "fallbackVitalRecords": int(sum(1 for vital in vitals if self._is_fallback_vital(vital))),
            "presence": None if presence is None else int(presence),
        }
        self.web_client.send_realtime_data(
            display_points,
            web_tracks,
            skeletons=web_skeletons,
            vitals=visible_vitals,
            meta=realtime_meta,
        )


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception:
        import traceback

        traceback.print_exc()
