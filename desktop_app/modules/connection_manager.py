import os

import serial.tools.list_ports
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from modules.radar_stream import RadarStream


class ConnectionManager:
    def __init__(self, main_window):
        self.mw = main_window

    def scan_ports(self):
        mw = self.mw
        mw.cli_combo.clear()
        mw.data_combo.clear()

        ports = serial.tools.list_ports.comports()
        port_descriptions = {}
        for port in ports:
            if "ttyS" in port.device:
                continue

            mw.cli_combo.addItem(port.device)
            mw.data_combo.addItem(port.device)
            port_descriptions[port.device] = port.description

        mw.log(f"Port scan complete: {mw.cli_combo.count()} found")

        settings = QSettings("FallDetector", "RadarApp")
        last_cli = settings.value("last_cli_port", "")
        last_data = settings.value("last_data_port", "")

        mw.log(f"[DEBUG] Loaded Persistence: CLI='{last_cli}', Data='{last_data}'")

        if last_cli:
            index = mw.cli_combo.findText(last_cli)
            if index >= 0:
                mw.cli_combo.setCurrentIndex(index)
                mw.log(f"[DEBUG] Restored CLI Port: {last_cli}")

        if last_data:
            index = mw.data_combo.findText(last_data)
            if index >= 0:
                mw.data_combo.setCurrentIndex(index)
                mw.log(f"[DEBUG] Restored Data Port: {last_data}")

        recommended_cli = next(
            (device for device, desc in port_descriptions.items() if "Enhanced COM Port" in desc),
            None,
        )
        recommended_data = next(
            (device for device, desc in port_descriptions.items() if "Standard COM Port" in desc),
            None,
        )
        if recommended_cli and recommended_data:
            cli_index = mw.cli_combo.findText(recommended_cli)
            data_index = mw.data_combo.findText(recommended_data)
            if cli_index >= 0 and data_index >= 0:
                mw.cli_combo.setCurrentIndex(cli_index)
                mw.data_combo.setCurrentIndex(data_index)
                mw.log(f"[DEBUG] Auto-mapped CLI={recommended_cli}, Data={recommended_data}")

    def toggle_connection(self):
        mw = self.mw
        is_file_source = hasattr(mw, "source_combo") and mw.source_combo.currentIndex() == 1

        mw.log("toggle_connection entered")
        try:
            if not mw.is_connected and not getattr(mw, "is_connecting", False):
                if is_file_source:
                    self._start_file_playback()
                    mw.is_connecting = False
                    mw.is_connected = True
                    mw.connect_btn.setText("Stop")
                    mw.connect_btn.setStyleSheet("background-color: #dc3545; font-weight: bold;")
                    mw.connect_btn.repaint()
                else:
                    self._start_radar_stream()
            else:
                mw._set_disconnected_ui("Disconnected")

                if hasattr(mw, "playback_panel"):
                    mw.playback_panel.setVisible(False)

        except Exception as exc:
            mw.is_connected = False
            mw.is_connecting = False
            if hasattr(mw, "connect_btn"):
                mw.connect_btn.setText("Start")
                mw.connect_btn.setStyleSheet("background-color: #198754; font-weight: bold;")
                mw.connect_btn.repaint()
            if "Errno 13" in str(exc) or "Permission denied" in str(exc):
                cmd = f"sudo chmod 666 {mw.cli_combo.currentText()} {mw.data_combo.currentText()}"
                msg = (
                    "Port permission denied.\n\n"
                    "Run the following command and try again:\n\n"
                    f"{cmd}"
                )
                mw.log(f"[Permission Error] {msg}")
                QMessageBox.critical(mw, "Permission Denied", msg)
            else:
                mw.log(f"Connection error: {exc}")
                import traceback

                traceback.print_exc()

    def _start_radar_stream(self):
        mw = self.mw
        cli = mw.cli_combo.currentText()
        data = mw.data_combo.currentText()
        cfg = mw.cfg_input.text()

        if hasattr(mw, "cfg_manager"):
            resolved_cfg = mw.cfg_manager.resolve_cfg_path(cfg)
            if resolved_cfg and resolved_cfg != cfg:
                mw.log(f"[cfg] Switching to bundled C3CD config: {resolved_cfg}")
                cfg = resolved_cfg
                mw.cfg_input.setText(cfg)
                mw.cfg_manager.load_cfg_file()

        mw.log(f"Connecting: CLI={cli}, Data={data}, CFG={cfg}")

        if not cli or not data:
            mw.log("Please select both ports.")
            raise RuntimeError("ports not selected")

        cli_name = cli.split()[0]
        data_name = data.split()[0]
        if cli_name == data_name:
            mw.log("CLI/Data ports must be different.")
            raise RuntimeError("CLI/Data ports must be different")

        if not cfg or not os.path.exists(cfg):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if hasattr(mw, "cfg_manager"):
                default_cfg = mw.cfg_manager.recommended_people_tracking_cfg()
            else:
                default_cfg = os.path.join(base_dir, "configs", "vital_signs_ISK_6m_multi.cfg")

            if os.path.exists(default_cfg):
                mw.log(f"Using default config: {default_cfg}")
                cfg = default_cfg
                mw.cfg_input.setText(cfg)
                settings = QSettings("FallDetector", "RadarApp")
                settings.setValue("last_cfg_path", cfg)
                if hasattr(mw, "cfg_manager"):
                    mw.cfg_manager.load_cfg_file()
            else:
                mw.log(f"Config file not found: {cfg}")
                raise FileNotFoundError(cfg)

        settings = QSettings("FallDetector", "RadarApp")
        settings.setValue("last_cli_port", cli_name)
        settings.setValue("last_data_port", data_name)
        settings.setValue("last_cfg_path", cfg)

        mw.is_connecting = True
        mw.is_connected = False
        if hasattr(mw, "connect_btn"):
            mw.connect_btn.setText("Starting...")
            mw.connect_btn.setStyleSheet("background-color: #fd7e14; font-weight: bold;")
            mw.connect_btn.repaint()

        mw.stream = RadarStream(
            cli_name,
            data_name,
            cfg,
            mw.handle_data,
            mw.log_signal.emit,
            state_callback=mw.stream_state_signal.emit,
        )
        mw.stream.start()
        mw.log(f"Radar stream thread started: CLI={cli_name}, Data={data_name}; waiting for valid TLV frames")

    def _start_file_playback(self):
        mw = self.mw
        file_path = mw.csv_input.text().strip()
        physical_path = file_path.split("!", 1)[0] if "!" in file_path else file_path

        if not file_path or not os.path.exists(physical_path):
            mw.log(f"File not found: {file_path}")
            raise FileNotFoundError(file_path)

        if file_path.lower().endswith(".npz"):
            from modules.npz_stream import NpzStream

            mw.stream = NpzStream(
                file_path,
                mw.handle_data,
                mw.log_signal.emit,
                progress_callback=mw.playback_progress_signal.emit,
            )
            mw.log(f"NPZ playback started: {file_path}")
        elif file_path.lower().endswith(".bin"):
            from modules.bin_stream import BinStream

            mw.stream = BinStream(
                file_path,
                mw.handle_data,
                mw.log_signal.emit,
                progress_callback=mw.playback_progress_signal.emit,
            )
            mw.log(f"BIN playback started: {file_path}")
        else:
            from modules.csv_stream import CsvStream

            mw.stream = CsvStream(
                file_path,
                mw.handle_data,
                mw.log_signal.emit,
                progress_callback=mw.playback_progress_signal.emit,
            )
            mw.log(f"CSV playback started: {file_path}")

        mw.stream.start()

        if hasattr(mw, "playback_panel"):
            mw.playback_panel.setVisible(True)
            mw.play_pause_btn.setText("Pause")
            mw.play_pause_btn.setStyleSheet("")
            mw.timeline_slider.setValue(0)
            mw.timeline_label.setText("0 / 0")
            mw.speed_combo.setCurrentIndex(1)

    def browse_csv_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.mw,
            "Select data file",
            "",
            "Data Files (*.bin *.npz *.csv *.zip);;All Files (*)",
        )
        if file_path:
            self.mw.csv_input.setText(file_path)
