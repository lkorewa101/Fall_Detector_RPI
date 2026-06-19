import os

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QFileDialog


class ConfigManager:
    def __init__(self, main_window):
        self.mw = main_window

    def app_base_dir(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def recommended_people_tracking_cfg(self) -> str:
        return os.path.join(self.app_base_dir(), "configs", "vital_signs_ISK_6m_multi.cfg")

    @staticmethod
    def _is_current_firmware_content(content: str) -> bool:
        if not ("trackingCfg" in content and "sensorPosition" in content and "vitalsign" in content):
            return False
        return ConfigManager._tracking_max_tracks(content) >= 6

    @staticmethod
    def _tracking_max_tracks(content: str) -> int:
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "trackingCfg":
                try:
                    return int(parts[4])
                except (TypeError, ValueError):
                    return 0
        return 0

    def resolve_cfg_path(self, cfg_path: str) -> str:
        cfg_path = (cfg_path or "").strip()
        recommended = self.recommended_people_tracking_cfg()
        if os.path.exists(recommended):
            if not cfg_path or not os.path.exists(cfg_path):
                return recommended

            # Saved QSettings paths are absolute. When the package is moved,
            # prefer the bundled, verified C3CD config instead of an older
            # config that happens to still exist on this machine.
            try:
                app_base = os.path.realpath(self.app_base_dir())
                cfg_real = os.path.realpath(cfg_path)
                if not os.path.commonpath([app_base, cfg_real]) == app_base:
                    return recommended
            except (OSError, ValueError):
                return recommended

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return recommended if os.path.exists(recommended) else cfg_path

        if os.path.exists(recommended) and not self._is_current_firmware_content(content):
            return recommended

        return cfg_path

    def browse_cfg_file(self):
        fname, _ = QFileDialog.getOpenFileName(self.mw, "Select config file", "", "Config Files (*.cfg);;All Files (*)")
        if fname:
            resolved = self.resolve_cfg_path(fname)
            self.mw.cfg_input.setText(resolved)
            settings = QSettings("FallDetector", "RadarApp")
            settings.setValue("last_cfg_path", resolved)
            if resolved != fname:
                self.mw.log(f"[cfg] Rejected config for older firmware: {fname}")
                self.mw.log(f"[cfg] Using current fall/vital config: {resolved}")

    def load_cfg_file(self):
        mw = self.mw
        cfg_path = self.resolve_cfg_path(mw.cfg_input.text())
        if cfg_path and cfg_path != mw.cfg_input.text():
            mw.cfg_input.setText(cfg_path)
            settings = QSettings("FallDetector", "RadarApp")
            settings.setValue("last_cfg_path", cfg_path)
            mw.log(f"[cfg] Using current fall/vital config: {cfg_path}")

        if not os.path.exists(cfg_path):
            mw.log(f"Config file not found: {cfg_path}")
            return

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                content = f.read()
            mw.current_cfg_content = content
            settings = QSettings("FallDetector", "RadarApp")
            settings.setValue("last_cfg_path", cfg_path)

            cfg_sensor_height = None
            cfg_sensor_tilt = None
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0] == "sensorPosition":
                    try:
                        cfg_sensor_height = float(parts[1])
                        cfg_sensor_tilt = float(parts[3])
                    except (TypeError, ValueError):
                        pass
                    break

            mw.cfg_sensor_height = cfg_sensor_height
            mw.cfg_sensor_tilt = cfg_sensor_tilt

            mw.log(f"Loaded config into memory: {os.path.basename(cfg_path)}")
            if cfg_sensor_height is not None and cfg_sensor_tilt is not None:
                mw.log(
                    f"[cfg] sensorPosition reference: height={cfg_sensor_height:.2f}m, "
                    f"tilt={cfg_sensor_tilt:.2f}deg"
                )

        except Exception as exc:
            mw.log(f"Config load error: {exc}")
