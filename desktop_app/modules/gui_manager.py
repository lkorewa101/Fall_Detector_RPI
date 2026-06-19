import sys
import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QStackedWidget, QWidget

from modules.ui_clock import ClockWidget


class GUIManager:
    """
    Manages GUI logic such as initialization, page switching, flow control, and styling.
    """

    def __init__(self, main_window):
        self.mw = main_window
        self.dark_style = """
            QMainWindow { background-color: #2b2b2b; color: #ffffff; }
            QLabel { color: #ffffff; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #333;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #0b5ed7; }
            QPushButton:disabled { background-color: #555; color: #888; }
            QComboBox {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #202020;
                color: #ffb000;
                selection-background-color: #145a32;
                selection-color: #00ff66;
                border: 1px solid #777;
                outline: 0;
            }
            QComboBox QAbstractItemView::item {
                min-height: 24px;
                padding: 4px 8px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: Consolas;
                border: 1px solid #444;
            }
            QLineEdit {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                padding: 5px;
            }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab {
                background: #444;
                color: #ccc;
                padding: 8px 12px;
            }
            QTabBar::tab:selected {
                background: #0d6efd;
                color: white;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999;
                height: 8px;
                background: #666;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0d6efd;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QCheckBox {
                color: white;
                spacing: 8px;
            }
        """

    def init_ui(self):
        mw = self.mw
        mw.stack = QStackedWidget()
        mw.setCentralWidget(mw.stack)

        mw.clock_page = ClockWidget(switch_callback=self.switch_to_monitor)
        mw.stack.addWidget(mw.clock_page)

        mw.monitor_page = QWidget()
        self.setup_monitor_ui(mw.monitor_page)
        mw.stack.addWidget(mw.monitor_page)

        mw.stack.setCurrentIndex(0)
        self.apply_dark_theme()

    def setup_monitor_ui(self, parent_widget):
        from modules.ui_builder import MonitorUIBuilder

        MonitorUIBuilder.setup_ui(self.mw, parent_widget)

    def apply_dark_theme(self):
        mw = self.mw
        if hasattr(mw, "viewer"):
            mw.viewer.setBackgroundColor("#1e1e1e")
        mw.setStyleSheet(self.dark_style)
        if hasattr(mw, "config_scroll"):
            mw.config_scroll.setStyleSheet("border: none; background-color: #2b2b2b;")
        if hasattr(mw, "config_scroll_content"):
            mw.config_scroll_content.setStyleSheet("background-color: #2b2b2b;")

    def _current_screen_geometry(self):
        app = QApplication.instance()
        screen = None
        if app is not None:
            try:
                screen = app.screenAt(QCursor.pos())
            except Exception:
                screen = None
            if screen is None and self.mw.windowHandle() is not None:
                screen = self.mw.windowHandle().screen()
            if screen is None:
                screen = app.primaryScreen()
        if screen is not None:
            return screen.availableGeometry()
        return self.mw.frameGeometry()

    def apply_screen_layout(self):
        mw = self.mw
        geom = self._current_screen_geometry()
        width = max(640, int(geom.width()))
        height = max(480, int(geom.height()))

        if hasattr(mw, "left_panel"):
            panel_width = max(300, min(390, int(width * 0.34)))
            mw.left_panel.setMinimumWidth(panel_width)
            mw.left_panel.setMaximumWidth(max(panel_width, int(width * 0.45)))

        compact = height < 760
        if hasattr(mw, "status_text"):
            mw.status_text.setMinimumHeight(55 if compact else 80)
            mw.status_text.setMaximumHeight(max(70, min(180, int(height * (0.13 if compact else 0.18)))))
        if hasattr(mw, "log_text"):
            mw.log_text.setMinimumHeight(45 if compact else 60)
            mw.log_text.setMaximumHeight(max(60, min(150, int(height * (0.11 if compact else 0.15)))))
        if hasattr(mw, "tabs"):
            mw.tabs.setMinimumHeight(170 if compact else 260)

    def switch_to_monitor(self):
        mw = self.mw
        mw.hide()
        mw.stack.setCurrentIndex(1)
        self.apply_dark_theme()
        self.apply_screen_layout()
        mw.setWindowFlags(Qt.FramelessWindowHint)
        mw.setGeometry(self._current_screen_geometry())
        mw.showFullScreen()
        QTimer.singleShot(300, self.apply_screen_layout)
        self.log("Developer monitor mode")

    def switch_to_clock(self):
        mw = self.mw
        mw.hide()
        mw.stack.setCurrentIndex(0)
        self.apply_dark_theme()
        self.apply_screen_layout()
        mw.setWindowFlags(Qt.FramelessWindowHint)
        mw.setGeometry(self._current_screen_geometry())
        mw.showFullScreen()
        QTimer.singleShot(300, self.apply_screen_layout)
        self.log("Clock mode")

    def log(self, msg):
        mw = self.mw
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {msg}"
        try:
            print(formatted)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            safe_formatted = formatted.encode(encoding, errors="replace").decode(encoding, errors="replace")
            print(safe_formatted)
        if hasattr(mw, "log_text"):
            mw.log_text.append(formatted)
            sb = mw.log_text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def toggle_web_client(self, state):
        mw = self.mw
        mw.web_client.enabled = state == Qt.Checked
        mw.web_client.server_url = mw.web_url_input.text()

    def update_sensitivity(self):
        mw = self.mw
        s_val = mw.speed_slider.value()
        speed_thresh = -1.5 + (s_val / 100.0) * 1.4
        h_val = mw.height_slider.value()
        height_thresh = 0.4 + (h_val / 100.0) * 1.1
        mw.detector.set_sensitivity(speed_thresh, height_thresh)
        mw.speed_label.setText(f"속도 임계값: {speed_thresh:.2f} m/s")
        mw.height_label.setText(f"높이 임계값: {height_thresh:.2f} m")

    def apply_settings(self, speed, height):
        mw = self.mw
        mw.detector.set_sensitivity(float(speed), float(height))
        self.log(f"Sensitivity updated: speed={speed}, height={height}")

    def close_event(self, event):
        mw = self.mw
        if mw.stream:
            mw.stream.stop()
        if mw.web_client:
            mw.web_client.disconnect()
        event.accept()
