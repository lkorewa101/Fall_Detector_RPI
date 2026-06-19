from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QFrame, QGridLayout, QSpacerItem, QSizePolicy, QGraphicsDropShadowEffect, QPushButton)
from PyQt5.QtCore import Qt, QTimer, QTime, QDate
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor
import os

try:
    from modules.env_sensor import DhtEnvironmentSensor
except Exception:
    DhtEnvironmentSensor = None


def _safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="replace").decode("ascii"))


class ClockWidget(QWidget):
    def __init__(self, parent=None, switch_callback=None):
        super().__init__(parent)
        self.switch_callback = switch_callback
        self.click_count = 0
        self.last_click_time = 0
        self.env_sensor = DhtEnvironmentSensor() if DhtEnvironmentSensor else None
        
        # Themes configuration
        self.themes = [
            {"name": "Dynamic", "bg": "dynamic", "text_color": "#ffffff", "shadow_color": "#000000"},
            {"name": "Nature", "bg": "images/bg_nature.png", "text_color": "#ffffff", "shadow_color": "#000000"},
            {"name": "Modern", "bg": "images/bg_modern.png", "text_color": "#ffffff", "shadow_color": "#000000"},
            {"name": "Warm", "bg": "images/bg_warm.png", "text_color": "#1c1c3c", "shadow_color": "#ffffff"},
            {"name": "Illustration", "bg": "images/bg_illustration.png", "text_color": "#1c1c3c", "shadow_color": "#ffffff"},
            {"name": "Default", "bg": None, "text_color": "#1c1c3c", "shadow_color": None}
        ]
        self.current_theme_idx = 0
        self.last_hour = -1 
        self.bg_pixmap = None # For paintEvent
        
        self.init_ui()
        
        self.colon_visible = True
        self.signal_step = 0
        
        # Timer for updating time
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000) 

        self.env_timer = QTimer(self)
        self.env_timer.timeout.connect(self.update_environment)
        self.env_timer.start(2000)
        
        self.update_time()
        self.update_environment()
        self.apply_theme()

    def init_ui(self):
        self.setObjectName("ClockWidget")
        
        # Determine Project Root for Images
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.image_root = os.path.join(base_dir, "images")
        _safe_print(f"[Clock] Image Root: {self.image_root}")
        
        # Main Layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(60, 60, 60, 60)
        
        # [NEW] Debug Button (Top-Right)
        top_layout = QHBoxLayout()
        top_layout.addStretch()
        self.debug_btn = QPushButton("🛠 DEBUG")
        self.debug_btn.setFixedSize(100, 40)
        self.debug_btn.setStyleSheet("""
            QPushButton { 
                background-color: rgba(255, 0, 0, 0.3); 
                color: rgba(255, 255, 255, 0.7); 
                border: 1px solid rgba(255, 255, 255, 0.5); 
                border-radius: 5px; font-weight: bold; 
            }
            QPushButton:hover { background-color: rgba(255, 0, 0, 0.8); color: white; }
        """)
        self.debug_btn.clicked.connect(lambda: self.switch_callback() if self.switch_callback else None)
        top_layout.addWidget(self.debug_btn)
        
        main_layout.addLayout(top_layout)

        # --- Main Clock Section (Center) ---
        clock_container = QWidget()
        clock_layout = QHBoxLayout(clock_container)
        clock_layout.setAlignment(Qt.AlignCenter)
        
        # Left Side (AM/PM + Signal)
        left_box = QWidget()
        left_v_layout = QVBoxLayout(left_box)
        left_v_layout.setAlignment(Qt.AlignCenter)
        left_v_layout.setContentsMargins(0, 20, 30, 0) 
        
        self.ampm_label = QLabel("AM")
        self.ampm_label.setFont(QFont("Segoe UI", 32, QFont.Bold)) 
        
        self.signal_label = QLabel("RADAR") 
        self.signal_label.setFont(QFont("Segoe UI", 24)) 
        self.signal_label.setAlignment(Qt.AlignCenter)

        left_v_layout.addWidget(self.ampm_label)
        left_v_layout.addWidget(self.signal_label)
        
        # Right Side (Time)
        self.time_label = QLabel("12:00")
        self.time_label.setFont(QFont("Segoe UI", 180, QFont.Bold)) 
        
        clock_layout.addWidget(left_box)
        clock_layout.addWidget(self.time_label)
        
        main_layout.addStretch(1)
        main_layout.addWidget(clock_container)
        
        # --- Indoor Environment (Temp/Humidity) ---
        env_layout = QHBoxLayout()
        env_layout.setAlignment(Qt.AlignCenter)
        
        self.temp_label = QLabel("온도 23.5C") 
        self.temp_label.setFont(QFont("Segoe UI", 48)) 
        
        self.humid_label = QLabel("습도 45%")
        self.humid_label.setFont(QFont("Segoe UI", 48))
        
        env_layout.addWidget(self.temp_label)
        env_layout.addWidget(self.humid_label)
        
        main_layout.addLayout(env_layout)
        main_layout.addStretch(1)
        
        # --- Bottom Section: Weather Info ---
        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Plain)
        main_layout.addWidget(self.line)
        main_layout.addSpacing(30)
        
        bottom_layout = QHBoxLayout()
        
        self.bs_label = QLabel("외부 환경 정보")
        self.bs_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        
        bottom_layout.addWidget(self.bs_label)
        bottom_layout.addStretch()
        
        weather_text = "대구   맑음   25C   공기 좋음"
        self.weather_info = QLabel(weather_text)
        self.weather_info.setFont(QFont("Segoe UI", 18)) 
        self.weather_info.setAlignment(Qt.AlignRight)
        
        bottom_layout.addWidget(self.weather_info)
        main_layout.addLayout(bottom_layout)

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.bg_pixmap and not self.bg_pixmap.isNull():
            painter.drawPixmap(self.rect(), self.bg_pixmap)
        else:
            # Fallback Gradient or Color
            painter.fillRect(self.rect(), QColor(20, 20, 20)) # Dark Gray

    def apply_theme(self):
        theme = self.themes[self.current_theme_idx]
        text_color = theme["text_color"]
        shadow_color = theme["shadow_color"]
        
        # Background Logic
        rel_path = theme["bg"]
        bg_path = None
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if rel_path == "dynamic":
            curr_h = QTime.currentTime().hour()
            # Map hours to existing background images
            if 6 <= curr_h < 9:
                rel_path = "images/bg_dawn.png"
            elif 9 <= curr_h < 17:
                rel_path = "images/bg_day.png"
            elif 17 <= curr_h < 21:
                rel_path = "images/bg_dusk.png"
            else:
                rel_path = "images/bg_night.png"
                
            # Adjust text colors for dynamic theme if needed
            if 6 <= curr_h < 18:
                text_color = "#ffffff" 
                shadow_color = "#000000"
            else:
                text_color = "#ffffff" 
                shadow_color = "#000000"

        if rel_path:
             # Split path by / and join with os.path.join for platform compatibility
             path_parts = rel_path.replace("\\", "/").split("/")
             bg_path = os.path.join(base_dir, *path_parts)

        # Load Image for paintEvent
        if bg_path and os.path.exists(bg_path):
            self.bg_pixmap = QPixmap(bg_path)
            if self.bg_pixmap.isNull():
                 _safe_print(f"[Clock] Failed to load pixmap from {bg_path}")
        else:
            # Final fallback: if specifically mapped image is missing, try bg_day.png
            fallback_path = os.path.join(base_dir, "images", "bg_day.png")
            if os.path.exists(fallback_path):
                self.bg_pixmap = QPixmap(fallback_path)
            else:
                _safe_print(f"[Clock] Image not found: {bg_path}")
                self.bg_pixmap = None
            
        self.update() # Trigger paintEvent

        # Text Styles
        common_style = f"color: {text_color}; background: transparent;"
        
        self.ampm_label.setStyleSheet(f"color: {text_color}; margin-bottom: 5px; background: transparent;")
        self.signal_label.setStyleSheet(common_style)
        self.time_label.setStyleSheet(common_style)
        self.temp_label.setStyleSheet(f"color: {text_color}; margin-right: 60px; background: transparent;")
        self.humid_label.setStyleSheet(common_style)
        self.bs_label.setStyleSheet(common_style)
        self.weather_info.setStyleSheet(common_style)
        self.line.setStyleSheet(f"background-color: {text_color}; max-height: 2px; opacity: 0.5;")

        # Shadow
        def add_shadow(widget):
            if shadow_color:
                effect = QGraphicsDropShadowEffect()
                effect.setBlurRadius(10)
                effect.setColor(QColor(shadow_color))
                effect.setOffset(2, 2)
                widget.setGraphicsEffect(effect)
            else:
                widget.setGraphicsEffect(None)

        add_shadow(self.time_label)
        add_shadow(self.ampm_label)
        add_shadow(self.temp_label)
        add_shadow(self.humid_label)

    def update_time(self):
        curr_time = QTime.currentTime()
        h = curr_time.hour()
        
        if self.themes[self.current_theme_idx]["name"] == "Dynamic":
            if h != self.last_hour:
                self.last_hour = h
                self.apply_theme()

        m = curr_time.minute()
        ampm = "AM" if h < 12 else "PM"
        display_h = h if h <= 12 else h - 12
        if display_h == 0: display_h = 12
        
        self.colon_visible = not self.colon_visible
        colon = ":" if self.colon_visible else " "
        self.time_label.setText(f'{display_h:02d}{colon}{m:02d}')
        self.ampm_label.setText(ampm)
        
        self.signal_step = (self.signal_step + 1) % 3
        sig_text = "RADAR" if self.signal_step == 1 else ("RADAR ON" if self.signal_step == 2 else "ON")
        self.signal_label.setText(sig_text)       

    def update_environment(self):
        if self.env_sensor is None:
            self.temp_label.setText("온도 --.-C")
            self.humid_label.setText("습도 --%")
            return

        reading = self.env_sensor.read()
        if reading.ok and reading.temperature_c is not None and reading.humidity_percent is not None:
            self.temp_label.setText(f"온도 {reading.temperature_c:.1f}C")
            self.humid_label.setText(f"습도 {reading.humidity_percent:.0f}%")
            return

        self.temp_label.setText("온도 --.-C")
        self.humid_label.setText("습도 --%")
        if reading.status:
            self.temp_label.setToolTip(reading.status)
            self.humid_label.setToolTip(reading.status)

    def mouseDoubleClickEvent(self, event):
        self.current_theme_idx = (self.current_theme_idx + 1) % len(self.themes)
        self.apply_theme()

    def mousePressEvent(self, event):
        w = self.width()
        if event.x() > w - 100 and event.y() < 100:
            self.handle_secret_click()
            
    def handle_secret_click(self):
        import time
        curr = time.time()
        if curr - self.last_click_time < 0.5:
            self.click_count += 1
        else:
            self.click_count = 1
        self.last_click_time = curr
        
        if self.click_count >= 3:
            if self.switch_callback:
                self.switch_callback()
            self.click_count = 0

    def closeEvent(self, event):
        if self.env_sensor is not None:
            self.env_sensor.close()
        super().closeEvent(event)
