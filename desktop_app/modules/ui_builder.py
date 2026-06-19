import os
import numpy as np
import serial.tools.list_ports
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, 
                             QLabel, QGroupBox, QTextEdit, QLineEdit, QCheckBox, 
                             QGridLayout, QTabWidget, QDoubleSpinBox, QSlider, QSpinBox, QMessageBox,
                             QSizePolicy, QScrollArea)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

# pyqtgraph import handled carefully
import pyqtgraph.opengl as gl

class VisibleCheckBox(QCheckBox):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(24)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        indicator_size = 18
        indicator_x = 1
        indicator_y = (self.height() - indicator_size) // 2
        indicator_rect = self.rect().adjusted(indicator_x, indicator_y, 0, 0)
        indicator_rect.setWidth(indicator_size)
        indicator_rect.setHeight(indicator_size)

        checked = self.isChecked()
        enabled = self.isEnabled()
        border = QColor("#ffffff" if checked else "#cfd4da")
        fill = QColor("#0d6efd" if checked else "#2b2b2b")
        if not enabled:
            border = QColor("#666666")
            fill = QColor("#3a3a3a")

        painter.setPen(QPen(border, 1.6))
        painter.setBrush(fill)
        painter.drawRoundedRect(indicator_rect, 3, 3)

        if checked:
            check_path = QPainterPath()
            check_path.moveTo(indicator_rect.left() + 4.0, indicator_rect.top() + 9.5)
            check_path.lineTo(indicator_rect.left() + 7.5, indicator_rect.top() + 13.0)
            check_path.lineTo(indicator_rect.left() + 14.0, indicator_rect.top() + 5.0)
            painter.setPen(QPen(QColor("#ffffff"), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(check_path)

        text_x = indicator_rect.right() + 9
        text_rect = self.rect().adjusted(text_x, 0, 0, 0)
        painter.setPen(QColor("#ffffff" if enabled else "#888888"))
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.text())

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setHeight(max(hint.height(), 24))
        hint.setWidth(hint.width() + 8)
        return hint


QCheckBox = VisibleCheckBox

class NoWheelSlider(QSlider):
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
    def wheelEvent(self, event):
        event.ignore()

class VerticalOnlyScrollArea(QScrollArea):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        widget = self.widget()
        if widget is not None:
            widget.setFixedWidth(max(1, self.viewport().width()))

class NoWheelComboBox(QComboBox):
    """콤보박스에서 마우스 휠 스크롤 무시"""
    def wheelEvent(self, event):
        event.ignore()
    def showPopup(self):
        self.setMaxVisibleItems(8)
        super().showPopup()
        popup = self.view().window()
        popup.raise_()
        popup.activateWindow()

class NoWheelSpinBox(QSpinBox):
    """스핀박스에서 마우스 휠 스크롤 무시"""
    def wheelEvent(self, event):
        event.ignore()

class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """더블스핀박스에서 마우스 휠 스크롤 무시"""
    def wheelEvent(self, event):
        event.ignore()

class NoWheelComboBox(QComboBox):
    popup_style = """
        QAbstractItemView {
            background-color: #202020;
            color: #ffb000;
            selection-background-color: #145a32;
            selection-color: #00ff66;
            border: 1px solid #777;
            outline: 0;
        }
        QAbstractItemView::item {
            min-height: 24px;
            padding: 4px 8px;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view().setStyleSheet(self.popup_style)

    def wheelEvent(self, event):
        event.ignore()

    def showPopup(self):
        self.setMaxVisibleItems(8)
        self.view().setStyleSheet(self.popup_style)
        super().showPopup()
        popup = self.view().window()
        popup.raise_()
        popup.activateWindow()


class MonitorUIBuilder:
    """
    Handles the creation of the Monitor UI (Radar Control & Visualization).
    Separates GUI construction code from main logic.
    """
    @staticmethod
    def _expand_input(widget):
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return widget

    @staticmethod
    def _allow_large_entry(widget):
        widget.setKeyboardTracking(False)
        return widget

    @staticmethod
    def _as_scroll_tab(content_widget):
        scroll = VerticalOnlyScrollArea()
        scroll.setObjectName("DarkTabScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content_widget.setMinimumWidth(0)
        content_widget.setMaximumWidth(16777215)
        content_widget.setAutoFillBackground(False)
        content_widget.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #eeeeee;
            }
            QLabel {
                color: #eeeeee;
                background-color: transparent;
            }
            QGroupBox {
                color: #eeeeee;
                background-color: #2b2b2b;
                border: 1px solid #666666;
                border-radius: 3px;
                margin-top: 8px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
                left: 6px;
                color: #ffffff;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                background-color: #303030;
                color: #f0f0f0;
                border: 1px solid #666666;
            }
            QPushButton {
                color: #ffffff;
            }
        """)
        scroll.setWidget(content_widget)
        content_widget.setFixedWidth(max(1, scroll.viewport().width()))
        scroll.viewport().setAutoFillBackground(False)
        scroll.setStyleSheet("""
            QScrollArea#DarkTabScrollArea {
                background-color: #2b2b2b;
                border: none;
            }
            QScrollArea#DarkTabScrollArea > QWidget > QWidget {
                background-color: #2b2b2b;
            }
            QScrollBar:vertical {
                background: #242424;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #777777;
                min-height: 24px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar:horizontal {
                height: 0;
            }
        """)
        return scroll

    @staticmethod
    def _discover_ai_models():
        module_dir = os.path.dirname(os.path.abspath(__file__))
        app_root = os.path.dirname(module_dir)
        search_roots = [
            os.path.join(app_root, "models"),
            module_dir,
        ]
        found = []
        seen = set()
        for root in search_roots:
            if not os.path.isdir(root):
                continue
            for current_root, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d not in {".venv", "__pycache__"}]
                for name in files:
                    if not name.lower().endswith(".pkl"):
                        continue
                    name_lower = name.lower()
                    if not name_lower.startswith("fall_model"):
                        continue
                    path = os.path.abspath(os.path.join(current_root, name))
                    if path in seen:
                        continue
                    seen.add(path)
                    rel = os.path.relpath(path, app_root)
                    found.append((rel, path))
        found.sort(key=lambda item: item[0].lower())
        return found

    @staticmethod
    def setup_ui(main_window, parent_widget):
        """
        Builds the monitor UI on the given parent_widget.
        Attaches created widgets to main_window for logic access.
        """
        main_layout = QHBoxLayout(parent_widget)
        
        # --- Left Panel (Control & Status) ---
        left_panel = QWidget()
        main_window.left_panel = left_panel
        left_panel.setMinimumWidth(350) # Default compact width
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 10)
        left_layout.setSpacing(6)
        
        # 1. Top Button Group (Back)
        top_btn_layout = QHBoxLayout()
        
        back_btn = QPushButton("⬅ 시계 화면")
        back_btn.clicked.connect(main_window.switch_to_clock)
        back_btn.setStyleSheet("background-color: #6610f2; color: white; font-weight: bold; padding: 10px;")
        
        top_btn_layout.addWidget(back_btn)
        
        # [NEW] Exit Button
        exit_btn = QPushButton("종료 (Exit)")
        exit_btn.clicked.connect(main_window.close)
        exit_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 10px;")
        top_btn_layout.addWidget(exit_btn)
        
        top_btn_layout.addStretch()
        left_layout.addLayout(top_btn_layout)
        
        # 2. Tabs
        main_window.tabs = QTabWidget()

        # [Layout Adjustment] Give Tabs a stretch factor of 1 to occupy remaining space (approx 50%+)
        left_layout.addWidget(main_window.tabs, 1)
        
        # --- Tab 1: Connection ---
        MonitorUIBuilder._setup_connection_tab(main_window)
        
        # --- Tab 2: Settings (Merged Detection & Settings & Tracker) ---
        MonitorUIBuilder._setup_settings_tab(main_window)
        
        # --- Tab 3: Visualization ---
        MonitorUIBuilder._setup_viz_tab(main_window)
        
        # 3. Status Group
        status_group = QGroupBox("Status")
        # Reduced margin-top (10 -> 6), padding (5 -> 3), title indent
        status_group.setStyleSheet("""
            QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 6px; padding-top: 5px; font-weight: bold; } 
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; left: 5px; }
        """)
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(2, 5, 2, 2) # Highly minimized margins
        
        main_window.status_text = QTextEdit()
        main_window.status_text.setReadOnly(True)
        main_window.status_text.setFont(QFont("Consolas", 10))
        main_window.status_text.setMinimumHeight(80)
        main_window.status_text.setMaximumHeight(200) 
        status_layout.addWidget(main_window.status_text)
        
        left_layout.addWidget(status_group)
        
        # 4. Log Group
        log_group = QGroupBox("시스템 로그")
        log_group.setStyleSheet("""
            QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 6px; padding-top: 5px; font-weight: bold; } 
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; left: 5px; }
        """)
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(2, 5, 2, 2) 
        
        main_window.log_text = QTextEdit()
        main_window.log_text.setReadOnly(True)
        main_window.log_text.setMinimumHeight(60)
        main_window.log_text.setMaximumHeight(170)
        log_layout.addWidget(main_window.log_text)
        
        left_layout.addWidget(log_group)
        left_layout.addSpacing(4)
        
        # 5. 3D Viewer (Right Side)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        MonitorUIBuilder._setup_viewer(main_window)
        right_layout.addWidget(main_window.viewer, 1)

        main_window.playback_panel = QWidget()
        playback_layout = QHBoxLayout(main_window.playback_panel)
        playback_layout.setContentsMargins(5, 5, 5, 5)

        main_window.replay_btn = QPushButton("다시 재생")
        main_window.replay_btn.clicked.connect(main_window.replay_playback)
        main_window.replay_btn.setFixedWidth(90)
        playback_layout.addWidget(main_window.replay_btn)

        main_window.play_pause_btn = QPushButton("일시정지")
        main_window.play_pause_btn.setFixedWidth(90)
        main_window.play_pause_btn.clicked.connect(main_window.toggle_playback)
        playback_layout.addWidget(main_window.play_pause_btn)

        main_window.timeline_slider = NoWheelSlider(Qt.Horizontal)
        main_window.timeline_slider.setRange(0, 100)
        main_window.timeline_slider.setValue(0)
        main_window.timeline_slider.sliderMoved.connect(main_window.seek_playback)
        playback_layout.addWidget(main_window.timeline_slider)

        main_window.timeline_label = QLabel("0 / 0")
        main_window.timeline_label.setFixedWidth(80)
        main_window.timeline_label.setAlignment(Qt.AlignCenter)
        playback_layout.addWidget(main_window.timeline_label)

        main_window.speed_combo = NoWheelComboBox()
        main_window.speed_combo.addItems(["0.1x", "0.5x", "1.0x", "1.5x", "2.0x"])
        main_window.speed_combo.setCurrentIndex(1)
        main_window.speed_combo.currentIndexChanged.connect(main_window.change_playback_speed)
        playback_layout.addWidget(main_window.speed_combo)

        main_window.playback_panel.setVisible(False)
        right_layout.addWidget(main_window.playback_panel)
        
        # Add to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

        # [Dynamic Layout] Expand left panel only for TI Config Tab
        def on_tab_changed(index):
            # The "Radar Config" tab (index 3) has been removed.
            # The left panel will now always maintain its compact width.
            left_panel.setMinimumWidth(350)
        
        main_window.tabs.currentChanged.connect(on_tab_changed)
        
        # Initial sizing
        on_tab_changed(main_window.tabs.currentIndex())

    @staticmethod
    def _setup_connection_tab(mw):
        tab_conn = QWidget()
        layout_conn = QVBoxLayout(tab_conn)

        source_group = QGroupBox("데이터 소스 선택")
        source_layout = QVBoxLayout(source_group)

        mw.source_combo = NoWheelComboBox()
        mw.source_combo.addItems(["실제 레이더", "데이터 파일 재생 (.bin/.npz/.csv/.zip)"])
        mw.source_combo.setToolTip(
            "저장된 데이터 파일을 재생할 수 있습니다.\n"
            "지원 형식: .bin 원시 TLV, .npz, .csv, .zip\n\n"
            ".bin은 레이더에서 저장한 원시 프레임을 다시 파싱해서 재생합니다.\n"
            ".npz/.csv/.zip은 후처리된 포인트 클라우드 파일 재생에 사용합니다."
        )
        mw.source_combo.currentIndexChanged.connect(lambda idx: MonitorUIBuilder._on_source_changed(mw, idx))
        source_layout.addWidget(mw.source_combo)

        layout_conn.addWidget(source_group)

        mw.radar_group = QGroupBox("1. 레이더 연결 설정")
        conn_layout = QVBoxLayout(mw.radar_group)
        
        mw.cli_combo = NoWheelComboBox()
        mw.cli_combo.setEditable(True)
        mw.data_combo = NoWheelComboBox()
        mw.data_combo.setEditable(True)
        
        port_layout = QGridLayout()
        port_layout.addWidget(QLabel("CLI 포트 (명령):"), 0, 0)
        port_layout.addWidget(mw.cli_combo, 0, 1)
        port_layout.addWidget(QLabel("Data 포트 (데이터):"), 1, 0)
        port_layout.addWidget(mw.data_combo, 1, 1)
        
        conn_layout.addLayout(port_layout)
        
        refresh_btn = QPushButton("포트 새로고침")
        refresh_btn.clicked.connect(mw.scan_ports)
        refresh_btn.setStyleSheet("background-color: #6c757d;")
        conn_layout.addWidget(refresh_btn)
        
        # Config File - Load last used path from QSettings
        settings = QSettings("FallDetector", "RadarApp")
        last_cfg = settings.value("last_cfg_path", "")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bundled_cfg = os.path.join(base_dir, "configs", "vital_signs_ISK_6m_multi.cfg")
        default_cfg = ""
        if last_cfg:
            last_cfg = os.path.abspath(str(last_cfg))
            if os.path.exists(last_cfg):
                default_cfg = last_cfg
        if not default_cfg and os.path.exists(bundled_cfg):
            default_cfg = bundled_cfg
        
        mw.cfg_input = QLineEdit(default_cfg)
        
        cfg_layout = QHBoxLayout()
        cfg_layout.addWidget(QLabel("설정 파일:"))
        cfg_layout.addWidget(mw.cfg_input)
        
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(30)
        browse_btn.clicked.connect(mw.browse_cfg_file)
        cfg_layout.addWidget(browse_btn)
        
        # Edit btn removed (Use Radar Config Tab)
        
        conn_layout.addLayout(cfg_layout)
        
        layout_conn.addWidget(mw.radar_group)

        mw.record_group = QGroupBox("데이터 하드 저장 (Raw Data Recording)")
        record_layout = QVBoxLayout(mw.record_group)
        record_layout.setContentsMargins(0, 2, 0, 2) # Added slight vertical margin
        record_layout.setSpacing(0)

        # Ultra-Compact 'Stuck Together' Recording Row
        compact_row_layout = QHBoxLayout()
        compact_row_layout.setSpacing(0) # Absolute zero spacing
        compact_row_layout.setContentsMargins(0, 0, 0, 0)
        compact_row_layout.setAlignment(Qt.AlignLeft)

        mw.record_status_label = QLabel("상태: 대기 중 (IDLE)")
        mw.record_status_label.setStyleSheet("color: #666; font-weight: bold; margin-right: 0px; padding: 0;")
        compact_row_layout.addWidget(mw.record_status_label)

        icon_btn_style = """
            QGroupBox {
                border: 1px solid #ddd; border-radius: 4px; 
                margin-top: 8px; padding-top: 4px; font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 5px; padding: 0 2px; }
            QPushButton {
                background-color: transparent; border: none; font-size: 16px;
                min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px;
                padding: 0px; margin: 0px;
            }
            QPushButton:hover { background-color: transparent; }
            QPushButton#recStartBtn { color: #d32f2f; font-size: 20px; padding-bottom: 1px; } /* Dot: Slightly Up */
            QPushButton#recPauseBtn { color: #d32f2f; font-size: 11px; padding-bottom: 0px; } 
            QPushButton#recSaveBtn { color: #d32f2f; font-size: 18px; padding-bottom: 3px; }  /* Square: More Up */
            QPushButton#recCancelBtn { color: #d32f2f; font-size: 13px; padding-top: 1px; }   /* Cross: Slightly Down */
            
            QPushButton#recStartBtn:hover { color: #ff5252; } 
            QPushButton#recPauseBtn:hover { color: #ff5252; } 
            QPushButton#recSaveBtn:hover { color: #ff5252; }  
            QPushButton#recCancelBtn:hover { color: #ff5252; }
            QPushButton:pressed { background-color: rgba(0,0,0,0.05); }
        """

        mw.rec_start_btn = QPushButton("●")
        mw.rec_start_btn.setObjectName("recStartBtn")
        mw.rec_start_btn.setToolTip("녹화 시작 (Record)")
        
        mw.rec_pause_btn = QPushButton("||")
        mw.rec_pause_btn.setObjectName("recPauseBtn")
        mw.rec_pause_btn.setToolTip("일시정지 (Pause)")
        
        mw.rec_save_btn = QPushButton("■")
        mw.rec_save_btn.setObjectName("recSaveBtn")
        mw.rec_save_btn.setToolTip("정지 및 저장 (Stop & Save)")
        
        mw.rec_cancel_btn = QPushButton("✕")
        mw.rec_cancel_btn.setObjectName("recCancelBtn")
        mw.rec_cancel_btn.setToolTip("취소 (Cancel)")

        mw.rec_start_btn.clicked.connect(mw.on_rec_start)
        mw.rec_pause_btn.clicked.connect(mw.on_rec_pause)
        mw.rec_save_btn.clicked.connect(mw.on_rec_save)
        mw.rec_cancel_btn.clicked.connect(mw.on_rec_cancel)

        mw.record_group.setStyleSheet(icon_btn_style)

        compact_row_layout.addWidget(mw.rec_start_btn)
        compact_row_layout.addWidget(mw.rec_pause_btn)
        compact_row_layout.addWidget(mw.rec_save_btn)
        compact_row_layout.addWidget(mw.rec_cancel_btn)
        record_layout.addLayout(compact_row_layout)

        layout_conn.addWidget(mw.record_group)

        mw.csv_group = QGroupBox("재생 파일 선택 (.bin/.npz/.csv/.zip)")
        csv_layout = QVBoxLayout(mw.csv_group)

        mw.csv_input = QLineEdit()
        mw.csv_input.setPlaceholderText("재생할 .bin, .npz, .csv 또는 .zip 파일 경로")
        mw.csv_input.setToolTip(
            "재생할 데이터 파일을 선택합니다.\n\n"
            ".bin: 레이더 원시 TLV 프레임 파일입니다. 실제 수신 데이터에 가장 가깝게 재생합니다.\n"
            ".npz: 저장된 NumPy 포인트/프레임 데이터입니다.\n"
            ".csv/.zip: 프레임별 포인트 클라우드 테이블입니다."
        )
        csv_file_layout = QHBoxLayout()
        csv_file_label = QLabel("재생 데이터 파일 (.bin/.npz/.csv/.zip):")
        csv_file_label.setToolTip(mw.csv_input.toolTip())
        csv_file_layout.addWidget(csv_file_label)
        csv_file_layout.addWidget(mw.csv_input)

        csv_browse_btn = QPushButton("...")
        csv_browse_btn.setFixedWidth(30)
        csv_browse_btn.setToolTip("재생할 .bin, .npz, .csv 또는 .zip 파일을 선택합니다.")
        csv_browse_btn.clicked.connect(mw.browse_csv_file)
        csv_file_layout.addWidget(csv_browse_btn)

        csv_layout.addLayout(csv_file_layout)
        layout_conn.addWidget(mw.csv_group)

        mw.connect_btn = QPushButton("연결 시작")
        mw.connect_btn.clicked.connect(mw.toggle_connection)
        mw.connect_btn.setStyleSheet("background-color: #198754; font-weight: bold; min-height: 40px;")
        layout_conn.addWidget(mw.connect_btn)

        # Web Server
        web_group = QGroupBox("웹 서버 연동")
        web_layout = QVBoxLayout(web_group)
        
        mw.web_url_input = QLineEdit("http://localhost:8000/api/alert")
        mw.web_enable_check = QCheckBox("낙상 감지 시 웹으로 알림 전송")
        mw.web_enable_check.setChecked(False)
        mw.web_enable_check.stateChanged.connect(mw.toggle_web_client)
        
        web_layout.addWidget(QLabel("서버 URL:"))
        web_layout.addWidget(mw.web_url_input)
        web_layout.addWidget(mw.web_enable_check)
        
        layout_conn.addWidget(web_group)
        layout_conn.addStretch()

        MonitorUIBuilder._on_source_changed(mw, 0)
        mw.tabs.addTab(MonitorUIBuilder._as_scroll_tab(tab_conn), "연결")

    @staticmethod
    def _on_source_changed(mw, index):
        if index == 0:
            mw.radar_group.setVisible(True)
            mw.record_group.setVisible(True)
            mw.csv_group.setVisible(False)
        else:
            mw.radar_group.setVisible(False)
            mw.record_group.setVisible(False)
            mw.csv_group.setVisible(True)

    @staticmethod
    def _setup_settings_tab(mw):
        tab_set = QWidget()
        layout_set = QVBoxLayout(tab_set)
        
        # 1. Sensitivity
        sens_group = QGroupBox("1. AI 감지 민감도")
        sens_layout = QGridLayout(sens_group)
        
        mw.speed_slider = NoWheelSlider(Qt.Horizontal)
        mw.speed_slider.setRange(0, 100)
        mw.speed_slider.setValue(70) 
        mw.speed_slider.setToolTip(
            "하강 속도 기준 (vz <= 임계값)\n"
            "오른쪽: 민감, 느린 낙상 감지, 오탐 증가\n"
            "왼쪽: 보수적, 빠른 하강만 감지, 오탐 감소\n"
            "영향: 룰 후보, 화면 상태, 보조 판정"
        )
        mw.speed_slider.valueChanged.connect(mw.update_sensitivity)
        mw.speed_label = QLabel("속도 임계값:")
        mw.speed_label.setToolTip(mw.speed_slider.toolTip())
        sens_layout.addWidget(mw.speed_label, 0, 0)
        sens_layout.addWidget(mw.speed_slider, 0, 1)
        
        mw.height_slider = NoWheelSlider(Qt.Horizontal)
        mw.height_slider.setRange(0, 100)
        mw.height_slider.setValue(36)
        mw.height_slider.setToolTip(
            "낮은 자세/접지 높이 기준\n"
            "오른쪽: 민감, 침대/매트 낙상 감지, 오탐 증가\n"
            "왼쪽: 보수적, 바닥 근접만 감지, 오탐 감소\n"
            "영향: 룰 후보, AI 호출, GROUND/LOW POSTURE 표시"
        )
        mw.height_slider.valueChanged.connect(mw.update_sensitivity)
        mw.height_label = QLabel("높이 임계값:")
        mw.height_label.setToolTip(mw.height_slider.toolTip())
        sens_layout.addWidget(mw.height_label, 1, 0)
        sens_layout.addWidget(mw.height_slider, 1, 1)

        # Falling 카테고리 확률 임계값
        mw.falling_prob_spin = NoWheelSpinBox()
        mw.falling_prob_spin.setRange(1, 99999)
        MonitorUIBuilder._allow_large_entry(mw.falling_prob_spin)
        mw.falling_prob_spin.setValue(50)
        mw.falling_prob_spin.setSuffix(" %")
        mw.falling_prob_spin.setMinimumWidth(70)
        mw.falling_prob_spin.setMaximumWidth(90)
        mw.falling_prob_spin.setToolTip(
            "AI 모델이 계산한 Falling 확률을 낙상으로 인정하는 기준입니다.\n\n"
            "예: 50%라면 AI의 Falling 확률이 0.50 이상일 때 낙상 쪽으로 판단합니다.\n"
            "예: 20%라면 AI가 약하게라도 Falling 가능성을 보면 더 쉽게 반응합니다.\n\n"
            "값을 낮추면 민감도가 올라갑니다.\n"
            "  - 장점: 애매한 낙상, 일부만 보이는 낙상, 시뮬/실데이터 차이가 있는 상황을 더 잘 잡을 수 있습니다.\n"
            "  - 단점: 앉기, 눕기, 숙이기, 빠른 움직임을 낙상으로 오탐할 수 있습니다.\n\n"
            "값을 높이면 더 확실한 Falling일 때만 반응합니다.\n"
            "  - 장점: 오탐이 줄어듭니다.\n"
            "  - 단점: 실제 낙상이어도 AI 확률이 낮게 나온 경우 놓칠 수 있습니다.\n\n"
            "룰 후보 필터 ON: 룰이 낙상 후보를 추린 뒤 AI 점수가 보조로 반영됩니다.\n"
            "룰 후보 필터 OFF: 룰 latch 없이 이 AI 확률 기준이 최종 낙상 판단에 직접 사용됩니다."
        )
        mw.falling_prob_spin.valueChanged.connect(mw.update_ai_params)
        mw.falling_prob_label = QLabel("Falling 판정 임계 확률:")
        mw.falling_prob_label.setToolTip(mw.falling_prob_spin.toolTip())

        mw.falling_prob_slider = NoWheelSlider(Qt.Horizontal)
        mw.falling_prob_slider.setRange(1, 100)
        mw.falling_prob_slider.setValue(mw.falling_prob_spin.value())
        mw.falling_prob_slider.setToolTip(mw.falling_prob_spin.toolTip())
        mw.falling_prob_slider.valueChanged.connect(mw.falling_prob_spin.setValue)
        mw.falling_prob_spin.valueChanged.connect(mw.falling_prob_slider.setValue)

        falling_prob_row = QWidget()
        falling_prob_layout = QHBoxLayout(falling_prob_row)
        falling_prob_layout.setContentsMargins(0, 0, 0, 0)
        falling_prob_layout.setSpacing(8)
        falling_prob_layout.addWidget(mw.falling_prob_spin)
        falling_prob_layout.addWidget(mw.falling_prob_slider, 1)
        sens_layout.addWidget(mw.falling_prob_label, 2, 0)
        sens_layout.addWidget(falling_prob_row, 2, 1)

        mw.ai_model_combo = NoWheelComboBox()
        model_tooltip = (
            "추론에 사용할 AI 모델(.pkl)을 선택합니다.\n\n"
            "목록은 앱의 models 폴더와 modules 폴더에서 찾은 .pkl 파일입니다.\n"
            "선택을 바꾸면 즉시 모델을 다시 로드합니다.\n\n"
            "새 모델을 추가하려면 models/latest 또는 models 하위 폴더에 .pkl 파일을 넣은 뒤 앱을 다시 실행하세요."
        )
        mw.ai_model_combo.setToolTip(model_tooltip)
        discovered_models = MonitorUIBuilder._discover_ai_models()
        current_model_path = None
        settings = QSettings("FallDetector", "RadarApp")
        saved_model_path = settings.value("selected_ai_model_path", "")
        if hasattr(mw, "detector") and getattr(mw.detector, "ai_model", None) is not None:
            current_path = getattr(mw.detector.ai_model, "model_path", None)
            if current_path and os.path.exists(str(current_path)):
                current_model_path = os.path.abspath(str(current_path))
        if current_model_path is None and saved_model_path and os.path.exists(saved_model_path):
            current_model_path = os.path.abspath(saved_model_path)

        if discovered_models:
            for label, path in discovered_models:
                mw.ai_model_combo.addItem(label, path)
            if current_model_path:
                for index in range(mw.ai_model_combo.count()):
                    if os.path.abspath(mw.ai_model_combo.itemData(index)) == current_model_path:
                        mw.ai_model_combo.setCurrentIndex(index)
                        break
        else:
            mw.ai_model_combo.addItem("사용 가능한 .pkl 모델 없음", "")
            mw.ai_model_combo.setEnabled(False)

        mw.ai_model_combo.currentIndexChanged.connect(mw.update_ai_params)
        mw.ai_model_label = QLabel("사용할 AI 모델:")
        mw.ai_model_label.setToolTip(model_tooltip)
        sens_layout.addWidget(mw.ai_model_label, 3, 0)
        sens_layout.addWidget(mw.ai_model_combo, 3, 1)
        
        layout_set.addWidget(sens_group)

        mw.advanced_settings_toggle = QPushButton("세부 설정 열기")
        mw.advanced_settings_toggle.setCheckable(True)
        mw.advanced_settings_toggle.setChecked(False)
        mw.advanced_settings_container = QWidget()
        advanced_layout = QVBoxLayout(mw.advanced_settings_container)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(6)
        mw.advanced_settings_container.setVisible(False)

        def _toggle_advanced_settings(checked):
            mw.advanced_settings_container.setVisible(bool(checked))
            mw.advanced_settings_toggle.setText("세부 설정 닫기" if checked else "세부 설정 열기")

        mw.advanced_settings_toggle.toggled.connect(_toggle_advanced_settings)
        layout_set.addWidget(mw.advanced_settings_toggle)
        layout_set.addWidget(mw.advanced_settings_container)
        
        # 2. Tracking Config 
        track_group = QGroupBox("2. 추적 알고리즘 유지 시간")
        track_layout = QGridLayout(track_group)
        
        mw.max_missing_spin = NoWheelSpinBox()
        mw.max_missing_spin.setRange(1, 99999)
        MonitorUIBuilder._allow_large_entry(mw.max_missing_spin)
        mw.max_missing_spin.setValue(20)
        MonitorUIBuilder._expand_input(mw.max_missing_spin)
        if hasattr(mw, 'update_tracker_params'):
            mw.max_missing_spin.valueChanged.connect(mw.update_tracker_params)
        
        track_layout.addWidget(QLabel("신호 소실 유지 프레임 (Shadow 그림자):"), 0, 0)
        track_layout.addWidget(mw.max_missing_spin, 0, 1)

        mw.predict_missing_motion_check = QCheckBox("신호 소실 중 등속도 예측 이동")
        mw.predict_missing_motion_check.setChecked(False)
        mw.predict_missing_motion_check.setToolTip(
            "OFF: 신호가 끊긴 트랙은 마지막 위치 근처에 그대로 남깁니다.\n"
            "ON: 신호가 끊긴 동안 칼만필터 속도값으로 위치를 등속도 예측 이동합니다.\n"
            "mmWave 포인트가 드문드문 끊기는 환경에서는 실제 위치와 다르게 떠밀릴 수 있어 기본값은 OFF입니다."
        )
        if hasattr(mw, 'update_tracker_params'):
            mw.predict_missing_motion_check.stateChanged.connect(mw.update_tracker_params)
        track_layout.addWidget(mw.predict_missing_motion_check, 1, 0, 1, 2)

        mw.lost_descent_fall_check = QCheckBox("급하강 후 신호 소실 낙상 후보")
        mw.lost_descent_fall_check.setChecked(False)
        mw.lost_descent_fall_check.setToolTip(
            "OFF (기본값): 신호가 끊긴 것만으로는 낙상으로 보지 않습니다.\n"
            "ON: 사람 클러스터/트랙 중심 높이가 빠르게 내려간 직후 신호가 끊기면 보조 낙상 룰로 봅니다.\n\n"
            "가까운 거리에서는 바닥에 앉거나 넘어진 뒤 포인트가 시야각 아래로 빠져 사라질 수 있습니다.\n"
            "이 기능은 그런 상황을 보완하지만, 센서 상하 각도/거리/가려짐에 따라 오탐 가능성이 있어 기본값은 OFF입니다."
        )
        if hasattr(mw, 'update_tracker_params'):
            mw.lost_descent_fall_check.stateChanged.connect(mw.update_tracker_params)
        track_layout.addWidget(mw.lost_descent_fall_check, 2, 0, 1, 2)

        lost_descent_tip = (
            "급하강 후 신호 소실 룰의 세부 기준입니다.\n"
            "신호 소실 확인 프레임: 몇 프레임 연속으로 트랙 갱신이 끊기면 소실로 볼지 정합니다.\n"
            "급하강 기억 프레임: 급격한 하강을 본 뒤 몇 프레임 안의 소실까지 연결할지 정합니다.\n"
            "중심 하강 기준: 한 프레임 사이 중심 높이가 이 값 이상 내려가면 급하강 후보로 기억합니다."
        )

        mw.lost_descent_missing_spin = NoWheelSpinBox()
        mw.lost_descent_missing_spin.setRange(1, 99999)
        MonitorUIBuilder._allow_large_entry(mw.lost_descent_missing_spin)
        mw.lost_descent_missing_spin.setValue(3)
        mw.lost_descent_missing_spin.setSuffix(" 프레임")
        MonitorUIBuilder._expand_input(mw.lost_descent_missing_spin)
        mw.lost_descent_missing_spin.setToolTip(lost_descent_tip)
        if hasattr(mw, 'update_tracker_params'):
            mw.lost_descent_missing_spin.valueChanged.connect(mw.update_tracker_params)
        lost_missing_label = QLabel("신호 소실 확인 프레임:")
        lost_missing_label.setToolTip(lost_descent_tip)
        track_layout.addWidget(lost_missing_label, 3, 0)
        track_layout.addWidget(mw.lost_descent_missing_spin, 3, 1)

        mw.lost_descent_window_spin = NoWheelSpinBox()
        mw.lost_descent_window_spin.setRange(1, 99999)
        MonitorUIBuilder._allow_large_entry(mw.lost_descent_window_spin)
        mw.lost_descent_window_spin.setValue(10)
        mw.lost_descent_window_spin.setSuffix(" 프레임")
        MonitorUIBuilder._expand_input(mw.lost_descent_window_spin)
        mw.lost_descent_window_spin.setToolTip(lost_descent_tip)
        if hasattr(mw, 'update_tracker_params'):
            mw.lost_descent_window_spin.valueChanged.connect(mw.update_tracker_params)
        lost_window_label = QLabel("급하강 기억 프레임:")
        lost_window_label.setToolTip(lost_descent_tip)
        track_layout.addWidget(lost_window_label, 4, 0)
        track_layout.addWidget(mw.lost_descent_window_spin, 4, 1)

        mw.lost_descent_drop_spin = NoWheelDoubleSpinBox()
        mw.lost_descent_drop_spin.setRange(0.01, 99999.0)
        MonitorUIBuilder._allow_large_entry(mw.lost_descent_drop_spin)
        mw.lost_descent_drop_spin.setSingleStep(0.01)
        mw.lost_descent_drop_spin.setValue(0.08)
        mw.lost_descent_drop_spin.setDecimals(2)
        mw.lost_descent_drop_spin.setSuffix(" m")
        MonitorUIBuilder._expand_input(mw.lost_descent_drop_spin)
        mw.lost_descent_drop_spin.setToolTip(lost_descent_tip)
        if hasattr(mw, 'update_tracker_params'):
            mw.lost_descent_drop_spin.valueChanged.connect(mw.update_tracker_params)
        lost_drop_label = QLabel("중심 하강 기준:")
        lost_drop_label.setToolTip(lost_descent_tip)
        track_layout.addWidget(lost_drop_label, 5, 0)
        track_layout.addWidget(mw.lost_descent_drop_spin, 5, 1)
        
        advanced_layout.addWidget(track_group)

        # 3. Sensor Config
        calib_group = QGroupBox("3. 센서 환경 보정")
        calib_layout = QGridLayout(calib_group)
        calib_layout.setColumnStretch(1, 1)
        calib_layout.setColumnStretch(3, 1)

        mw.sensor_tilt_spin = NoWheelDoubleSpinBox()
        mw.sensor_tilt_spin.setRange(-99999.0, 99999.0)
        MonitorUIBuilder._allow_large_entry(mw.sensor_tilt_spin)
        mw.sensor_tilt_spin.setSingleStep(1.0)
        mw.sensor_tilt_spin.setValue(0.0)
        MonitorUIBuilder._expand_input(mw.sensor_tilt_spin)
        mw.sensor_tilt_spin.setToolTip("Point cloud tilt correction only. Tracker boxes are not adjusted.")
        if hasattr(mw, 'update_calibration_params'):
            mw.sensor_tilt_spin.valueChanged.connect(mw.update_calibration_params)
        
        calib_layout.addWidget(QLabel("센서 상하 각도 (-하향):"), 0, 0)
        calib_layout.addWidget(mw.sensor_tilt_spin, 0, 1)

        mw.sensor_height_spin = NoWheelDoubleSpinBox()
        mw.sensor_height_spin.setRange(-99999.0, 99999.0)
        MonitorUIBuilder._allow_large_entry(mw.sensor_height_spin)
        mw.sensor_height_spin.setSingleStep(0.1)
        mw.sensor_height_spin.setValue(0.0)
        MonitorUIBuilder._expand_input(mw.sensor_height_spin)
        mw.sensor_height_spin.setToolTip("Point cloud height offset only. Tracker boxes are not adjusted.")
        if hasattr(mw, 'update_calibration_params'):
            mw.sensor_height_spin.valueChanged.connect(mw.update_calibration_params)

        calib_layout.addWidget(QLabel("센서 바닥 높이 (m):"), 0, 2)
        calib_layout.addWidget(mw.sensor_height_spin, 0, 3)

        mw.enable_firmware_box_height_check = QCheckBox("펌웨어 상자에도 높이/각도 보정 적용")
        mw.enable_firmware_box_height_check.setChecked(False)
        mw.enable_firmware_box_height_check.setToolTip("체크 시 펌웨어에서 받은 추적 상자의 위치 좌표에도 위 보정값을 적용합니다.")
        if hasattr(mw, 'update_calibration_params'):
            mw.enable_firmware_box_height_check.stateChanged.connect(mw.update_calibration_params)
        calib_layout.addWidget(mw.enable_firmware_box_height_check, 1, 0, 1, 4)

        advanced_layout.addWidget(calib_group)
        
        # 4. Clustering Config
        cluster_group = QGroupBox("4. 영상 속성 클러스터링 기반 분석")
        cluster_layout = QGridLayout(cluster_group)
        cluster_layout.setColumnStretch(1, 1)
        cluster_layout.setColumnStretch(3, 1)
        
        mw.eps_spin = NoWheelDoubleSpinBox()
        mw.eps_spin.setRange(0.1, 99999.0)
        MonitorUIBuilder._allow_large_entry(mw.eps_spin)
        mw.eps_spin.setSingleStep(0.1)
        mw.eps_spin.setValue(0.8)
        MonitorUIBuilder._expand_input(mw.eps_spin)
        mw.eps_spin.valueChanged.connect(mw.update_viz_params) # Reusing existing param update
        cluster_layout.addWidget(QLabel("DBSCAN 반경 (Eps):"), 0, 0)
        cluster_layout.addWidget(mw.eps_spin, 0, 1)
        
        mw.min_samples_spin = NoWheelSpinBox()
        mw.min_samples_spin.setRange(2, 99999)
        MonitorUIBuilder._allow_large_entry(mw.min_samples_spin)
        mw.min_samples_spin.setValue(6)
        MonitorUIBuilder._expand_input(mw.min_samples_spin)
        mw.min_samples_spin.valueChanged.connect(mw.update_viz_params)
        cluster_layout.addWidget(QLabel("최소 포인트 (MinPts):"), 0, 2)
        cluster_layout.addWidget(mw.min_samples_spin, 0, 3)

        mw.max_tracks_spin = NoWheelSpinBox()
        mw.max_tracks_spin.setRange(1, 99999)
        MonitorUIBuilder._allow_large_entry(mw.max_tracks_spin)
        mw.max_tracks_spin.setValue(6)
        MonitorUIBuilder._expand_input(mw.max_tracks_spin)
        mw.max_tracks_spin.valueChanged.connect(mw.update_viz_params)
        cluster_layout.addWidget(QLabel("최대 개체 수 (Max Tracks):"), 1, 0)
        cluster_layout.addWidget(mw.max_tracks_spin, 1, 1, 1, 3)

        mw.history_cluster_check = QCheckBox("다중 프레임 잔상 통합 클러스터링 (Better Tracking)")
        mw.history_cluster_check.setChecked(False)
        mw.history_cluster_check.stateChanged.connect(mw.update_viz_params)
        cluster_layout.addWidget(mw.history_cluster_check, 2, 0, 1, 4)

        # Trail Effect
        mw.trail_slider = NoWheelSlider(Qt.Horizontal)
        mw.trail_slider.setRange(0, 60)
        mw.trail_slider.setValue(0)
        mw.trail_slider.valueChanged.connect(mw.update_viz_params)
        
        mw.trail_label = QLabel("화면 잔상 (Trace 프레임): 0")
        cluster_layout.addWidget(mw.trail_label, 3, 0)
        cluster_layout.addWidget(mw.trail_slider, 3, 1, 1, 3)

        mw.point_height_filter_check = QCheckBox("포인트 높이 제한 사용")
        mw.point_height_filter_check.setChecked(False)
        mw.point_height_filter_check.setToolTip(
            "OFF: 받은 포인트를 높이 조건으로 버리지 않고 표시/처리합니다.\n"
            "ON: 아래 Z min/Z max 범위 안의 포인트만 클러스터링과 추적 연산에 사용합니다.\n"
            "화면에서는 범위 밖 포인트도 숨기지 않고 작은 십자가로 표시합니다."
        )
        mw.point_height_filter_check.stateChanged.connect(mw.update_viz_params)
        cluster_layout.addWidget(mw.point_height_filter_check, 4, 0, 1, 4)

        mw.pt_min_height_spin = NoWheelDoubleSpinBox()
        mw.pt_min_height_spin.setRange(-99999.0, 99999.0)
        MonitorUIBuilder._allow_large_entry(mw.pt_min_height_spin)
        mw.pt_min_height_spin.setSingleStep(0.05)
        mw.pt_min_height_spin.setValue(-0.3)
        mw.pt_min_height_spin.setDecimals(2)
        mw.pt_min_height_spin.setSuffix(" m")
        MonitorUIBuilder._expand_input(mw.pt_min_height_spin)
        mw.pt_min_height_spin.setEnabled(False)
        mw.pt_min_height_spin.valueChanged.connect(mw.update_viz_params)
        cluster_layout.addWidget(QLabel("포인트 최소 높이 (Z min):"), 5, 0)
        cluster_layout.addWidget(mw.pt_min_height_spin, 5, 1)

        mw.pt_max_height_spin = NoWheelDoubleSpinBox()
        mw.pt_max_height_spin.setRange(0.0, 99999.0)
        MonitorUIBuilder._allow_large_entry(mw.pt_max_height_spin)
        mw.pt_max_height_spin.setSingleStep(0.05)
        mw.pt_max_height_spin.setValue(3.0)
        mw.pt_max_height_spin.setDecimals(2)
        mw.pt_max_height_spin.setSuffix(" m")
        MonitorUIBuilder._expand_input(mw.pt_max_height_spin)
        mw.pt_max_height_spin.setEnabled(False)
        mw.pt_max_height_spin.valueChanged.connect(mw.update_viz_params)
        cluster_layout.addWidget(QLabel("포인트 최대 높이 (Z max):"), 5, 2)
        cluster_layout.addWidget(mw.pt_max_height_spin, 5, 3)
        
        mw.enable_clustering_check = QCheckBox("앱 자체 클러스터링 (빈틈 채우기 상자) 활성화")
        mw.enable_clustering_check.setChecked(True)
        mw.enable_clustering_check.setToolTip(
            "펌웨어 tracker target-list가 없는 프레임에서도 포인트 클러스터로 사람 상자를 만듭니다. "
            "C3CD 한 명 vital fallback 환경에서는 기본 ON이 더 안정적입니다."
        )
        if hasattr(mw, 'update_viz_params'):
            mw.enable_clustering_check.stateChanged.connect(mw.update_viz_params)
        cluster_layout.addWidget(mw.enable_clustering_check, 6, 0, 1, 4)
        
        advanced_layout.addWidget(cluster_group)
        
        # 5. AI / 기타 옵션
        test_group = QGroupBox("5. AI 추론 및 기타 옵션")
        test_layout = QGridLayout(test_group)
        test_layout.setContentsMargins(5, 5, 5, 5)
        test_layout.setVerticalSpacing(4)

        mw.sim_ai_check = QCheckBox("속도 기반 보조 판정")
        mw.sim_ai_check.setToolTip(
            "낙상 후보와 룰 점수가 충분할 때 속도 기반 보조 판정을 더 강하게 반영합니다."
        )
        test_layout.addWidget(mw.sim_ai_check, 0, 0, 1, 2)

        # AI 추론 빈도 (프레임 간격)
        mw.ai_interval_spin = NoWheelSpinBox()
        mw.ai_interval_spin.setRange(1, 99999)
        MonitorUIBuilder._allow_large_entry(mw.ai_interval_spin)
        mw.ai_interval_spin.setValue(3)
        mw.ai_interval_spin.setSuffix(" 프레임")
        MonitorUIBuilder._expand_input(mw.ai_interval_spin)
        mw.ai_interval_spin.setToolTip(
            "AI 추론이 실행되는 프레임 간격입니다.\n"
            "값이 1이면 매 프레임마다, 3이면 3프레임마다 AI가 동작합니다.\n"
            "낮을수록 반응이 빠르지만 CPU 부하가 증가합니다."
        )
        mw.ai_interval_spin.valueChanged.connect(mw.update_ai_params)
        test_layout.addWidget(QLabel("AI 추론 간격:"), 1, 0)
        test_layout.addWidget(mw.ai_interval_spin, 1, 1)

        # 룰 후보 필터 토글
        mw.ai_strict_mode_check = QCheckBox("룰 후보 필터 사용 (후보일 때만 AI)")
        mw.ai_strict_mode_check.setChecked(True)
        mw.ai_strict_mode_check.setToolTip(
            "ON (기본값): 룰 기반 코드가 직립 기준선, 하강, 접지, 낮은 자세,\n"
            "포인트 수를 보고 낙상 후보를 먼저 추립니다.\n"
            "후보로 추려진 경우에만 AI를 실행하고, 룰 확인과 AI 점수를 함께 사용합니다.\n\n"
            "OFF: 후보 필터를 쓰지 않고 AI를 지정 간격마다 직접 실행합니다.\n"
            "이때 낙상 판단은 룰 latch 없이 AI 확률 기준만 사용합니다.\n\n"
            "ON은 더 보수적이고 오탐이 적은 편이며, OFF는 더 민감하지만\n"
            "일상 자세 변화도 낙상으로 볼 가능성이 있습니다."
        )
        mw.ai_strict_mode_check.stateChanged.connect(mw.update_ai_params)
        test_layout.addWidget(mw.ai_strict_mode_check, 2, 0, 1, 2)

        advanced_layout.addWidget(test_group)
        layout_set.addStretch()
        
        mw.tabs.addTab(MonitorUIBuilder._as_scroll_tab(tab_set), "분석 및 설정")

    @staticmethod
    def _setup_viz_tab(mw):
        tab_viz = QWidget()
        layout_viz = QVBoxLayout(tab_viz)
        
        viz_group = QGroupBox("시각화 전용 설정 (Rendering)")
        viz_layout = QGridLayout(viz_group)
        viz_layout.setContentsMargins(5, 5, 5, 5)
        viz_layout.setVerticalSpacing(2)
        
        # Pt Size -> Index 0
        mw.pt_size_slider = NoWheelSlider(Qt.Horizontal)
        mw.pt_size_slider.setRange(2, 50)
        mw.pt_size_slider.setValue(10)
        mw.pt_size_slider.valueChanged.connect(mw.update_viz_params)
        
        mw.pt_size_label = QLabel("점 크기: 10")
        viz_layout.addWidget(mw.pt_size_label, 0, 0)
        viz_layout.addWidget(mw.pt_size_slider, 0, 1)
        
        # Style
        mw.style_combo = NoWheelComboBox()
        mw.style_combo.addItems(["점 (Points)", "스켈레톤 (Skeleton)"])
        mw.style_combo.setCurrentIndex(0)
        mw.style_combo.setMinimumWidth(220)
        mw.style_combo.setMaxVisibleItems(8)
        mw.style_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        mw.style_combo.currentIndexChanged.connect(mw.update_viz_params)
        viz_layout.addWidget(QLabel("시각화 스타일:"), 1, 0)
        viz_layout.addWidget(mw.style_combo, 1, 1)

        mw.vel_legend_label = QLabel("Doppler: 적=접근 / 청=이탈 / 회=정지")
        viz_layout.addWidget(mw.vel_legend_label, 2, 0, 1, 2)

        mw.doppler_color_scale_slider = NoWheelSlider(Qt.Horizontal)
        mw.doppler_color_scale_slider.setRange(5, 500)
        mw.doppler_color_scale_slider.setValue(430)
        mw.doppler_color_scale_spin = NoWheelDoubleSpinBox()
        mw.doppler_color_scale_spin.setRange(0.05, 5.00)
        mw.doppler_color_scale_spin.setSingleStep(0.05)
        mw.doppler_color_scale_spin.setDecimals(2)
        mw.doppler_color_scale_spin.setValue(4.30)
        mw.doppler_color_scale_spin.setSuffix(" m/s")
        MonitorUIBuilder._allow_large_entry(mw.doppler_color_scale_spin)
        mw.doppler_color_scale_slider.valueChanged.connect(
            lambda value: mw.doppler_color_scale_spin.setValue(float(value) / 100.0)
        )
        mw.doppler_color_scale_spin.valueChanged.connect(
            lambda value: mw.doppler_color_scale_slider.setValue(int(round(float(value) * 100.0)))
        )
        mw.doppler_color_scale_spin.valueChanged.connect(mw.update_viz_params)
        doppler_color_layout = QHBoxLayout()
        doppler_color_layout.setContentsMargins(0, 0, 0, 0)
        doppler_color_layout.addWidget(mw.doppler_color_scale_slider, 1)
        doppler_color_layout.addWidget(mw.doppler_color_scale_spin)
        viz_layout.addWidget(QLabel("Doppler color scale:"), 6, 0)
        viz_layout.addLayout(doppler_color_layout, 6, 1)

        # Highlight Objects Toggle
        mw.highlight_objects_check = QCheckBox("감지된 객체 포인트 강조 (Highlight Objects)")
        mw.highlight_objects_check.setChecked(False) 
        mw.highlight_objects_check.stateChanged.connect(mw.update_viz_params)
        viz_layout.addWidget(mw.highlight_objects_check, 3, 0, 1, 2)

        # Cluster Distance Filter
        mw.cluster_dist_filter_check = QCheckBox("클러스터 외곽 점 제외 (Cluster Distance Filter)")
        mw.cluster_dist_filter_check.setChecked(False)
        mw.cluster_dist_filter_check.stateChanged.connect(mw.update_viz_params)
        viz_layout.addWidget(mw.cluster_dist_filter_check, 4, 0, 1, 2)

        mw.cluster_dist_spin = NoWheelDoubleSpinBox()
        mw.cluster_dist_spin.setRange(0.1, 99999.0)
        MonitorUIBuilder._allow_large_entry(mw.cluster_dist_spin)
        mw.cluster_dist_spin.setSingleStep(0.1)
        mw.cluster_dist_spin.setValue(0.5)
        mw.cluster_dist_spin.setDecimals(2)
        mw.cluster_dist_spin.setSuffix(" m")
        mw.cluster_dist_spin.setEnabled(False)  # 체크 여부에 따라 활성
        mw.cluster_dist_spin.valueChanged.connect(mw.update_viz_params)

        mw.cluster_dist_filter_check.stateChanged.connect(
            lambda state: mw.cluster_dist_spin.setEnabled(bool(state))
        )

        mw.cluster_dist_label = QLabel("허용 최대 거리:")
        viz_layout.addWidget(mw.cluster_dist_label, 5, 0)
        viz_layout.addWidget(mw.cluster_dist_spin, 5, 1)

        layout_viz.addWidget(viz_group)
        
        layout_viz.addStretch()
        mw.tabs.addTab(MonitorUIBuilder._as_scroll_tab(tab_viz), "시각화")

    @staticmethod
    def _setup_viewer(mw):
        mw.viewer = gl.GLViewWidget()
        mw.viewer.opts['distance'] = 5
        mw.viewer.opts['elevation'] = 20
        mw.viewer.opts['azimuth'] = -45
        mw.viewer.setBackgroundColor('#1e1e1e')
        
        g = gl.GLGridItem()
        g.setSize(x=10, y=10, z=1)
        g.setSpacing(x=1, y=1, z=1)
        mw.viewer.addItem(g)
        
        axis = gl.GLAxisItem()
        axis.setSize(1, 1, 1)
        mw.viewer.addItem(axis)
        
        mw.scatter = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(1,1,1,0.5), size=5.0, pxMode=True)
        mw.scatter.setGLOptions('translucent')
        mw.viewer.addItem(mw.scatter)
        
        mw.trail_scatter = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0.5,0.5,0.5,0.3), size=3.0, pxMode=True)
        mw.trail_scatter.setGLOptions('translucent')
        mw.viewer.addItem(mw.trail_scatter)

        mw.skeleton_plot = gl.GLLinePlotItem(pos=np.empty((0,3)), color=(0, 0.7, 1, 0.8), width=3, mode='lines')
        mw.viewer.addItem(mw.skeleton_plot)
        
        mw.track_items = {}
