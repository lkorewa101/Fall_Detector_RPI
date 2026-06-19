import threading
import time
import numpy as np
import re
from .tlv_parser import ParsedFrame
from .pose_estimator import load_sidecar_skeletons

class NpzStream(threading.Thread):
    def __init__(self, npz_path, callback, log_callback=None, fps=30, progress_callback=None):
        super().__init__()
        self.npz_path = npz_path
        self.callback = callback
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.fps = fps
        self.running = False
        self.paused = False
        self.current_idx = 0
        self.total_frames = 0
        self.speed_multiplier = 1.0
        self.daemon = True
        self.sidecar_skeletons = {}

    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def toggle_pause(self):
        self.paused = not self.paused

    def seek(self, percent):
        if self.total_frames > 0:
            self.current_idx = int((percent / 100.0) * (self.total_frames - 1))
            self.current_idx = max(0, min(self.current_idx, self.total_frames - 1))

    def set_speed(self, speed):
        self.speed_multiplier = speed

    def _sidecar_for_frame(self, frame_key):
        try:
            return self.sidecar_skeletons.get(int(frame_key))
        except Exception:
            return self.sidecar_skeletons.get(self.current_idx)

    def run(self):
        try:
            self.sidecar_skeletons = load_sidecar_skeletons(self.npz_path)
            if self.sidecar_skeletons:
                self.log(f"[NpzStream] Loaded sidecar joints.npz skeletons: {len(self.sidecar_skeletons)} frames")
            data_dict = np.load(self.npz_path, allow_pickle=True)
            keys = sorted(data_dict.files)
            if not keys:
                self.log("[NpzStream] Empty NPZ file.")
                return

            if len(keys) == 1 and len(data_dict[keys[0]].shape) == 2 and data_dict[keys[0]].shape[1] >= 6:
                data = data_dict[keys[0]]
                self.log(f"[NpzStream] Loaded {len(data)} rows from {self.npz_path}")
                self._play_single_array(data)
            elif 'data' in keys and len(data_dict['data'].shape) == 2 and data_dict['data'].shape[1] >= 6:
                data = data_dict['data']
                self.log(f"[NpzStream] Loaded {len(data)} rows from {self.npz_path}")
                self._play_single_array(data)
            else:
                self.log(f"[NpzStream] Loaded {len(keys)} frames (multi-key) from {self.npz_path}")
                self._play_multi_array(data_dict, keys)
                
            data_dict.close()
        except Exception as e:
            self.log(f"[NpzStream] Error loading NPZ: {e}")
            return

    def _play_single_array(self, data):
        if data.shape[1] < 6:
            self.log("[NpzStream] Invalid single-array NPZ shape. Expected at least 6 columns.")
            return
            
        frames_col = data[:, 0]
        unique_frames = np.unique(frames_col)
        self.total_frames = len(unique_frames)
        self.current_idx = 0
        
        self.running = True
        self.log(f"[NpzStream] Starting playback: {self.total_frames} frames at {self.fps} FPS")

        while self.running:
            if self.current_idx >= self.total_frames:
                if not self.paused:
                    self.paused = True
                    if self.progress_callback:
                        self.progress_callback(self.total_frames, self.total_frames)
                time.sleep(0.1)
                continue

            if self.paused:
                time.sleep(0.1)
                continue
                
            start_time = time.time()
            frame_idx = unique_frames[self.current_idx]
            
            frame_mask = (frames_col == frame_idx)
            frame_data = data[frame_mask]
            
            points = frame_data[:, 2:6].astype(np.float32)
            
            parsed = ParsedFrame(
                ts=time.time(),
                points=points,
                tracks=[],
                raw=b"",
                skeleton_lines=self._sidecar_for_frame(frame_idx),
            )
            
            self.callback(parsed)
            
            if self.progress_callback:
                # current, total
                self.progress_callback(self.current_idx + 1, self.total_frames)
            
            self.current_idx += 1
            
            elapsed = time.time() - start_time
            current_interval = 1.0 / (self.fps * self.speed_multiplier)
            sleep_time = max(0, current_interval - elapsed)
            time.sleep(sleep_time)

        self.running = False
        self.log("[NpzStream] Playback finished")

    def _play_multi_array(self, data_dict, keys):
        self.total_frames = len(keys)
        self.current_idx = 0
        self.running = True
        self.log(f"[NpzStream] Starting playback: {self.total_frames} frames at {self.fps} FPS")
        
        # 파일명에서 각도 추출하여 센서 정면 기준으로 회전할 수 있도록 준비 (정면 고정)
        angle_rad = 0.0
        match = re.search(r'ang([0-9.]+)', self.npz_path)
        if match:
            angle_rad = np.radians(float(match.group(1)))
        
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        while self.running:
            if self.current_idx >= self.total_frames:
                if not self.paused:
                    self.paused = True
                    if self.progress_callback:
                        self.progress_callback(self.total_frames, self.total_frames)
                time.sleep(0.1)
                continue

            if self.paused:
                time.sleep(0.1)
                continue
                
            start_time = time.time()
            key = keys[self.current_idx]
            frame_data = data_dict[key]
            
            if len(frame_data.shape) >= 2 and frame_data.shape[1] >= 4:
                # 4차원 데이터 복사 및 Float32 지정
                points = frame_data[:, 0:4].copy().astype(np.float32)
                
                # 1. 뷰어에 맞게 X(우측), Y(정면_Depth) 축 회전 적용 (월드좌표 -> 센서좌표)
                wx = points[:, 0].copy()
                wy = points[:, 1].copy()
                points[:, 0] = -wx * c + wy * s  # Radar X (Right)
                points[:, 1] = -wx * s - wy * c  # Radar Y (Forward/Depth)
                
                # 2. 노이즈(바닥) 필터링: 높이(Z)가 지면 기준 0.2m 이상인 영역(사람)만 남김
                human_mask = points[:, 2] > 0.2
                points = points[human_mask]
            else:
                points = np.zeros((0, 4), dtype=np.float32)
            
            parsed = ParsedFrame(
                ts=time.time(),
                points=points,
                tracks=[],
                raw=b"",
                skeleton_lines=self._sidecar_for_frame(key),
            )
            
            self.callback(parsed)
            
            if self.progress_callback:
                self.progress_callback(self.current_idx + 1, self.total_frames)
                
            self.current_idx += 1
            
            elapsed = time.time() - start_time
            current_interval = 1.0 / (self.fps * self.speed_multiplier)
            sleep_time = max(0, current_interval - elapsed)
            time.sleep(sleep_time)

        self.running = False
        self.log("[NpzStream] Playback finished")

    def stop(self):
        self.running = False
