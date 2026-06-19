import threading
import time
import os
from .tlv_parser import TlvFrameParser, ParsedFrame

class BinStream(threading.Thread):
    def __init__(self, bin_path, callback, log_callback=None, fps=30, progress_callback=None):
        super().__init__()
        self.bin_path = bin_path
        self.callback = callback
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.fps = fps
        self.running = False
        self.paused = False
        self.current_idx = 0
        self.frames = []
        self.total_frames = 0
        self.speed_multiplier = 1.0
        self.daemon = True

        self._load_frames()

    def _load_frames(self):
        try:
            with open(self.bin_path, "rb") as f:
                file_bytes = f.read()

            parser = TlvFrameParser()
            self.frames = parser.push_and_extract(file_bytes)
            self.total_frames = len(self.frames)
            
            if self.log_callback:
                self.log_callback(f"[BinStream] Loaded {self.total_frames} valid TLV frames from {os.path.basename(self.bin_path)}")
            else:
                print(f"[BinStream] Loaded {self.total_frames} frames.")
        except Exception as e:
            if self.log_callback:
                self.log_callback(f"[BinStream] Error reading BIN file: {e}")
            else:
                print(f"Error reading BIN file: {e}")

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

    def run(self):
        if self.total_frames == 0:
            return

        self.running = True
        self.log(f"[BinStream] Starting playback: {self.total_frames} frames at {self.fps} FPS")
        parser = TlvFrameParser()

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
            
            # Get raw TLV frame bytes and parse it
            raw_frame_bytes = self.frames[self.current_idx]
            parsed = parser.parse_frame(raw_frame_bytes, ts=time.time())
            
            if parsed is None:
                parsed = ParsedFrame(ts=time.time(), points=None, tracks=[], raw=raw_frame_bytes)

            self.callback(parsed)
            
            if self.progress_callback:
                self.progress_callback(self.current_idx + 1, self.total_frames)
                
            self.current_idx += 1
            
            elapsed = time.time() - start_time
            current_interval = 1.0 / (self.fps * self.speed_multiplier)
            sleep_time = max(0, current_interval - elapsed)
            time.sleep(sleep_time)

        self.running = False
        self.log("[BinStream] Playback finished")

    def stop(self):
        self.running = False
