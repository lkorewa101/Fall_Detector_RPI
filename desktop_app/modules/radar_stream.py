import threading
import time

from .radar_interface import RadarInterface
from .tlv_parser import TlvFrameParser


class RadarStream(threading.Thread):
    def __init__(
        self,
        cli_port,
        data_port,
        cfg_path,
        callback,
        log_callback=None,
        state_callback=None,
    ):
        super().__init__()
        self.cli_port = cli_port
        self.data_port = data_port
        self.cfg_path = cfg_path
        self.callback = callback
        self.log_callback = log_callback
        self.state_callback = state_callback

        self.radar = RadarInterface()
        self.parser = TlvFrameParser()
        self.running = False
        self.stop_requested = False
        self.configured = False
        self.bytes_received = 0
        self.chunks_received = 0
        self.frames_parsed = 0
        self.last_chunk_time = 0.0
        self.first_data_timeout = 15.0
        self.first_frame_timeout = 15.0
        self.daemon = True

    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def emit_state(self, event, **info):
        if not self.state_callback:
            return
        payload = {
            "event": event,
            "stream_id": id(self),
            "cli_port": self.cli_port,
            "data_port": self.data_port,
            "bytes_received": self.bytes_received,
            "chunks_received": self.chunks_received,
            "frames_parsed": self.frames_parsed,
            **info,
        }
        try:
            self.state_callback(payload)
        except Exception:
            pass

    def run(self):
        self.emit_state("connecting")
        try:
            if not self.radar.connect(self.cli_port, self.data_port):
                self.log("[RadarStream] Connection failed")
                self.emit_state("connection_failed", message="serial connect failed")
                return

            self.emit_state("configuring")
            success, msg = self.radar.send_config(self.cfg_path, log_callback=self.log)
            if not success:
                self.log(f"[RadarStream] Config failed: {msg}")
                self.emit_state("config_failed", message=msg)
                return

            self.configured = True
            self.running = True
            wait_started_at = time.time()
            first_chunk_reported = False
            self.log("[RadarStream] Config sent. Waiting for Data UART bytes...")
            self.emit_state("waiting_data")

            while self.running:
                chunk = self.radar.read_frame()
                if not chunk:
                    if not self.radar.is_connected and not self.stop_requested:
                        err = getattr(self.radar, "last_error", None)
                        message = str(err) if err else "data port closed"
                        self.log(f"[RadarStream] Data port error: {message}")
                        self.emit_state("error", message=message)
                        break

                    elapsed = time.time() - wait_started_at
                    if (
                        not self.stop_requested
                        and self.bytes_received == 0
                        and elapsed >= self.first_data_timeout
                    ):
                        message = (
                            f"Data UART 0 bytes for {self.first_data_timeout:.0f}s. "
                            "Check functional SOP mode, NRST reset, firmware boot, and CLI/Data port order."
                        )
                        self.log(f"[RadarStream] {message}")
                        self.emit_state("no_data", message=message)
                        break

                    if (
                        not self.stop_requested
                        and self.bytes_received > 0
                        and self.frames_parsed == 0
                        and elapsed >= self.first_frame_timeout
                    ):
                        message = (
                            f"Data UART received {self.bytes_received} bytes but no TLV frames parsed. "
                            "Check data baud rate and firmware output protocol."
                        )
                        self.log(f"[RadarStream] {message}")
                        self.emit_state("parse_stalled", message=message)
                        break

                    time.sleep(0.005)
                    continue

                self.bytes_received += len(chunk)
                self.chunks_received += 1
                self.last_chunk_time = time.time()
                if not first_chunk_reported:
                    first_chunk_reported = True
                    self.emit_state("data_received")
                if self.chunks_received <= 3:
                    self.log(f"[RadarStream] Data UART chunk #{self.chunks_received}: {len(chunk)} bytes")

                frames = self.parser.push_and_extract(chunk)
                if frames and self.frames_parsed == 0:
                    self.log(f"[RadarStream] First parsed TLV frames: {len(frames)}")
                for frame_bytes in frames:
                    parsed = self.parser.parse_frame(frame_bytes)
                    if parsed:
                        self.frames_parsed += 1
                        if self.frames_parsed == 1:
                            self.log("[RadarStream] First valid TLV frame parsed. Started.")
                            self.emit_state("started")
                        self.callback(parsed)
        except Exception as exc:
            if not self.stop_requested:
                self.log(f"[RadarStream] Stream error: {exc}")
                self.emit_state("error", message=str(exc))
        finally:
            self.running = False
            self.radar.stop()
            self.emit_state("stopped", requested=self.stop_requested, configured=self.configured)

    def stop(self):
        self.stop_requested = True
        self.running = False
        self.radar.stop()
