import struct
import sys
import threading
import time
from pathlib import Path

MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
VENDOR_DIR = Path.home() / "iwr6843_firmware_release" / "tools" / "vendor"


try:
    import serial
except ImportError:
    if VENDOR_DIR.exists():
        sys.path.insert(0, str(VENDOR_DIR))
    import serial


def _hold_control_lines_low(ser):
    try:
        ser.dtr = False
    except Exception:
        pass
    try:
        ser.rts = False
    except Exception:
        pass
    try:
        ser.setDTR(False)
    except Exception:
        pass
    try:
        ser.setRTS(False)
    except Exception:
        pass


def _open_serial_port(port, baudrate, timeout=1, write_timeout=2):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.timeout = timeout
    ser.write_timeout = write_timeout
    ser.rtscts = False
    ser.dsrdtr = False
    ser.xonxoff = False
    _hold_control_lines_low(ser)
    ser.open()
    _hold_control_lines_low(ser)
    return ser


def _close_serial_port(ser):
    if not ser:
        return
    try:
        _hold_control_lines_low(ser)
    except Exception:
        pass
    try:
        ser.close()
    except Exception:
        pass


class RadarInterface:
    def __init__(self):
        self.cli_serial = None
        self.data_serial = None
        self.is_connected = False
        self.last_error = None
        self.stop_requested = False
        self._io_lock = threading.RLock()

    def connect(self, cli_port, data_port):
        try:
            with self._io_lock:
                if self.cli_serial:
                    _close_serial_port(self.cli_serial)
                if self.data_serial:
                    _close_serial_port(self.data_serial)

                self.last_error = None
                self.stop_requested = False
                self.cli_serial = _open_serial_port(cli_port, 115200, timeout=1, write_timeout=2)
                self.data_serial = _open_serial_port(data_port, 921600, timeout=0.1, write_timeout=2)
            time.sleep(0.5)
            self.is_connected = True
            return True
        except Exception as exc:
            print(f"[Interface Error] {exc}")
            self.last_error = exc
            self.is_connected = False
            return False

    def _log(self, msg, log_callback=None):
        if log_callback:
            log_callback(msg)
            return

        try:
            print(msg)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            safe_msg = msg.encode(encoding, errors="replace").decode(encoding, errors="replace")
            print(safe_msg)

    def _read_cli_response(self, timeout_sec=0.5, idle_break_sec=0.12):
        end_time = time.time() + timeout_sec
        last_data_time = None
        chunks = []

        while time.time() < end_time:
            ser = self.cli_serial
            if self.stop_requested or ser is None:
                break
            waiting = ser.in_waiting
            if waiting > 0:
                chunk = ser.read(waiting).decode(errors="ignore")
                if chunk:
                    chunks.append(chunk)
                    last_data_time = time.time()
                time.sleep(0.02)
                continue

            if last_data_time is not None and (time.time() - last_data_time) >= idle_break_sec:
                break
            time.sleep(0.02)

        return "".join(chunks)

    def _send_cli_command(self, cmd, response_timeout=0.35):
        if self.stop_requested or self.cli_serial is None:
            raise RuntimeError("CLI port closed")
        self.cli_serial.write((cmd + "\n").encode())
        self.cli_serial.flush()
        time.sleep(0.05)
        return self._read_cli_response(timeout_sec=response_timeout)

    def send_config(self, cfg_path, log_callback=None):
        if not self.is_connected or self.stop_requested:
            return False, "Not connected"

        log = lambda msg: self._log(msg, log_callback)
        log("[Config] Sending radar config...")

        try:
            if self.cli_serial:
                self.cli_serial.reset_input_buffer()
                self.cli_serial.reset_output_buffer()
            if self.data_serial:
                self.data_serial.reset_input_buffer()
        except Exception:
            pass

        # Some firmware builds do not answer an empty CLI line. Treat this as
        # optional and validate the port by sending real commands below.
        try:
            if self.stop_requested or self.cli_serial is None:
                return False, "Stopped before CLI probe"
            time.sleep(0.2)
            self.cli_serial.write(b"\n")
            self.cli_serial.flush()
            prompt_resp = self._read_cli_response(timeout_sec=0.8)
        except Exception as exc:
            return False, f"CLI probe failed: {exc}"

        if prompt_resp.strip():
            for line in prompt_resp.splitlines():
                if line.strip():
                    log(f"  >> {line.strip()}")

        try:
            stop_resp = self._send_cli_command("sensorStop", response_timeout=1.0)
            flush_resp = self._send_cli_command("flushCfg", response_timeout=1.0)
            if "Error" in stop_resp or "Error" in flush_resp:
                return False, (stop_resp + "\n" + flush_resp).strip()
        except Exception as exc:
            log(f"[Config] Initial sensorStop/flushCfg failed; continuing: {exc}")

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return False, f"File not found: {cfg_path}"
        except Exception as exc:
            return False, f"Config open error: {exc}"

        combined_resp = ""
        try:
            for line in lines:
                if self.stop_requested:
                    return False, "Stopped during config"
                cmd = line.strip()
                if not cmd or cmd.startswith("%") or cmd.startswith("#"):
                    continue
                if cmd in {"sensorStop", "flushCfg", "sensorStart"}:
                    continue

                log(f"[CFG] {cmd}")
                resp = self._send_cli_command(cmd, response_timeout=1.0)
                combined_resp += resp

                if resp.strip():
                    for resp_line in resp.splitlines():
                        if resp_line.strip():
                            log(f"  >> {resp_line.strip()}")

                if "Error" in resp:
                    return False, resp.strip()

            log("[CFG] sensorStart")
            if self.stop_requested:
                return False, "Stopped before sensorStart"
            sensor_resp = self._send_cli_command("sensorStart", response_timeout=3.0)
            combined_resp += sensor_resp

            if sensor_resp.strip():
                for resp_line in sensor_resp.splitlines():
                    if resp_line.strip():
                        log(f"  >> {resp_line.strip()}")
            else:
                log("[Config] No CLI response after sensorStart; waiting for Data UART.")

            if "Error" in sensor_resp:
                return False, sensor_resp.strip()

            if not combined_resp.strip():
                log(
                    "[Config] No CLI acknowledgements received. "
                    "If Data UART stays at 0 bytes, check SOP mode, NRST reset, and firmware boot."
                )

            log("[Config] Config send complete.")
            return True, "Configuration sent."
        except Exception as exc:
            return False, f"Error: {exc}"

    def stop(self, send_sensor_stop=False):
        self.stop_requested = True
        self.is_connected = False
        with self._io_lock:
            if self.cli_serial:
                if send_sensor_stop:
                    try:
                        self.cli_serial.write(b"sensorStop\n")
                        self.cli_serial.flush()
                        time.sleep(0.05)
                    except Exception:
                        pass
                try:
                    _close_serial_port(self.cli_serial)
                except Exception:
                    pass
                self.cli_serial = None
            if self.data_serial:
                try:
                    _close_serial_port(self.data_serial)
                except Exception:
                    pass
                self.data_serial = None

    def read_frame(self):
        if not self.is_connected or self.data_serial is None:
            return None
        try:
            return self.data_serial.read(8192)
        except Exception as exc:
            self.last_error = exc
            self.is_connected = False
            return None

    @staticmethod
    def parse_tlv(data):
        try:
            import numpy as np

            header = data[0:40]
            if len(header) < 40:
                return None

            num_tlvs = struct.unpack("<I", header[32:36])[0]
            current_idx = 40
            points = []

            for _ in range(num_tlvs):
                if current_idx + 8 > len(data):
                    break
                tlv_type, tlv_len = struct.unpack("<2I", data[current_idx : current_idx + 8])
                current_idx += 8

                if tlv_type == 1:
                    payload = data[current_idx : current_idx + tlv_len]
                    points = np.frombuffer(payload, dtype=np.float32).reshape(-1, 4)

                current_idx += tlv_len

            return points
        except Exception:
            return None
