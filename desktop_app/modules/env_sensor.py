import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvReading:
    temperature_c: Optional[float] = None
    humidity_percent: Optional[float] = None
    ok: bool = False
    status: str = "not started"


class DhtEnvironmentSensor:
    def __init__(self, pin_name: str = None, sensor_type: str = None, min_interval_sec: float = 2.0):
        self.pin_name = pin_name or os.environ.get("FALL_ENV_SENSOR_PIN", "D4")
        self.sensor_type = (sensor_type or os.environ.get("FALL_ENV_SENSOR_TYPE", "DHT22")).upper()
        self.min_interval_sec = max(1.0, float(min_interval_sec))
        self._device = None
        self._last_read_at = 0.0
        self._last = EnvReading(status="waiting")
        self._init_error = ""

    def close(self) -> None:
        device = self._device
        self._device = None
        if device is not None and hasattr(device, "exit"):
            try:
                device.exit()
            except Exception:
                pass

    def read(self) -> EnvReading:
        now = time.monotonic()
        if now - self._last_read_at < self.min_interval_sec:
            return self._last

        self._last_read_at = now
        if self._device is None and not self._init_device():
            self._last = EnvReading(status=self._init_error or "sensor unavailable")
            return self._last

        try:
            temperature = self._device.temperature
            humidity = self._device.humidity
        except RuntimeError as exc:
            self._last.status = str(exc) or "transient read error"
            return self._last
        except Exception as exc:
            self._last = EnvReading(status=str(exc) or "sensor read failed")
            return self._last

        if temperature is None or humidity is None:
            self._last.status = "no data"
            return self._last

        self._last = EnvReading(
            temperature_c=float(temperature),
            humidity_percent=float(humidity),
            ok=True,
            status="ok",
        )
        return self._last

    def _init_device(self) -> bool:
        try:
            import board
            import adafruit_dht
        except Exception as exc:
            self._init_error = f"missing sensor package: {exc}"
            return False

        pin = getattr(board, self.pin_name, None)
        if pin is None:
            self._init_error = f"unknown board pin: {self.pin_name}"
            return False

        try:
            sensor_cls = adafruit_dht.DHT11 if self.sensor_type == "DHT11" else adafruit_dht.DHT22
            self._device = sensor_cls(pin)
            self._init_error = ""
            return True
        except Exception as exc:
            self._init_error = f"sensor init failed: {exc}"
            return False
