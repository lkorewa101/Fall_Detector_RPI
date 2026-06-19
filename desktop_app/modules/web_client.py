from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from typing import Optional

import requests
import websocket

try:
    import firebase_admin
    from firebase_admin import credentials, messaging
except Exception:  # pragma: no cover - optional dependency
    firebase_admin = None
    credentials = None
    messaging = None


class WebClient:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000/api/alert",
        ws_url: str = "ws://127.0.0.1:8000/ws",
    ) -> None:
        self.server_url = os.getenv("FALL_BACKEND_ALERT_URL", server_url)
        self.status_url = os.getenv(
            "FALL_BACKEND_STATUS_URL",
            self.server_url.replace("/api/alert", "/api/status"),
        )
        self.ws_url = os.getenv("FALL_BACKEND_WS_URL", ws_url)
        self.auth_token = os.getenv("FALL_BACKEND_TOKEN", "").strip()
        self.enabled = True
        self.fcm_enabled = False
        self.alert_relay_url = os.getenv("FALL_ALERT_RELAY_URL", "https://ntfy.sh/falldetector-alerts").rstrip("/")
        self.alert_relay_enabled = os.getenv("FALL_ALERT_RELAY", "1").lower() not in {"0", "false", "no", "off"}
        self.max_realtime_points = int(os.getenv("FALL_MAX_WS_POINTS", "256"))
        self.min_realtime_interval = float(os.getenv("FALL_WS_INTERVAL", "0.1"))
        self.last_realtime_sent = 0.0
        self.status_http_interval = float(os.getenv("FALL_STATUS_HTTP_INTERVAL", "1.0"))
        self.last_status_http_sent = 0.0
        self.status_http_inflight = False
        self.status_relay_interval = float(os.getenv("FALL_STATUS_RELAY_INTERVAL", "5.0"))
        self.last_status_relay_sent = 0.0

        self.ws = None
        self.ws_thread: Optional[threading.Thread] = None
        self.ws_connected = False
        self.command_callback = None
        self.reconnect_delay = 5

        self.alert_queue: "queue.Queue[dict]" = queue.Queue(maxsize=16)
        self.alert_worker = threading.Thread(target=self._alert_worker, daemon=True)
        self.alert_worker.start()

        self._init_firebase()

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["x-api-key"] = self.auth_token
        return headers

    def _init_firebase(self) -> None:
        if firebase_admin is None or credentials is None:
            return

        key_path = os.getenv("FIREBASE_CREDENTIALS")
        if not key_path:
            return

        if not os.path.exists(key_path):
            print(f"[web_client] Firebase credentials not found: {key_path}")
            return

        try:
            if not firebase_admin._apps:
                firebase_admin.initialize_app(credentials.Certificate(key_path))
            self.fcm_enabled = True
            print(f"[web_client] Firebase initialized from {key_path}")
        except Exception as exc:
            print(f"[web_client] Firebase initialization failed: {exc}")
            self.fcm_enabled = False

    def connect_websocket(self) -> None:
        if self.ws_thread and self.ws_thread.is_alive():
            return
        self.ws_thread = threading.Thread(target=self._ws_run, daemon=True)
        self.ws_thread.start()

    def _ws_run(self) -> None:
        ws_url = self.ws_url
        if self.auth_token:
            separator = "&" if "?" in ws_url else "?"
            ws_url = f"{ws_url}{separator}token={self.auth_token}"

        while self.enabled:
            try:
                self.ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=self._on_ws_open,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                )
                self.ws.run_forever()
            except Exception as exc:
                print(f"[web_client] WebSocket error: {exc}")
                traceback.print_exc()

            if self.enabled:
                time.sleep(self.reconnect_delay)

    def _on_ws_open(self, ws) -> None:
        self.ws_connected = True
        try:
            ws.send(json.dumps({"type": "PYTHON_INIT"}))
        except Exception as exc:
            print(f"[web_client] Failed to send init message: {exc}")

    def _on_ws_message(self, ws, message: str) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        if data.get("type") in {"CONTROL", "SETTINGS"} and self.command_callback:
            self.command_callback(data)

    def _on_ws_error(self, ws, error) -> None:
        print(f"[web_client] WebSocket error: {error}")

    def _on_ws_close(self, ws, close_status_code, close_msg) -> None:
        self.ws_connected = False
        print(f"[web_client] WebSocket closed: {close_status_code} - {close_msg}")

    def register_command_callback(self, callback) -> None:
        self.command_callback = callback

    def send_status(self, data: dict) -> None:
        if not self.enabled:
            return

        payload = dict(data)
        payload["type"] = "STATUS"
        payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.ws and self.ws_connected:
            try:
                self.ws.send(json.dumps(payload))
            except Exception as exc:
                print(f"[web_client] Failed to send status: {exc}")

        now = time.time()
        if (
            self.status_url
            and not self.status_http_inflight
            and now - self.last_status_http_sent >= self.status_http_interval
        ):
            self.last_status_http_sent = now
            self.status_http_inflight = True
            threading.Thread(target=self._post_status, args=(payload.copy(),), daemon=True).start()

        if (
            self.alert_relay_enabled
            and self.alert_relay_url
            and now - self.last_status_relay_sent >= self.status_relay_interval
        ):
            self.last_status_relay_sent = now
            threading.Thread(target=self._send_status_relay, args=(payload.copy(),), daemon=True).start()

    def _post_status(self, payload: dict) -> None:
        try:
            response = requests.post(
                self.status_url,
                json=payload,
                headers=self._headers(),
                timeout=1.0,
            )
            if response.status_code != 200:
                return

            try:
                command = response.json().get("command")
            except Exception:
                command = None
            if command and self.command_callback:
                self.command_callback({"type": "CONTROL", "action": str(command).upper()})
        except Exception as exc:
            print(f"[web_client] Failed to post status: {exc}")
        finally:
            self.status_http_inflight = False

    def send_alert(self, data: dict) -> None:
        if not self.enabled:
            return

        payload = dict(data)
        payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        payload["type"] = "ALERT"

        try:
            self.alert_queue.put_nowait(payload)
        except queue.Full:
            try:
                self.alert_queue.get_nowait()
            except queue.Empty:
                pass
            self.alert_queue.put_nowait(payload)

    def _alert_worker(self) -> None:
        while True:
            try:
                payload = self.alert_queue.get(timeout=0.5)
            except queue.Empty:
                if not self.enabled:
                    break
                continue
            self._post_data(payload)

    def _post_data(self, payload: dict) -> None:
        delivered = False

        if self.ws and self.ws_connected:
            try:
                self.ws.send(json.dumps(payload))
                delivered = True
            except Exception as exc:
                print(f"[web_client] Failed to send alert over WebSocket: {exc}")

        if not delivered:
            try:
                response = requests.post(
                    self.server_url,
                    json=payload,
                    headers=self._headers(),
                    timeout=2.0,
                )
                delivered = response.status_code == 200
            except Exception as exc:
                print(f"[web_client] Failed to send alert over HTTP: {exc}")

        if self.fcm_enabled:
            self._send_fcm(payload)

        self._send_alert_relay(payload)

    def _send_fcm(self, payload: dict) -> None:
        if not self.fcm_enabled or messaging is None:
            return
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title="낙상 감지 알림",
                    body=f"낙상 의심 이벤트가 감지되었습니다. 점수 {payload.get('score', 0):.2f}",
                ),
                data={key: str(value) for key, value in payload.items()},
                topic="all",
            )
            messaging.send(message)
        except Exception as exc:
            print(f"[web_client] Failed to send FCM notification: {exc}")

    def _send_alert_relay(self, payload: dict) -> None:
        if not self.alert_relay_enabled or not self.alert_relay_url:
            return
        try:
            requests.post(
                self.alert_relay_url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Title": "fall-alert",
                    "Priority": "urgent",
                },
                timeout=2.0,
            )
        except Exception as exc:
            print(f"[web_client] Failed to send alert relay: {exc}")

    def _send_status_relay(self, payload: dict) -> None:
        if not self.alert_relay_enabled or not self.alert_relay_url:
            return
        try:
            requests.post(
                self.alert_relay_url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Title": "fall-status",
                    "Priority": "low",
                },
                timeout=2.0,
            )
        except Exception as exc:
            print(f"[web_client] Failed to send status relay: {exc}")

    def disconnect(self) -> None:
        self.enabled = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

    def send_realtime_data(self, points, tracks, skeletons=None, vitals=None, meta=None) -> None:
        if not self.enabled or not self.ws or not self.ws_connected:
            return

        now = time.time()
        if now - self.last_realtime_sent < self.min_realtime_interval:
            return
        self.last_realtime_sent = now

        points_list = self._sample_points(points)
        skeletons_list = skeletons.tolist() if hasattr(skeletons, "tolist") else (skeletons or [])
        vitals_list = list(vitals or [])
        tracks_list = list(tracks or [])
        meta_payload = dict(meta or {})

        payload = {
            "type": "realtime_data",
            "points": points_list,
            "tracks": tracks_list,
            "skeletons": skeletons_list,
            "vitals": vitals_list,
            "personCount": int(meta_payload.get("personCount", len(tracks_list))),
            "pointCount": int(meta_payload.get("pointCount", len(points_list))),
            "vitalCount": int(meta_payload.get("visibleVitalRecords", len(vitals_list))),
            "meta": meta_payload,
            "timestamp": now,
        }

        try:
            self.ws.send(json.dumps(payload))
        except Exception as exc:
            print(f"[web_client] Failed to send realtime payload: {exc}")

    def _sample_points(self, points):
        if points is None:
            return []

        if hasattr(points, "tolist"):
            import numpy as np

            array = np.asarray(points)
            if array.ndim != 2 or array.shape[0] == 0:
                return []
            if array.shape[0] > self.max_realtime_points:
                indices = np.linspace(0, array.shape[0] - 1, num=self.max_realtime_points, dtype=int)
                array = array[indices]
            return array.tolist()

        if len(points) > self.max_realtime_points:
            step = max(1, len(points) // self.max_realtime_points)
            return list(points)[::step][: self.max_realtime_points]
        return list(points)
