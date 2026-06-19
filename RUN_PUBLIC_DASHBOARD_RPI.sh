#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$ROOT"

if [ "$(id -u)" = "0" ] && [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER:-}" != "root" ]; then
    USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    USER_ID=$(id -u "$SUDO_USER")
    echo "[rpi] Do not run the whole app as root. Re-running as $SUDO_USER..."
    exec sudo -u "$SUDO_USER" env \
        HOME="$USER_HOME" \
        DISPLAY="${DISPLAY:-:0}" \
        XAUTHORITY="${XAUTHORITY:-$USER_HOME/.Xauthority}" \
        XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$USER_ID}" \
        PATH="$USER_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
        "$0" "$@"
fi

STATE_DIR="$ROOT/.launcher_state"
mkdir -p "$STATE_DIR"

fix_local_permissions() {
    chmod u+rwx "$ROOT" 2>/dev/null || true
    find "$ROOT" -type d -exec chmod u+rwx {} + 2>/dev/null || true
    find "$ROOT" -type f -exec chmod u+rw {} + 2>/dev/null || true
    chmod +x "$ROOT"/RUN_*.sh "$STATE_DIR/cloudflared" 2>/dev/null || true
}
fix_local_permissions

if [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=:0
fi
if [ -z "${XDG_RUNTIME_DIR:-}" ] && [ -d "/run/user/$(id -u)" ]; then
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi
if [ -z "${XAUTHORITY:-}" ] && [ -f "$HOME/.Xauthority" ]; then
    export XAUTHORITY="$HOME/.Xauthority"
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_USER=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PATH="$HOME/.local/bin:$PATH"
export FALL_STATIC_WEB=1

find_cloudflared() {
    if command -v cloudflared >/dev/null 2>&1; then
        command -v cloudflared
        return 0
    fi

    local_tool="$STATE_DIR/cloudflared"
    if [ -x "$local_tool" ]; then
        printf '%s\n' "$local_tool"
        return 0
    fi

    mkdir -p "$STATE_DIR"
    arch=$(uname -m)
    case "$arch" in
        aarch64|arm64)
            url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
            ;;
        armv7l|armv6l|armhf|arm)
            url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm"
            ;;
        *)
            echo "Unsupported Raspberry Pi architecture for auto cloudflared install: $arch" >&2
            echo "Install cloudflared manually, then run this script again." >&2
            exit 1
            ;;
    esac

    echo "[rpi] cloudflared not found. Downloading $url" >&2
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail -o "$local_tool" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$local_tool" "$url"
    else
        echo "curl or wget is required to download cloudflared." >&2
        exit 1
    fi
    chmod +x "$local_tool"
    printf '%s\n' "$local_tool"
}

wait_for_backend() {
    attempts=45
    while [ "$attempts" -gt 0 ]; do
        if python3 - <<'PY' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2).read()
PY
        then
            return 0
        fi
        attempts=$((attempts - 1))
        sleep 2
    done
    return 1
}

cleanup() {
    if [ -n "${LAUNCHER_PID:-}" ]; then
        kill "$LAUNCHER_PID" >/dev/null 2>&1 || true
        wait "$LAUNCHER_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup INT TERM EXIT

echo "[1/3] Starting desktop app and backend web server..."
python3 -B launcher.py >"$STATE_DIR/rpi-launcher.out.log" 2>"$STATE_DIR/rpi-launcher.err.log" &
LAUNCHER_PID=$!

echo "[2/3] Waiting for http://127.0.0.1:8000 ..."
if wait_for_backend; then
    echo "[rpi] Backend is ready."
else
    echo "[rpi] Backend did not answer yet." >&2
    echo "[rpi] --- launcher stdout ---" >&2
    tail -n 120 "$STATE_DIR/rpi-launcher.out.log" >&2 || true
    echo "[rpi] --- launcher stderr ---" >&2
    tail -n 120 "$STATE_DIR/rpi-launcher.err.log" >&2 || true
    exit 1
fi

echo "[3/3] Starting Cloudflare Tunnel..."
echo "Copy the printed https://*.trycloudflare.com URL into the Android app or another browser."
CLOUDFLARED=$(find_cloudflared)
"$CLOUDFLARED" tunnel --url http://127.0.0.1:8000 --no-autoupdate
