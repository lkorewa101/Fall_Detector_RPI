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

if [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=:0
fi
if [ -z "${XDG_RUNTIME_DIR:-}" ] && [ -d "/run/user/$(id -u)" ]; then
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi
if [ -z "${XAUTHORITY:-}" ] && [ -f "$HOME/.Xauthority" ]; then
    export XAUTHORITY="$HOME/.Xauthority"
fi

fix_local_permissions() {
    chmod u+rwx "$ROOT" 2>/dev/null || true
    find "$ROOT" -type d -exec chmod u+rwx {} + 2>/dev/null || true
    find "$ROOT" -type f -exec chmod u+rw {} + 2>/dev/null || true
    chmod +x "$ROOT"/RUN_*.sh 2>/dev/null || true
}
fix_local_permissions

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_USER=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PATH="$HOME/.local/bin:$PATH"

# Raspberry Pi mode serves the already-built web UI through the backend on port 8000.
export FALL_STATIC_WEB=1

python3 -B launcher.py
