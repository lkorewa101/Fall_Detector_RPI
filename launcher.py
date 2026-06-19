from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
ACCESS_INFO_FILE = ROOT_DIR / "외부접속주소.txt"


def find_child_dir(description: str, predicate) -> Path:
    for child in ROOT_DIR.iterdir():
        try:
            if child.is_dir() and predicate(child):
                return child
        except OSError as exc:
            print(
                f"[launcher] Skipping unreadable folder while finding {description}: {child} ({exc})",
                flush=True,
            )
            continue
    raise FileNotFoundError(f"Missing {description} folder under {ROOT_DIR}")


def find_named_or_child_dir(description: str, names: tuple[str, ...], predicate) -> Path:
    for name in names:
        child = ROOT_DIR / name
        try:
            if child.is_dir() and predicate(child):
                return child
        except OSError as exc:
            print(
                f"[launcher] Skipping unreadable folder while finding {description}: {child} ({exc})",
                flush=True,
            )
    return find_child_dir(description, predicate)


DESKTOP_DIR = find_named_or_child_dir(
    "desktop app",
    ("desktop_app", "낙상감지기_데스크탑앱"),
    lambda path: (path / "main.py").is_file() and (path / "requirements.txt").is_file(),
)
WEB_DIR = find_named_or_child_dir(
    "web dashboard",
    ("web_dashboard", "웹_대시보드"),
    lambda path: (path / "backend" / "server.js").is_file()
    and (path / "frontend" / "package.json").is_file(),
)
WEB_BACKEND_DIR = WEB_DIR / "backend"
WEB_FRONTEND_DIR = WEB_DIR / "frontend"
NO_OUTPUT_TIMEOUT_SECONDS = 100
STATE_DIR = ROOT_DIR / ".launcher_state"
RPI_GUI_DEPENDENCIES = {
    "pyqt5",
    "pyqt5-qt5",
    "pyqt5-sip",
    "pyqt6",
    "pyside2",
    "pyside6",
    "pyqtgraph",
    "pyopengl",
    "pyopengl-accelerate",
    "pyopengl_accelerate",
}
RPI_APT_BASE_PACKAGES = ["python3-pip"]
RPI_APT_GUI_PACKAGES = [
    "python3-pyqt5",
    "python3-pyqt5.qtopengl",
    "python3-pyqtgraph",
    "python3-opengl",
]
RPI_APT_WEB_PACKAGES = ["nodejs", "npm"]
RPI_GUI_IMPORTS = ["PyQt5.QtWidgets", "pyqtgraph.opengl", "OpenGL"]
WINDOWS_OPTIONAL_PACKAGES = {
    "adafruit-circuitpython-dht",
}
FIREBASE_CREDENTIAL_NAMES = {
    "firebase-service-account.json",
    "service-account.json",
    "serviceAccountKey.json",
    "firebase-adminsdk.json",
}


@dataclass(frozen=True)
class LaunchTask:
    name: str
    command: list[str]
    cwd: Path
    env: dict[str, str] | None = None


@dataclass
class RunningTask:
    task: LaunchTask
    process: subprocess.Popen[str]


def project_venv_dir(project_dir: Path) -> Path:
    if os.name == "nt":
        path_hash = hashlib.sha1(str(project_dir).encode("utf-8")).hexdigest()[:12]
        return ROOT_DIR.parent / ".fall_launcher_venvs" / f"py{sys.version_info.major}{sys.version_info.minor}_{path_hash}"
    return project_dir / ".venv"


def venv_python(project_dir: Path) -> Path:
    return Path(sys.executable)


def is_raspberry_pi() -> bool:
    if platform.system().lower() != "linux":
        return False

    model_path = Path("/proc/device-tree/model")
    try:
        model = model_path.read_text(encoding="utf-8", errors="ignore").lower()
        if "raspberry pi" in model:
            return True
    except OSError:
        pass

    machine = platform.machine().lower()
    return machine.startswith("arm") or machine in {"aarch64", "arm64"}


def static_web_mode() -> bool:
    value = os.environ.get("FALL_STATIC_WEB", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def child_process_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def find_firebase_credentials() -> Path | None:
    candidates: list[Path] = []
    for directory in (ROOT_DIR, DESKTOP_DIR, ROOT_DIR / "firebase", ROOT_DIR / "credentials"):
        if not directory.exists():
            continue
        for name in FIREBASE_CREDENTIAL_NAMES:
            candidates.append(directory / name)
        candidates.extend(directory.glob("*firebase*.json"))
        candidates.extend(directory.glob("*service*.json"))
        candidates.extend(directory.glob("*adminsdk*.json"))

    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("type") == "service_account" and data.get("client_email") and data.get("private_key"):
            return path
    return None


def ensure_venv(project_name: str, project_dir: Path) -> Path:
    return Path(sys.executable)


def backup_incompatible_venv(project_name: str, venv_dir: Path) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = venv_dir.with_name(f".venv_incompatible_{stamp}")
    suffix = 1
    while backup_dir.exists():
        backup_dir = venv_dir.with_name(f".venv_incompatible_{stamp}_{suffix}")
        suffix += 1

    print(
        f"[launcher] Backing up incompatible {project_name} venv: {venv_dir} -> {backup_dir}",
        flush=True,
    )
    shutil.move(str(venv_dir), str(backup_dir))


def enable_system_site_packages(project_dir: Path) -> None:
    cfg_path = project_venv_dir(project_dir) / "pyvenv.cfg"
    if not cfg_path.exists():
        return

    text = cfg_path.read_text(encoding="utf-8", errors="replace")
    target_line = "include-system-site-packages = true"
    if target_line in text:
        return

    backup_path = cfg_path.with_name(f"pyvenv.cfg.bak_{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(cfg_path, backup_path)

    lines = text.splitlines()
    changed = False
    for index, line in enumerate(lines):
        if line.lower().startswith("include-system-site-packages"):
            lines[index] = target_line
            changed = True
            break
    if not changed:
        lines.append(target_line)

    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def npm_command() -> str:
    command = shutil.which("npm.cmd") or shutil.which("npm")
    if not command:
        raise FileNotFoundError("Missing npm command. Install Node.js or add npm to PATH.")
    return command


def local_ip_addresses() -> list[str]:
    addresses: set[str] = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = item[4][0]
            if address and not address.startswith("127."):
                addresses.add(address)
    except OSError:
        pass

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            address = probe.getsockname()[0]
            if address and not address.startswith("127."):
                addresses.add(address)
        finally:
            probe.close()
    except OSError:
        pass

    return sorted(addresses)


def write_access_info() -> str:
    addresses = local_ip_addresses()
    return addresses[0] if addresses else "<device-ip>"


def print_access_urls() -> None:
    address = write_access_info()
    for local_address in local_ip_addresses():
        print(f"[launcher] Web dashboard: http://{local_address}:8000", flush=True)
    if address == "<device-ip>":
        print("[launcher] Web dashboard: http://<device-ip>:8000", flush=True)
    print("[launcher] App alert subscription: https://ntfy.sh/falldetector-alerts", flush=True)


def read_state(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(path: Path, data: dict[str, object]) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def emit_prefixed_output(name: str, text: str) -> None:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for line in normalized.splitlines():
        if line.strip():
            print(f"[{name}] {line}", flush=True)


def sudo_prefix() -> list[str]:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []
    sudo = shutil.which("sudo")
    if not sudo:
        raise FileNotFoundError("Missing sudo. Install apt packages manually or run as root.")
    return [sudo]


def apt_package_installed(package_name: str) -> bool:
    result = subprocess.run(
        ["dpkg-query", "-W", "-f=${Status}", package_name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=NO_OUTPUT_TIMEOUT_SECONDS,
        check=False,
    )
    return result.returncode == 0 and "install ok installed" in result.stdout


def current_hostname() -> str:
    try:
        result = subprocess.run(
            ["hostname"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=NO_OUTPUT_TIMEOUT_SECONDS,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def ensure_rpi_fixed_hostname() -> None:
    return


def ensure_rpi_apt_dependencies(include_desktop: bool, include_web: bool) -> None:
    if not is_raspberry_pi():
        return

    if not shutil.which("apt-get") or not shutil.which("dpkg-query"):
        raise FileNotFoundError("Missing apt-get or dpkg-query on Raspberry Pi.")

    package_names = list(RPI_APT_BASE_PACKAGES)
    if include_desktop:
        package_names.extend(RPI_APT_GUI_PACKAGES)
    if include_web:
        package_names.extend(RPI_APT_WEB_PACKAGES)

    missing = [
        package_name
        for package_name in package_names
        if not apt_package_installed(package_name)
    ]
    if not missing:
        print("[launcher] Raspberry Pi apt packages already installed.", flush=True)
        ensure_rpi_fixed_hostname()
        return

    print(
        f"[launcher] Missing Raspberry Pi apt packages: {', '.join(missing)}",
        flush=True,
    )
    prefix = sudo_prefix()
    run_setup_command("rpi-apt-update", [*prefix, "apt-get", "update"], ROOT_DIR)
    run_setup_command(
        "rpi-apt-deps",
        [*prefix, "apt-get", "install", "-y", *missing],
        ROOT_DIR,
    )
    ensure_rpi_fixed_hostname()


def run_setup_command(name: str, command: list[str], cwd: Path) -> None:
    print(f"[launcher] Installing {name}: {subprocess.list2cmdline(command)}", flush=True)

    popen_kwargs: dict[str, object] = {}
    process_env = child_process_env()
    process_env.pop("PIP_NO_INDEX", None)
    process_env.pop("PIP_FIND_LINKS", None)
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=process_env,
        **popen_kwargs,
    )

    output_queue: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        if process.stdout is None:
            output_queue.put(None)
            return
        for line in process.stdout:
            output_queue.put(line)
        output_queue.put(None)

    threading.Thread(target=reader, daemon=True).start()

    last_output_time = time.monotonic()
    reader_done = False
    while True:
        try:
            line = output_queue.get(timeout=0.2)
            if line is None:
                reader_done = True
            else:
                last_output_time = time.monotonic()
                emit_prefixed_output(name, line)
        except queue.Empty:
            pass

        code = process.poll()
        if code is not None and reader_done:
            if code != 0:
                raise RuntimeError(f"{name} install failed with code {code}")
            return

        if time.monotonic() - last_output_time > NO_OUTPUT_TIMEOUT_SECONDS:
            stop_process(process)
            raise TimeoutError(
                f"{name} install had no output for {NO_OUTPUT_TIMEOUT_SECONDS} seconds"
            )


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except Exception:
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()


def normalize_package_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def requirement_name(line: str) -> str | None:
    clean = line.split("#", 1)[0].strip()
    if not clean or clean.startswith("-") or "://" in clean:
        return None

    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", ";", "[", " "):
        if separator in clean:
            clean = clean.split(separator, 1)[0].strip()
    return clean or None


def prepare_requirements(
    project_name: str,
    requirements: Path,
    include_gui_deps: bool,
) -> tuple[Path, str, list[str], list[str]]:
    original_lines = requirements.read_text(encoding="utf-8").splitlines()
    filtered_lines: list[str] = []
    skipped: list[str] = []
    package_names: list[str] = []

    for line in original_lines:
        name = requirement_name(line)
        normalized_name = normalize_package_name(name) if name else None

        if (
            normalized_name
            and is_raspberry_pi()
            and not include_gui_deps
            and normalized_name in RPI_GUI_DEPENDENCIES
        ):
            skipped.append(line.strip())
            continue
        if normalized_name and os.name == "nt" and normalized_name in WINDOWS_OPTIONAL_PACKAGES:
            skipped.append(line.strip())
            continue

        filtered_lines.append(line)
        if name:
            package_names.append(name)

    filtered_text = "\n".join(filtered_lines).strip() + "\n"
    filtered_path = STATE_DIR / f"{project_name}_requirements.txt"
    STATE_DIR.mkdir(exist_ok=True)
    if not filtered_path.exists() or filtered_path.read_text(encoding="utf-8") != filtered_text:
        filtered_path.write_text(filtered_text, encoding="utf-8")

    return filtered_path, filtered_text, skipped, package_names


def missing_python_packages(python_path: Path, package_names: list[str]) -> list[str]:
    if not package_names:
        return []

    script = (
        "import importlib.metadata as metadata, sys\n"
        "missing = []\n"
        "for name in sys.argv[1:]:\n"
        "    try:\n"
        "        metadata.distribution(name)\n"
        "    except metadata.PackageNotFoundError:\n"
        "        missing.append(name)\n"
        "print('\\n'.join(missing))\n"
        "raise SystemExit(1 if missing else 0)\n"
    )
    result = subprocess.run(
        [str(python_path), "-c", script, *package_names],
        cwd=python_path.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=NO_OUTPUT_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode == 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def verify_python_imports(project_name: str, python_path: Path, import_names: list[str]) -> None:
    script = (
        "import importlib, sys\n"
        "missing = []\n"
        "for name in sys.argv[1:]:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        missing.append(f'{name}: {exc}')\n"
        "print('\\n'.join(missing))\n"
        "raise SystemExit(1 if missing else 0)\n"
    )
    result = subprocess.run(
        [str(python_path), "-c", script, *import_names],
        cwd=python_path.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=NO_OUTPUT_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        details = result.stdout.strip() or result.stderr.strip()
        raise RuntimeError(f"{project_name} imports failed: {details}")


def ensure_python_requirements(
    project_name: str,
    project_dir: Path,
    include_gui_deps: bool,
) -> None:
    requirements = project_dir / "requirements.txt"
    if not requirements.exists():
        print(f"[launcher] No Python requirements for {project_name}.", flush=True)
        return
    print(f"[launcher] Using system Python for {project_name}. No .venv will be created.", flush=True)

    python_path = Path(sys.executable)
    filtered_path, filtered_text, skipped, package_names = prepare_requirements(
        project_name,
        requirements,
        include_gui_deps,
    )
    requirements_hash = sha256_text(filtered_text)
    state_path = STATE_DIR / f"{project_name}_python_deps.json"
    state = read_state(state_path)
    missing = missing_python_packages(python_path, package_names)

    if skipped:
        print(
            f"[launcher] Using Raspberry Pi apt packages instead of pip for {project_name}: "
            f"{', '.join(skipped)}",
            flush=True,
        )

    if not missing:
        print(f"[launcher] Python dependencies already installed for {project_name}.", flush=True)
        if state.get("requirements_hash") != requirements_hash:
            write_state(
                state_path,
                {
                    "requirements_hash": requirements_hash,
                    "python": str(python_path),
                    "skipped": skipped,
                },
            )
        return

    print(
        f"[launcher] Missing Python dependencies for {project_name}: {', '.join(missing)}",
        flush=True,
    )

    run_setup_command(
        f"{project_name}-python-deps",
        [str(python_path), "-m", "pip", "install", "-r", str(filtered_path)],
        project_dir,
    )
    write_state(
        state_path,
        {
            "requirements_hash": requirements_hash,
            "python": str(python_path),
            "skipped": skipped,
        },
    )


def read_node_dependencies(project_dir: Path) -> list[str]:
    package_json = project_dir / "package.json"
    if not package_json.exists():
        return []

    data = json.loads(package_json.read_text(encoding="utf-8"))
    dependencies: dict[str, str] = {}
    dependencies.update(data.get("dependencies", {}))
    dependencies.update(data.get("devDependencies", {}))
    return sorted(dependencies)


def node_module_path(project_dir: Path, package_name: str) -> Path:
    parts = package_name.split("/")
    return project_dir / "node_modules" / Path(*parts)


def ensure_node_dependencies(project_name: str, project_dir: Path) -> None:
    package_json = project_dir / "package.json"
    if not package_json.exists():
        print(f"[launcher] No Node package.json for {project_name}.", flush=True)
        return

    dependencies = read_node_dependencies(project_dir)
    missing = [
        package_name
        for package_name in dependencies
        if not node_module_path(project_dir, package_name).exists()
    ]
    package_json_text = package_json.read_text(encoding="utf-8")
    lock_path = project_dir / "package-lock.json"
    lock_text = lock_path.read_text(encoding="utf-8") if lock_path.exists() else ""
    package_hash = sha256_text(package_json_text + "\n" + lock_text)
    state_path = STATE_DIR / f"{project_name}_node_deps.json"
    state = read_state(state_path)

    if not missing:
        if state.get("package_hash") and state.get("package_hash") != package_hash:
            print(f"[launcher] Node package files changed for {project_name}.", flush=True)
        else:
            print(f"[launcher] Node dependencies already installed for {project_name}.", flush=True)
            if state.get("package_hash") != package_hash:
                write_state(
                    state_path,
                    {
                        "package_hash": package_hash,
                        "npm": npm_command(),
                    },
                )
            return
    else:
        print(
            f"[launcher] Missing Node dependencies for {project_name}: {', '.join(missing)}",
            flush=True,
        )

    npm = npm_command()
    if is_raspberry_pi():
        install_command = [npm, "install", "--no-audit", "--no-fund"]
    else:
        install_command = [npm, "ci", "--no-audit", "--no-fund"] if lock_path.exists() else [npm, "install", "--no-audit", "--no-fund"]
    try:
        run_setup_command(f"{project_name}-node-deps", install_command, project_dir)
    except RuntimeError:
        if install_command[1] != "ci":
            raise
        print(f"[launcher] npm ci failed for {project_name}; retrying npm install.", flush=True)
        run_setup_command(f"{project_name}-node-deps", [npm, "install", "--no-audit", "--no-fund"], project_dir)
    write_state(
        state_path,
        {
            "package_hash": package_hash,
            "npm": npm,
        },
    )


def ensure_dependencies(
    include_desktop: bool,
    include_web: bool,
    include_gui_deps: bool,
    install_rpi_apt: bool,
) -> None:
    if is_raspberry_pi() and install_rpi_apt:
        ensure_rpi_apt_dependencies(include_desktop, include_web)

    if include_desktop:
        ensure_python_requirements("desktop", DESKTOP_DIR, include_gui_deps)
        if is_raspberry_pi():
            verify_python_imports("desktop-rpi-gui", venv_python(DESKTOP_DIR), RPI_GUI_IMPORTS)

    if include_web:
        ensure_node_dependencies("web-backend", WEB_BACKEND_DIR)
        if static_web_mode():
            dist_index = WEB_FRONTEND_DIR / "dist" / "index.html"
            if dist_index.exists():
                print("[launcher] Static web mode enabled. Frontend dev server will be skipped.", flush=True)
            else:
                print(
                    "[launcher] Static web mode enabled, but frontend/dist/index.html is missing.",
                    flush=True,
                )
        else:
            ensure_node_dependencies("web-frontend", WEB_FRONTEND_DIR)


def backend_health_ok() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=0.5) as response:
            return 200 <= int(response.status) < 500
    except Exception:
        return False


def build_tasks(include_desktop: bool, include_web: bool) -> list[LaunchTask]:
    tasks: list[LaunchTask] = []

    if include_web:
        npm = npm_command()
        web_env = child_process_env()
        web_env.setdefault("FALL_BLACKBOX_DIR", str(DESKTOP_DIR / "records"))

        if backend_health_ok():
            print("[launcher] Backend is already running on port 8000. Reusing it.", flush=True)
        else:
            tasks.append(
                LaunchTask(
                    name="web-backend",
                    command=[npm, "start"],
                    cwd=WEB_BACKEND_DIR,
                    env=web_env,
                )
            )
        if static_web_mode():
            print("[launcher] Static web mode: using backend on port 8000 for the web UI.", flush=True)
        else:
            tasks.append(
                LaunchTask(
                    name="web-frontend",
                    command=[npm, "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"],
                    cwd=WEB_FRONTEND_DIR,
                    env=web_env,
                )
            )

    if include_desktop:
        desktop_env = child_process_env()
        desktop_env.pop("FIREBASE_CREDENTIALS", None)
        if os.name != "nt":
            desktop_env.setdefault("DISPLAY", ":0")
            runtime_dir = Path("/run/user") / str(os.getuid())
            if runtime_dir.exists():
                desktop_env.setdefault("XDG_RUNTIME_DIR", str(runtime_dir))
            xauthority = Path.home() / ".Xauthority"
            if xauthority.exists():
                desktop_env.setdefault("XAUTHORITY", str(xauthority))
            if desktop_env.get("QT_QPA_PLATFORM", "").lower() in {"offscreen", "minimal"}:
                desktop_env.pop("QT_QPA_PLATFORM", None)
        desktop_env.setdefault("FALL_ALERT_RELAY", "1")
        desktop_env.setdefault("FALL_ALERT_RELAY_URL", "https://ntfy.sh/falldetector-alerts")
        desktop_env.setdefault("FALL_STATUS_RELAY_INTERVAL", "5.0")
        if include_web:
            desktop_env.setdefault("FALL_BACKEND_ALERT_URL", "http://127.0.0.1:8000/api/alert")
            desktop_env.setdefault("FALL_BACKEND_STATUS_URL", "http://127.0.0.1:8000/api/status")
            desktop_env.setdefault("FALL_BACKEND_WS_URL", "ws://127.0.0.1:8000/ws")
            desktop_env.setdefault("FALL_MAX_WS_POINTS", "384")
            desktop_env.setdefault("FALL_WS_INTERVAL", "0.15")
            desktop_env.setdefault("FALL_STATUS_HTTP_INTERVAL", "1.0")
        tasks.append(
            LaunchTask(
                name="desktop",
                command=[str(venv_python(DESKTOP_DIR)), "main.py"],
                cwd=DESKTOP_DIR,
                env=desktop_env,
            )
        )

    return tasks


def validate_task(task: LaunchTask) -> None:
    if not task.cwd.exists():
        raise FileNotFoundError(f"Missing working directory for {task.name}: {task.cwd}")


def stream_output(running: RunningTask) -> None:
    process = running.process

    if process.stdout is None:
        return

    for line in process.stdout:
        emit_prefixed_output(running.task.name, line)


def start_task(task: LaunchTask) -> RunningTask:
    validate_task(task)
    display_command = subprocess.list2cmdline(task.command)
    print(f"[launcher] Starting {task.name}: {display_command}", flush=True)

    popen_kwargs: dict[str, object] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(
        task.command,
        cwd=task.cwd,
        env=task.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **popen_kwargs,
    )

    running = RunningTask(task=task, process=process)
    thread = threading.Thread(target=stream_output, args=(running,), daemon=True)
    thread.start()
    return running


def stop_task(running: RunningTask) -> None:
    process = running.process
    if process.poll() is not None:
        return

    print(f"[launcher] Stopping {running.task.name}...", flush=True)
    stop_process(process)


def monitor(running_tasks: list[RunningTask]) -> int:
    exit_code = 0
    alive = set(range(len(running_tasks)))

    try:
        while alive:
            finished: list[int] = []
            for index in alive:
                running = running_tasks[index]
                code = running.process.poll()
                if code is not None:
                    finished.append(index)
                    print(
                        f"[launcher] {running.task.name} exited with code {code}",
                        flush=True,
                    )
                    if code != 0 and exit_code == 0:
                        exit_code = code

            for index in finished:
                alive.remove(index)

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[launcher] Ctrl+C received. Shutting down all processes.", flush=True)
        exit_code = 130
    finally:
        for running in running_tasks:
            stop_task(running)

    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the desktop app and web dashboard together."
    )
    parser.add_argument(
        "--desktop-only",
        action="store_true",
        help="Launch only the desktop app.",
    )
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Launch only the web dashboard backend and frontend.",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip automatic local dependency installation.",
    )
    parser.add_argument(
        "--deps-only",
        action="store_true",
        help="Install missing local dependencies, then exit.",
    )
    parser.add_argument(
        "--skip-apt",
        action="store_true",
        help="On Raspberry Pi, skip automatic apt package installation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    include_desktop = not args.web_only
    include_web = not args.desktop_only
    rpi_mode = is_raspberry_pi()

    if args.desktop_only and args.web_only:
        print("[launcher] Choose either --desktop-only or --web-only, not both.", file=sys.stderr)
        return 2

    try:
        if not args.skip_deps:
            ensure_dependencies(
                include_desktop=include_desktop,
                include_web=include_web,
                include_gui_deps=not rpi_mode,
                install_rpi_apt=not args.skip_apt,
            )
        if args.deps_only:
            print("[launcher] Dependency setup finished.", flush=True)
            return 0

        tasks = build_tasks(include_desktop=include_desktop, include_web=include_web)
        running_tasks = [start_task(task) for task in tasks]
    except Exception as exc:
        print(f"[launcher] Failed to start: {exc}", file=sys.stderr)
        return 1

    print("[launcher] All requested processes started. Press Ctrl+C to stop.", flush=True)
    if include_web:
        print_access_urls()
    return monitor(running_tasks)


if __name__ == "__main__":
    raise SystemExit(main())
