"""Lifecycle helpers for inference services owned by the desktop app."""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

log = logging.getLogger("rula.local_services")
_lock = threading.Lock()
_tabby_process: subprocess.Popen | None = None


def _is_reachable(url: str, api_key: str) -> bool:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(f"{url}/models", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=1.0):
            return True
    except urllib.error.HTTPError as exc:
        return exc.code in (401, 403)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _tabby_home() -> Path:
    configured = os.environ.get("TABBYAPI_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "Programs" / "TabbyAPI"
    raise RuntimeError(
        "TabbyAPIの場所が分かりません。TABBYAPI_HOME を設定してください。"
    )


def _log_tail(path: Path) -> str:
    try:
        return "\n".join(path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-30:])
    except OSError:
        return ""


def ensure_tabbyapi_started(base_url: str, api_key: str = "") -> None:
    """Start the installed local TabbyAPI if it is not already responding."""
    global _tabby_process
    if os.name != "nt":
        return
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        return
    service_url = base_url.rstrip("/")
    if _is_reachable(service_url, api_key):
        return

    with _lock:
        if _is_reachable(service_url, api_key):
            return
        if _tabby_process is not None and _tabby_process.poll() is None:
            process = _tabby_process
        else:
            home = _tabby_home()
            python = home / "venv" / "Scripts" / "python.exe"
            start_script = home / "start.py"
            if not python.is_file() or not start_script.is_file():
                raise RuntimeError(
                    f"TabbyAPIを自動起動できません。{home} にTabbyAPIと"
                    "そのPython環境があることを確認してください。"
                )
            app_data = Path(os.environ.get("APPDATA") or Path.home())
            log_path = app_data / "Rula" / "logs" / "tabbyapi.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            child_env = os.environ.copy()
            child_env["PYTHONUTF8"] = "1"
            child_env["PYTHONIOENCODING"] = "utf-8"
            with log_path.open("ab") as output:
                _tabby_process = subprocess.Popen(
                    [str(python), str(start_script)],
                    cwd=home,
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    creationflags=(
                        getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    ),
                )
            process = _tabby_process
            log.info("Started local TabbyAPI (pid=%s)", process.pid)

    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if _is_reachable(service_url, api_key):
            return
        if process.poll() is not None:
            break
        time.sleep(0.5)

    log_path = (
        Path(os.environ.get("APPDATA") or Path.home())
        / "Rula" / "logs" / "tabbyapi.log"
    )
    details = _log_tail(log_path)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    _tabby_process = None
    raise RuntimeError(
        "TabbyAPIの自動起動に失敗しました。"
        + (f"\n{details}" if details else "")
    )


def stop_owned_tabbyapi() -> None:
    """Stop only a TabbyAPI process started by this Rula desktop session."""
    global _tabby_process
    with _lock:
        process = _tabby_process
        _tabby_process = None
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    log.info("Stopped local TabbyAPI (pid=%s)", process.pid)
