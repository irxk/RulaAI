"""RulaCoder adapter for its isolated ExLlamaV2 server."""
import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from .strata_backend import StrataEngine, _normalize_base_url

log = logging.getLogger("rula.coder_v2")


class ExLlamaV2Engine(StrataEngine):
    """OpenAI-compatible client that starts and controls the local V2 server."""

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get(
            "RULA_CODER_V2_MODEL",
            "Qwen2.5-Coder-7B-Instruct-4.0bpw-exl2",
        ).strip() or "Qwen2.5-Coder-7B-Instruct-4.0bpw-exl2"
        self.base_url = _normalize_base_url(os.environ.get(
            "RULA_CODER_V2_BASE_URL", "http://127.0.0.1:5101/v1"
        ))
        self._server_process: subprocess.Popen | None = None
        self._ensure_server()
        super().__init__(
            base_url=self.base_url,
            api_key=os.environ.get("RULA_CODER_V2_API_KEY", ""),
            model=self.model,
            backend="exllamav2",
            display_name=os.environ.get(
                "RULA_CODER_V2_DISPLAY_NAME",
                "Qwen2.5-Coder-7B-Instruct (EXL2)",
            ).strip(),
            model_env="RULA_CODER_V2_MODEL",
            service_name="RulaCoder ExLlamaV2",
        )

    @property
    def _service_url(self) -> str:
        return self.base_url[:-3]

    def _health(self) -> bool:
        try:
            with urllib.request.urlopen(
                    f"{self._service_url}/health", timeout=1.0) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                "RULA_CODER_V2_BASE_URL のポートに別のサービスが応答しています。"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError):
            return False
        if body.get("service") != "rula-coder-exllamav2":
            raise RuntimeError(
                "RULA_CODER_V2_BASE_URL のポートに別のサービスが応答しています。"
            )
        return True

    @staticmethod
    def _default_python() -> str:
        tabby_home = os.environ.get("TABBYAPI_HOME", "").strip()
        if not tabby_home:
            local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
            if not local_app_data:
                raise RuntimeError(
                    "TabbyAPIのPython環境が見つかりません。"
                    "RULA_CODER_V2_PYTHON を設定してください。"
                )
            tabby_home = os.path.join(local_app_data, "Programs", "TabbyAPI")
        return os.path.join(tabby_home, "venv", "Scripts", "python.exe")

    @staticmethod
    def _log_tail(path: Path) -> str:
        try:
            lines = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-30:])

    def _ensure_server(self) -> None:
        parsed = urlsplit(self.base_url)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
                or parsed.path != "/v1" or parsed.query or parsed.fragment):
            raise RuntimeError(
                "ExLlamaV2 CoderサーバーはこのPC内のURLだけに接続できます。"
            )
        try:
            port = parsed.port or 5101
        except ValueError as exc:
            raise RuntimeError("RULA_CODER_V2_BASE_URL のポートが不正です。") from exc
        if self._health():
            return

        python = os.environ.get("RULA_CODER_V2_PYTHON", "").strip()
        if not python:
            python = self._default_python()
        if not os.path.isfile(python):
            raise RuntimeError(
                f"ExLlamaV2用Pythonが見つかりません: {python}。"
                "install-coder-v2.ps1 を実行するか、"
                "RULA_CODER_V2_PYTHON を設定してください。"
            )

        source_root = Path(__file__).resolve().parents[1]
        child_env = os.environ.copy()
        existing_python_path = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = os.pathsep.join(
            path for path in (str(source_root), existing_python_path) if path
        )
        child_env["RULA_CODER_V2_SERVER_PORT"] = str(port)
        log_dir = Path(
            os.environ.get("APPDATA") or Path.home()
        ) / "RulaCoder" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "coder-v2-server.log"
        with log_path.open("ab") as log_file:
            self._server_process = subprocess.Popen(
                [python, "-m", "rula.coder_v2_server"],
                cwd=source_root.parent,
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self._health():
                return
            if self._server_process.poll() is not None:
                break
            time.sleep(0.25)

        details = self._log_tail(log_path)
        if self._server_process.poll() is None:
            self._server_process.terminate()
            self._server_process.wait(timeout=5)
        raise RuntimeError(
            "ExLlamaV2 Coderサーバーを起動できませんでした。"
            "install-coder-v2.ps1 を実行してEXL2推論環境を準備してください。"
            + (f"\n{details}" if details else "")
        )

    def ensure_loaded(self) -> None:
        with self._request("/model/load", {}, timeout=1800.0) as response:
            response.read()

    def generate(self, messages, cfg=None) -> str:
        self.ensure_loaded()
        return super().generate(messages, cfg)

    def close(self) -> None:
        try:
            with self._request("/model/unload", {}, timeout=180.0) as response:
                response.read()
        except RuntimeError as exc:
            log.error("Could not unload ExLlamaV2 during shutdown: %s", exc)

    def shutdown(self) -> None:
        if self._server_process is None or self._server_process.poll() is not None:
            return
        try:
            with self._request("/shutdown", {}, timeout=5.0) as response:
                response.read()
            self._server_process.wait(timeout=10)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            log.error("Could not stop the ExLlamaV2 server cleanly: %s", exc)
            if self._server_process.poll() is None:
                self._server_process.terminate()
                self._server_process.wait(timeout=5)
