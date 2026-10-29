"""Rula adapter for a running TabbyAPI / ExLlamaV3 server."""
import json
import logging
import os

from .strata_backend import StrataEngine, _normalize_base_url

log = logging.getLogger("rula.tabbyapi")


class TabbyAPIEngine(StrataEngine):
    """Same generation surface as StrataEngine, backed by TabbyAPI."""

    def __init__(self, model: str | None = None, *,
                 model_env: str = "RULA_TABBYAPI_MODEL",
                 display_name: str | None = None,
                 settings_prefix: str = "RULA_TABBYAPI"):
        requested_model = model or os.environ.get(
            model_env,
            "Llama-3.1-8B-Instruct-exl3",
        ).strip()
        base_url = _normalize_base_url(os.environ.get(
            "RULA_TABBYAPI_BASE_URL", "http://127.0.0.1:5000/v1"
        ))
        api_key = os.environ.get("RULA_TABBYAPI_API_KEY", "").strip()
        self.admin_key = os.environ.get("RULA_TABBYAPI_ADMIN_KEY", "").strip()
        self.base_url = base_url
        self.api_key = api_key
        self.service_name = "TabbyAPI"
        self.model = requested_model
        self.model_env = model_env
        self.settings_prefix = settings_prefix
        if os.environ.get("RULA_APP_MODE", "").strip().lower() in {
                "general", "coder"}:
            from .local_services import ensure_tabbyapi_started
            ensure_tabbyapi_started(base_url, api_key)
        self._ensure_model(requested_model)
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=requested_model,
            backend="tabbyapi",
            display_name=display_name or os.environ.get(
                f"{settings_prefix}_DISPLAY_NAME",
                "Llama 3.1 8B Instruct",
            ),
            model_env=model_env,
            vision_env="RULA_TABBYAPI_VISION",
            service_name="TabbyAPI",
        )

    def _ensure_model(self, requested_model: str) -> None:
        with self._request("/models", timeout=5.0) as response:
            body = json.loads(response.read().decode("utf-8"))
        loaded = [
            item.get("id") for item in body.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        if requested_model in loaded:
            return
        if not self.admin_key:
            raise RuntimeError(
                f"TabbyAPIで {requested_model!r} がロードされていません。"
                "RULA_TABBYAPI_ADMIN_KEY を設定してモデルを自動ロードするか、"
                "TabbyAPI側でこのモデルをロードしてください。"
            )

        payload = {
            "model_name": requested_model,
            "max_seq_len": int(os.environ.get(
                f"{self.settings_prefix}_MAX_SEQ_LEN",
                os.environ.get("RULA_TABBYAPI_MAX_SEQ_LEN", "8192"),
            )),
            "cache_size": int(os.environ.get(
                f"{self.settings_prefix}_CACHE_SIZE",
                os.environ.get("RULA_TABBYAPI_CACHE_SIZE", "8192"),
            )),
            "cache_mode": os.environ.get(
                f"{self.settings_prefix}_CACHE_MODE",
                os.environ.get("RULA_TABBYAPI_CACHE_MODE", "Q4"),
            ),
        }
        original_key = self.api_key
        self.api_key = self.admin_key
        try:
            with self._request("/model/load", payload, timeout=1800.0) as response:
                response.read()
        finally:
            self.api_key = original_key

    def ensure_loaded(self) -> None:
        """Restore this engine's model after another TabbyAPI model was used."""
        self._ensure_model(self.model)

    def generate(self, messages, cfg=None) -> str:
        self.ensure_loaded()
        return super().generate(messages, cfg)

    def close(self) -> None:
        if not self.admin_key:
            log.warning(
                "RULA_TABBYAPI_ADMIN_KEY is not set; the TabbyAPI model remains loaded."
            )
            return
        original_key = self.api_key
        self.api_key = self.admin_key
        try:
            if self.model not in self._models():
                return
            with self._request("/model/unload", {}, timeout=60.0) as response:
                response.read()
        except RuntimeError as exc:
            log.error("Could not unload the TabbyAPI model during shutdown: %s", exc)
        finally:
            self.api_key = original_key
