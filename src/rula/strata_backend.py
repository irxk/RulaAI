"""Rula adapter for a running Strata OpenAI-compatible API server."""
import base64
import json
import logging
import os
import urllib.error
import urllib.request

log = logging.getLogger("rula.strata")


def _normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise RuntimeError("推論APIのURLは http:// または https:// で指定してください。")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


class StrataEngine:
    """Same generation surface as the local engines, backed by Strata."""

    echoes_stream = True
    supports_thinking = False

    def __init__(self, *, base_url: str | None = None,
                 api_key: str | None = None, model: str | None = None,
                 backend: str = "strata", display_name: str | None = None,
                 model_env: str = "RULA_STRATA_MODEL",
                 vision_env: str = "RULA_STRATA_VISION",
                 service_name: str = "Strata"):
        self.base_url = _normalize_base_url(
            base_url if base_url is not None else os.environ.get(
                "RULA_STRATA_BASE_URL", "http://127.0.0.1:8080/v1"
            )
        )
        self.api_key = (
            api_key if api_key is not None
            else os.environ.get("RULA_STRATA_API_KEY", "")
        ).strip()
        self.device = "cpu"
        self.quantized = False
        self.backend = backend
        self.service_name = service_name
        self.size = None
        self.cancel = False
        self.hit_token_limit = False
        self.last_reasoning: str | None = None
        self.echo_sink = None
        self.echo_guard = None
        self.echo_reset = None
        self.supports_vision = os.environ.get(
            vision_env, ""
        ).strip().lower() in {"1", "true", "yes", "on"}

        model_ids = self._models()
        requested = (model or os.environ.get(model_env, "")).strip()
        if requested and requested not in model_ids:
            raise RuntimeError(
                f"{self.service_name} にモデル {requested!r} がありません。"
                f"利用可能: {', '.join(model_ids)}"
            )
        if not requested and len(model_ids) != 1:
            raise RuntimeError(
                f"{self.service_name}で複数モデルが利用可能です。"
                f"使用するモデルIDを {model_env} に指定してください: "
                + ", ".join(model_ids)
            )
        self.model = requested or model_ids[0]
        self.display_name = display_name or os.environ.get(
            "RULA_DISPLAY_NAME", self.model
        )

    def _request(self, path: str, payload: dict | None = None,
                 timeout: float = 600.0):
        headers = {
            "Accept": "text/event-stream"
            if payload and payload.get("stream") else "application/json"
        }
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers,
            method="POST" if payload is not None else "GET",
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            message = detail or exc.reason
            raise RuntimeError(
                f"{self.service_name} API が HTTP {exc.code} を返しました: {message}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"{self.service_name} API に接続できません ({self.base_url})。"
                f"{self.service_name} を起動し、URLとポートを確認してください。"
            ) from exc

    def _models(self) -> list[str]:
        with self._request("/models", timeout=5.0) as response:
            body = json.loads(response.read().decode("utf-8"))
        models = [
            item.get("id") for item in body.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        if not models:
            raise RuntimeError(
                f"{self.service_name} API は応答しましたが、利用可能なモデルがありません。"
                "モデルを読み込んでからRulaを起動してください。"
            )
        return models

    def generate(self, messages, cfg=None) -> str:
        from .engine import GenConfig
        cfg = cfg or GenConfig()

        chat = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]
        if cfg.prefill:
            chat.append({"role": "assistant", "content": cfg.prefill})
        streaming = bool(cfg.stream or cfg.on_token or cfg.abort_check)
        payload = {
            "model": self.model,
            "messages": chat,
            "stream": streaming,
            "max_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        }
        if cfg.response_format is not None:
            payload["response_format"] = cfg.response_format
        if cfg.stop:
            payload["stop"] = cfg.stop

        content: list[str] = []
        thinking: list[str] = []
        finish_reason = None
        echo_printed = None
        with self._request("/chat/completions", payload) as response:
            if not streaming:
                body = json.loads(response.read().decode("utf-8"))
                choices = body.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    content.extend(self._text_parts(message.get("content")))
                    thinking.extend(self._reasoning_parts(message))
                    finish_reason = choices[0].get("finish_reason")
            else:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    for piece in self._reasoning_parts(delta):
                        thinking.append(piece)
                        self._notify(cfg.on_token, "thinking", piece)
                        if self.cancel:
                            finish_reason = "cancelled"
                            break
                    if finish_reason == "cancelled":
                        break
                    for piece in self._text_parts(delta.get("content")):
                        content.append(piece)
                        self._notify(cfg.on_token, "content", piece)
                        if cfg.stream:
                            echo_content = ([cfg.prefill] if cfg.prefill else []) + content
                            echo_printed = self._echo(
                                echo_content, cfg.prefill, echo_printed)
                        if cfg.abort_check is not None:
                            try:
                                if cfg.abort_check("".join(content)):
                                    finish_reason = "cancelled"
                                    break
                            except Exception:
                                log.exception("Strata abort check failed")
                        if self.cancel:
                            finish_reason = "cancelled"
                            break
                    if finish_reason == "cancelled":
                        break
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                        break

        self.last_reasoning = "".join(thinking).strip() or None
        self.hit_token_limit = finish_reason == "length"
        return "".join(content).strip()

    @staticmethod
    def _text_parts(value) -> list[str]:
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [
                item["text"] for item in value
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
        return []

    @staticmethod
    def _reasoning_parts(message: dict) -> list[str]:
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = message.get(key)
            if isinstance(value, str) and value:
                return [value]
        return []

    @staticmethod
    def _notify(callback, kind: str, piece: str) -> None:
        if callback is not None:
            try:
                callback(kind, piece)
            except Exception:
                log.exception("Strata token callback failed")

    def _echo(self, content: list[str], prefill: str,
              printed: int | None) -> int | None:
        if printed == -1:
            return -1
        joined = "".join(content)
        if printed is None:
            if prefill and len(joined) < len(prefill) and prefill.startswith(joined):
                return None
            start = len(prefill) if prefill and joined.startswith(prefill) else 0
            if self.echo_guard is not None:
                verdict = self.echo_guard(joined[start:])
                if verdict is None:
                    return None
                if verdict is False:
                    return -1
            if start == 0 and self.echo_reset is not None:
                try:
                    self.echo_reset()
                except Exception:
                    log.exception("Strata echo reset failed")
            printed = start
        if len(joined) > printed:
            if self.echo_sink is not None:
                try:
                    self.echo_sink(joined[printed:])
                except Exception:
                    log.exception("Strata echo sink failed")
            else:
                import sys
                sys.stdout.write(joined[printed:])
                sys.stdout.flush()
        return len(joined)

    def describe_image(self, ref: str, question: str | None = None,
                       max_new_tokens: int = 320) -> str:
        from .vision import DEFAULT_QUESTION, _open_image
        try:
            image = _open_image(ref)
        except Exception as exc:
            return f"画像を読み込めませんでした: {type(exc).__name__}: {exc}"

        import io
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        image_url = "data:image/jpeg;base64," + base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": question or DEFAULT_QUESTION},
                ],
            }],
            "stream": False,
            "max_tokens": max_new_tokens,
            "temperature": 0.2,
        }
        try:
            with self._request("/chat/completions", payload) as response:
                body = json.loads(response.read().decode("utf-8"))
            choices = body.get("choices") or []
            if not choices:
                return "画像の内容を説明できませんでした。"
            text = self._text_parts(
                (choices[0].get("message") or {}).get("content")
            )
            return "".join(text).strip() or "画像の内容を説明できませんでした。"
        except Exception as exc:
            return f"画像の解析に失敗しました: {type(exc).__name__}: {exc}"
