import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

source_root = Path(os.environ.get(
    "RULA_TEST_SOURCE", str(Path(__file__).resolve().parents[1] / "src")
)).resolve()
sys.path.insert(0, str(source_root))

from rula.engine import GenConfig, build_coder_engine, build_engine
from rula.strata_backend import StrataEngine
from rula.tabbyapi_backend import TabbyAPIEngine
from rula.agent import Agent


class FakeResponse:
    def __init__(self, body=b"", lines=None):
        self.body = body
        self.lines = lines or []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False if args else None

    def read(self):
        return self.body

    def __iter__(self):
        return iter(self.lines)


class StrataBackendTests(unittest.TestCase):
    def test_main_engine_defaults_to_llama_31_8b_on_tabbyapi(self):
        response = FakeResponse(json.dumps({
            "data": [{"id": "Llama-3.1-8B-Instruct-exl3"}],
        }).encode())
        with patch.dict(os.environ, {
            "RULA_MAIN_BACKEND": "tabbyapi",
            "RULA_TABBYAPI_MODEL": "Llama-3.1-8B-Instruct-exl3",
            "RULA_TABBYAPI_BASE_URL": "http://127.0.0.1:5000/v1",
            "RULA_TABBYAPI_API_KEY": "test-key",
        }), patch("urllib.request.urlopen", return_value=response):
            engine = build_engine()

        self.assertEqual(engine.backend, "tabbyapi")
        self.assertEqual(engine.model, "Llama-3.1-8B-Instruct-exl3")
        self.assertEqual(engine.display_name, "Llama 3.1 8B Instruct")
        self.assertEqual(engine.base_url, "http://127.0.0.1:5000/v1")
        self.assertEqual(engine.api_key, "test-key")

    def test_default_main_engine_uses_tabbyapi(self):
        response = FakeResponse(json.dumps({
            "data": [{"id": "Llama-3.1-8B-Instruct-exl3"}],
        }).encode())
        with patch.dict(os.environ, {
            "RULA_TABBYAPI_MODEL": "Llama-3.1-8B-Instruct-exl3",
        }, clear=True), patch("urllib.request.urlopen", return_value=response):
            engine = build_engine()

        self.assertEqual(engine.backend, "tabbyapi")

    def test_tabbyapi_requests_use_chat_api_key(self):
        requests = []
        models = FakeResponse(json.dumps({
            "data": [{"id": "Llama-3.1-8B-Instruct-exl3"}],
        }).encode())
        completion = FakeResponse(json.dumps({
            "choices": [{"message": {"content": "ready"}, "finish_reason": "stop"}],
        }).encode())

        def fake_urlopen(request, timeout):
            requests.append(request)
            return models if request.full_url.endswith("/models") else completion

        with patch.dict(os.environ, {
            "RULA_TABBYAPI_BASE_URL": "http://127.0.0.1:5000",
            "RULA_TABBYAPI_API_KEY": "tabby-test-key",
            "RULA_TABBYAPI_MODEL": "Llama-3.1-8B-Instruct-exl3",
        }), patch("urllib.request.urlopen", side_effect=fake_urlopen):
            engine = build_engine()
            answer = engine.generate(
                [{"role": "user", "content": "test"}],
                GenConfig(max_new_tokens=8),
            )

        self.assertEqual(answer, "ready")
        self.assertEqual(
            requests[-1].get_header("Authorization"),
            "Bearer tabby-test-key",
        )
        self.assertEqual(
            requests[-1].full_url,
            "http://127.0.0.1:5000/v1/chat/completions",
        )

    def test_coder_engine_reloads_its_model_after_planner_switch(self):
        requests = []
        model_responses = iter([
            FakeResponse(json.dumps({
                "data": [{"id": "Qwen2.5-Coder-7B-Instruct-exl3"}],
            }).encode()),
            FakeResponse(json.dumps({
                "data": [{"id": "Qwen2.5-Coder-7B-Instruct-exl3"}],
            }).encode()),
            FakeResponse(json.dumps({
                "data": [{"id": "Llama-3.1-8B-Instruct-exl3"}],
            }).encode()),
        ])
        completion = FakeResponse(json.dumps({
            "choices": [{"message": {"content": "coder-ready"}}],
        }).encode())

        def fake_urlopen(request, timeout):
            requests.append(request)
            if request.full_url.endswith("/models"):
                return next(model_responses)
            return completion

        with patch.dict(os.environ, {
            "RULA_CODER_BACKEND": "tabbyapi",
            "RULA_CODER_TABBYAPI_MODEL": "Qwen2.5-Coder-7B-Instruct-exl3",
            "RULA_CODER_TABBYAPI_MAX_SEQ_LEN": "4096",
            "RULA_TABBYAPI_ADMIN_KEY": "tabby-admin-key",
        }), patch("urllib.request.urlopen", side_effect=fake_urlopen):
            engine = build_coder_engine()
            answer = engine.generate(
                [{"role": "user", "content": "test"}],
                GenConfig(max_new_tokens=8),
            )

        self.assertEqual(answer, "coder-ready")
        self.assertTrue(requests[3].full_url.endswith("/model/load"))
        self.assertEqual(
            json.loads(requests[3].data)["max_seq_len"],
            4096,
        )
        self.assertTrue(requests[4].full_url.endswith("/chat/completions"))

    def test_tabbyapi_connection_failure_is_reported_without_fallback(self):
        with patch.dict(os.environ, {
            "RULA_MAIN_BACKEND": "tabbyapi",
            "RULA_TABBYAPI_BASE_URL": "http://127.0.0.1:1/v1",
            "RULA_TABBYAPI_API_KEY": "test-key",
        }), patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            with self.assertRaisesRegex(RuntimeError, "TabbyAPI API に接続できません"):
                build_engine()

    def test_desktop_tabbyapi_backend_starts_its_local_service(self):
        with patch.dict(os.environ, {
                "RULA_APP_MODE": "general",
                "RULA_TABBYAPI_BASE_URL": "http://127.0.0.1:5000/v1",
                "RULA_TABBYAPI_API_KEY": "test-key",
        }), patch("rula.local_services.ensure_tabbyapi_started") as start, \
                patch.object(TabbyAPIEngine, "_ensure_model"), \
                patch.object(StrataEngine, "_models",
                             return_value=["Llama-3.1-8B-Instruct-exl3"]):
            engine = TabbyAPIEngine()

        start.assert_called_once_with(engine.base_url, "test-key")

    def test_tabbyapi_loads_and_unloads_model_with_admin_key(self):
        requests = []
        model_responses = iter([
            FakeResponse(json.dumps({"data": []}).encode()),
            FakeResponse(json.dumps({
                "data": [{"id": "Llama-3.1-8B-Instruct-exl3"}],
            }).encode()),
            FakeResponse(json.dumps({
                "data": [{"id": "Llama-3.1-8B-Instruct-exl3"}],
            }).encode()),
        ])

        def fake_urlopen(request, timeout):
            requests.append(request)
            if request.full_url.endswith("/models"):
                return next(model_responses)
            return FakeResponse()

        with patch.dict(os.environ, {
            "RULA_TABBYAPI_BASE_URL": "http://127.0.0.1:5000/v1",
            "RULA_TABBYAPI_API_KEY": "tabby-user-key",
            "RULA_TABBYAPI_ADMIN_KEY": "tabby-admin-key",
            "RULA_TABBYAPI_MODEL": "Llama-3.1-8B-Instruct-exl3",
        }), patch("urllib.request.urlopen", side_effect=fake_urlopen):
            engine = TabbyAPIEngine()
            engine.close()

        self.assertEqual(requests[1].full_url, "http://127.0.0.1:5000/v1/model/load")
        self.assertEqual(requests[1].get_header("Authorization"), "Bearer tabby-admin-key")
        self.assertEqual(
            json.loads(requests[1].data),
            {
                "model_name": "Llama-3.1-8B-Instruct-exl3",
                "max_seq_len": 8192,
                "cache_size": 8192,
                "cache_mode": "Q4",
            },
        )
        self.assertEqual(requests[2].full_url, "http://127.0.0.1:5000/v1/models")
        self.assertEqual(requests[3].full_url, "http://127.0.0.1:5000/v1/models")
        self.assertEqual(requests[4].full_url, "http://127.0.0.1:5000/v1/model/unload")
        self.assertEqual(requests[3].get_header("Authorization"), "Bearer tabby-admin-key")

    def test_coder_engine_uses_its_own_ollama_model(self):
        response = FakeResponse(json.dumps({
            "models": [{"name": "qwen2.5-coder:7b"}],
        }).encode())
        with patch.dict(os.environ, {
            "RULA_CODER_BACKEND": "ollama",
            "RULA_CODER_OLLAMA_MODEL": "qwen2.5-coder:7b",
            "RULA_CODER_NUM_CTX": "8192",
            "RULA_CODER_KEEP_ALIVE": "0",
        }), patch("urllib.request.urlopen", return_value=response):
            engine = build_coder_engine()

        self.assertEqual(engine.backend, "ollama")
        self.assertEqual(engine.model, "qwen2.5-coder:7b")
        self.assertEqual(engine.display_name, "Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(engine.num_ctx, 8192)
        self.assertEqual(engine.keep_alive, "0")

    def test_coder_engine_rejects_unknown_backend(self):
        with patch.dict(os.environ, {"RULA_CODER_BACKEND": "unknown"}):
            with self.assertRaisesRegex(RuntimeError, "RULA_CODER_BACKEND"):
                build_coder_engine()

    def test_coder_engine_defaults_to_qwen_exl2_on_exllamav2(self):
        response = FakeResponse(json.dumps({
            "data": [{"id": "Qwen2.5-Coder-7B-Instruct-4.0bpw-exl2"}],
        }).encode())
        with patch("rula.engine.load_env_file"), patch.dict(
                os.environ, {}, clear=True), patch(
                "rula.coder_v2_backend.ExLlamaV2Engine._ensure_server"), patch(
                "urllib.request.urlopen", return_value=response):
            engine = build_coder_engine()

        self.assertEqual(engine.backend, "exllamav2")
        self.assertEqual(engine.model, "Qwen2.5-Coder-7B-Instruct-4.0bpw-exl2")
        self.assertEqual(
            engine.display_name,
            "Qwen2.5-Coder-7B-Instruct (EXL2)",
        )

    def test_streaming_maps_config_reasoning_and_token_callbacks(self):
        requests = []
        events = []
        models = FakeResponse(json.dumps({
            "data": [{"id": "qwen-coder"}],
        }).encode())
        stream = FakeResponse(lines=[
            b'data: {"choices":[{"delta":{"reasoning_content":"checking"},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{"content":"done"},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n',
            b"data: [DONE]\n",
        ])

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return models if request.full_url.endswith("/models") else stream

        with patch.dict(os.environ, {
            "RULA_STRATA_BASE_URL": "http://strata.local",
            "RULA_STRATA_API_KEY": "local-test-key",
            "RULA_STRATA_MODEL": "qwen-coder",
        }), patch("urllib.request.urlopen", side_effect=fake_urlopen):
            engine = StrataEngine()
            answer = engine.generate(
                [{"role": "user", "content": "write code"}],
                GenConfig(
                    max_new_tokens=90,
                    temperature=0.25,
                    top_p=0.8,
                    stop=["END"],
                    on_token=lambda kind, text: events.append((kind, text)),
                    abort_check=lambda text: text == "cancel",
                ),
            )

        self.assertEqual(answer, "done")
        self.assertEqual(engine.last_reasoning, "checking")
        self.assertTrue(engine.hit_token_limit)
        self.assertEqual(events, [
            ("thinking", "checking"),
            ("content", "done"),
        ])
        request, _ = requests[-1]
        self.assertEqual(request.full_url, "http://strata.local/v1/chat/completions")
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer " + "local-test-key",
        )
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen-coder")
        self.assertEqual(payload["max_tokens"], 90)
        self.assertEqual(payload["stop"], ["END"])
        self.assertTrue(payload["stream"])

    def test_structured_response_format_is_sent_to_compatible_api(self):
        requests = []
        models = FakeResponse(json.dumps({
            "data": [{"id": "structured-model"}],
        }).encode())
        answer = FakeResponse(json.dumps({
            "choices": [{"message": {"content": "{}"}}],
        }).encode())

        def fake_urlopen(request, timeout):
            requests.append(request)
            return models if request.full_url.endswith("/models") else answer

        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "plan", "schema": {"type": "object"}},
        }
        with patch.dict(os.environ, {
            "RULA_STRATA_MODEL": "structured-model",
        }), patch("urllib.request.urlopen", side_effect=fake_urlopen):
            engine = StrataEngine()
            engine.generate(
                [{"role": "user", "content": "plan"}],
                GenConfig(response_format=response_format),
            )

        self.assertEqual(json.loads(requests[-1].data)["response_format"], response_format)

    def test_image_requests_use_inline_image_and_require_explicit_vision(self):
        from PIL import Image
        import tempfile

        models = FakeResponse(json.dumps({
            "data": [{"id": "vision-model"}],
        }).encode())
        answer = FakeResponse(json.dumps({
            "choices": [{"message": {"content": "a small test image"}}],
        }).encode())
        requests = []

        def fake_urlopen(request, timeout):
            self.assertGreater(timeout, 0)
            requests.append(request)
            return models if request.full_url.endswith("/models") else answer

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            Image.new("RGB", (8, 8), "white").save(path)
            with patch.dict(os.environ, {
                "RULA_STRATA_BASE_URL": "http://127.0.0.1:8080/v1",
                "RULA_STRATA_VISION": "1",
            }), patch("urllib.request.urlopen", side_effect=fake_urlopen):
                engine = StrataEngine()
                self.assertTrue(engine.supports_vision)
                self.assertEqual(
                    engine.describe_image(str(path), "What is shown?"),
                    "a small test image",
                )

        payload = json.loads(requests[-1].data)
        content = payload["messages"][0]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(content[1]["text"], "What is shown?")

    def test_pinned_backend_reports_connection_failure_instead_of_fallback(self):
        with patch.dict(os.environ, {
            "RULA_MAIN_BACKEND": "strata",
            "RULA_STRATA_BASE_URL": "http://127.0.0.1:1/v1",
        }), patch(
            "urllib.request.urlopen",
            side_effect=OSError("connection refused"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Strata API に接続できません"):
                build_engine()

    def test_explicit_main_strata_backend_is_used(self):
        response = FakeResponse(json.dumps({
            "data": [{"id": "qwen-coder"}],
        }).encode())

        with patch.dict(os.environ, {
            "RULA_MAIN_BACKEND": "strata",
            "RULA_STRATA_BASE_URL": "http://127.0.0.1:8080/v1",
        }), patch("urllib.request.urlopen", return_value=response):
            engine = build_engine(prefer_fast=False)

        self.assertIsInstance(engine, StrataEngine)

    def test_model_id_selects_strata_model_and_adapter_is_rejected(self):
        response = FakeResponse(json.dumps({
            "data": [{"id": "qwen-coder"}],
        }).encode())
        with patch.dict(os.environ, {
            "RULA_MAIN_BACKEND": "strata",
            "RULA_STRATA_BASE_URL": "http://127.0.0.1:8080/v1",
        }), patch("urllib.request.urlopen", return_value=response):
            engine = build_engine(model_id="qwen-coder")
            self.assertEqual(engine.model, "qwen-coder")
            with self.assertRaisesRegex(RuntimeError, "LoRA adapter指定"):
                build_engine(adapter="local-lora")

    def test_multiple_models_require_an_explicit_choice(self):
        response = FakeResponse(json.dumps({
            "data": [{"id": "model-a"}, {"id": "model-b"}],
        }).encode())
        with patch.dict(os.environ, {
            "RULA_STRATA_BASE_URL": "http://127.0.0.1:8080/v1",
        }), patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "RULA_STRATA_MODEL"):
                StrataEngine()

    def test_strata_without_image_capability_never_falls_back_to_local_vision(self):
        agent = object.__new__(Agent)
        agent.engine = type("StrataTextOnly", (), {
            "backend": "strata",
            "supports_vision": False,
        })()
        with self.assertRaisesRegex(RuntimeError, "画像入力に対応していません"):
            agent.look("unused.png")


if __name__ == "__main__":
    unittest.main()
