import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

source_root = Path(os.environ.get(
    "RULA_TEST_SOURCE", str(Path(__file__).resolve().parents[1] / "src")
)).resolve()
sys.path.insert(0, str(source_root))

from rula.coder_v2_backend import ExLlamaV2Engine
from rula.coder_agent import CoderAgent
from rula.engine import GenConfig


class FakeResponse:
    def __init__(self, body=b""):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False if args else None

    def read(self):
        return self.body


class ExLlamaV2BackendTests(unittest.TestCase):
    def test_generation_loads_model_before_openai_compatible_request(self):
        requests = []
        model_response = FakeResponse(json.dumps({
            "data": [{
                "id": "Qwen2.5-Coder-7B-Instruct-4.0bpw-exl2",
            }],
        }).encode())
        completion_response = FakeResponse(json.dumps({
            "choices": [{
                "message": {"content": "coder-ready"},
                "finish_reason": "stop",
            }],
        }).encode())

        def fake_urlopen(request, timeout):
            requests.append(request)
            if request.full_url.endswith("/models"):
                return model_response
            return completion_response

        with patch.object(ExLlamaV2Engine, "_ensure_server"), patch.dict(
                os.environ, {
                    "RULA_CODER_V2_BASE_URL": "http://127.0.0.1:5101/v1",
                    "RULA_CODER_V2_MODEL":
                        "Qwen2.5-Coder-7B-Instruct-4.0bpw-exl2",
                }), patch(
                "urllib.request.urlopen", side_effect=fake_urlopen):
            engine = ExLlamaV2Engine()
            answer = engine.generate(
                [{"role": "user", "content": "write code"}],
                GenConfig(max_new_tokens=8),
            )

        self.assertEqual(answer, "coder-ready")
        self.assertEqual(
            requests[1].full_url,
            "http://127.0.0.1:5101/v1/model/load",
        )
        self.assertEqual(
            requests[2].full_url,
            "http://127.0.0.1:5101/v1/chat/completions",
        )
        self.assertEqual(
            json.loads(requests[2].data)["model"],
            "Qwen2.5-Coder-7B-Instruct-4.0bpw-exl2",
        )

    def test_close_releases_the_exllama_v2_model(self):
        requests = []
        models = FakeResponse(json.dumps({
            "data": [{
                "id": "Qwen2.5-Coder-7B-Instruct-4.0bpw-exl2",
            }],
        }).encode())

        def fake_urlopen(request, timeout):
            requests.append(request)
            return models if request.full_url.endswith("/models") else FakeResponse()

        with patch.object(ExLlamaV2Engine, "_ensure_server"), patch.dict(
                os.environ, {}, clear=True), patch(
                "urllib.request.urlopen", side_effect=fake_urlopen):
            engine = ExLlamaV2Engine()
            engine.close()

        self.assertEqual(
            requests[-1].full_url,
            "http://127.0.0.1:5101/v1/model/unload",
        )

    def test_server_url_must_be_loopback(self):
        with patch.dict(os.environ, {
            "RULA_CODER_V2_BASE_URL": "https://example.com/v1",
        }):
            with self.assertRaisesRegex(RuntimeError, "このPC内"):
                ExLlamaV2Engine()

    def test_auto_planner_unloads_v2_model_before_loading_main_model(self):
        events = []
        plan = {
            "task_type": "feature",
            "target_files": ["src/main.py"],
            "requirements": ["Implement the feature"],
            "context_summary": "Implement the requested capability.",
            "validation": ["Run relevant tests"],
            "blocking_question": None,
        }

        class FakeEngine:
            backend = "exllamav2"
            supports_thinking = False

            @staticmethod
            def close():
                events.append("coder_unloaded")

        class FakePlanner:
            @staticmethod
            def close():
                events.append("planner_unloaded")

        with patch("rula.engine.load_env_file"), patch.dict(
                os.environ, {
                    "RULA_CODER_PLANNER_MODEL": "Llama-3.1-8B-Instruct-exl3",
                }), patch(
                "rula.tabbyapi_backend.TabbyAPIEngine",
                return_value=FakePlanner()), patch(
                "rula.coder_planner.create_coding_plan",
                side_effect=lambda *args: events.append("plan_created") or plan):
            agent = CoderAgent(FakeEngine())
            agent.configure_request(
                "auto", "small", Mock(), request_text="実装して"
            )

        self.assertEqual(
            events,
            ["coder_unloaded", "plan_created", "planner_unloaded"],
        )


if __name__ == "__main__":
    unittest.main()
