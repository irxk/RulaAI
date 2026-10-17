import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

source_root = Path(os.environ.get(
    "RULA_TEST_SOURCE", str(Path(__file__).resolve().parents[1] / "src")
)).resolve()
sys.path.insert(0, str(source_root))

from fastapi import Request

from rula.agent import AgentConfig
from rula.coder_agent import CoderAgent
from rula.coder_workflow import CoderWorkflow
from rula.tools import default_registry
from rula.gui import server
from rula.version import __version__


class GuiVersionTests(unittest.TestCase):
    def test_failed_write_activity_is_not_marked_complete(self):
        self.assertFalse(server._coder_tool_succeeded(
            "write_file", "形式エラー: 相対パス :: ファイル全文"))
        self.assertTrue(server._coder_tool_succeeded(
            "write_file", "保存しました: index.html"))
        self.assertEqual(
            server._coder_activity_result("write_file", "形式エラー: 引数を確認してください。"),
            "形式エラー: 引数を確認してください。",
        )

    def test_local_server_activity_shows_readiness_instead_of_generic_running(self):
        result = server._coder_activity_result(
            "run_command",
            "サーバー起動完了: http://127.0.0.1:8000/\nプロセスID: 4321")

        self.assertIn("サーバー起動完了", result)
        self.assertIn("127.0.0.1:8000", result)

    def test_coder_prompt_uses_bounded_four_phase_workflow(self):
        workflow = CoderWorkflow("auto", "medium").system_prompt()

        self.assertIn("依頼の種類・完了条件・制約", workflow)
        self.assertIn("同じ計画や調査を繰り返しません", workflow)
        self.assertIn("差分レビュー", workflow)
        self.assertIn("無関係な成功コマンド", workflow)

    def test_coder_prompt_classifies_different_software_tasks(self):
        cases = {
            "login is broken; fix the error": "bugfix",
            "既存APIに検索機能を追加して": "feature",
            "Add tests for the parser": "test",
            "READMEの手順を直して": "documentation",
            "コードをリファクタして整理": "refactor",
            "Discord用のwebsiteを作って": "website",
        }
        for request, expected in cases.items():
            with self.subTest(request=request):
                self.assertEqual(CoderWorkflow.classify_task(request), expected)
                self.assertIn(
                    f"依頼タイプ: {expected}",
                    CoderWorkflow("auto", "medium").system_prompt(request),
                )

    def test_dedicated_coder_agent_owns_its_mode_and_effort_policy(self):
        agent = CoderAgent(
            SimpleNamespace(supports_thinking=True),
            AgentConfig(candid=True, stream=True, preview=True),
            default_registry(),
        )

        agent.configure_request(
            "auto", "large", default_registry(), memory_context="再開メモ")

        self.assertTrue(agent.config.coder_auto)
        self.assertFalse(agent.config.stream)
        self.assertFalse(agent.config.preview)
        self.assertEqual(agent.config.max_tool_calls, 32)
        self.assertEqual(agent.config.max_new_tokens, 6144)
        self.assertTrue(agent.config.thinking)
        self.assertIn("RulaCoder専用", agent.config.extra_system)
        self.assertIn("再開メモ", agent.config.extra_system)

    def test_coder_profile_gets_separate_agent_with_shared_model_engine(self):
        engine = SimpleNamespace(supports_thinking=True)
        chat = {"id": "test-coder-agent", "profile": "coder", "messages": []}

        with patch.dict(server._agents, {}, clear=True), patch.dict(
                server.STATE, {"engine": engine}):
            agent = server._agent_for(chat)

        self.assertIsInstance(agent, CoderAgent)
        self.assertIs(agent.engine, engine)

    def test_stream_guard_recognizes_each_new_coder_tool(self):
        for name in ("find_files", "find_symbol", "read_files", "project_memory",
                     "file_outline", "check_python"):
            with self.subTest(name=name):
                self.assertIn(name, server._TOOL_WORDS)

    def test_only_large_coder_effort_enables_supported_native_reasoning(self):
        engine = SimpleNamespace(supports_thinking=True)

        self.assertTrue(CoderWorkflow.thinking_enabled("large", engine))
        self.assertFalse(CoderWorkflow.thinking_enabled("medium", engine))
        self.assertFalse(
            CoderWorkflow.thinking_enabled(
                "large", SimpleNamespace(supports_thinking=False)))

    def test_info_endpoint_reports_application_version(self):
        state = {
            "token": "test-token",
            "port": 43127,
            "status": "ready",
            "error": None,
            "engine": SimpleNamespace(
                display_name="test", size="9B", backend="test",
                mtp=False, supports_vision=False, supports_thinking=False,
            ),
        }
        request = Request({
            "type": "http",
            "headers": [
                (b"host", b"127.0.0.1:43127"),
                (b"x-rula-token", b"test-token"),
            ],
        })

        with patch.dict(server.STATE, state), patch.object(
                server, "_vram_free", return_value=None):
            response = server.info(request)

        self.assertEqual(response["version"], __version__)
        self.assertRegex(response["version"], r"^\d+\.\d{2}$")
        self.assertIn(response["app_mode"], ("general", "coder"))
        self.assertFalse(response["switching"])

    def test_coder_app_always_routes_requests_to_coder_profile(self):
        with patch.object(server, "APP_MODE", "coder"):
            self.assertEqual(server._profile_for_app("general"), "coder")
            self.assertEqual(server._profile_for_app("fast"), "coder")

    def test_switch_endpoint_requests_window_close_with_local_token(self):
        request = Request({
            "type": "http",
            "headers": [
                (b"host", b"127.0.0.1:43127"),
                (b"x-rula-token", b"test-token"),
            ],
        })
        state = {
            "token": "test-token",
            "port": 43127,
            "switching": False,
        }

        with patch.dict(server.STATE, state):
            result = server.switch_app(request, "test-token")
            ping = server.ping(request)

        self.assertEqual(result, {"ok": True})
        self.assertTrue(ping["switching"])

    def test_dedicated_coder_ui_is_locked_to_coder_profile(self):
        static = source_root / "rula" / "gui" / "static"
        app_js = (static / "app.js").read_text(encoding="utf-8")
        server_source = (source_root / "rula" / "gui" / "server.py").read_text(
            encoding="utf-8")
        root = source_root.parent

        self.assertIn('option.hidden = coder ? option.value !== "coder"', app_js)
        self.assertIn('req.profile = _profile_for_app(req.profile)', server_source)
        self.assertIn('"RulaCoder" if APP_MODE == "coder" else "Rula"', server_source)
        launcher = (root / "launcher" / "RulaLauncher.cs").read_text(encoding="utf-8")
        build = (root / "build.ps1").read_text(encoding="utf-8-sig")
        self.assertIn('const string MutexName = "Local\\\\RulaDesktopApp"', launcher)
        self.assertIn("RulaCoder.exe", build)
        self.assertIn("/define:CODER_APP", build)

    def test_version_badge_is_fixed_to_the_visible_window_corner(self):
        css = (source_root / "rula" / "gui" / "static" / "app.css").read_text(
            encoding="utf-8")

        self.assertRegex(
            css,
            r"\.app-version\s*\{[^}]*position:\s*fixed;[^}]*"
            r"right:\s*12px;[^}]*bottom:\s*5px;",
        )


if __name__ == "__main__":
    unittest.main()
