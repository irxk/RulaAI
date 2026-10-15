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

from rula.gui import server
from rula.version import __version__


class GuiVersionTests(unittest.TestCase):
    def test_coder_prompt_uses_bounded_four_phase_workflow(self):
        workflow = server.CODER_WORKFLOW_INSTRUCTIONS

        self.assertIn("依頼の種類・完了条件・制約", workflow)
        self.assertIn("同じ計画や調査を繰り返しません", workflow)
        self.assertIn("失敗したら出力を根拠に修正して再検証", workflow)
        self.assertIn("未解決点だけを報告", workflow)

    def test_stream_guard_recognizes_each_new_coder_tool(self):
        for name in ("find_files", "find_symbol", "read_files",
                     "file_outline", "check_python"):
            with self.subTest(name=name):
                self.assertIn(name, server._TOOL_WORDS)

    def test_only_large_coder_effort_enables_supported_native_reasoning(self):
        engine = SimpleNamespace(supports_thinking=True)

        self.assertTrue(server._coder_thinking_enabled("large", engine))
        self.assertFalse(server._coder_thinking_enabled("medium", engine))
        self.assertFalse(
            server._coder_thinking_enabled(
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
