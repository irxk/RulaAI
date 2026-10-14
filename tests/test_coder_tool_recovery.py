import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

source_root = Path(os.environ.get(
    "RULA_TEST_SOURCE", str(Path(__file__).resolve().parents[1] / "src")
)).resolve()
sys.path.insert(0, str(source_root))

from rula.agent import Agent, AgentConfig, Trace
from rula.tools.coder import CoderTools


class CaptureEngine:
    display_name = "test"
    supports_thinking = False
    hit_token_limit = False

    def generate(self, _messages, config):
        self.config = config
        return "完了しました。"


class ToolProtocolRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.coder_tools = CoderTools(self.temp_dir.name, "auto")
        self.agent = Agent(
            engine=object(),
            config=AgentConfig(use_vision=False, max_tool_calls=8),
            tools=self.coder_tools.registry(),
        )

    def test_plan_then_malformed_calls_recover_and_write_file(self):
        outputs = iter([
            "ゲームの操作と画面構成を整理します。",
            ('<tool>update_plan: [{"step":"ゲームを実装する","status":"in_progress"}]</tool>'),
            "<tool>update_plan</tool>",
            "<tool>write_file</tool>",
            "<tool>write_file: game.py :: print('ready')\n</tool>",
            "ゲームのファイルを作成しました。必要なファイルを保存済みです。",
        ])
        prompts = []

        def generate(scratch, **kwargs):
            prompts.append([dict(message) for message in scratch])
            return next(outputs)

        self.agent._generate = generate
        trace = Trace()
        thinking, answer = self.agent._think_then_answer([], trace)

        self.assertEqual(thinking, "ゲームの操作と画面構成を整理します。")
        self.assertEqual(answer, "ゲームのファイルを作成しました。必要なファイルを保存済みです。")
        self.assertEqual(
            Path(self.temp_dir.name, "game.py").read_text(encoding="utf-8"),
            "print('ready')",
        )
        self.assertEqual(
            [name for name, _, _ in trace.tool_calls],
            ["update_plan", "update_plan", "write_file", "write_file"],
        )
        self.assertIn("引数が必要", prompts[4][-1]["content"])
        self.assertNotIn("<tool>", answer)

    def test_tool_budget_never_returns_unexecuted_tool_markup(self):
        self.agent.config.max_tool_calls = 1
        outputs = iter([
            '<tool>update_plan: [{"step":"調査する","status":"pending"}]</tool>',
            "<tool>write_file</tool>",
        ])
        self.agent._generate = lambda _scratch, **_kwargs: next(outputs)

        answer = self.agent._answer_or_tool([], Trace())

        self.assertIn("上限", answer)
        self.assertNotIn("<tool>", answer)
        self.assertFalse(Path(self.temp_dir.name, "game.py").exists())

    def test_invalid_plan_is_not_reported_as_shared(self):
        result = self.coder_tools.update_plan("")

        prompt = self.agent._tool_result_prompt(result, "update_plan")

        self.assertIn("更新できませんでした", prompt)
        self.assertIn("JSON配列", prompt)
        self.assertNotIn("共有されました", prompt)

    def test_disabled_streaming_never_echoes_partial_tool_source(self):
        engine = CaptureEngine()
        agent = Agent(
            engine=engine,
            config=AgentConfig(
                stream=False, preview=False, use_vision=False,
            ),
        )

        answer = agent._generate([], stream=True)

        self.assertEqual(answer, "完了しました。")
        self.assertFalse(engine.config.stream)

    def test_plan_is_not_recreated_or_regressed(self):
        published = []
        self.coder_tools.on_plan = lambda tasks: published.append(tasks)
        initial = '[{"step":"ゲームを作る","status":"in_progress"}]'
        self.coder_tools.update_plan(initial)

        repeated = self.coder_tools.update_plan(
            '[{"step":"設計から再確認してゲームを作る","status":"pending"}]')

        self.assertIn("維持", repeated)
        self.assertEqual(len(published), 1)
        self.assertEqual(self.coder_tools._plan, [
            {"step": "ゲームを作る", "status": "in_progress"},
        ])

    def test_auto_mode_bounds_read_only_inspection(self):
        for _ in range(5):
            self.coder_tools.registry().get("list_files").run(".")

        Path(self.temp_dir.name, "new-file.txt").write_text("now exists", encoding="utf-8")
        result = self.coder_tools.registry().get("list_files").run(".")

        self.assertTrue(result.startswith("調査上限:"))
        self.assertNotIn("new-file.txt", result)
        guidance = self.agent._tool_result_prompt(result, "list_files")
        self.assertIn("実装を開始", guidance)

    def test_project_info_uses_the_auto_inspection_budget(self):
        for _ in range(5):
            self.coder_tools.project_info()

        result = self.coder_tools.registry().get("project_info").run("")

        self.assertTrue(result.startswith("調査上限:"))

    def test_duplicate_plan_steps_are_rejected(self):
        result = self.coder_tools.update_plan(
            '[{"step":"実装する","status":"pending"},'
            '{"step":"実装する","status":"completed"}]')

        self.assertIn("重複", result)
        self.assertEqual(self.coder_tools._plan, [])

    def test_patch_validation_failure_does_not_change_earlier_files(self):
        first = Path(self.temp_dir.name, "first.txt")
        first.write_text("old", encoding="utf-8")
        result = self.coder_tools.apply_patch(
            "@@ first.txt\n--- old\n+++ new\n"
            "@@ missing.txt\n--- old\n+++ new")

        self.assertIn("ファイルが見つかりません", result)
        self.assertEqual(first.read_text(encoding="utf-8"), "old")

    def test_patch_rolls_back_when_a_later_replace_fails(self):
        first = Path(self.temp_dir.name, "first.txt")
        second = Path(self.temp_dir.name, "second.txt")
        first.write_text("old one", encoding="utf-8")
        second.write_text("old two", encoding="utf-8")
        real_replace = os.replace
        replace_count = 0

        def fail_second_replace(source, destination):
            nonlocal replace_count
            replace_count += 1
            if replace_count == 2:
                raise OSError("injected replacement failure")
            return real_replace(source, destination)

        with patch("rula.tools.coder.os.replace", side_effect=fail_second_replace):
            result = self.coder_tools.apply_patch(
                "@@ first.txt\n--- old one\n+++ new one\n"
                "@@ second.txt\n--- old two\n+++ new two")

        self.assertIn("ロールバックしました", result)
        self.assertEqual(first.read_text(encoding="utf-8"), "old one")
        self.assertEqual(second.read_text(encoding="utf-8"), "old two")
        self.assertEqual(self.coder_tools.changed_files(), [])
        self.assertEqual(list(Path(self.temp_dir.name).glob(".*.tmp")), [])
        self.assertEqual(list(Path(self.temp_dir.name).glob(".*.bak")), [])

    def test_patch_applies_multiple_crlf_files_together(self):
        first = Path(self.temp_dir.name, "first.txt")
        second = Path(self.temp_dir.name, "second.txt")
        first.write_bytes(b"old one\r\n")
        second.write_bytes(b"old two\r\n")

        result = self.coder_tools.apply_patch(
            "@@ first.txt\r\n--- old one\r\n+++ new one\r\n"
            "@@ second.txt\r\n--- old two\r\n+++ new two")

        self.assertIn("first.txt", result)
        self.assertIn("second.txt", result)
        self.assertEqual(first.read_bytes(), b"new one\r\n")
        self.assertEqual(second.read_bytes(), b"new two\r\n")
        self.assertEqual(self.coder_tools.changed_files(), ["first.txt", "second.txt"])

    def test_write_preserves_existing_crlf_and_reports_noop(self):
        path = Path(self.temp_dir.name, "crlf.txt")
        path.write_bytes(b"before\r\nsecond\r\n")

        result = self.coder_tools.write_file("crlf.txt :: after\nsecond\n")

        self.assertTrue(result.startswith("保存しました:"))
        self.assertEqual(path.read_bytes(), b"after\r\nsecond\r\n")
        self.assertEqual(self.coder_tools.changed_files(), ["crlf.txt"])
        unchanged = self.coder_tools.write_file("crlf.txt :: after\nsecond\n")
        self.assertTrue(unchanged.startswith("変更なし:"))

    def test_write_rejects_paths_outside_workspace(self):
        result = self.coder_tools.write_file("../outside.txt :: unsafe")

        self.assertIn("外", result)
        self.assertFalse(Path(self.temp_dir.name).parent.joinpath("outside.txt").exists())

    def test_repeated_identical_tool_calls_are_stopped(self):
        self.agent.config.max_tool_calls = 5
        call = '<tool>update_plan: [{"step":"実装する","status":"in_progress"}]</tool>'
        outputs = iter([call, call, call])
        self.agent._generate = lambda _scratch, **_kwargs: next(outputs)
        trace = Trace()

        answer = self.agent._answer_or_tool([], trace)

        self.assertIn("ループを停止", answer)
        self.assertEqual(len(trace.tool_calls), 1)

    def test_tool_markup_inside_code_example_is_not_executed(self):
        code_example = "例:\n```xml\n<tool>write_file: x.py :: pass</tool>\n```"

        self.assertIsNone(self.agent._extract_tool_call(code_example))
        self.assertEqual(
            self.agent._extract_tool_call("<tool>write_file: x.py :: pass</tool>"),
            ("write_file", "x.py :: pass"),
        )

    def test_tool_budget_is_cumulative_across_answer_retries(self):
        self.agent.config.max_tool_calls = 1
        trace = Trace(tool_calls=[("write_file", "x.py :: pass", "保存しました")])
        self.agent._generate = lambda _scratch, **_kwargs: (
            "<tool>write_file: x.py :: pass</tool>")

        answer = self.agent._answer_or_tool([], trace)

        self.assertIn("上限", answer)
        self.assertEqual(len(trace.tool_calls), 1)
        self.assertFalse(Path(self.temp_dir.name, "x.py").exists())

    def test_rejected_command_is_never_started(self):
        self.coder_tools.interact = lambda *_args, **_kwargs: "rejected"
        with patch("rula.tools.coder.subprocess.run") as run:
            result = self.coder_tools.run_command("echo should-not-run")

        self.assertIn("拒否", result)
        run.assert_not_called()

    def test_invalid_path_character_is_reported_without_writing(self):
        result = self.coder_tools.write_file("bad\0name.txt :: unsafe")

        self.assertIn("無効", result)
        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
