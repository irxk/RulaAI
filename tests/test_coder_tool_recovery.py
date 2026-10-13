import os
import sys
import tempfile
import unittest
from pathlib import Path

source_root = Path(os.environ.get(
    "RULA_TEST_SOURCE", str(Path(__file__).resolve().parents[1] / "src")
)).resolve()
sys.path.insert(0, str(source_root))

from rula.agent import Agent, AgentConfig, Trace
from rula.tools.coder import CoderTools


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


if __name__ == "__main__":
    unittest.main()
