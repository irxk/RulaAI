import os
import sys
import unittest
from pathlib import Path

source_root = Path(os.environ.get(
    "RULA_TEST_SOURCE", str(Path(__file__).resolve().parents[1] / "src")
)).resolve()
sys.path.insert(0, str(source_root))

from rula.persona import build_system_prompt
from rula.planner import needs_structured_reasoning
from rula.agent import Agent, AgentConfig


class ThinkingCaptureEngine:
    display_name = "test"
    supports_thinking = True
    hit_token_limit = False

    def __init__(self):
        self.thinking_flags = []

    def generate(self, _messages, config):
        self.thinking_flags.append(config.think)
        return "結論は条件を満たします。"


class CognitivePipelinePromptTests(unittest.TestCase):
    def test_default_prompt_includes_intent_evidence_and_final_consistency_checks(self):
        prompt = build_system_prompt()

        self.assertIn("依頼の種類・明示された条件・必要な成果", prompt)
        self.assertIn("検索結果は出典や確かさを見て扱い", prompt)
        self.assertIn("前提・処理・結果が一致するか", prompt)
        self.assertIn("回答内の矛盾を点検", prompt)
        self.assertIn("簡体字が混ざっていないか", prompt)

    def test_language_and_code_are_not_blindly_rewritten(self):
        prompt = build_system_prompt()

        self.assertIn("ユーザーが英語で書いたとき", prompt)
        self.assertIn("必要な固有名詞・引用・コードは変えない", prompt)

    def test_symbolic_router_enables_only_explicit_multi_condition_reasoning(self):
        self.assertTrue(needs_structured_reasoning(
            "このパズルを場合分けして、条件をすべて確認して"))
        self.assertFalse(needs_structured_reasoning(
            "Pythonで最短手順を返す関数を作って"))
        self.assertFalse(needs_structured_reasoning(
            "この条件を場合分けして調べて"))
        self.assertFalse(needs_structured_reasoning("こんにちは"))

    def test_agent_only_auto_thinks_for_explicit_complex_reasoning(self):
        engine = ThinkingCaptureEngine()
        agent = Agent(
            engine=engine,
            config=AgentConfig(
                auto_think=True, use_tools=False, use_vision=False,
                max_new_tokens=256,
            ),
        )

        agent.ask("このパズルを場合分けして、条件をすべて確認して")
        self.assertTrue(engine.thinking_flags[0])

        engine.thinking_flags.clear()
        agent.ask("こんにちは")
        self.assertFalse(engine.thinking_flags[0])


if __name__ == "__main__":
    unittest.main()
