import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

source_root = Path(os.environ.get(
    "RULA_TEST_SOURCE", str(Path(__file__).resolve().parents[1] / "src")
)).resolve()
sys.path.insert(0, str(source_root))

from rula.coder_planner import create_coding_plan, format_coding_plan


class CoderPlannerTests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "task_type": "feature",
            "target_files": ["src/main.py"],
            "requirements": ["Implement the feature"],
            "context_summary": "Add the requested capability.",
            "validation": ["Run relevant tests"],
            "blocking_question": None,
        }

    def test_plan_request_uses_json_schema_and_returns_validated_context(self):
        engine = Mock()
        engine.generate.return_value = json.dumps(self.plan)

        result = create_coding_plan(engine, "Add a feature", "previous note")

        self.assertEqual(result, self.plan)
        messages, config = engine.generate.call_args.args
        self.assertEqual(messages[1]["content"].splitlines()[1], "Add a feature")
        self.assertIn("previous note", messages[1]["content"])
        self.assertEqual(config.response_format["type"], "json_schema")
        self.assertIn("target_files", format_coding_plan(result))

    def test_plan_rejects_paths_outside_the_workspace(self):
        engine = Mock()
        plan = {**self.plan, "target_files": ["..\\outside.py"]}
        engine.generate.return_value = json.dumps(plan)

        with self.assertRaisesRegex(ValueError, "作業フォルダ外"):
            create_coding_plan(engine, "Change the app")

    def test_plan_rejects_invalid_json(self):
        engine = Mock()
        engine.generate.return_value = "not json"

        with self.assertRaisesRegex(ValueError, "有効なJSON"):
            create_coding_plan(engine, "Change the app")


if __name__ == "__main__":
    unittest.main()
