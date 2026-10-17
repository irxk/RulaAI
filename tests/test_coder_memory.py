import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

source_root = Path(os.environ.get(
    "RULA_TEST_SOURCE", str(Path(__file__).resolve().parents[1] / "src")
)).resolve()
sys.path.insert(0, str(source_root))

from rula.coder_memory import ProjectMemory
from rula.tools.coder import CoderTools


class ProjectMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.storage = self.root / "app-data"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_task_notes_resume_only_in_the_same_workspace(self):
        memory = ProjectMemory(self.workspace, self.storage)
        memory.record_task(
            "検索APIを追加して", ["src/api.py"], ["pytest: 終了コード 0"],
            plan=[{"step": "既存APIを確認", "status": "completed"},
                  {"step": "検索処理を実装", "status": "pending"}])
        resumed = ProjectMemory(self.workspace, self.storage).prompt()
        other = self.root / "other"
        other.mkdir()

        self.assertIn("検索APIを追加して", resumed)
        self.assertIn("src/api.py", resumed)
        self.assertIn("終了コード 0", resumed)
        self.assertIn("未完了の計画", resumed)
        self.assertIn("検索処理を実装", resumed)
        self.assertEqual(ProjectMemory(other, self.storage).prompt(), "")

    def test_explicit_notes_are_redacted_and_can_be_forgotten(self):
        tools = CoderTools(
            str(self.workspace), "auto", request_text="次回もこの構成を覚えておいて",
            memory_root=str(self.storage))

        saved = tools.project_memory_tool(
            'remember :: API入口は src/api.py、OPENAI_API_KEY="private value"')
        prompt = tools.project_memory.prompt()
        delete_tools = CoderTools(
            str(self.workspace), "auto", request_text="このプロジェクトの記憶を削除して",
            memory_root=str(self.storage))
        deleted = delete_tools.project_memory_tool("forget")

        self.assertIn("保存しました", saved)
        self.assertIn("src/api.py", prompt)
        self.assertNotIn("private value", prompt)
        self.assertIn("[REDACTED]", prompt)
        self.assertIn("削除しました", deleted)
        self.assertEqual(delete_tools.project_memory.prompt(), "")

    def test_memory_writes_require_an_explicit_user_request(self):
        tools = CoderTools(
            str(self.workspace), "auto", request_text="APIを修正して",
            memory_root=str(self.storage))

        result = tools.project_memory_tool("remember :: 保存しないメモ")

        self.assertIn("保存しませんでした", result)
        self.assertEqual(tools.project_memory.prompt(), "")

    def test_negative_forget_request_does_not_authorize_deletion(self):
        tools = CoderTools(
            str(self.workspace), "auto", request_text="記憶を削除しないで",
            memory_root=str(self.storage))
        tools.project_memory.remember("API入口は src/api.py")

        result = tools.project_memory_tool("forget")

        self.assertIn("削除しませんでした", result)
        self.assertIn("src/api.py", tools.project_memory.prompt())

    def test_corrupt_memory_is_reported_instead_of_silently_replaced(self):
        memory = ProjectMemory(self.workspace, self.storage)
        memory.path.parent.mkdir(parents=True)
        memory.path.write_text("{invalid json", encoding="utf-8")

        with self.assertRaises(json.JSONDecodeError):
            memory.prompt()

    def test_existing_file_must_be_read_before_full_replacement(self):
        target = self.workspace / "app.py"
        target.write_text("original = True\n", encoding="utf-8")
        tools = CoderTools(
            str(self.workspace), "auto", memory_root=str(self.storage))

        self.assertIn(
            "既存ファイルの全文上書きを中止",
            tools.write_file("app.py :: replaced = True\n"))
        self.assertIn("original = True", tools.read_file("app.py :: 1 :: 1"))
        blocked = tools.write_file("app.py :: replaced = True\n")
        unchanged = target.read_text(encoding="utf-8")
        tools.read_file("app.py")
        saved = tools.write_file("app.py :: replaced = True\n")

        self.assertIn("先に read_file / read_files", blocked)
        self.assertEqual(unchanged, "original = True\n")
        self.assertIn("保存しました", saved)
        self.assertEqual(target.read_text(encoding="utf-8"), "replaced = True\n")


if __name__ == "__main__":
    unittest.main()
