import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

source_root = Path(os.environ.get(
    "RULA_TEST_SOURCE", str(Path(__file__).resolve().parents[1] / "src")
)).resolve()
sys.path.insert(0, str(source_root))

from rula.agent import Agent, AgentConfig, Trace, Turn
from rula.tools.coder import CoderTools


class CaptureEngine:
    display_name = "test"
    supports_thinking = False
    hit_token_limit = False

    def generate(self, _messages, config):
        self.config = config
        return "完了しました。"


class ContextOverflowEngine:
    display_name = "context test"
    supports_thinking = False
    hit_token_limit = False

    def __init__(self, overflow_when=None):
        self.overflow_when = overflow_when
        self.calls = []

    def generate(self, messages, _config):
        self.calls.append(messages)
        if self.overflow_when is None or self.overflow_when(messages):
            raise RuntimeError(
                'llama-server error: {"type":"exceed_context_size_error",'
                '"n_prompt_tokens":16449,"n_ctx":16384}')
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

    def test_auto_mode_never_returns_source_after_failed_file_tool(self):
        self.agent.config.coder_auto = True
        self.agent.config.max_tool_calls = 1
        self.agent._turn_text = "index.htmlでサイトを作って"
        outputs = iter([
            "<tool>write_file: index.html</tool>",
            "<!doctype html><html><body>source only</body></html>",
        ])
        self.agent._generate = lambda _scratch, **_kwargs: next(outputs)

        answer = self.agent._answer_or_tool([], Trace())

        self.assertIn("コードを回答に貼るだけで完了とは扱いませんでした", answer)
        self.assertNotIn("source only", answer)
        self.assertFalse(Path(self.temp_dir.name, "index.html").exists())

    def test_auto_mode_retries_source_answer_as_a_file_tool_call(self):
        self.agent.config.coder_auto = True
        self.agent.config.max_tool_calls = 3
        self.agent._turn_text = "3DのWebサイトを作って"
        outputs = iter([
            "<!doctype html><html><body>source only</body></html>",
            "<tool>write_file: index.html :: <!doctype html><html><body>done</body></html></tool>",
            "index.htmlを作成しました。",
        ])
        prompts = []

        def generate(scratch, **_kwargs):
            prompts.append([dict(message) for message in scratch])
            return next(outputs)

        self.agent._generate = generate
        self.agent._coder_completion_stage = lambda _trace: None
        trace = Trace()

        answer = self.agent._answer_or_tool([], trace)

        self.assertEqual(answer, "index.htmlを作成しました。")
        self.assertEqual(
            Path(self.temp_dir.name, "index.html").read_text(encoding="utf-8"),
            "<!doctype html><html><body>done</body></html>",
        )
        self.assertIn("次の応答は説明文ではなく", prompts[1][-1]["content"])

    def test_failed_mutation_does_not_count_as_a_file_write(self):
        trace = Trace(tool_calls=[
            ("write_file", "index.html", "形式エラー: 相対パス :: ファイル全文")
        ])

        self.assertFalse(self.agent._has_file_write(trace))

    def test_successful_mutation_counts_as_a_file_write(self):
        trace = Trace(tool_calls=[
            ("write_file", "index.html :: <html></html>", "保存しました: index.html")
        ])

        self.assertTrue(self.agent._has_file_write(trace))

    def test_coder_image_description_reaches_the_model_context(self):
        image_ref = "data:image/jpeg;base64,dGVzdA=="
        description = "画面にはcreateEnemy関数の未完成なコードが表示されています。"
        captured = {}
        self.agent.engine = CaptureEngine()
        self.agent.config.use_vision = True
        def read_image(ref, _question=None):
            captured["image_ref"] = ref
            return description

        self.agent.look = read_image

        def answer_from_context(scratch, _trace):
            captured["system"] = self.agent._system_prompt()
            return "画面の内容を確認しました。"

        self.agent._answer_or_tool = answer_from_context

        answer, trace = self.agent.ask(
            "この画面の問題を確認してください。", image_ref=image_ref)

        self.assertEqual(answer, "画面の内容を確認しました。")
        self.assertEqual(trace.image, (image_ref, description))
        self.assertEqual(captured["image_ref"], image_ref)
        self.assertIn(description, captured["system"])

    def test_noop_file_tool_does_not_allow_source_as_auto_mode_answer(self):
        self.agent.config.coder_auto = True
        self.agent.config.max_tool_calls = 1
        self.agent._turn_text = "3DのWebサイトを作って"
        outputs = iter([
            "<tool>write_file: index.html :: <html></html></tool>",
            "<html><body>generated source</body></html>",
        ])
        self.agent._generate = lambda _scratch, **_kwargs: next(outputs)
        trace = Trace()
        self.agent._use_tool = lambda name, arguments, _scratch, trace: trace.tool_calls.append(
            (name, arguments, "変更なし: 指定内容はすでに保存されています。")
        )

        answer = self.agent._answer_or_tool([], trace)

        self.assertIn("ファイル変更を完了できませんでした", answer)
        self.assertNotIn("generated source", answer)
        self.assertFalse(self.agent._has_file_write(trace))

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

    def test_completion_gate_tool_name_is_parseable(self):
        self.assertEqual(
            self.agent._extract_tool_call("<tool>review_changes</tool>"),
            ("review_changes", ""),
        )

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

    def test_context_overflow_retries_without_old_history_and_keeps_current_work(self):
        engine = ContextOverflowEngine(
            lambda messages: any("古い会話" in message["content"] for message in messages)
        )
        self.agent.engine = engine
        self.agent.history.extend([
            Turn("user", "古い会話の依頼"),
            Turn("assistant", "古い会話の回答"),
            Turn("user", "今回の依頼"),
        ])
        progress = []
        self.agent.on_progress = progress.append
        tool_context = [{"role": "user", "content": "現在のツール結果"}]

        answer = self.agent._generate(tool_context)

        self.assertEqual(answer, "完了しました。")
        self.assertEqual(len(engine.calls), 2)
        self.assertTrue(any("古い会話" in message["content"]
                            for message in engine.calls[0]))
        self.assertFalse(any("古い会話" in message["content"]
                             for message in engine.calls[1]))
        self.assertTrue(any("今回の依頼" in message["content"]
                            for message in engine.calls[1]))
        self.assertTrue(any("現在のツール結果" in message["content"]
                            for message in engine.calls[1]))
        self.assertTrue(any("以前の会話を一部省略" in message["content"]
                            for message in engine.calls[1]))
        self.assertEqual(len(self.agent.history), 3)
        self.assertTrue(progress)

    def test_context_overflow_without_droppable_history_has_actionable_error(self):
        self.agent.engine = ContextOverflowEngine()
        self.agent.history.append(Turn("user", "大きすぎる今回の依頼"))

        with self.assertRaisesRegex(RuntimeError, "今回の依頼と必須指示だけで"):
            self.agent._generate([])

        self.assertEqual(len(self.agent.engine.calls), 1)

    def test_compressed_history_archives_full_text_and_recalls_relevant_chunks(self):
        self.agent.engine = CaptureEngine()
        self.agent.config.compress_after = 4
        self.agent.config.keep_recent = 2
        long_old_message = ("古い設計メモです。\n" * 160) + (
            "決定ID NOVA-742: データ保存形式はSQLiteを使用する。")
        self.agent.history.extend([
            Turn("user", long_old_message),
            Turn("assistant", "NOVA-742ではSQLiteを採用します。"),
            Turn("user", "最近の依頼です。"),
            Turn("assistant", "最近の回答です。"),
            Turn("user", "NOVA-742のデータ保存形式は何でしたか？"),
        ])

        self.agent._compress_history()
        self.agent._turn_text = "NOVA-742のデータ保存形式は何でしたか？"
        messages = self.agent._messages()
        latest_user = next(
            message for message in reversed(messages) if message["role"] == "user")

        self.assertEqual(len(self.agent.history), 2)
        self.assertEqual(len(self.agent._archived_turns), 3)
        self.assertEqual(self.agent._archived_turns[0].content, long_old_message)
        self.assertIn("決定ID NOVA-742", latest_user["content"])
        self.assertIn("SQLite", latest_user["content"])
        self.assertIn("最近の依頼です。", latest_user["content"])

    def test_save_and_load_keeps_compressed_archive_verbatim(self):
        self.agent.engine = CaptureEngine()
        self.agent.config.compress_after = 4
        self.agent.config.keep_recent = 2
        original = "長期記憶に残す正確な仕様ID ARCHIVE-90210"
        self.agent.history.extend([
            Turn("user", original),
            Turn("assistant", "仕様IDを記録しました。"),
            Turn("user", "最近の質問"),
            Turn("assistant", "最近の回答"),
        ])
        self.agent._compress_history()

        path = Path(self.temp_dir.name, "conversation.json")
        self.agent.save(str(path))
        restored = Agent(
            engine=CaptureEngine(),
            config=AgentConfig(use_vision=False),
            tools=self.coder_tools.registry(),
        )
        restored.config.history_limit = 2
        restored.load(str(path))
        restored.history.append(Turn("user", "ARCHIVE-90210の仕様を確認して"))
        restored._turn_text = "ARCHIVE-90210の仕様を確認して"
        messages = restored._messages()
        latest_user = next(
            message for message in reversed(messages) if message["role"] == "user")

        self.assertEqual(len(restored.history), 3)
        self.assertEqual(len(restored._archived_turns), 2)
        self.assertIn(original, latest_user["content"])

    def test_native_coder_reasoning_uses_separate_bounded_budget(self):
        engine = CaptureEngine()
        engine.supports_thinking = True
        agent = Agent(
            engine=engine,
            config=AgentConfig(
                thinking=True, think_cap=2000, max_new_tokens=4096,
                use_vision=False,
            ),
        )

        agent._generate([])

        self.assertTrue(engine.config.think)
        self.assertEqual(engine.config.max_new_tokens, 4096)
        self.assertEqual(engine.config.think_cap, 2000)

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

    def test_empty_website_request_rejects_python_only_first_write(self):
        tools = CoderTools(
            self.temp_dir.name, "auto", request_text="Discord向けのsiteを作って")

        rejected = tools.write_file("app.py :: def broken(:\n    pass\n")

        self.assertIn("index.html", rejected)
        self.assertFalse(Path(self.temp_dir.name, "app.py").exists())
        self.assertFalse(tools.changed_files())

        saved = tools.write_file("index.html :: <!doctype html><title>Site</title>")

        self.assertEqual(saved, "保存しました: index.html")
        self.assertTrue(Path(self.temp_dir.name, "index.html").is_file())

    def test_site_request_is_explicit_in_dedicated_coder_prompt(self):
        from rula.coder_workflow import CoderWorkflow

        prompt = CoderWorkflow("auto", "medium").system_prompt(
            "Create a website in this folder")

        self.assertIn("index.html", prompt)
        self.assertIn("app.pyだけ", prompt)

    def test_project_info_includes_bounded_relative_file_structure(self):
        project = Path(self.temp_dir.name)
        (project / "src" / "feature").mkdir(parents=True)
        (project / "tests").mkdir()
        (project / "node_modules").mkdir()
        (project / "src" / "feature" / "service.py").write_text(
            "def run():\n    return True\n", encoding="utf-8")
        (project / "tests" / "test_service.py").write_text(
            "def test_run():\n    pass\n", encoding="utf-8")
        (project / "node_modules" / "ignored.js").write_text(
            "ignored", encoding="utf-8")

        result = self.coder_tools.project_info()

        self.assertIn("--- ファイル構成 ---", result)
        self.assertIn("- src/", result)
        self.assertIn("  - feature/", result)
        self.assertIn("    - service.py", result)
        self.assertIn("- tests/", result)
        self.assertIn("  - test_service.py", result)
        self.assertNotIn("node_modules", result)
        self.assertNotIn("ignored.js", result)

    def test_project_info_recommends_native_checks_for_detected_toolchains(self):
        project = Path(self.temp_dir.name)
        (project / "package.json").write_text(
            '{"scripts":{"test":"vitest run","build":"vite build"}}',
            encoding="utf-8")
        (project / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n",
                                                encoding="utf-8")
        (project / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
            encoding="utf-8")
        (project / "tests").mkdir()
        (project / "tests" / "test_api.py").write_text(
            "def test_api():\n    assert True\n", encoding="utf-8")
        (project / "Cargo.toml").write_text(
            "[package]\nname='demo'\nversion='0.1.0'\n", encoding="utf-8")

        result = self.coder_tools.project_info()

        self.assertIn("pnpm run test", result)
        self.assertIn("pnpm run build", result)
        self.assertIn("python -m pytest", result)
        self.assertIn("cargo test", result)

    def test_find_files_finds_matching_files_without_following_skipped_dirs(self):
        Path(self.temp_dir.name, "src").mkdir()
        Path(self.temp_dir.name, "src", "main.py").write_text(
            "print('ok')", encoding="utf-8")
        hidden = Path(self.temp_dir.name, "node_modules")
        hidden.mkdir()
        Path(hidden, "ignored.py").write_text("ignored", encoding="utf-8")

        result = self.coder_tools.find_files("**/*.py")

        self.assertIn("src/main.py", result)
        self.assertNotIn("ignored.py", result)
        self.assertIn("作業フォルダ内の相対パス", self.coder_tools.find_files("../*.py"))

    def test_read_files_batches_related_files_as_one_auto_inspection(self):
        for name in ("one.py", "two.py"):
            Path(self.temp_dir.name, name).write_text(
                f"print('{name}')", encoding="utf-8")

        result = self.coder_tools.registry().get("read_files").run("one.py\ntwo.py")

        self.assertIn("--- one.py ---", result)
        self.assertIn("print('one.py')", result)
        self.assertIn("--- two.py ---", result)
        self.assertEqual(self.coder_tools._inspection_count, 1)

    def test_find_symbol_and_file_outline_locate_python_definitions(self):
        source = Path(self.temp_dir.name, "module.py")
        source.write_text(
            "class Example:\n"
            "    def method(self):\n"
            "        return 1\n"
            "\n"
            "def helper():\n"
            "    return 2\n",
            encoding="utf-8",
        )

        matches = self.coder_tools.find_symbol("helper")
        outline = self.coder_tools.file_outline("module.py")

        self.assertIn("module.py:5: def helper", matches)
        self.assertIn("module.py:1: class Example", outline)
        self.assertIn("module.py:2: def method", outline)
        self.assertIn("module.py:5: def helper", outline)

    def test_check_python_reports_syntax_without_executing_source(self):
        safe = Path(self.temp_dir.name, "safe.py")
        bad = Path(self.temp_dir.name, "bad.py")
        safe.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
        bad.write_text("def broken(:\n    pass\n", encoding="utf-8")

        result = self.coder_tools.check_python("safe.py\nbad.py")

        self.assertIn("bad.py:1:", result)
        self.assertIn("Python構文エラー", result)
        self.assertTrue(safe.is_file())

    def test_coder_registry_excludes_irrelevant_general_tools(self):
        registry = self.coder_tools.registry()

        self.assertIn("search", registry.names())
        for name in ("find_files", "find_symbol", "read_files", "file_outline",
                     "check_python", "write_file"):
            self.assertIn(name, registry.names())
        for name in ("calc", "date", "smart_search"):
            self.assertNotIn(name, registry.names())

    def test_auto_coder_requires_review_and_post_review_verification(self):
        self.agent.config.coder_auto = True
        self.coder_tools.auto_approved = True
        prompts = []
        outputs = iter([
            "<tool>write_file: app.py :: print('ready')\n</tool>",
            "ファイルを作成しました。",
            "<tool>review_changes</tool>",
            "差分を確認しました。",
            "<tool>run_command: python -m compileall .</tool>",
            "変更を確認し、検証コマンドも成功しました。",
        ])

        def generate(scratch, **_kwargs):
            prompts.append([dict(message) for message in scratch])
            return next(outputs)

        self.agent._generate = generate
        trace = Trace()

        answer = self.agent._answer_or_tool([], trace)

        self.assertEqual(
            [name for name, _, _ in trace.tool_calls],
            ["write_file", "review_changes", "run_command"],
        )
        self.assertTrue(any("review_changes" in message["content"]
                            for messages in prompts for message in messages))
        self.assertTrue(any("run_command" in message["content"]
                            for messages in prompts for message in messages))
        self.assertIn("終了コード: 0", trace.tool_calls[-1][2])
        self.assertEqual(answer, "変更を確認し、検証コマンドも成功しました。")

    def test_auto_coder_does_not_claim_success_when_review_is_skipped(self):
        self.agent.config.coder_auto = True
        outputs = iter([
            "<tool>write_file: app.py :: print('ready')\n</tool>",
            "変更完了です。",
            "レビューは後で行います。完了しました。",
        ])
        self.agent._generate = lambda _scratch, **_kwargs: next(outputs)

        answer = self.agent._answer_or_tool([], Trace())

        self.assertIn("差分レビューを完了できませんでした", answer)
        self.assertNotIn("変更完了です", answer)

    def test_failed_post_review_command_does_not_unlock_completion(self):
        trace = Trace(tool_calls=[
            ("write_file", "app.py :: print('ready')", "保存しました: app.py"),
            ("review_changes", "", "--- a/app.py\n+++ b/app.py"),
            ("run_command", "pytest", "終了コード: 1\nテスト失敗"),
        ])

        self.assertEqual(self.agent._coder_completion_stage(trace), "repair")

        trace.tool_calls.append(
            ("run_command", "pytest", "終了コード: 0\nOK"))

        self.assertIsNone(self.agent._coder_completion_stage(trace))

    def test_arbitrary_successful_command_does_not_unlock_completion(self):
        trace = Trace(tool_calls=[
            ("write_file", "app.py :: print('ready')", "保存しました: app.py"),
            ("review_changes", "", "--- a/app.py\n+++ b/app.py"),
            ("run_command", "echo verified", "終了コード: 0\nverified"),
        ])

        self.assertEqual(self.agent._coder_completion_stage(trace), "verify")

    def test_failed_python_syntax_check_requests_repair_not_generic_verification(self):
        trace = Trace(tool_calls=[
            ("write_file", "app.py :: def broken(:\n    pass", "保存しました: app.py"),
            ("review_changes", "", "--- a/app.py\n+++ b/app.py"),
            ("check_python", "app.py", "Python構文エラー:\napp.py:1:12: invalid syntax"),
        ])

        self.assertEqual(self.agent._coder_completion_stage(trace), "repair")

    def test_auto_coder_repairs_syntax_failure_then_rechecks(self):
        self.agent.config.coder_auto = True
        outputs = iter([
            "<tool>write_file: app.py :: def broken(:\n    pass</tool>",
            "<tool>review_changes</tool>",
            "<tool>check_python: app.py</tool>",
            "<tool>write_file: app.py :: print('fixed')</tool>",
            "<tool>review_changes</tool>",
            "<tool>check_python: app.py</tool>",
            "構文を修正して検査しました。",
        ])
        prompts = []

        def generate(scratch, **_kwargs):
            prompts.append([dict(message) for message in scratch])
            return next(outputs)

        self.agent._generate = generate

        answer = self.agent._answer_or_tool([], Trace())

        self.assertEqual(
            Path(self.temp_dir.name, "app.py").read_text(encoding="utf-8"),
            "print('fixed')",
        )
        self.assertEqual(answer, "構文を修正して検査しました。")
        self.assertTrue(any(
            "Python構文エラー" in message["content"] and "ソースを修正"
            in message["content"]
            for messages in prompts for message in messages
        ))

    def test_test_build_and_static_check_commands_unlock_completion(self):
        for command in (
            "pytest -q",
            "python -m unittest discover -s tests",
            "npm run build",
            "cargo test",
            r"C:\Python\python.exe -m compileall .",
            "uv run pytest",
        ):
            with self.subTest(command=command):
                trace = Trace(tool_calls=[
                    ("write_file", "app.py :: print('ready')", "保存しました: app.py"),
                    ("review_changes", "", "--- a/app.py\n+++ b/app.py"),
                    ("run_command", command, "終了コード: 0\nOK"),
                ])
                self.assertIsNone(self.agent._coder_completion_stage(trace))

    def test_latest_relevant_validation_result_controls_completion(self):
        trace = Trace(tool_calls=[
            ("write_file", "app.py :: print('ready')", "保存しました: app.py"),
            ("review_changes", "", "--- a/app.py\n+++ b/app.py"),
            ("run_command", "pytest -q", "終了コード: 0\nOK"),
            ("run_command", "pytest -q", "終了コード: 1\nFAILED"),
            ("run_command", "echo done", "終了コード: 0\ndone"),
        ])

        self.assertEqual(self.agent._coder_completion_stage(trace), "repair")

    def test_auto_coder_keeps_write_tools_enabled_after_project_info(self):
        self.agent.engine = CaptureEngine()
        self.agent.config.coder_auto = True
        self.agent.history.append(Turn("user", "簡単な企業Webサイトを作って"))
        self.agent._turn_text = "簡単な企業Webサイトを作って"
        self.agent._grounded = True

        messages = self.agent._messages()

        self.assertNotIn("ツールは使わずに答えてください", messages[-1]["content"])
        self.assertIn("追加調査は不要", messages[-1]["content"])
        self.assertIn("write_file / edit_file / apply_patch", messages[-1]["content"])
        self.assertIn("write_file", messages[0]["content"])

    def test_full_auto_turn_writes_file_after_project_info(self):
        self.agent.engine = CaptureEngine()
        self.agent.config.coder_auto = True
        self.coder_tools.auto_approved = True
        self.agent._improve_answer = lambda answer, _issues, _scratch: answer
        generated_messages = []
        outputs = iter([
            "<tool>write_file: index.html :: <!doctype html>\n"
            "<html><body>企業サイト</body></html></tool>",
            "<tool>review_changes</tool>",
            "<tool>run_command: python -m compileall .</tool>",
            "index.html を作成しました。",
        ])

        def generate(scratch, **_kwargs):
            generated_messages.append(self.agent._messages(scratch))
            return next(outputs)

        self.agent._generate = generate
        with patch("rula.agent.plan_tool", return_value=("project_info", "")):
            answer, trace = self.agent.ask("簡単な企業Webサイトを作って")

        self.assertEqual(
            Path(self.temp_dir.name, "index.html").read_text(encoding="utf-8"),
            "<!doctype html>\n<html><body>企業サイト</body></html>",
        )
        self.assertEqual([tool for tool, _, _ in trace.tool_calls],
                         ["project_info", "write_file", "review_changes", "run_command"])
        self.assertEqual(answer, "index.html を作成しました。")
        latest_user_message = next(
            message for message in generated_messages[0]
            if message["role"] == "user")
        self.assertNotIn("ツールは使わずに答えてください",
                         latest_user_message["content"])
        self.assertIn("write_file / edit_file / apply_patch",
                      latest_user_message["content"])

    def test_non_coder_grounded_answer_still_discourages_unneeded_tools(self):
        self.agent.engine = CaptureEngine()
        self.agent.history.append(Turn("user", "調査結果を要約して"))
        self.agent._turn_text = "調査結果を要約して"
        self.agent._grounded = True

        self.assertIn("ツールは使わずに答えてください",
                      self.agent._messages()[-1]["content"])

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

        rejected = self.coder_tools.write_file("crlf.txt :: after\nsecond\n")
        self.assertIn("先に read_file / read_files", rejected)
        self.assertIn(
            "既存の内容を read_file / read_files で全文読み",
            self.agent._tool_result_prompt(rejected, "write_file"),
        )
        self.assertEqual(path.read_bytes(), b"before\r\nsecond\r\n")
        self.coder_tools.read_file("crlf.txt")
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
        self.assertIsNone(self.agent._extract_trailing_tool_call(code_example))
        self.assertEqual(
            self.agent._extract_tool_call("<tool>write_file: x.py :: pass</tool>"),
            ("write_file", "x.py :: pass"),
        )

    def test_auto_mode_executes_trailing_tool_call_after_prose(self):
        self.agent.config.coder_auto = True
        self.coder_tools.auto_approved = True
        self.agent._turn_text = "サイトを作って"
        outputs = iter([
            "ファイルを保存します。\n"
            "<tool>write_file: index.html :: <!doctype html><title>企業</title></tool>",
            "<tool>review_changes</tool>",
            "<tool>run_command: python -m compileall .</tool>",
            "index.html を保存しました。",
        ])
        self.agent._generate = lambda _scratch, **_kwargs: next(outputs)

        answer = self.agent._answer_or_tool([], Trace())

        self.assertEqual(
            Path(self.temp_dir.name, "index.html").read_text(encoding="utf-8"),
            "<!doctype html><title>企業</title>",
        )
        self.assertEqual(answer, "index.html を保存しました。")
        self.assertNotIn("<tool>", answer)

    def test_tool_budget_is_cumulative_across_answer_retries(self):
        self.agent.config.max_tool_calls = 1
        trace = Trace(tool_calls=[("write_file", "x.py :: pass", "保存しました")])
        self.agent._generate = lambda _scratch, **_kwargs: (
            "<tool>write_file: x.py :: pass</tool>")

        answer = self.agent._answer_or_tool([], trace)

        self.assertIn("上限", answer)
        self.assertEqual(len(trace.tool_calls), 1)
        self.assertFalse(Path(self.temp_dir.name, "x.py").exists())

    def test_auto_command_requires_one_time_workspace_approval(self):
        with patch("rula.tools.coder.subprocess.run") as run:
            result = self.coder_tools.run_command("echo should-not-run")

        self.assertIn("開始確認", result)
        run.assert_not_called()

    def test_approved_auto_command_runs_without_an_interaction_prompt(self):
        self.coder_tools.auto_approved = True
        with patch("rula.tools.coder.subprocess.run") as run:
            run.return_value = type("Result", (), {"returncode": 0, "stdout": "ok"})()
            result = self.coder_tools.run_command("echo ok")

        self.assertIn("終了コード: 0", result)
        run.assert_called_once()
        self.assertIsNone(self.coder_tools.interact)

    def test_local_server_command_is_recognized_without_blocking_shell_commands(self):
        from rula.tools.coder import _local_http_port

        self.assertEqual(_local_http_port("python3 -m http.server 8000"), 8000)
        self.assertEqual(_local_http_port("python -m http.server"), 8000)
        self.assertEqual(_local_http_port("uvicorn app:app --port 8123"), 8123)
        self.assertEqual(_local_http_port("flask run"), 5000)
        self.assertIsNone(_local_http_port("python -m pytest"))
        self.assertIsNone(_local_http_port("echo python -m http.server 8000"))
        self.assertIsNone(_local_http_port("python -m http.server 8000 && echo done"))
        self.assertIsNone(_local_http_port("python -m http.server 70000"))

    def test_node_dev_server_port_is_inferred_from_project_script(self):
        from rula.tools.coder import _local_http_port

        Path(self.temp_dir.name, "package.json").write_text(
            '{"scripts":{"dev":"vite --host 127.0.0.1",'
            '"preview":"vite preview"}}',
            encoding="utf-8",
        )

        self.assertEqual(
            _local_http_port("npm run dev", Path(self.temp_dir.name)), 5173)
        self.assertEqual(
            _local_http_port("npm run preview", Path(self.temp_dir.name)), 4173)
        self.assertEqual(
            _local_http_port(
                "npm run dev -- --port 8123", Path(self.temp_dir.name)), 8123)

    def test_local_http_probe_detects_a_real_localhost_response(self):
        from rula.tools.coder import _local_http_responds, _wait_for_local_http

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()

            def log_message(self, _format, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)

        self.assertTrue(_local_http_responds(server.server_address[1]))
        process = type("Process", (), {"poll": lambda _self: None})()
        self.assertTrue(_wait_for_local_http(
            process, server.server_address[1], timeout=1))
        self.assertFalse(_local_http_responds(0))

    def test_approved_local_server_returns_after_localhost_is_ready(self):
        self.coder_tools.auto_approved = True
        process = type("Process", (), {"pid": 4321, "poll": lambda _self: None})()
        with patch("rula.tools.coder.subprocess.Popen", return_value=process) as popen, \
                patch("rula.tools.coder._local_http_responds", return_value=False), \
                patch("rula.tools.coder._wait_for_local_http", return_value=True), \
                patch("rula.tools.coder.subprocess.run") as run:
            result = self.coder_tools.run_command("python3 -m http.server 8000")

        self.assertIn("サーバー起動完了: http://127.0.0.1:8000/", result)
        self.assertIn("プロセスID: 4321", result)
        popen.assert_called_once()
        run.assert_not_called()

    def test_existing_local_server_is_reported_without_duplicate_launch(self):
        self.coder_tools.auto_approved = True
        with patch("rula.tools.coder._local_http_responds", return_value=True), \
                patch("rula.tools.coder.subprocess.Popen") as popen:
            result = self.coder_tools.run_command("python3 -m http.server 8000")

        self.assertIn("既存のローカルサーバー応答を確認", result)
        self.assertIn("127.0.0.1:8000", result)
        popen.assert_not_called()

    def test_local_server_startup_failure_is_reported_and_process_is_stopped(self):
        self.coder_tools.auto_approved = True
        process = type("Process", (), {"pid": 4322, "poll": lambda _self: None})()
        with patch("rula.tools.coder.subprocess.Popen", return_value=process), \
                patch("rula.tools.coder._local_http_responds", return_value=False), \
                patch("rula.tools.coder._wait_for_local_http", return_value=False), \
                patch("rula.tools.coder._stop_process_tree", return_value=True) as stop:
            result = self.coder_tools.run_command("python3 -m http.server 8001")

        self.assertIn("サーバー起動失敗", result)
        self.assertIn("localhost:8001", result)
        stop.assert_called_once_with(process)

    def test_tool_result_callback_receives_completed_operation(self):
        completed = []
        self.agent.on_tool_result = lambda name, arg, result: completed.append(
            (name, arg, result))

        self.agent._use_tool("write_file", "sample.py :: print('ok')", [], Trace())

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0][0:2], ("write_file", "sample.py :: print('ok')"))
        self.assertTrue(completed[0][2].startswith("保存しました:"))

    def test_ready_local_server_satisfies_post_review_verification(self):
        trace = Trace(tool_calls=[
            ("write_file", "index.html :: <h1>Ready</h1>", "保存しました: index.html"),
            ("review_changes", "", "--- a/index.html\n+++ b/index.html"),
            ("run_command", "python3 -m http.server 8000",
             "サーバー起動完了: http://127.0.0.1:8000/\nプロセスID: 4321"),
        ])

        self.assertIsNone(self.agent._coder_completion_stage(trace))

        trace.tool_calls[-1] = (
            "run_command", "python3 -m http.server 8000",
            "既存のローカルサーバー応答を確認: http://127.0.0.1:8000/",
        )
        self.assertIsNone(self.agent._coder_completion_stage(trace))

    def test_auto_coding_request_retries_as_file_tool_after_prose(self):
        self.agent.config.coder_auto = True
        self.coder_tools.auto_approved = True
        outputs = iter([
            "こちらがHTMLです。\n<html><body>会社サイト</body></html>",
            "<tool>write_file: index.html :: <!doctype html>\n<html>done</html></tool>",
            "<tool>review_changes</tool>",
            "<tool>run_command: python -m compileall .</tool>",
            "index.html を作成しました。",
        ])
        progress = []
        self.agent.on_progress = progress.append
        self.agent._generate = lambda _scratch, **_kwargs: next(outputs)
        trace = Trace()
        self.agent._turn_text = "簡単な企業Webサイトを作って。"

        answer = self.agent._answer_or_tool([], trace)

        self.assertEqual(
            Path(self.temp_dir.name, "index.html").read_text(encoding="utf-8"),
            "<!doctype html>\n<html>done</html>",
        )
        self.assertEqual(
            [tool for tool, _, _ in trace.tool_calls],
            ["write_file", "review_changes", "run_command"],
        )
        self.assertEqual(answer, "index.html を作成しました。")
        self.assertTrue(any("実装を要求" in message for message in progress))

    def test_auto_coding_recognizes_copy_request_and_writes_original_site(self):
        self.agent.config.coder_auto = True
        self.coder_tools.auto_approved = True
        request = ("じゃあ、Netflixという有名サービスの、Webサイトをパクって、"
                   "演出から、UIまで、全部パクって。")
        self.assertTrue(self.agent._is_coder_action_request(request))
        self.agent._turn_text = request
        trace = Trace(tool_calls=[("search", "Netflix brand guidelines", "検索結果")])
        outputs = iter([
            "検索結果に基づく構成案とHTMLコードを以下に示します。",
            ("<tool>write_file: index.html :: <!doctype html>"
             "<title>Streamly</title><main>オリジナル動画サービス</main></tool>"),
            "<tool>review_changes</tool>",
            "<tool>run_command: python -m compileall .</tool>",
            "オリジナルの動画サービスサイトを作業フォルダに保存しました。",
        ])
        prompts = []

        def generate(scratch, **_kwargs):
            prompts.append([dict(message) for message in scratch])
            return next(outputs)

        self.agent._generate = generate
        answer = self.agent._answer_or_tool([], trace)

        self.assertEqual(
            Path(self.temp_dir.name, "index.html").read_text(encoding="utf-8"),
            "<!doctype html><title>Streamly</title>"
            "<main>オリジナル動画サービス</main>",
        )
        self.assertEqual([name for name, _, _ in trace.tool_calls],
                         ["search", "write_file", "review_changes", "run_command"])
        self.assertEqual(answer, "オリジナルの動画サービスサイトを作業フォルダに保存しました。")
        self.assertTrue(any("write_file" in message["content"]
                            for messages in prompts for message in messages))

    def test_auto_coding_recovers_after_project_info_and_two_prose_replies(self):
        self.agent.config.coder_auto = True
        self.coder_tools.auto_approved = True
        trace = Trace()
        scratch = []
        self.agent._use_tool("project_info", "", scratch, trace)
        outputs = iter([
            "プロジェクト情報を確認しました。企業サイトを作成します。",
            "以下のHTMLを使えます。\n<html><body>企業サイト</body></html>",
            "<tool>write_file: index.html :: <!doctype html>\n<html><body>会社</body></html></tool>",
            "<tool>review_changes</tool>",
            "<tool>run_command: python -m compileall .</tool>",
            "index.html を保存しました。",
        ])
        prompts = []
        progress = []

        def generate(scratch, **_kwargs):
            prompts.append([dict(message) for message in scratch])
            return next(outputs)

        self.agent._generate = generate
        self.agent.on_progress = progress.append
        self.agent._turn_text = "簡単な企業Webサイト作って"

        answer = self.agent._answer_or_tool(scratch, trace)

        self.assertEqual(
            Path(self.temp_dir.name, "index.html").read_text(encoding="utf-8"),
            "<!doctype html>\n<html><body>会社</body></html>",
        )
        self.assertEqual([tool for tool, _, _ in trace.tool_calls],
                         ["project_info", "write_file", "review_changes", "run_command"])
        self.assertEqual(answer, "index.html を保存しました。")
        self.assertTrue(any(
            "既存プロジェクトではファイル構成と既存コードを基準に" in message["content"]
            for messages in prompts for message in messages
        ))
        self.assertTrue(any("実装を再試行" in message for message in progress))

    def test_auto_coding_request_never_returns_unsaved_source_as_success(self):
        self.agent.config.coder_auto = True
        outputs = iter([
            "```html\n<html>未保存</html>\n```",
            "```html\n<html>まだ未保存</html>\n```",
            "```html\n<html>まだ未保存</html>\n```",
        ])
        self.agent._generate = lambda _scratch, **_kwargs: next(outputs)
        self.agent._turn_text = "企業Webサイトを作って"

        answer = self.agent._answer_or_tool([], Trace())

        self.assertIn("まだ作業フォルダへのファイル変更", answer)
        self.assertNotIn("<html>", answer)

    def test_invalid_path_character_is_reported_without_writing(self):
        result = self.coder_tools.write_file("bad\0name.txt :: unsafe")

        self.assertIn("無効", result)
        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
