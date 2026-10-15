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

from rula.agent import Agent, AgentConfig, Trace, Turn
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
            "<tool>run_command: echo verified</tool>",
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

        self.assertEqual(self.agent._coder_completion_stage(trace), "verify")

        trace.tool_calls.append(
            ("run_command", "pytest", "終了コード: 0\nOK"))

        self.assertIsNone(self.agent._coder_completion_stage(trace))

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
            "<tool>run_command: echo verified</tool>",
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
            "<tool>run_command: echo verified</tool>",
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

    def test_tool_result_callback_receives_completed_operation(self):
        completed = []
        self.agent.on_tool_result = lambda name, arg, result: completed.append(
            (name, arg, result))

        self.agent._use_tool("write_file", "sample.py :: print('ok')", [], Trace())

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0][0:2], ("write_file", "sample.py :: print('ok')"))
        self.assertTrue(completed[0][2].startswith("保存しました:"))

    def test_auto_coding_request_retries_as_file_tool_after_prose(self):
        self.agent.config.coder_auto = True
        self.coder_tools.auto_approved = True
        outputs = iter([
            "こちらがHTMLです。\n<html><body>会社サイト</body></html>",
            "<tool>write_file: index.html :: <!doctype html>\n<html>done</html></tool>",
            "<tool>review_changes</tool>",
            "<tool>run_command: echo verified</tool>",
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
            "<tool>run_command: echo verified</tool>",
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
            "<tool>run_command: echo verified</tool>",
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
            "index.html に完成したページ" in message["content"]
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
