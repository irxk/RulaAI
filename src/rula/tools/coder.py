"""Workspace-scoped tools for the desktop coding agent."""
import ast
import http.client
import json
import os
import re
import stat
import subprocess
import tempfile
import time
from difflib import unified_diff
from pathlib import Path, PureWindowsPath
from typing import Callable

from ..coder_workflow import CoderWorkflow
from ..coder_memory import ProjectMemory
from .base import Tool, ToolRegistry

_SKIP_DIRS = {
    ".git", ".next", ".venv", ".tox", ".cache", "venv", "__pycache__", "node_modules",
    "dist", "build", "coverage",
}
_MAX_FILE_BYTES = 100_000
_MAX_PATCH_FILES = 20
_MAX_PATCH_BYTES = 1_000_000
_MAX_OUTPUT_CHARS = 20_000
_MAX_BATCH_READ_FILES = 8
_MAX_PYTHON_CHECK_FILES = 20
_MAX_FIND_FILES = 200
_MEMORY_WRITE_REQUEST = re.compile(
    r"覚えて|記憶(?:して|に残|へ保存)|メモ(?:して|を残|に保存)|"
    r"記録して|次回(?:も|は)?(?:使|覚)|remember this|save (?:this|a note)",
    re.I,
)
_MEMORY_FORGET_REQUEST = re.compile(
    r"記憶(?:を|の)(?:削除|消|忘)"
    r"(?!しない|しません|しなく|したくない|せず|さない|さず|"
    r"れない|れず|れません|ないで)|"
    r"forget (?:the )?(?:project )?memory",
    re.I,
)
_TEXT_SUFFIXES = {
    ".c", ".cc", ".cfg", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html",
    ".ini", ".java", ".js", ".json", ".jsx", ".kt", ".md", ".php", ".ps1",
    ".py", ".rb", ".rs", ".sh", ".sql", ".toml", ".ts", ".tsx", ".txt", ".xml",
    ".yaml", ".yml",
}
_PROJECT_FILES = (
    "CLAUDE.md", "README.md", "pyproject.toml", "package.json",
    "Cargo.toml", "go.mod", "requirements.txt", "composer.json", "pom.xml",
    "build.gradle", "CMakeLists.txt",
)
_MAX_PROJECT_TREE_ENTRIES = 120
_MAX_PROJECT_TREE_CHARS = 6000
_MAX_AUTO_INSPECTIONS = 5
_PLAN_STATUS_RANK = {"pending": 0, "in_progress": 1, "completed": 2}
_TOOL_ARGUMENT_SEPARATOR = re.compile(r"[ \t]*(?:::|：：)[ \t]*")
_HTTP_SERVER_COMMANDS = (
    (re.compile(
        r"^\s*(?:python(?:3(?:\.\d+)?)?|py(?:\s+-\d+(?:\.\d+)?)?)"
        r"\s+-m\s+http\.server(?:\s+(?P<port>\d+))?(?:\s+.*)?$",
        re.I), 8000),
    (re.compile(r"^\s*uvicorn\b.*$", re.I), 8000),
    (re.compile(r"^\s*flask\s+run\b.*$", re.I), 5000),
    (re.compile(r"^\s*npx\s+vite(?:\s+.*)?$", re.I), 5173),
    (re.compile(r"^\s*npx\s+http-server\b.*$", re.I), 8080),
    (re.compile(r"^\s*npx\s+serve\b.*$", re.I), 3000),
)
_SERVER_START_TIMEOUT = 30
_SHELL_CONTROL = re.compile(r"[;&|<>]")


def _local_http_port(command: str, workspace: Path | None = None) -> int | None:
    """Return the expected local HTTP port for known long-running server commands."""
    if _SHELL_CONTROL.search(command):
        return None
    for pattern, default_port in _HTTP_SERVER_COMMANDS:
        match = pattern.fullmatch(command)
        if not match:
            continue
        explicit = re.search(r"(?:--port(?:=|\s+)|-p\s+)(\d{1,5})\b", command)
        port_text = (explicit.group(1) if explicit else
                     match.groupdict().get("port"))
        port = int(port_text) if port_text else default_port
        return port if 1 <= port <= 65535 else None
    node_script = re.fullmatch(
        r"\s*(?:npm|pnpm|yarn|bun)\s+run\s+(?P<script>dev|start|preview)"
        r"(?:\s+--\s+.*)?\s*", command, re.I)
    if node_script and workspace is not None:
        try:
            package = json.loads((workspace / "package.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            package = {}
        script = str((package.get("scripts") or {}).get(
            node_script.group("script"), "")).lower()
        if "vite" in script:
            default_port = 4173 if node_script.group("script").lower() == "preview" else 5173
        elif any(framework in script for framework in ("next", "nuxt", "webpack-dev-server")):
            default_port = 3000
        else:
            return None
        explicit = re.search(r"(?:--port(?:=|\s+)|-p\s+)(\d{1,5})\b", command)
        port = int(explicit.group(1)) if explicit else default_port
        return port if 1 <= port <= 65535 else None
    php_server = re.fullmatch(r"\s*php\s+-S\s+(?:\S+:)?(?P<port>\d{1,5})(?:\s+.*)?",
                              command, re.I)
    if php_server:
        port = int(php_server.group("port"))
        return port if 1 <= port <= 65535 else None
    return None


def _local_http_responds(port: int) -> bool:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        return 100 <= response.status < 600
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _wait_for_local_http(process: subprocess.Popen, port: int,
                         timeout: float = _SERVER_START_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        if _local_http_responds(port):
            return True
        time.sleep(0.2)
    return False


def _stop_process_tree(process: subprocess.Popen) -> bool:
    if process.poll() is not None:
        return True
    tree_stopped = os.name != "nt"
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10,
                check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            tree_stopped = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return False
    return tree_stopped


class CoderTools:
    def __init__(self, workspace: str, mode: str, auto_approved: bool = False,
                 request_text: str = "", memory_root: str | None = None,
                 command_rules: dict[str, str] | None = None):
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("作業フォルダが見つかりません。")
        self.mode = mode
        self.auto_approved = auto_approved
        self.command_rules = dict(command_rules or {})
        self.project_memory = ProjectMemory(
            self.workspace,
            memory_root or os.path.join(
                os.environ.get("APPDATA") or os.path.expanduser("~"),
                "RulaCoder", "project-memory"))
        self._memory_write_allowed = bool(_MEMORY_WRITE_REQUEST.search(request_text))
        self._memory_forget_allowed = bool(_MEMORY_FORGET_REQUEST.search(request_text))
        self.memory_cleared = False
        self._website_request = CoderWorkflow.asks_for_website(request_text)
        self._workspace_started_empty = self._is_workspace_empty()
        self.interact: Callable[..., str] | None = None
        self.on_plan: Callable[[list[dict[str, str]]], None] | None = None
        self._originals: dict[str, str | None] = {}
        self._plan: list[dict[str, str]] = []
        self._inspection_count = 0
        self._explicitly_read: set[Path] = set()

    def _is_workspace_empty(self) -> bool:
        for root, dirs, files in os.walk(self.workspace, followlinks=False):
            dirs[:] = [name for name in dirs if name not in _SKIP_DIRS]
            if any(not (Path(root) / name).is_symlink() for name in files):
                return False
        return True

    def _path(self, relative: str) -> Path:
        if "\0" in relative:
            raise ValueError("ファイルパスに無効な文字が含まれています。")
        candidate = (self.workspace / relative).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError("作業フォルダの外は操作できません。") from exc
        return candidate

    def registry(self) -> ToolRegistry:
        from .search import DEEP_SEARCH_TOOL, READ_URL_TOOL, SEARCH_TOOL

        registry = ToolRegistry()
        for tool in (SEARCH_TOOL, DEEP_SEARCH_TOOL, READ_URL_TOOL):
            registry.register(tool)
        registry.register(Tool(
            "list_files", "作業フォルダ内のファイル一覧を取得する",
            "相対パス (例: . または src)",
            lambda argument: self._inspect(self.list_files, argument)))
        registry.register(Tool(
            "find_files", "名前やglobパターンで作業フォルダ内のファイルを探す",
            "相対globパターン (例: src/**/*.py または **/*test*.js)",
            lambda argument: self._inspect(self.find_files, argument)))
        registry.register(Tool(
            "find_symbol", "関数・クラスなどの定義箇所をソースから探す",
            "シンボル名", lambda argument: self._inspect(self.find_symbol, argument)))
        registry.register(Tool(
            "read_file", "作業フォルダ内のテキストファイルを読む",
            "相対ファイルパス または 相対ファイルパス :: 開始行 :: 終了行",
            lambda argument: self._inspect(self.read_file, argument)))
        registry.register(Tool(
            "read_files", "関連する複数のテキストファイルを一度に読む",
            "作業フォルダ相対パスを1行に1つ (最大8ファイル)",
            lambda argument: self._inspect(self.read_files, argument)))
        registry.register(Tool(
            "file_outline", "Pythonファイルの関数・クラス・メソッドと行番号を一覧表示する",
            "作業フォルダ相対の.pyファイルパス",
            lambda argument: self._inspect(self.file_outline, argument)))
        registry.register(Tool(
            "search_code", "作業フォルダ内のソースコードから文字列を検索する",
            "検索文字列", lambda argument: self._inspect(self.search_code, argument)))
        registry.register(Tool(
            "check_python", "Pythonファイルの構文を実行せずに検査する",
            "作業フォルダ相対の.pyファイルを1行に1つ (最大20ファイル)",
            self.check_python))
        if self.mode in ("generate", "auto"):
            registry.register(Tool(
                "verify_project", "検出したテスト・ビルド・静的検査を実行して結果を集約する",
                "引数なし", self.verify_project))
        registry.register(Tool(
            "project_info", "プロジェクト構成・開発指示・変更状態を確認する",
            "引数なし", self.project_info))
        registry.register(Tool(
            "project_memory", "この作業フォルダの過去の作業記録を読む。ユーザーが明示したメモだけ保存する",
            "read、remember :: 保存するメモ、または forget", self.project_memory_tool))
        registry.register(Tool(
            "update_plan", "現在の作業計画と進捗をユーザーに共有する",
            "引数なし (現在の計画を維持) または JSON配列: "
            "[{\"step\":\"内容\",\"status\":\"pending|in_progress|completed\"}]",
            self.update_plan))
        registry.register(Tool(
            "git_status", "Gitの変更状態を確認する (読み取り専用)",
            "引数なし", lambda argument: self._inspect(self.git_status, argument)))
        registry.register(Tool(
            "git_diff", "Gitの追跡対象ファイルの差分を確認する (読み取り専用)",
            "任意の作業フォルダ相対パス",
            lambda argument: self._inspect(self.git_diff, argument)))
        registry.register(Tool(
            "ask_user", "要件が曖昧で作業を続けられないときにユーザーへ質問する",
            "質問文", self.ask_user))
        if self.mode in ("generate", "auto"):
            registry.register(Tool(
                "write_file", "作業フォルダ内にファイルを新規作成または上書きする",
                "相対パス :: ファイル全文", self.write_file))
            registry.register(Tool(
                "edit_file", "作業フォルダ内のファイルの一意な文字列を置換する",
                "相対パス :: 置換前の文字列 :: 置換後の文字列", self.edit_file))
            registry.register(Tool(
                "apply_patch", "複数ファイルの一意な変更をまとめて適用する",
                "@@ 相対パス の後に --- 旧内容 と +++ 新内容 を書く", self.apply_patch))
            registry.register(Tool(
                "review_changes", "この依頼で実施したファイル変更の差分を確認する",
                "引数なし", self.review_changes))
            registry.register(Tool(
                "run_command", "承認を得て作業フォルダ内でコマンドを実行する",
                "実行するコマンド", self.run_command))
        return registry

    def _inspect(self, operation: Callable[[str], str], argument: str) -> str:
        if self.mode == "auto":
            if self._inspection_count >= _MAX_AUTO_INSPECTIONS:
                return ("調査上限: 必要な範囲の確認は済んでいます。"
                        "追加の読み取りをせず、確認済みの内容から最小の動く実装を"
                        "write_file / edit_file / apply_patch で今すぐ作成してください。")
            self._inspection_count += 1
        return operation(argument)

    def list_files(self, relative: str) -> str:
        try:
            folder = self._path(relative or ".")
        except ValueError as exc:
            return str(exc)
        if not folder.is_dir():
            return f"フォルダが見つかりません: {relative}"
        found: list[str] = []
        for root, dirs, files in os.walk(folder, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
            for name in sorted(files):
                path = Path(root) / name
                if path.is_symlink():
                    continue
                found.append(path.relative_to(self.workspace).as_posix())
                if len(found) >= 200:
                    found.append("… (200件で打ち切り)")
                    return "\n".join(found)
        return "\n".join(found) or "ファイルはありません。"

    def find_files(self, pattern: str) -> str:
        pattern = pattern.strip().replace("\\", "/")
        if not pattern:
            return "検索パターンが空です。"
        if (Path(pattern).is_absolute() or PureWindowsPath(pattern).is_absolute()
                or any(part == ".." for part in pattern.split("/"))):
            return "検索パターンには作業フォルダ内の相対パスを指定してください。"
        found: list[str] = []
        visited = 0
        for root, dirs, files in os.walk(self.workspace, followlinks=False):
            dirs[:] = sorted(
                name for name in dirs
                if name not in _SKIP_DIRS and not (Path(root) / name).is_symlink())
            for name in sorted(files):
                path = Path(root) / name
                if path.is_symlink():
                    continue
                visited += 1
                relative = path.relative_to(self.workspace).as_posix()
                if Path(relative).match(pattern):
                    found.append(relative)
                    if len(found) >= _MAX_FIND_FILES:
                        return "\n".join(found) + f"\n… ({_MAX_FIND_FILES}件で打ち切り)"
                if visited >= 20_000:
                        truncated = "\n… (20,000ファイルを調査)"
                        return ("\n".join(found) + truncated if found
                                else "該当ファイルはありません。" + truncated)
        return "\n".join(found) or "該当ファイルはありません。"

    def read_file(self, relative: str) -> str:
        parts = relative.rsplit(" :: ", 2)
        path_arg = parts[0]
        start_line, end_line = 1, None
        if len(parts) == 3:
            try:
                start_line, end_line = int(parts[1]), int(parts[2])
            except ValueError:
                return "行番号は整数で指定してください。"
            if start_line < 1 or end_line < start_line:
                return "行番号の範囲が正しくありません。"
        try:
            path = self._path(path_arg.strip())
        except ValueError as exc:
            return str(exc)
        if not path.is_file():
            return f"ファイルが見つかりません: {path_arg}"
        if path.stat().st_size > _MAX_FILE_BYTES:
            return f"ファイルが大きすぎるため読めません (上限 {_MAX_FILE_BYTES} bytes): {path_arg}"
        try:
            content = self._read_text(path)
        except (OSError, UnicodeError) as exc:
            return f"ファイルを読めません ({path_arg}): {exc}"
        if end_line is None:
            self._explicitly_read.add(path)
        if end_line is None:
            return content
        lines = content.splitlines()
        if start_line > len(lines):
            return f"開始行がファイルの範囲外です (全 {len(lines)} 行)。"
        return "\n".join(f"{i}: {line}" for i, line in enumerate(
            lines[start_line - 1:end_line], start_line))

    def project_memory_tool(self, argument: str) -> str:
        action, separator, note = argument.partition(" :: ")
        if action.strip().lower() == "read":
            return self.project_memory.prompt() or "この作業フォルダの記憶はまだありません。"
        if action.strip().lower() == "forget" and not separator:
            if not self._memory_forget_allowed:
                return "削除しませんでした。ユーザーが今回の依頼で記憶の削除を明示した場合だけ実行できます。"
            self.project_memory.clear()
            self.memory_cleared = True
            return "この作業フォルダの記憶を削除しました。"
        if action.strip().lower() != "remember" or not separator:
            return "形式エラー: read または remember :: 保存するメモ を指定してください。"
        if not self._memory_write_allowed:
            return "保存しませんでした。ユーザーが今回の依頼で記憶を明示的に求めた場合だけ保存できます。"
        return self.project_memory.remember(note)

    def find_symbol(self, symbol: str) -> str:
        symbol = symbol.strip()
        if not symbol or len(symbol) > 120 or not re.fullmatch(
                r"[A-Za-z_$][A-Za-z0-9_$]*", symbol):
            return "シンボル名は英数字・アンダースコアで指定してください。"
        escaped = re.escape(symbol)
        end = rf"{escaped}(?![\w$])"
        patterns = {
            ".py": re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{end}"),
            ".js": re.compile(
                rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
                rf"(?:function|class|const|let|var)\s+{end}"),
            ".jsx": re.compile(
                rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
                rf"(?:function|class|const|let|var)\s+{end}"),
            ".ts": re.compile(
                rf"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?"
                rf"(?:async\s+)?(?:function|class|interface|type|enum|const|let|var)"
                rf"\s+{end}"),
            ".tsx": re.compile(
                rf"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?"
                rf"(?:async\s+)?(?:function|class|interface|type|enum|const|let|var)"
                rf"\s+{end}"),
            ".go": re.compile(rf"^\s*func\s+(?:\([^)]*\)\s*)?{end}"),
            ".rs": re.compile(
                rf"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
                rf"(?:fn|struct|enum|trait|type|const|static)\s+{end}"),
        }
        matches: list[str] = []
        visited = 0
        scanned = 0
        for root, dirs, files in os.walk(self.workspace, followlinks=False):
            dirs[:] = sorted(
                name for name in dirs
                if name not in _SKIP_DIRS and not (Path(root) / name).is_symlink())
            for name in sorted(files):
                path = Path(root) / name
                if path.is_symlink():
                    continue
                scanned += 1
                if scanned > 20_000:
                    break
                pattern = patterns.get(path.suffix.lower())
                if pattern is None:
                    continue
                try:
                    if path.stat().st_size > _MAX_FILE_BYTES:
                        continue
                    with path.open(encoding="utf-8") as source:
                        for number, line in enumerate(source, 1):
                            if pattern.search(line):
                                relative = path.relative_to(self.workspace).as_posix()
                                matches.append(
                                    f"{relative}:{number}: {line.strip()[:240]}")
                                if len(matches) >= 100:
                                    return "\n".join(matches) + "\n… (100件で打ち切り)"
                except (OSError, UnicodeError):
                    continue
                visited += 1
                if visited >= 3000 or scanned >= 20_000:
                    break
            if visited >= 3000 or scanned >= 20_000:
                break
        return "\n".join(matches) or "定義が見つかりません。"

    def file_outline(self, relative: str) -> str:
        try:
            path = self._path(relative.strip())
        except ValueError as exc:
            return str(exc)
        if path.suffix.lower() != ".py":
            return "file_outline はPythonファイル (.py) に対応しています。"
        if not path.is_file():
            return f"ファイルが見つかりません: {relative}"
        if path.stat().st_size > _MAX_FILE_BYTES:
            return f"ファイルが大きすぎるため解析できません: {relative}"
        try:
            module = ast.parse(self._read_text(path), filename=relative)
        except SyntaxError as exc:
            return f"Python構文エラー ({relative}:{exc.lineno or 1}): {exc.msg}"
        except (OSError, UnicodeError) as exc:
            return f"ファイルを解析できません ({relative}): {exc}"
        symbols = [
            (node.lineno, node.name, "class" if isinstance(node, ast.ClassDef) else "def")
            for node in ast.walk(module)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        symbols.sort(key=lambda item: (item[0], item[1]))
        if not symbols:
            return "クラス・関数の定義はありません。"
        return "\n".join(
            f"{relative}:{line}: {kind} {name}"
            for line, name, kind in symbols[:300]
        )

    def read_files(self, paths: str) -> str:
        raw_paths = paths.strip()
        if not raw_paths:
            return "ファイルパスが空です。"
        if raw_paths.startswith("["):
            try:
                requested = json.loads(raw_paths)
            except json.JSONDecodeError as exc:
                return f"ファイル一覧のJSONが不正です: {exc.msg}"
            if not isinstance(requested, list) or not all(
                    isinstance(path, str) and path.strip() for path in requested):
                return "JSON配列には相対ファイルパス文字列を指定してください。"
        else:
            requested = [path.strip() for path in raw_paths.splitlines() if path.strip()]
        if len(requested) > _MAX_BATCH_READ_FILES:
            return f"一度に読めるファイルは最大 {_MAX_BATCH_READ_FILES} 件です。"
        if len(set(requested)) != len(requested):
            return "ファイル一覧に同じパスが重複しています。"

        sections: list[str] = []
        remaining = _MAX_OUTPUT_CHARS
        for relative in requested:
            result = self.read_file(relative)
            section = f"--- {relative} ---\n{result}"
            if len(section) > remaining:
                sections.append(section[:remaining] + "\n… (一括読み込みの出力上限)")
                break
            sections.append(section)
            remaining -= len(section)
            if remaining <= 0:
                break
        return "\n\n".join(sections) or "ファイルパスが空です。"

    def check_python(self, paths: str) -> str:
        requested = [path.strip() for path in paths.splitlines() if path.strip()]
        if not requested:
            return "検査するPythonファイルを指定してください。"
        if len(requested) > _MAX_PYTHON_CHECK_FILES:
            return f"一度に検査できるPythonファイルは最大 {_MAX_PYTHON_CHECK_FILES} 件です。"
        errors: list[str] = []
        checked = 0
        for relative in requested:
            if Path(relative).suffix.lower() != ".py":
                return f"Pythonファイル (.py) ではありません: {relative}"
            try:
                path = self._path(relative)
            except ValueError as exc:
                return str(exc)
            if not path.is_file():
                return f"ファイルが見つかりません: {relative}"
            if path.stat().st_size > _MAX_FILE_BYTES:
                return f"ファイルが大きすぎるため検査できません: {relative}"
            try:
                source = self._read_text(path)
                ast.parse(source, filename=relative)
            except SyntaxError as exc:
                errors.append(
                    f"{relative}:{exc.lineno or 1}:{exc.offset or 1}: {exc.msg}")
            except (OSError, UnicodeError) as exc:
                return f"ファイルを検査できません ({relative}): {exc}"
            checked += 1
        if errors:
            return "Python構文エラー:\n" + "\n".join(errors[:100])
        return f"Python構文OK: {checked}ファイル (コードは実行していません)"

    def search_code(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "検索文字列が空です。"
        matches: list[str] = []
        visited = 0
        for root, dirs, files in os.walk(self.workspace, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
            for name in sorted(files):
                path = Path(root) / name
                if path.is_symlink():
                    continue
                if path.suffix.lower() not in _TEXT_SUFFIXES:
                    continue
                try:
                    if path.stat().st_size > _MAX_FILE_BYTES:
                        continue
                    lines = path.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeError):
                    continue
                visited += 1
                for number, line in enumerate(lines, 1):
                    if query.casefold() in line.casefold():
                        relative = path.relative_to(self.workspace).as_posix()
                        matches.append(f"{relative}:{number}: {line.strip()[:240]}")
                        if len(matches) >= 100:
                            return "\n".join(matches) + "\n… (100件で打ち切り)"
                if visited >= 3000:
                    matches.append("… (3,000ファイルを調査)")
                    return "\n".join(matches) if len(matches) > 1 else "該当箇所はありません。"
        return "\n".join(matches) or "該当箇所はありません。"

    def project_info(self, argument: str = "") -> str:
        return self._inspect(self._project_info, argument)

    def _project_tree(self) -> str:
        entries: list[str] = []
        total_chars = 0
        truncated = False
        for root, dirs, files in os.walk(self.workspace, followlinks=False):
            dirs[:] = sorted(
                name for name in dirs
                if name not in _SKIP_DIRS and not (Path(root) / name).is_symlink())
            relative_root = Path(root).relative_to(self.workspace)
            depth = 0 if relative_root == Path(".") else len(relative_root.parts)
            children = [(name, True) for name in dirs]
            children.extend((name, False) for name in sorted(files))
            for name, is_dir in children:
                if not is_dir and (Path(root) / name).is_symlink():
                    continue
                entry = f"{'  ' * depth}- {name}{'/' if is_dir else ''}"
                if (len(entries) >= _MAX_PROJECT_TREE_ENTRIES
                        or total_chars + len(entry) + 1 > _MAX_PROJECT_TREE_CHARS):
                    truncated = True
                    break
                entries.append(entry)
                total_chars += len(entry) + 1
            if truncated:
                break
        if not entries:
            return "（ファイル構成は空です）"
        if truncated:
            entries.append("… (構成表示の上限)")
        return "\n".join(entries)

    def _project_checks(self) -> list[str]:
        files: set[str] = set()
        basenames: set[str] = set()
        for root, dirs, names in os.walk(self.workspace, followlinks=False):
            dirs[:] = [name for name in dirs
                       if name not in _SKIP_DIRS and not (Path(root) / name).is_symlink()]
            for name in names:
                path = Path(root) / name
                if path.is_symlink():
                    continue
                relative = path.relative_to(self.workspace).as_posix()
                files.add(relative)
                basenames.add(name.casefold())

        checks: list[str] = []
        package_files = [name for name in files
                         if Path(name).name.casefold() == "package.json"]
        for package_file in sorted(package_files)[:4]:
            try:
                package = json.loads(self._read_text(self._path(package_file)))
            except (OSError, UnicodeError, ValueError):
                continue
            scripts = package.get("scripts", {}) if isinstance(package, dict) else {}
            if not isinstance(scripts, dict):
                continue
            directory = Path(package_file).parent.as_posix()
            prefix = "" if directory == "." else f"cd {directory} && "
            lock_names = {
                (Path(package_file).parent / name).as_posix()
                for name in ("pnpm-lock.yaml", "yarn.lock", "bun.lockb", "bun.lock",
                             "package-lock.json")
            }
            manager = next((Path(name).name for name in sorted(lock_names)
                            if name in files), "package-lock.json")
            runner = ("pnpm" if manager == "pnpm-lock.yaml" else
                      "yarn" if manager == "yarn.lock" else
                      "bun" if manager.startswith("bun.lock") else "npm")
            for script in ("test", "build", "lint", "typecheck", "check"):
                if isinstance(scripts.get(script), str):
                    checks.append(f"{prefix}{runner} run {script}")
        if "cargo.toml" in basenames:
            checks.append("cargo test")
        if "go.mod" in basenames:
            checks.append("go test ./...")
        python_files = [name for name in files if name.endswith(".py")]
        if "pyproject.toml" in basenames or python_files:
            pytest_configured = "pytest.ini" in basenames or "tox.ini" in basenames
            for name in files:
                if Path(name).name != "pyproject.toml":
                    continue
                try:
                    if self._path(name).stat().st_size <= _MAX_FILE_BYTES:
                        pytest_configured |= "pytest" in self._read_text(
                            self._path(name)).casefold()
                except (OSError, UnicodeError):
                    continue
            if pytest_configured:
                checks.append("python -m pytest")
            elif any(
                    Path(name).name.startswith("test_") and name.endswith(".py")
                    for name in files):
                checks.append("python -m unittest discover")
            elif python_files:
                checks.append("Python変更はcheck_pythonで構文検査")
        dotnet_projects = sorted(
            name for name in files
            if name.casefold().endswith((".sln", ".slnx", ".csproj", ".fsproj", ".vbproj")))
        if dotnet_projects:
            checks.append(f'dotnet test "{dotnet_projects[0]}"')
        if "pom.xml" in basenames:
            checks.append("mvn test")
        if "gradlew" in basenames:
            checks.append(".\\gradlew test")
        elif "build.gradle" in basenames or "build.gradle.kts" in basenames:
            checks.append("gradle test")
        makefiles = [name for name in files if Path(name).name.casefold() == "makefile"]
        for makefile in makefiles[:4]:
            try:
                contents = self._read_text(self._path(makefile))
            except (OSError, UnicodeError):
                continue
            if re.search(r"(?m)^test(?::|\s*::)", contents):
                directory = Path(makefile).parent.as_posix()
                prefix = "" if directory == "." else f"cd {directory} && "
                checks.append(prefix + "make test")
        return list(dict.fromkeys(checks))[:12]

    def verify_project(self, argument: str = "") -> str:
        if argument.strip():
            return "このツールに引数は不要です。"
        checks = self._project_checks()
        python_syntax = any(path.endswith(".py") for path in self.changed_files())
        if python_syntax:
            python_files = [
                path for path in self.changed_files() if path.endswith(".py")
            ][:_MAX_PYTHON_CHECK_FILES]
            syntax_result = self.check_python("\n".join(python_files))
            if not syntax_result.startswith("Python構文OK:"):
                return "検証失敗: Python構文検査\n" + syntax_result
        else:
            syntax_result = ""
        python_only = any("check_pythonで構文検査" in check for check in checks)
        commands = [check for check in checks if "check_pythonで構文検査" not in check]
        results: list[tuple[str, str, str]] = (
            [("Python構文検査", syntax_result, "ok")] if syntax_result else [])
        if python_only and not commands:
            python_files: list[str] = []
            for root, dirs, files in os.walk(self.workspace, followlinks=False):
                dirs[:] = sorted(
                    name for name in dirs
                    if name not in _SKIP_DIRS
                    and not (Path(root) / name).is_symlink())
                for name in sorted(files):
                    path = Path(root) / name
                    if name.endswith(".py") and not path.is_symlink():
                        python_files.append(path.relative_to(self.workspace).as_posix())
                        if len(python_files) >= _MAX_PYTHON_CHECK_FILES:
                            break
                if len(python_files) >= _MAX_PYTHON_CHECK_FILES:
                    break
            if python_files:
                result = self.check_python("\n".join(python_files))
                results.append(("Python構文検査", result, "ok" if
                                result.startswith("Python構文OK:") else "failed"))
        if not commands:
            if results and all(status == "ok" for _, _, status in results):
                return "検証完了: Python構文検査\n" + "\n".join(
                    f"- {command}: {result}" for command, result, _ in results)
            return ("検証未完了: プロジェクト設定から実行可能なテストや検査を検出できませんでした。"
                    "関連テストや適切な静的検査を確認し、実行する場合はrun_commandを使ってください。")

        limit = 8
        skipped = max(0, len(commands) - limit)
        for command in commands[:limit]:
            result = self.run_command(command)
            results.append((command, result, "ok" if (
                re.search(r"(?m)^終了コード: 0\s*$", result)
                or result.startswith((
                    "サーバー起動完了: http://127.0.0.1:",
                    "既存のローカルサーバー応答を確認: http://127.0.0.1:",
                ))) else "blocked" if result.startswith((
                    "ユーザーがコマンドを拒否しました。",
                    "この作業フォルダの許可設定により、コマンドを禁止しました。",
                    "コマンドを実行できません:",
                )) else "failed"))
        failed = sum(status == "failed" for _, _, status in results)
        blocked = sum(status == "blocked" for _, _, status in results)
        outcome = "検証完了" if not failed and not blocked and not skipped else (
            "検証失敗" if failed else "検証未完了")
        details = [
            f"- {command}: {result[:3000]}"
            for command, result, _ in results
        ]
        if skipped:
            details.append(f"- 上限により未実行: 残り{skipped}件")
        if blocked:
            details.append(f"- 承認・禁止ルールにより実行されなかった検査: {blocked}件")
        return f"{outcome}: {len(results)}件実行\n" + "\n".join(details)

    def _project_info(self, argument: str = "") -> str:
        if argument.strip():
            return "このツールに引数は不要です。"
        tree_section = f"\n--- ファイル構成 ---\n{self._project_tree()}"
        checks = self._project_checks()
        checks_section = (
            "\n--- 検証候補 (変更に関連するものだけ実行) ---\n"
            + ("\n".join(f"- {command}" for command in checks)
               if checks else "- 固有のテストコマンドは検出できません。関連テストまたは対象言語の静的検査を確認してください。")
        )
        parts = [f"作業フォルダ: {self.workspace}", tree_section, checks_section]
        remaining = max(0, 12_000 - sum(map(len, parts)))
        project_files = list(_PROJECT_FILES)
        for root, dirs, files in os.walk(self.workspace, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
            if "AGENTS.md" in files:
                instruction_file = Path(root) / "AGENTS.md"
                if instruction_file.is_symlink():
                    continue
                relative = instruction_file.relative_to(self.workspace).as_posix()
                project_files.append(relative)
                if len(project_files) >= len(_PROJECT_FILES) + 20:
                    break
        for name in project_files:
            try:
                path = self._path(name)
            except ValueError:
                continue
            if path.is_file() and path.stat().st_size <= _MAX_FILE_BYTES:
                try:
                    text = self._read_text(path).strip()
                except (OSError, UnicodeError):
                    continue
                section = f"\n--- {name} ---\n{text[:min(4000, remaining)]}"
                parts.append(section)
                remaining -= len(section)
                if remaining <= 0:
                    parts.append("\n… (プロジェクト情報の上限)")
                    break
        return "\n".join(parts)

    def update_plan(self, argument: str) -> str:
        if not argument.strip():
            if self._plan:
                return "現在の作業計画を維持しました。次の作業を続けてください。"
            return ("計画がありません。JSON配列で手順を指定してください。"
                    "例: [{\"step\":\"関連コードを調査する\",\"status\":\"pending\"}]")
        try:
            data = json.loads(argument)
        except json.JSONDecodeError as exc:
            return f"計画のJSONが不正です: {exc.msg}"
        if not isinstance(data, list) or not 1 <= len(data) <= 12:
            return "計画は1〜12項目のJSON配列で指定してください。"
        tasks: list[dict[str, str]] = []
        seen_steps: set[str] = set()
        for item in data:
            if not isinstance(item, dict):
                return "各項目は step と status を持つJSONオブジェクトにしてください。"
            step = item.get("step")
            status = item.get("status", "pending")
            if not isinstance(step, str) or not step.strip() or len(step) > 240:
                return "各項目の step は1〜240文字の文字列にしてください。"
            if status not in ("pending", "in_progress", "completed"):
                return "status は pending / in_progress / completed のいずれかです。"
            normalized = re.sub(r"\s+", " ", step).strip().casefold()
            if normalized in seen_steps:
                return "計画に同じ手順が重複しています。手順を一度だけ指定してください。"
            seen_steps.add(normalized)
            tasks.append({"step": step.strip(), "status": status})

        if self._plan:
            incoming_by_step = {
                re.sub(r"\s+", " ", task["step"]).strip().casefold(): task
                for task in tasks
            }
            next_plan = [task.copy() for task in self._plan]
            changed = False
            for task in next_plan:
                key = re.sub(r"\s+", " ", task["step"]).strip().casefold()
                incoming = incoming_by_step.get(key)
                if incoming is None:
                    continue
                old_status = task["status"]
                new_status = incoming["status"]
                if _PLAN_STATUS_RANK[new_status] > _PLAN_STATUS_RANK[old_status]:
                    task["status"] = new_status
                    changed = True
            if not changed:
                return ("既存の作業計画を維持しました。手順の作り直しや追加は不要です。"
                        "現在の計画に沿ってファイルを実装してください。")
            tasks = next_plan

        self._plan = tasks
        if self.on_plan is not None:
            self.on_plan([task.copy() for task in tasks])
        return ("作業計画を共有しました。計画の再作成は不要です。"
                "現在の手順の実作業を続け、状態が実際に変わったときだけ更新してください。")

    def git_status(self, argument: str = "") -> str:
        if argument.strip():
            return "このツールに引数は不要です。"
        return self._git_readonly(["status", "--short", "--branch"])

    def git_diff(self, relative: str = "") -> str:
        args = ["diff", "--no-ext-diff", "--unified=3", "--"]
        if relative.strip():
            try:
                path = self._path(relative.strip())
            except ValueError as exc:
                return str(exc)
            args.append(path.relative_to(self.workspace).as_posix())
        return self._git_readonly(args)

    def _git_readonly(self, args: list[str]) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(self.workspace), *args],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace",
                timeout=15, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except FileNotFoundError:
            return "Gitが見つかりません。"
        except subprocess.TimeoutExpired:
            return "Gitの読み取り処理が15秒でタイムアウトしました。"
        if result.returncode:
            return f"Gitを実行できません (終了コード {result.returncode}):\n{result.stdout[-2000:]}"
        return result.stdout[-_MAX_OUTPUT_CHARS:] or "(差分なし)"

    def ask_user(self, question: str) -> str:
        if self.interact is None:
            return "ユーザーに質問できません。追加の質問が必要な場合は、回答として質問を提示してください。"
        return self.interact("question", message=question)

    def write_file(self, argument: str) -> str:
        parts = _TOOL_ARGUMENT_SEPARATOR.split(argument, maxsplit=1)
        if len(parts) != 2:
            return "形式エラー: 相対パス :: ファイル全文 の形式で指定してください。"
        relative, content = parts
        try:
            content_size = len(content.encode("utf-8"))
        except UnicodeError as exc:
            return f"ファイルにUTF-8として保存できない文字が含まれています: {exc}"
        if content_size > _MAX_FILE_BYTES:
            return f"新しい内容が大きすぎるため保存を中止しました (上限 {_MAX_FILE_BYTES} bytes)。"
        try:
            path = self._write_path(relative.strip())
        except ValueError as exc:
            return str(exc)
        return self._write_files([(path, content)])

    def edit_file(self, argument: str) -> str:
        parts = _TOOL_ARGUMENT_SEPARATOR.split(argument, maxsplit=2)
        if len(parts) != 3:
            return "形式エラー: 相対パス :: 置換前 :: 置換後 の形式で指定してください。"
        relative, old, new = parts
        try:
            path = self._path(relative.strip())
        except ValueError as exc:
            return str(exc)
        if not path.is_file():
            return f"ファイルが見つかりません: {relative}"
        if path.stat().st_size > _MAX_FILE_BYTES:
            return f"ファイルが大きすぎるため編集を中止しました: {relative}"
        try:
            original = self._read_text(path)
        except (OSError, UnicodeError) as exc:
            return f"ファイルを読めません ({relative}): {exc}"
        self._explicitly_read.add(path)
        original, old, new = self._match_newlines(original, old, new)
        occurrences = original.count(old)
        if occurrences != 1:
            return f"置換を中止しました: 対象文字列は {occurrences} 件です (一意な場合のみ変更します)。"
        return self.write_file(f"{relative.strip()} :: {original.replace(old, new, 1)}")

    def apply_patch(self, patch: str) -> str:
        """Apply a sequence of exact file replacements encoded as patch blocks."""
        blocks = re.split(r"(?m)^@@ ([^\r\n]+)\r?\n", patch)
        if len(blocks) < 3 or len(blocks) % 2 == 0:
            return "パッチ形式エラー: @@ 相対パス の行でファイルごとに区切ってください。"
        if (len(blocks) - 1) // 2 > _MAX_PATCH_FILES:
            return f"パッチは最大 {_MAX_PATCH_FILES} ファイルまでです。"
        try:
            patch_size = len(patch.encode("utf-8"))
        except UnicodeError as exc:
            return f"パッチにUTF-8として保存できない文字が含まれています: {exc}"
        if patch_size > _MAX_PATCH_BYTES:
            return f"パッチが大きすぎるため適用を中止しました (上限 {_MAX_PATCH_BYTES} bytes)。"
        changes: list[tuple[Path, str]] = []
        seen_paths: set[Path] = set()
        for index in range(1, len(blocks), 2):
            try:
                path = self._write_path(blocks[index].strip())
            except ValueError as exc:
                return str(exc)
            if path in seen_paths:
                return f"パッチ形式エラー: ファイルを重複指定しています: {blocks[index]}"
            seen_paths.add(path)
            body = blocks[index + 1].replace("\r\n", "\n").strip("\n")
            old, separator, new = body.partition("\n+++ ")
            if not separator or not old.startswith("--- "):
                return f"パッチ形式エラー ({blocks[index]}): --- 旧内容 と +++ 新内容 が必要です。"
            old = old[4:]
            if not path.is_file():
                return f"ファイルが見つかりません: {blocks[index]}"
            if path.stat().st_size > _MAX_FILE_BYTES:
                return f"ファイルが大きすぎるため編集を中止しました: {blocks[index]}"
            try:
                original = self._read_text(path)
            except (OSError, UnicodeError) as exc:
                return f"ファイルを読めません ({blocks[index]}): {exc}"
            self._explicitly_read.add(path)
            original, old, new = self._match_newlines(original, old, new)
            if original.count(old) != 1:
                return (f"適用を中止しました ({blocks[index]}): 旧内容は"
                        f" {original.count(old)} 件一致しました。一意な場合のみ変更します。")
            changes.append((path, original.replace(old, new, 1)))
        result = self._write_files(changes)
        if result.startswith("変更なし:") or not result.startswith("保存しました:"):
            return result
        return "変更しました: " + ", ".join(
            path.relative_to(self.workspace).as_posix() for path, _ in changes)

    def _read_text(self, path: Path) -> str:
        return path.read_bytes().decode("utf-8")

    @staticmethod
    def _match_newlines(original: str, old: str, new: str) -> tuple[str, str, str]:
        if original.count(old):
            return original, old, new
        normalized_original = original.replace("\r\n", "\n").replace("\r", "\n")
        normalized_old = old.replace("\r\n", "\n").replace("\r", "\n")
        normalized_new = new.replace("\r\n", "\n").replace("\r", "\n")
        return normalized_original, normalized_old, normalized_new

    def _write_path(self, relative: str) -> Path:
        requested = Path(relative)
        if "\0" in relative:
            raise ValueError("ファイルパスに無効な文字が含まれています。")
        if not relative or requested.is_absolute() or requested.drive:
            raise ValueError("作業フォルダ内の相対ファイルパスを指定してください。")
        lexical = Path(os.path.abspath(self.workspace / requested))
        try:
            parts = lexical.relative_to(self.workspace).parts
        except ValueError as exc:
            raise ValueError("作業フォルダの外は操作できません。") from exc
        current = self.workspace
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("シンボリックリンク経由の書き込みはできません。")
        return self._path(relative)

    def _write_files(self, changes: list[tuple[Path, str]]) -> str:
        if (self._website_request and self._workspace_started_empty
                and not self._originals
                and not any(path.name.casefold() == "index.html"
                            for path, _ in changes)):
            return ("サイト依頼の最初の変更には、ブラウザーで開ける index.html が必要です。"
                    "app.pyだけではWebサイトになりません。index.htmlを入口としてHTML/CSS/"
                    "JavaScriptで作成してください。まだファイルは変更していません。")
        prepared: list[tuple[Path, str, bytes, bytes | None]] = []
        originals: dict[str, str | None] = {}
        total_bytes = 0
        for path, content in changes:
            if path.exists() and not path.is_file():
                return f"ファイルではなくフォルダです: {path.relative_to(self.workspace).as_posix()}"
            if path.is_file() and path not in self._explicitly_read:
                return ("既存ファイルの全文上書きを中止しました。先に read_file / read_files で"
                        f"内容を確認してください: {path.relative_to(self.workspace).as_posix()}")
            try:
                original_bytes = path.read_bytes() if path.exists() else None
                original = original_bytes.decode("utf-8") if original_bytes is not None else None
            except (OSError, UnicodeError) as exc:
                return f"既存ファイルを安全に読めないため保存を中止しました ({path}): {exc}"
            if original_bytes is not None and len(original_bytes) > _MAX_FILE_BYTES:
                return f"既存ファイルが大きすぎるため上書きを中止しました: {path.name}"
            if original is not None:
                content = self._preserve_newlines(content, original)
            try:
                encoded = content.encode("utf-8")
            except UnicodeError as exc:
                return f"ファイルにUTF-8として保存できない文字が含まれています: {exc}"
            if len(encoded) > _MAX_FILE_BYTES:
                return f"新しい内容が大きすぎるため保存を中止しました (上限 {_MAX_FILE_BYTES} bytes)。"
            total_bytes += len(encoded)
            originals[path.relative_to(self.workspace).as_posix()] = original
            if original_bytes != encoded:
                prepared.append((path, content, encoded, original_bytes))
        if total_bytes > _MAX_PATCH_BYTES:
            return f"変更内容が大きすぎるため保存を中止しました (上限 {_MAX_PATCH_BYTES} bytes)。"
        if not prepared:
            return "変更なし: 指定内容はすでに保存されています。"

        staged: list[tuple[Path, Path, Path | None]] = []
        temporary_files: list[Path] = []
        replaced: list[tuple[Path, Path | None]] = []
        try:
            for path, _, encoded, original_bytes in prepared:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                        "wb", dir=path.parent, prefix=f".{path.name}.",
                        suffix=".tmp", delete=False) as stream:
                    new_temp = Path(stream.name)
                    temporary_files.append(new_temp)
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                if original_bytes is not None and os.name != "nt":
                    os.chmod(new_temp, stat.S_IMODE(path.stat().st_mode))
                old_temp = None
                if original_bytes is not None:
                    with tempfile.NamedTemporaryFile(
                            "wb", dir=path.parent, prefix=f".{path.name}.",
                            suffix=".bak", delete=False) as stream:
                        old_temp = Path(stream.name)
                        temporary_files.append(old_temp)
                        stream.write(original_bytes)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if os.name != "nt":
                        os.chmod(old_temp, stat.S_IMODE(path.stat().st_mode))
                staged.append((path, new_temp, old_temp))
            for path, new_temp, old_temp in staged:
                os.replace(new_temp, path)
                replaced.append((path, old_temp))
        except OSError as exc:
            rollback_errors: list[str] = []
            for path, old_temp in reversed(replaced):
                try:
                    if old_temp is None:
                        path.unlink(missing_ok=True)
                    else:
                        os.replace(old_temp, path)
                except OSError as rollback_exc:
                    rollback_errors.append(f"{path.name}: {rollback_exc}")
            message = f"保存に失敗しました。変更をロールバックしました: {exc}"
            if rollback_errors:
                message = ("保存に失敗し、一部のロールバックも失敗しました: "
                           + "; ".join(rollback_errors))
            return message
        finally:
            for temporary in temporary_files:
                temporary.unlink(missing_ok=True)

        for relative, original in originals.items():
            self._originals.setdefault(relative, original)
        self._explicitly_read.update(path for path, _, _, _ in prepared)
        return "保存しました: " + ", ".join(
            path.relative_to(self.workspace).as_posix() for path, _, _, _ in prepared)

    @staticmethod
    def _preserve_newlines(content: str, original: str) -> str:
        crlf = original.count("\r\n")
        lf = original.count("\n") - crlf
        if crlf <= lf:
            return content
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        return normalized.replace("\n", "\r\n")

    def review_changes(self, argument: str = "") -> str:
        if argument.strip():
            return "このツールに引数は不要です。"
        if not self._originals:
            return "この依頼ではまだファイルを変更していません。"
        diffs: list[str] = []
        remaining = _MAX_OUTPUT_CHARS
        for relative, original in self._originals.items():
            current = self._read_text(self._path(relative))
            before = original.splitlines(keepends=True) if original is not None else []
            after = current.splitlines(keepends=True)
            diff = "".join(unified_diff(
                before, after, fromfile=f"a/{relative}" if original is not None else "/dev/null",
                tofile=f"b/{relative}", lineterm="\n"))
            if diff:
                diffs.append(diff[:remaining])
                remaining -= len(diffs[-1])
                if remaining <= 0:
                    diffs.append("\n… (差分の上限)")
                    break
        return "".join(diffs) or "この依頼ではまだファイルを変更していません。"

    def changed_files(self) -> list[str]:
        """Return files this turn actually changed through the file tools."""
        changed = []
        for relative, original in self._originals.items():
            path = self._path(relative)
            if not path.exists():
                if original is not None:
                    changed.append(relative)
                continue
            if self._read_text(path) != original:
                changed.append(relative)
        return sorted(changed)

    def run_command(self, command: str) -> str:
        if not command.strip():
            return "コマンドが空です。"
        if len(command) > 4000:
            return "コマンドが長すぎるため実行を中止しました (上限 4,000文字)。"
        if self.mode not in ("generate", "auto"):
            return "このCoderモードではターミナルを利用できません。"
        if self.mode == "auto" and not self.auto_approved:
            return "オートモードの開始確認がないため、コマンドを実行しませんでした。"
        action = self.command_rules.get(command.strip(), "ask")
        if action == "deny":
            return "この作業フォルダの許可設定により、コマンドを禁止しました。"
        if action != "allow":
            if self.interact is None:
                return "コマンドを実行できません: ユーザー承認が利用できません。"
            approved = self.interact("command", command=command)
            if approved != "approved":
                return "ユーザーがコマンドを拒否しました。変更を加えずに続行してください。"
        port = _local_http_port(command, self.workspace)
        if port is not None:
            return self._start_local_server(command, port)
        try:
            result = subprocess.run(
                command, cwd=self.workspace, shell=True, text=True,
                encoding="utf-8", errors="replace",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
                check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            return f"コマンドは120秒でタイムアウトしました。\n{output[-_MAX_OUTPUT_CHARS:]}"
        except OSError as exc:
            return f"コマンドを起動できませんでした: {exc}"
        output = result.stdout[-_MAX_OUTPUT_CHARS:]
        return f"終了コード: {result.returncode}\n{output or '(出力なし)'}"

    def _start_local_server(self, command: str, port: int) -> str:
        """Run a known dev server without waiting for its long-lived process to exit."""
        if _local_http_responds(port):
            return (f"既存のローカルサーバー応答を確認: http://127.0.0.1:{port}/\n"
                    "HTTP応答済みのため、同じポートでの二重起動は行っていません。")
        try:
            with tempfile.TemporaryFile(mode="w+b") as log_file:
                process = subprocess.Popen(
                    command, cwd=self.workspace, shell=True,
                    stdout=log_file, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                ready = _wait_for_local_http(process, port)
                log_file.flush()
                log_file.seek(0)
                output = log_file.read()[-2000:].decode("utf-8", "replace").strip()
                if ready:
                    result = (f"サーバー起動完了: http://127.0.0.1:{port}/\n"
                              f"プロセスID: {process.pid}")
                    return result

                return_code = process.poll()
                if return_code is None:
                    stopped = _stop_process_tree(process)
                    reason = f"{_SERVER_START_TIMEOUT}秒以内にlocalhost:{port}の応答がありません"
                    if not stopped:
                        reason += "。起動プロセスの停止を確認できませんでした"
                else:
                    reason = f"起動コマンドが終了しました (終了コード: {return_code})"
                return (f"サーバー起動失敗: {reason}。\n"
                        f"コマンド: {command}"
                        + (f"\n起動ログ:\n{output}" if output else ""))
        except OSError as exc:
            return f"サーバーを起動できませんでした: {exc}"
