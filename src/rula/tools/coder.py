"""Workspace-scoped tools for the desktop coding agent."""
import json
import os
import re
import stat
import subprocess
import tempfile
from difflib import unified_diff
from pathlib import Path
from typing import Callable

from .base import Tool, ToolRegistry

_SKIP_DIRS = {
    ".git", ".next", ".venv", ".tox", ".cache", "venv", "__pycache__", "node_modules",
    "dist", "build", "coverage",
}
_MAX_FILE_BYTES = 100_000
_MAX_PATCH_FILES = 20
_MAX_PATCH_BYTES = 1_000_000
_MAX_OUTPUT_CHARS = 20_000
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
_MAX_AUTO_INSPECTIONS = 5
_PLAN_STATUS_RANK = {"pending": 0, "in_progress": 1, "completed": 2}


class CoderTools:
    def __init__(self, workspace: str, mode: str, auto_approved: bool = False):
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("作業フォルダが見つかりません。")
        self.mode = mode
        self.auto_approved = auto_approved
        self.interact: Callable[..., str] | None = None
        self.on_plan: Callable[[list[dict[str, str]]], None] | None = None
        self._originals: dict[str, str | None] = {}
        self._plan: list[dict[str, str]] = []
        self._inspection_count = 0

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
        from . import default_registry

        registry = default_registry()
        registry.register(Tool(
            "list_files", "作業フォルダ内のファイル一覧を取得する",
            "相対パス (例: . または src)",
            lambda argument: self._inspect(self.list_files, argument)))
        registry.register(Tool(
            "read_file", "作業フォルダ内のテキストファイルを読む",
            "相対ファイルパス または 相対ファイルパス :: 開始行 :: 終了行",
            lambda argument: self._inspect(self.read_file, argument)))
        registry.register(Tool(
            "search_code", "作業フォルダ内のソースコードから文字列を検索する",
            "検索文字列", lambda argument: self._inspect(self.search_code, argument)))
        registry.register(Tool(
            "project_info", "プロジェクト構成・開発指示・変更状態を確認する",
            "引数なし", self.project_info))
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
        if self.mode == "auto":
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
            return content
        lines = content.splitlines()
        if start_line > len(lines):
            return f"開始行がファイルの範囲外です (全 {len(lines)} 行)。"
        return "\n".join(f"{i}: {line}" for i, line in enumerate(
            lines[start_line - 1:end_line], start_line))

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

    def _project_info(self, argument: str = "") -> str:
        if argument.strip():
            return "このツールに引数は不要です。"
        parts = [f"作業フォルダ: {self.workspace}"]
        remaining = 12_000
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
        relative, separator, content = argument.partition(" :: ")
        if not separator:
            return "形式エラー: 相対パス :: ファイル全文 の形式で指定してください。"
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
        parts = argument.split(" :: ", 2)
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
        prepared: list[tuple[Path, str, bytes, bytes | None]] = []
        originals: dict[str, str | None] = {}
        total_bytes = 0
        for path, content in changes:
            if path.exists() and not path.is_file():
                return f"ファイルではなくフォルダです: {path.relative_to(self.workspace).as_posix()}"
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
        if self.mode == "auto" and not self.auto_approved:
            return "オートモードの開始確認がないため、コマンドを実行しませんでした。"
        if self.mode != "auto":
            if self.interact is None:
                return "コマンドを実行できません: ユーザー承認が利用できません。"
            approved = self.interact("command", command=command)
            if approved != "approved":
                return "ユーザーがコマンドを拒否しました。変更を加えずに続行してください。"
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
