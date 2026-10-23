"""Workspace-scoped exact-command permissions for RulaCoder."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

_ACTIONS = {"ask", "allow", "deny"}
_MAX_RULES = 100
_MAX_COMMAND_LENGTH = 4000


class CommandPermissionStore:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)

    def _path(self, workspace: str | os.PathLike[str]) -> tuple[Path, str]:
        resolved = str(Path(workspace).resolve(strict=True))
        key = os.path.normcase(resolved)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json", resolved

    def get(self, workspace: str | os.PathLike[str]) -> list[dict[str, str]]:
        path, resolved = self._path(workspace)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"コマンド許可設定を読み込めません: {exc}") from exc
        if (not isinstance(data, dict)
                or not isinstance(data.get("workspace"), str)
                or os.path.normcase(data["workspace"]) != os.path.normcase(resolved)
                or not isinstance(data.get("rules"), list)):
            raise ValueError("コマンド許可設定の形式が不正です。設定を修正または削除してください。")
        return self.validate(data["rules"])

    def set(self, workspace: str | os.PathLike[str],
            rules: list[dict[str, str]]) -> list[dict[str, str]]:
        path, resolved = self._path(workspace)
        validated = self.validate(rules)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=self.root, suffix=".tmp",
                    delete=False) as stream:
                temporary = stream.name
                json.dump({"workspace": resolved, "rules": validated}, stream,
                          ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except OSError as exc:
            raise ValueError(f"コマンド許可設定を保存できません: {exc}") from exc
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        return validated

    @staticmethod
    def validate(rules: list[dict[str, str]]) -> list[dict[str, str]]:
        if len(rules) > _MAX_RULES:
            raise ValueError(f"コマンド許可ルールは最大 {_MAX_RULES} 件です。")
        validated: list[dict[str, str]] = []
        seen: set[str] = set()
        for rule in rules:
            if not isinstance(rule, dict):
                raise ValueError("コマンド許可ルールの形式が不正です。")
            raw_command = rule.get("command")
            if not isinstance(raw_command, str):
                raise ValueError("コマンドは文字列で指定してください。")
            command = raw_command.strip()
            action = rule.get("action", "ask")
            if not command or len(command) > _MAX_COMMAND_LENGTH:
                raise ValueError("コマンドは1〜4,000文字で指定してください。")
            if not isinstance(action, str) or action not in _ACTIONS:
                raise ValueError("許可設定は ask・allow・deny のいずれかを指定してください。")
            if command in seen:
                raise ValueError(f"同じコマンドが重複しています: {command[:120]}")
            seen.add(command)
            validated.append({"command": command, "action": action})
        return validated
