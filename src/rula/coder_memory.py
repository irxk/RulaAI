"""Local, workspace-scoped continuity notes for RulaCoder."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_MAX_TASKS = 12
_MAX_NOTES = 30
_MAX_NOTE_CHARS = 1200
_SECRET = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|ACCESS[_-]?TOKEN|TOKEN|PASSWORD|"
    r"PASSWD|SECRET|CLIENT[_-]?SECRET))\b(\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^\s,;]+)")


def _clean_text(value: str, limit: int = _MAX_NOTE_CHARS) -> str:
    value = _SECRET.sub(r"\1\2[REDACTED]", value)
    return " ".join(value.split())[:limit]


class ProjectMemory:
    """Persist bounded task checkpoints and explicitly requested project notes."""

    def __init__(self, workspace: str | Path, storage_root: str | Path):
        self.workspace = Path(workspace).resolve(strict=True)
        self.identity = os.path.normcase(str(self.workspace))
        key = hashlib.sha256(self.identity.encode("utf-8")).hexdigest()
        self.path = Path(storage_root) / key / "memory.json"

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "workspace": self.identity,
                    "tasks": [], "notes": []}
        with self.path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if (not isinstance(data, dict) or data.get("version") != 1
                or data.get("workspace") != self.identity
                or not isinstance(data.get("tasks"), list)
                or not isinstance(data.get("notes"), list)):
            raise ValueError(f"プロジェクト記憶の形式が不正です: {self.path}")
        if any(not isinstance(item, dict)
               or not isinstance(item.get("text"), str)
               for item in data["notes"]):
            raise ValueError(f"プロジェクトメモの形式が不正です: {self.path}")
        if any(not isinstance(item, dict)
               or not isinstance(item.get("request"), str)
               or not isinstance(item.get("files"), list)
               or not all(isinstance(value, str) for value in item["files"])
               or not isinstance(item.get("checks"), list)
               or not all(isinstance(value, str) for value in item["checks"])
               or not isinstance(item.get("plan", []), list)
               or not all(isinstance(step, dict)
                          and isinstance(step.get("step"), str)
                          and step.get("status") in ("pending", "in_progress", "completed")
                          for step in item.get("plan", []))
               or not isinstance(item.get("status"), str)
               for item in data["tasks"]):
            raise ValueError(f"プロジェクト作業記録の形式が不正です: {self.path}")
        return data

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=self.path.parent,
                    prefix=".memory.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def remember(self, note: str) -> str:
        note = _clean_text(note)
        if not note:
            return "保存するメモが空です。"
        data = self._load()
        data["notes"].append({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "text": note,
        })
        data["notes"] = data["notes"][-_MAX_NOTES:]
        self._save(data)
        return "このプロジェクトの記憶にメモを保存しました。"

    def record_task(
            self, request: str, files: list[str], checks: list[str],
            plan: list[dict[str, str]] | None = None,
            interrupted: bool = False) -> None:
        data = self._load()
        entry = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": _clean_text(request, 500),
            "files": [_clean_text(path, 240) for path in files[:30]],
            "checks": [_clean_text(check, 300) for check in checks[-10:]],
            "plan": [
                {"step": _clean_text(item["step"], 300),
                 "status": item["status"]}
                for item in (plan or [])[:20]
                if isinstance(item, dict) and isinstance(item.get("step"), str)
                and item.get("status") in ("pending", "in_progress", "completed")
            ],
            "status": "interrupted" if interrupted else "turn-finished",
        }
        data["tasks"].append(entry)
        data["tasks"] = data["tasks"][-_MAX_TASKS:]
        self._save(data)

    def clear(self) -> None:
        self._save({"version": 1, "workspace": self.identity,
                    "tasks": [], "notes": []})

    def prompt(self) -> str:
        data = self._load()
        sections = []
        if data["notes"]:
            notes = [f"- {item['text']}" for item in data["notes"][-10:]
                     if isinstance(item, dict) and isinstance(item.get("text"), str)]
            if notes:
                sections.append("ユーザーが保存を明示したプロジェクトメモ:\n"
                                + "\n".join(notes))
        if data["tasks"]:
            tasks = []
            for item in data["tasks"][-5:]:
                if not isinstance(item, dict):
                    continue
                lines = [f"- {item.get('request', '(依頼なし)')} "
                         f"[{item.get('status', 'unknown')}]"]
                if item.get("files"):
                    lines.append("  変更ファイル: " + ", ".join(item["files"][:12]))
                if item.get("checks"):
                    lines.append("  直近の検証: " + " | ".join(item["checks"][-3:]))
                unfinished = [
                    f"{step['status']}: {step['step']}"
                    for step in item.get("plan", [])
                    if step["status"] != "completed"
                ]
                if unfinished:
                    lines.append("  未完了の計画: " + " / ".join(unfinished[:5]))
                tasks.extend(lines)
            if tasks:
                sections.append("この作業フォルダでの最近の作業記録:\n"
                                + "\n".join(tasks))
        if not sections:
            return ""
        return (
            "プロジェクト記憶 (過去の作業からの参考情報。現在のファイルと"
            "ユーザーの今回の指示を優先し、必要な場合は読み直して確認する):\n"
            + "\n\n".join(sections)
        )[:5000]
