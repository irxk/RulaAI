"""Tool protocol shared by everything Rula can call."""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    usage: str
    run: Callable[[str], str]

    def spec(self) -> str:
        return f"- {self.name}: {self.description}\n  使い方: {self.usage}"


# The model is told to write "search: ...", but it often translates the tool name
# and writes "検索: ..." instead. Accept both rather than losing the call.
ALIASES = {
    "検索": "search", "web検索": "search", "ウェブ検索": "search", "調べる": "search",
    "詳しく調べる": "deep_search", "詳細検索": "deep_search", "深く検索": "deep_search",
    "比較検索": "smart_search", "多角検索": "smart_search", "比較": "smart_search",
    "計算": "calc", "calculate": "calc", "計算する": "calc",
    "日付": "date", "日付計算": "date",
    "read": "read_url", "url": "read_url", "ページ取得": "read_url", "読む": "read_url",
}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        key = name.strip().lower()
        return self._tools.get(ALIASES.get(key, key))

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> str:
        return "\n".join(t.spec() for t in self._tools.values())

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self):
        return len(self._tools)
