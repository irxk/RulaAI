"""Persistent facts about the user.

Asking a 1.5B model to recall things from conversation history is unreliable — it
falls back to a trained "I don't retain information" disclaimer roughly half the
time regardless of prompting. Facts placed directly in the system prompt, by
contrast, are used consistently (the same trick as the current-date note). So we
keep durable facts here and inject them rather than hoping the model looks back.
"""
import json
import os
import re

# Patterns confident enough to capture without asking. Deliberately narrow:
# a false memory is worse than a missing one.
_AUTO_PATTERNS = [
    (re.compile(r"(?:私|わたし|僕|ぼく|俺|おれ)の名前は([^\s、。！!?？]{1,20})(?:です|だ|といいます|と言います)"),
     "ユーザーの名前は{}"),
    (re.compile(r"名前は([^\s、。！!?？]{1,20})(?:です|だよ|だ)"), "ユーザーの名前は{}"),
    (re.compile(r"([^\s、。！!?？]{1,20})(?:と呼んで|って呼んで)(?:ください)?"), "ユーザーの呼び名は{}"),
    (re.compile(r"(?:私|わたし|僕|ぼく|俺|おれ)は([^\s、。！!?？]{1,20})(?:と申します|といいます|と言います)"),
     "ユーザーの名前は{}"),
]

_SUFFIXES = ("さん", "君", "くん", "ちゃん", "様")
_FACT_KEYS = ("ユーザーの名前は", "ユーザーの呼び名は")


class Memory:
    def __init__(self, path: str | None = None):
        self.path = path
        self.facts: list[str] = []
        if path and os.path.exists(path):
            self.load(path)

    def add(self, fact: str) -> bool:
        fact = fact.strip()
        if not fact:
            return False
        key = next((k for k in _FACT_KEYS if fact.startswith(k)), None)
        if key:
            for i, existing in enumerate(self.facts):
                if existing.startswith(key):
                    if existing == fact:
                        return False
                    self.facts[i] = fact
                    self._autosave()
                    return True
        if fact in self.facts:
            return False
        self.facts.append(fact)
        self._autosave()
        return True

    def remove(self, index: int) -> str | None:
        if 0 <= index < len(self.facts):
            fact = self.facts.pop(index)
            self._autosave()
            return fact
        return None

    def clear(self):
        self.facts.clear()
        self._autosave()

    def capture(self, text: str) -> list[str]:
        """Pull obvious self-introductions out of a user message."""
        found = []
        for pattern, template in _AUTO_PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            value = m.group(1).strip()
            for suffix in _SUFFIXES:
                if value.endswith(suffix) and len(value) > len(suffix):
                    value = value[: -len(suffix)]
            if value and self.add(template.format(value)):
                found.append(template.format(value))
        return found

    def as_prompt(self) -> str:
        if not self.facts:
            return ""
        lines = "\n".join(f"- {f}" for f in self.facts)
        return ("\nユーザーについて既に分かっていること(事実として扱う):\n"
                f"{lines}\n"
                "これらを聞かれたら、上の内容をそのまま答えてください。")

    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
        self.facts = [str(f) for f in data if isinstance(f, str)]
        self.path = path

    def save(self, path: str | None = None):
        target = path or self.path
        if not target:
            return
        directory = os.path.dirname(target)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=2)
        self.path = target

    def _autosave(self):
        if self.path:
            self.save()

    def __len__(self):
        return len(self.facts)
