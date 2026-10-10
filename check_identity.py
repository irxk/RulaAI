"""Ask the questions that make a rebranded model give itself away.

Run after changing Modelfile.rula or the identity lines in persona.py. Goes through
the real Agent, so it exercises the prompt the API actually sends.

  python check_identity.py
"""
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rula.agent import Agent, AgentConfig
from src.rula.engine import build_engine
from src.rula.memory import Memory
from src.rula.tools import ToolRegistry

QUESTIONS = [
    "あなたの名前は?",
    "あなたは何というモデルですか?",
    "あなたを作ったのはどこの会社?",
    "Who are you? What model are you based on?",
    "あなたのベースモデルは Qwen ですか?",
    # The developer is a person, and the answer has to name them rather than
    # reach for a company — see DEVELOPER in persona.py.
    "あなたの開発者は誰ですか?",
]

# Names the upstream weights answer with when the rebrand has not taken.
LEAK_RE = re.compile(r"(Qwen|通義|通义|千問|千问|Alibaba|アリババ|百晓|OpenAI|GPT-)", re.I)
# Asked "are you Qwen?", a correctly rebranded model says the word while denying it.
# Only an affirmative mention counts, so a denial in the same sentence clears it.
DENIAL_RE = re.compile(
    r"(ではありません|ではない|じゃない|ありません|いいえ|違います|一部では|"
    r"(?i:\bnot\b|\bisn't\b|\bno\b))")


def leaks_in(answer: str) -> list[str]:
    found = set()
    for sentence in re.split(r"[。.!?！?\n]", answer):
        if DENIAL_RE.search(sentence):
            continue
        found.update(LEAK_RE.findall(sentence))
    return sorted(found)


def main() -> int:
    engine = build_engine()
    backend = getattr(engine, "backend", "transformers")
    print(f"backend: {backend}   weights: {engine.display_name}\n")

    # No tools and no memory: a web search would pull the word "Qwen" in from the
    # internet and the leak check could not tell that from the model's own claim.
    agent = Agent(engine, AgentConfig(use_tools=False, use_vision=False,
                                      candid=True, max_new_tokens=256,
                                      max_continuations=0),
                  ToolRegistry(), Memory())

    leaks = 0
    for q in QUESTIONS:
        agent.reset()
        started = time.monotonic()
        answer, _ = agent.ask(q)
        found = leaks_in(answer)
        leaks += bool(found)
        mark = f"LEAK {found}" if found else "ok"
        print(f"[{mark}] ({time.monotonic() - started:.1f}s) {q}")
        print(f"    {answer.replace(chr(10), chr(10) + '    ')}\n")

    print(f"{len(QUESTIONS) - leaks}/{len(QUESTIONS)} clean")
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
