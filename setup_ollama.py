"""Register the Rula model with Ollama (about 2.5GB, once).

Ollama is the fast path for the chat model: it ships llama.cpp builds that pick
their CPU instruction set at runtime, so it works on machines where a pip-installed
llama-cpp-python wheel crashes outright. If Ollama is not installed the API still
runs — it falls back to the transformers backend — just more slowly.

The model registered here renames the upstream weights, pins the sampling values
the source no longer carries, and raises num_ctx; nothing else. See Modelfile.rula.
"""
import json
import os
import shutil
import subprocess
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Q4_K_M, from the upstream author's own repository — the same fine-tune the
# registry build carried, quantised smaller. Measured against the Q6_K it
# replaces: 571MB less VRAM, 19% faster generation, and no quality difference
# found (answer length comparable, no simplified-Chinese leakage in either).
#
# Two things come with the source change, both handled in Modelfile.rula: a bare
# GGUF does not carry the author's sampling values, so they are written out
# there; and the prompt template comes from the GGUF's own Jinja rather than
# Ollama's built-in qwen35 renderer. Reasoning still lands in a separate field
# and tool calls still parse — both verified before the switch.
BASE = "hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q4_K_M"
TARGET = "rula-1.2-4b"
MODELFILE = "Modelfile.rula"
HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not HOST.startswith("http"):
    HOST = f"http://{HOST}"


def installed() -> list[str]:
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=5) as resp:
            return [m["name"] for m in json.loads(resp.read()).get("models", [])]
    except Exception:
        return []


def has(name: str, names: list[str]) -> bool:
    # Ollama reports "model:tag"; a bare name means the :latest tag.
    return any(n == name or n.split(":")[0] == name for n in names)


def run(*args: str) -> int:
    print(f"$ ollama {' '.join(args)}")
    return subprocess.call(["ollama", *args])


def main() -> int:
    if not shutil.which("ollama"):
        print("Ollama が見つかりません。https://ollama.com からインストールしてください。")
        print("入れない場合もAPIは動きます(従来のバックエンドで、やや低速)。")
        return 1

    names = installed()
    if not names and not _server_up():
        print("Ollama は入っていますが、サーバーが動いていません。")
        print("`ollama serve` を実行するか、Ollamaアプリを起動してから再実行してください。")
        return 1

    if has(BASE, names):
        print(f"[{BASE}] 取得済み")
    else:
        print(f"[{BASE}] 取得しています(約2.5GB)")
        if run("pull", BASE) != 0:
            print("取得に失敗しました。")
            return 1

    here = os.path.dirname(os.path.abspath(__file__))
    modelfile = os.path.join(here, MODELFILE)
    if not os.path.isfile(modelfile):
        print(f"{MODELFILE} が見つかりません。")
        return 1

    # Re-created unconditionally: it is a few seconds (the weights are shared with
    # the base, not copied) and it means an edited Modelfile actually takes effect.
    print(f"[{TARGET}] 登録しています")
    if run("create", TARGET, "-f", modelfile) != 0:
        print("登録に失敗しました。")
        return 1

    print(f"\n完了。API は次回起動時に {TARGET} を使います。")
    print("確認: ollama run " + TARGET + '  "あなたの名前は?"')
    return 0


def _server_up() -> bool:
    try:
        urllib.request.urlopen(f"{HOST}/api/tags", timeout=5)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
