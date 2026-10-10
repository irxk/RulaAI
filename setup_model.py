"""Fetch the models the API needs (about 6.7GB in total, once).

  chat   — 会話モデル (約2.5GB)
  vision — 画像読み取りモデル (約4.2GB)

Both are registered as pointers into the shared Hugging Face cache rather than
copied here, so a machine that already has them downloads nothing.
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODELS = {
    "chat": {
        "repo": "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
        "out": os.path.join("models", "Rula1.1-4b"),
        "name": "Rula1.1-4b",
        "role": "日本語での対話とWeb検索",
        "size": "約2.5GB",
        "prequantized": True,
    },
    "vision": {
        "repo": "Qwen/Qwen2-VL-2B-Instruct",
        "out": os.path.join("models", "Rula1.0-vision-2b"),
        "name": "Rula1.0-vision-2b",
        "role": "画像の内容説明と、画像内の文字の読み取り(OCR)",
        "size": "約4.2GB",
        "prequantized": False,
    },
}

NOTICE = """\
{name}
{underline}

日本語アシスタント Rula が使うモデルです。

用途: {role}

由来と権利表示
--------------
重みは {repo} (Alibaba Cloud) に由来します。
上流モデルは Apache License 2.0 で公開されており、再配布・改変・名称変更が
許諾される一方、ライセンス本文と帰属表示の保持が求められます。

- 上流モデル: {repo}
- ライセンス: Apache License 2.0

重み自体は上流から変更していません。
"""


def fetch(kind: str, spec: dict):
    from huggingface_hub import snapshot_download

    existing = _already_local(spec["name"])
    if existing:
        # Weights already sitting elsewhere on this machine — point at them rather
        # than downloading several GB of an identical copy.
        _register(spec, pointer={"local_dir": existing})
        print(f"[{spec['name']}] 既存の重みを参照します: {existing}")
        return

    print(f"[{spec['name']}] 取得しています: {spec['repo']} ({spec['size']})")
    snapshot_download(spec["repo"])

    _register(spec)
    print(f"[{spec['name']}] 完了 (重みは共有キャッシュを参照)")


# Places a sibling install may already hold the weights, checked before downloading.
_SEARCH_ROOTS = (
    os.path.join("..", "..", "models"),
    os.path.join(os.path.expanduser("~"), "Downloads", "rula", "models"),
)


def _already_local(name: str) -> str | None:
    for root in _SEARCH_ROOTS:
        candidate = os.path.join(root, name)
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return os.path.abspath(candidate)
    return None


def _register(spec: dict, pointer: dict | None = None):
    os.makedirs(spec["out"], exist_ok=True)
    body = pointer or {"repo_id": spec["repo"], "prequantized": spec["prequantized"]}
    with open(os.path.join(spec["out"], "source.json"), "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
    with open(os.path.join(spec["out"], "NOTICE"), "w", encoding="utf-8") as f:
        f.write(NOTICE.format(name=spec["name"], underline="=" * len(spec["name"]),
                              repo=spec["repo"], role=spec["role"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=list(MODELS) + ["all"], default="all")
    args = p.parse_args()

    for kind in (list(MODELS) if args.kind == "all" else [args.kind]):
        fetch(kind, MODELS[kind])
    print("\n起動: .\\run_api.ps1")


if __name__ == "__main__":
    main()
