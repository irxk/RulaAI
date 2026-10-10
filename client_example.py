"""Minimal client for the Rula API.

    python client_example.py "日本で2番目に高い山は?"
    python client_example.py --url https://xxx.trycloudflare.com "こんにちは"

Reads the key from RULA_API_KEY, or --key.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ask(url: str, key: str, message: str, conversation_id: str | None) -> dict:
    payload = {"message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=310) as resp:
        return json.load(resp)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("message", nargs="+", help="送信するメッセージ")
    p.add_argument("--url", default=os.environ.get("RULA_API_URL", "http://127.0.0.1:8000"))
    p.add_argument("--key", default=os.environ.get("RULA_API_KEY"))
    p.add_argument("--conversation", default=None, help="会話を継続するID")
    args = p.parse_args()

    if not args.key:
        print("APIキーがありません。--key か環境変数 RULA_API_KEY で渡してください。")
        return 1

    try:
        result = ask(args.url, args.key, " ".join(args.message), args.conversation)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        print(f"エラー {e.code}: {detail}")
        return 1
    except urllib.error.URLError as e:
        print(f"接続できません: {e.reason}")
        return 1

    print(result["answer"])
    if result.get("searched"):
        print(f"\n(検索: {', '.join(result['searched'])})")
    print(f"(会話ID: {result['conversation_id']} / {result['elapsed_ms']}ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
