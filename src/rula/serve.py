"""Check the inference server used by Rula."""
import json
import os
import sys
import urllib.error
import urllib.request

from .engine import build_engine


def main():
    backend = os.environ.get("RULA_MAIN_BACKEND", "tabbyapi").strip().lower()
    if backend == "tabbyapi":
        from .strata_backend import _normalize_base_url
        url = _normalize_base_url(os.environ.get(
            "RULA_TABBYAPI_BASE_URL", "http://127.0.0.1:5000/v1"
        ))
        health_url = url[:-3] + "/health"
        print("TabbyAPIの稼働状態を確認しています…")
        try:
            with urllib.request.urlopen(health_url, timeout=5.0) as response:
                status = json.loads(response.read().decode("utf-8")).get("status")
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError) as exc:
            print(f"TabbyAPIに接続できません: {type(exc).__name__}: {exc}")
            sys.exit(1)
        if status != "healthy":
            print(f"TabbyAPIは正常ではありません: {status or '状態不明'}")
            sys.exit(1)
        print(f"TabbyAPIに接続しました: {health_url}")
        print("Main Rulaを開くと設定済みのモデルがロードされます。")
        return

    print(f"{backend or '推論'}モデルに接続しています…")
    try:
        engine = build_engine()
    except Exception as exc:
        print(f"推論サーバーに接続できません: {type(exc).__name__}: {exc}")
        sys.exit(1)
    print(f"\n準備完了: {engine.display_name} ({engine.backend})")
    print(f"接続先: {engine.base_url}")
    print("モデルの起動・停止とメモリ管理は推論サーバー側で行われます。")


if __name__ == "__main__":
    main()
