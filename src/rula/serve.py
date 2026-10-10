"""Keep the model warm so the app launches instantly.

    python -m src.rula.serve        (or: .\warm.ps1)

Starts llama-server once and leaves it running. While this is open, every Rula
launch connects to the already-loaded model in about a second, instead of
reloading the 9B into VRAM each time (~10-15 s). The window closing no longer
unloads the model — the app only ever stops a server it started itself.

The cost is VRAM: the model (about 6 GB for the 9B) stays resident the whole
time this is open, even when no Rula window is up. Press Ctrl+C, or close this
window, to stop it and free the VRAM.
"""
import sys
import threading

from .llamaserver_backend import HOST, LlamaServerEngine


def main():
    print("モデルを読み込んでいます… (初回は10秒ほど)")
    try:
        eng = LlamaServerEngine()
    except Exception as e:
        print(f"起動できませんでした: {e}")
        sys.exit(1)
    extra = []
    if getattr(eng, "mtp", None):
        extra.append("MTP")
    if getattr(eng, "supports_vision", False):
        extra.append("画像")
    tag = f" ({' / '.join(extra)})" if extra else ""
    print(f"\n準備完了: {eng.display_name}{tag} が {HOST} で待機しています。")
    print("このウィンドウを開いている間、Rula の起動は一瞬になります。")
    print("終了して VRAM を解放するには、Ctrl+C を押すか、このウィンドウを閉じてください。\n")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        eng.close()
        print("\n停止しました。VRAM を解放しました。")


if __name__ == "__main__":
    main()
