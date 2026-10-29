"""Rula の対話CLI。"""
import argparse
from importlib.util import find_spec
import sys

from .agent import Agent, AgentConfig
from .embedding import EmbeddingMemory
from .engine import build_engine
from .memory import Memory
from .persona import NAME
from .tools import default_registry

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stdin.reconfigure(encoding="utf-8")

DIM, BOLD, CYAN, YELLOW, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[0m"

HELP = f"""
{BOLD}コマンド{RESET}
  /help              このヘルプを表示
  /think             思考モードの ON/OFF を切り替え
  /candid            率直モード(無検閲)の ON/OFF を切り替え
  /tone <指定>       口調を固定(例: /tone 冷笑的で皮肉っぽく)。/tone off で解除
  /image <パス> [質問] 画像を読み取る(パスを文中に書くだけでも自動で読む)
  /draw <内容>       画像を生成する(「猫の絵を描いて」でも自動で生成)
  /tools             ツール使用の ON/OFF を切り替え
  /trace             内部動作(思考・ツール)の表示を切り替え
  /search <語>       明示的にWeb検索してから答える
  /read <URL>        指定URLを読んでから答える
  /calc <式>         計算だけ実行する
  /now               現在日時を表示
  /remember <事実>   ユーザーについて覚えさせる(例: /remember 好きな食べ物はラーメン)
  /memory            覚えている内容を一覧表示
  /forget <番号>     指定した記憶を削除(/forget all で全消去)
  /reset             会話履歴をクリア(記憶は残る)
  /save <file>       会話を保存
  /load <file>       会話を読み込み
  /status            現在の設定を表示
  /exit              終了
"""


def _banner(agent: Agent):
    print(f"{BOLD}{CYAN}{NAME}{RESET} — 日本語アシスタント")
    eng = agent.engine
    quant = " (4bit)" if eng.quantized else ""
    features = [getattr(eng, "backend", "transformers")]
    if getattr(eng, "mtp", False):
        features.append("MTP")
    if getattr(eng, "supports_vision", False):
        features.append("画像対応")
    print(f"{DIM}model: {eng.display_name}{quant} | {', '.join(features)} | "
          f"tools: {', '.join(agent.tools.names())}{RESET}")
    print(f"{DIM}/help でコマンド一覧。/exit で終了。{RESET}\n")


def _print_answer(answer: str, trace, trailing_newline: bool = False):
    # With streaming on, the text has already gone to the terminal token by token;
    # printing it again would just repeat the whole answer.
    if getattr(trace, "streamed", False):
        print()
    else:
        print(f"{BOLD}{CYAN}{NAME}:{RESET} {answer}")
    if trailing_newline:
        print()


def _show_trace(trace, show: bool):
    for fact in trace.remembered:
        print(f"{DIM}  [memory] 覚えました: {fact}{RESET}")
    if trace.tone:
        print(f"{DIM}  [tone] 「{trace.tone}」で話します(/tone off で解除){RESET}")
    if show and trace.drawn:
        print(f"{DIM}  [draw] prompt: {trace.drawn[1]}{RESET}")
    if not show:
        return
    if trace.image:
        ref, description = trace.image
        print(f"{DIM}  [vision] {ref}{RESET}")
        print(f"{DIM}  {description[:300]}{RESET}")
    for name, arg, result in trace.tool_calls:
        preview = result if len(result) <= 300 else result[:300] + "…"
        print(f"{DIM}  [tool] {name}: {arg}{RESET}")
        print(f"{DIM}  {preview.replace(chr(10), chr(10) + '  ')}{RESET}")
    if trace.thinking:
        print(f"{DIM}  [think] {trace.thinking}{RESET}")


def main():
    p = argparse.ArgumentParser(description=f"{NAME} — 日本語アシスタント")
    p.add_argument("--model-id", default=None,
                   help="Strataサーバーに読み込まれたモデルID")
    p.add_argument("--adapter", default=None,
                   help="未対応: LoRAはStrataサーバー側で設定してください")
    p.add_argument("--think", action="store_true", help="思考モードを有効にして起動")
    p.add_argument("--candid", action="store_true", help="率直モード(無検閲)を有効にして起動")
    p.add_argument("--no-tools", action="store_true", help="ツールを無効にして起動")
    p.add_argument("--no-vision", action="store_true", help="画像の自動読み取りを無効にする")
    imagegen = p.add_mutually_exclusive_group()
    imagegen.add_argument("--imagegen", dest="use_imagegen", action="store_true",
                          help="画像生成を有効にする(対応ビルドのみ)")
    imagegen.add_argument("--no-imagegen", dest="use_imagegen", action="store_false",
                          help="画像生成を無効にする")
    p.set_defaults(use_imagegen=False)
    p.add_argument("--stream", action="store_true",
                   help="長い回答を生成しながら表示する")
    p.add_argument("--no-image-filter", action="store_true",
                   help="生成画像の安全フィルタを外す(誤検知で黒画像になる場合)")
    p.add_argument("--quantize", dest="quantize", action="store_true", default=None,
                   help="未対応: 量子化はStrataサーバー側で設定してください")
    p.add_argument("--no-quantize", dest="quantize", action="store_false",
                   help="未対応: 量子化はStrataサーバー側で設定してください")
    p.add_argument("--trace", action="store_true", help="内部動作を表示して起動")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--prompts", default=None,
                   help="セミコロン区切りで非対話実行して終了(検証用)")
    p.add_argument("--memory-file", default="data/rula_memory.json",
                   help="ユーザー情報の保存先(--no-memory で無効化)")
    p.add_argument("--no-memory", action="store_true", help="記憶を保存しない")
    p.add_argument("--device", default=None,
                   help="未対応: GPU割当はStrataサーバー側で設定してください")
    args = p.parse_args()
    if args.use_imagegen and find_spec("rula.imagegen") is None:
        p.error("このビルドには画像生成機能が含まれていません。")
    if args.adapter:
        p.error("--adapter は未対応です。モデル側のLoRAをStrataで読み込んでください。")
    if args.device is not None or args.quantize is not None:
        p.error("--device/--quantize は未対応です。Strataの起動設定で選択してください。")

    print(f"{DIM}Strataモデルに接続しています…{RESET}")
    engine = build_engine(model_id=args.model_id)
    emb = None
    if not args.no_memory and EmbeddingMemory.is_available():
        import os
        emb_path = os.path.join(os.path.dirname(args.memory_file or "memory.json"),
                                "embedding_memory.json")
        emb = EmbeddingMemory(emb_path)
    agent = Agent(
        engine,
        AgentConfig(thinking=args.think, candid=args.candid,
                    use_tools=not args.no_tools, use_vision=not args.no_vision,
                    use_imagegen=args.use_imagegen,
                    image_filter=not args.no_image_filter, stream=args.stream,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature),
        default_registry(),
        Memory(None if args.no_memory else args.memory_file),
        embedding_memory=emb,
    )
    show_trace = args.trace

    if args.prompts:
        for prompt in args.prompts.split(";"):
            prompt = prompt.strip()
            if not prompt:
                continue
            print(f"\n{BOLD}あなた:{RESET} {prompt}")
            answer, trace = agent.ask(prompt)
            _show_trace(trace, show_trace)
            _print_answer(answer, trace)
        return

    _banner(agent)

    while True:
        try:
            user_input = input(f"{BOLD}あなた:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, _, rest = user_input.partition(" ")
            cmd, rest = cmd.lower(), rest.strip()

            if cmd in ("/exit", "/quit"):
                break
            if cmd == "/help":
                print(HELP)
                continue
            if cmd == "/think":
                agent.config.thinking = not agent.config.thinking
                print(f"{YELLOW}思考モード: {'ON' if agent.config.thinking else 'OFF'}{RESET}")
                continue
            if cmd == "/candid":
                agent.config.candid = not agent.config.candid
                print(f"{YELLOW}率直モード: {'ON' if agent.config.candid else 'OFF'}{RESET}")
                continue
            if cmd == "/draw":
                if not rest:
                    print(f"{YELLOW}使い方: /draw 夕焼けの海{RESET}")
                    continue
                print(f"{DIM}画像を生成しています…{RESET}")
                _, message, prompt = agent.draw(rest)
                if show_trace:
                    print(f"{DIM}  [draw] prompt: {prompt}{RESET}")
                print(f"{YELLOW}{message}{RESET}")
                continue
            if cmd == "/image":
                if not rest:
                    print(f"{YELLOW}使い方: /image data/photo.png 何が写っている?{RESET}")
                    continue
                ref, _, question = rest.partition(" ")
                print(f"{DIM}画像を読み取っています…{RESET}")
                description = agent.look(ref, question.strip() or None)
                if show_trace:
                    print(f"{DIM}  [vision] {description[:300]}{RESET}")
                answer = agent.answer_with_context(
                    question.strip() or "この画像について説明してください。",
                    description, "添付画像の内容")
                print(f"{BOLD}{CYAN}{NAME}:{RESET} {answer}")
                continue
            if cmd == "/tools":
                agent.config.use_tools = not agent.config.use_tools
                print(f"{YELLOW}ツール: {'ON' if agent.config.use_tools else 'OFF'}{RESET}")
                continue
            if cmd == "/trace":
                show_trace = not show_trace
                print(f"{YELLOW}内部動作の表示: {'ON' if show_trace else 'OFF'}{RESET}")
                continue
            if cmd == "/status":
                c = agent.config
                print(f"{YELLOW}思考:{'ON' if c.thinking else 'OFF'} / "
                      f"率直:{'ON' if c.candid else 'OFF'} / "
                      f"ツール:{'ON' if c.use_tools else 'OFF'} / "
                      f"画像:{'ON' if c.use_vision else 'OFF'} / "
                      f"trace:{'ON' if show_trace else 'OFF'} / "
                      f"履歴:{len(agent.history)}件{RESET}")
                continue
            if cmd == "/tone":
                if not rest:
                    current = agent.tone or "(指定なし)"
                    print(f"{YELLOW}現在の口調: {current}"
                          f"  — 変更: /tone 辛口で / 解除: /tone off{RESET}")
                elif rest.lower() in ("off", "なし", "解除"):
                    agent.tone = None
                    print(f"{YELLOW}口調の指定を解除しました。{RESET}")
                else:
                    agent.tone = rest
                    print(f"{YELLOW}口調を設定しました: {rest}{RESET}")
                continue
            if cmd == "/reset":
                agent.reset()
                print(f"{YELLOW}会話履歴をクリアしました。{RESET}")
                continue
            if cmd == "/remember":
                if not rest:
                    print(f"{YELLOW}使い方: /remember 好きな食べ物はラーメン{RESET}")
                elif agent.memory.add(rest):
                    print(f"{YELLOW}覚えました: {rest}{RESET}")
                else:
                    print(f"{YELLOW}すでに覚えています。{RESET}")
                continue
            if cmd == "/memory":
                if not len(agent.memory):
                    print(f"{YELLOW}まだ何も覚えていません。{RESET}")
                else:
                    print(f"{YELLOW}覚えていること:{RESET}")
                    for i, fact in enumerate(agent.memory.facts):
                        print(f"{YELLOW}  [{i}] {fact}{RESET}")
                continue
            if cmd == "/forget":
                if rest == "all":
                    agent.memory.clear()
                    print(f"{YELLOW}すべて忘れました。{RESET}")
                elif rest.isdigit():
                    removed = agent.memory.remove(int(rest))
                    print(f"{YELLOW}{'忘れました: ' + removed if removed else 'その番号はありません。'}{RESET}")
                else:
                    print(f"{YELLOW}使い方: /forget 0 または /forget all{RESET}")
                continue
            if cmd == "/now":
                from datetime import datetime
                try:
                    from zoneinfo import ZoneInfo
                    t = datetime.now(ZoneInfo("Asia/Tokyo"))
                except (ImportError, KeyError):
                    t = datetime.now().astimezone()
                _wd = ("月", "火", "水", "木", "金", "土", "日")
                print(f"{YELLOW}{t.year}年{t.month}月{t.day}日({_wd[t.weekday()]}) "
                      f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}{RESET}")
                continue
            if cmd == "/calc":
                if not rest:
                    print(f"{YELLOW}使い方: /calc 1+2*3{RESET}")
                    continue
                print(f"{YELLOW}{agent.run_tool_directly('calc', rest)}{RESET}")
                continue
            if cmd in ("/search", "/read"):
                if not rest:
                    print(f"{YELLOW}使い方: {cmd} <{'検索語' if cmd == '/search' else 'URL'}>{RESET}")
                    continue
                tool = "search" if cmd == "/search" else "read_url"
                label = "検索結果" if cmd == "/search" else "ページ本文"
                print(f"{DIM}{label}を取得しています…{RESET}")
                context = agent.run_tool_directly(tool, rest)
                if show_trace:
                    print(f"{DIM}{context[:400]}{RESET}")
                answer = agent.answer_with_context(rest, context, label)
                print(f"{BOLD}{CYAN}{NAME}:{RESET} {answer}")
                continue
            if cmd in ("/save", "/load"):
                if not rest:
                    print(f"{YELLOW}使い方: {cmd} conversation.json{RESET}")
                    continue
                try:
                    if cmd == "/save":
                        agent.save(rest)
                        print(f"{YELLOW}保存しました: {rest}{RESET}")
                    else:
                        agent.load(rest)
                        print(f"{YELLOW}読み込みました: {rest}({len(agent.history)}件){RESET}")
                except OSError as e:
                    print(f"{YELLOW}失敗しました: {e}{RESET}")
                continue

            print(f"{YELLOW}不明なコマンドです: {cmd}(/help で一覧){RESET}")
            continue

        answer, trace = agent.ask(user_input)
        _show_trace(trace, show_trace)
        _print_answer(answer, trace, trailing_newline=True)


if __name__ == "__main__":
    main()
