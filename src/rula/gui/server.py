"""The desktop app's backend: serves the window and runs the Agent.

Started by Rula.exe (launcher/RulaLauncher.cs) as

    pythonw -m src.rula.gui --port N --token T

It is the same Agent the CLI and the API use — search, calc, dates, images, MTP —
behind a page shown in an Edge app window. Binds to 127.0.0.1 only, and every
API call must carry the token the launcher generated, so neither another program
on this machine nor a web page that guesses the port can drive it.

Chats are the user's own, on their own machine, so they are kept — like any chat
app — in %APPDATA%\\Rula\\chats. The log file holds status and errors only, never
message text.
"""
import argparse
import asyncio
import json
import logging
import os
import re
import secrets
import subprocess
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..agent import Agent, AgentConfig, Turn
from ..engine import build_engine
from ..memory import Memory
from ..tools import default_registry

log = logging.getLogger("rula.gui")

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
DATA = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "Rula")
CHATS = os.path.join(DATA, "chats")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
STATE = {"status": "loading", "error": None, "engine": None, "token": "", "port": 0,
         "last_ping": None, "started": time.monotonic()}
GEN_LOCK = threading.Lock()  # one GPU: one answer at a time
_agents: dict[str, Agent] = {}
_memory: Memory | None = None

def _tool_words() -> list[str]:
    from ..tools.base import ALIASES
    return sorted({n.lower() for n in default_registry().names()} | {a.lower() for a in ALIASES},
                  key=len, reverse=True)


_TOOL_WORDS = _tool_words()
_TOOL_CALL_START = re.compile(
    r"^(?:<tool|<tool_call|(?:" + "|".join(map(re.escape, _TOOL_WORDS)) + r")\s*[:：])", re.I)


def _echo_guard(text: str):
    """Whether a pass's opening is an answer to show or a tool request to hide.

    Every pass is streamed to the window, and the model asks for a tool by
    writing "<tool>search: …" or just "検索: …" as its whole reply. Those would
    flash up as text; held back until the opening says which it is.
    """
    t = text.lstrip(" `\n").lower()
    if not t:
        return None
    if _TOOL_CALL_START.match(t):
        return False
    if len(t) < 16 and "\n" not in t and any(
            w.startswith(t) or t.startswith(w) for w in _TOOL_WORDS + ["<tool"]):
        return None  # could still turn into one
    return True


_TOOL_LABELS = {"search": "検索しています", "deep_search": "ページを読み込んで調べています",
                "smart_search": "いくつかの観点で検索しています", "read_url": "ページを読んでいます",
                "calc": "計算しています", "date": "日付を計算しています"}


# -- access control ---------------------------------------------------------
def _check(request: Request):
    host = (request.headers.get("host") or "").lower()
    # The Host check stops DNS rebinding: a page on some domain that resolves to
    # 127.0.0.1 would arrive with its own name here, not ours.
    if host not in (f"127.0.0.1:{STATE['port']}", f"localhost:{STATE['port']}"):
        raise HTTPException(403, "forbidden host")
    if not secrets.compare_digest(request.headers.get("x-rula-token", ""), STATE["token"]):
        raise HTTPException(403, "bad token")


# -- static -----------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"),
                        headers={"Cache-Control": "no-store"})


@app.get("/static/{name}")
def static(name: str):
    if not re.fullmatch(r"[\w.-]+", name):
        raise HTTPException(404)
    path = os.path.join(STATIC, name)
    if not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "no-store"})


# -- status -----------------------------------------------------------------
def _vram_free() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


@app.get("/api/info")
def info(request: Request):
    _check(request)
    eng = STATE["engine"]
    return {"status": STATE["status"], "error": STATE["error"],
            "model": getattr(eng, "display_name", None),
            "backend": getattr(eng, "backend", None),
            "mtp": bool(getattr(eng, "mtp", False)),
            "vision": bool(getattr(eng, "supports_vision", False)),
            "thinking": bool(getattr(eng, "supports_thinking", False)),
            "vram_free": _vram_free() if eng else None}


@app.post("/api/ping")
def ping(request: Request):
    _check(request)
    STATE["last_ping"] = time.monotonic()
    return {"ok": True}


@app.post("/api/bye")
def bye(request: Request, t: str = ""):
    """Sent by the page as it closes (navigator.sendBeacon on pagehide). This is
    what quits Rula when its window closes: waiting on the Edge process is not
    reliable, since that process can be holding other windows too. sendBeacon
    cannot set headers, so the token comes in the query string."""
    host = (request.headers.get("host") or "").lower()
    if host not in (f"127.0.0.1:{STATE['port']}", f"localhost:{STATE['port']}"):
        raise HTTPException(403, "forbidden host")
    if not secrets.compare_digest(t, STATE["token"]):
        raise HTTPException(403, "bad token")
    STATE["bye"] = time.monotonic()
    return {"ok": True}


# -- chats ------------------------------------------------------------------
def _chat_path(chat_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{12}", chat_id):
        raise HTTPException(404)
    return os.path.join(CHATS, f"{chat_id}.json")


def _load_chat(chat_id: str) -> dict:
    path = _chat_path(chat_id)
    if not os.path.isfile(path):
        raise HTTPException(404)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_chat(chat: dict):
    os.makedirs(CHATS, exist_ok=True)
    tmp = _chat_path(chat["id"]) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(chat, f, ensure_ascii=False)
    os.replace(tmp, _chat_path(chat["id"]))


@app.get("/api/chats")
def list_chats(request: Request):
    _check(request)
    out = []
    if os.path.isdir(CHATS):
        for name in os.listdir(CHATS):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(CHATS, name), encoding="utf-8") as f:
                    c = json.load(f)
                out.append({"id": c["id"], "title": c.get("title") or "新しいチャット",
                            "updated": c.get("updated", 0)})
            except (OSError, ValueError, KeyError):
                continue
    out.sort(key=lambda c: c["updated"], reverse=True)
    return out


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str, request: Request):
    _check(request)
    return _load_chat(chat_id)


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, request: Request):
    _check(request)
    path = _chat_path(chat_id)
    if os.path.isfile(path):
        os.remove(path)
    _agents.pop(chat_id, None)
    return {"ok": True}


# -- talking ----------------------------------------------------------------
class ChatIn(BaseModel):
    chat_id: str | None = None
    message: str = ""
    image: str | None = None   # data URI, already resized by the page
    thumb: str | None = None   # small copy kept in the saved chat
    think: bool = False


def _agent_for(chat: dict) -> Agent:
    agent = _agents.get(chat["id"])
    if agent is None:
        agent = Agent(STATE["engine"],
                      AgentConfig(candid=True, stream=True, preview=True, use_imagegen=False,
                                  max_new_tokens=1024, max_continuations=2),
                      default_registry(), _memory)
        agent.history = [Turn(m["role"], m["content"]) for m in chat["messages"]]
        if len(_agents) >= 20:
            _agents.pop(next(iter(_agents)))
        _agents[chat["id"]] = agent
    return agent


@app.post("/api/chat")
async def chat(req: ChatIn, request: Request):
    _check(request)
    if STATE["status"] != "ready":
        raise HTTPException(503, "モデルの準備ができていません。")
    text = req.message.strip()[:8000]
    if not text and not req.image:
        raise HTTPException(400, "empty")
    if req.image and not req.image.startswith("data:image/"):
        raise HTTPException(400, "image must be a data URI")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(kind: str, **data):
        loop.call_soon_threadsafe(queue.put_nowait, {"type": kind, **data})

    if req.chat_id:
        chat_doc = _load_chat(req.chat_id)
    else:
        now = time.time()
        chat_doc = {"id": uuid.uuid4().hex[:12], "title": None, "created": now,
                    "updated": now, "messages": []}
    if not chat_doc.get("title"):
        chat_doc["title"] = (text or "画像")[:40].replace("\n", " ")

    def work():
        eng = STATE["engine"]
        agent = _agent_for(chat_doc)
        with GEN_LOCK:
            started = time.perf_counter()
            try:
                emit("chat", id=chat_doc["id"], title=chat_doc["title"])
                eng.cancel = False
                eng.echo_guard = _echo_guard
                eng.echo_sink = lambda piece: emit("token", text=piece)
                # A new pass starts its text over, so what a discarded pass
                # streamed ("まず調べます。" before a tool call) is not kept.
                eng.echo_reset = lambda: emit("reset")
                agent.on_thinking = lambda piece: emit("thinking", text=piece)
                agent.on_tool = lambda name, arg: emit(
                    "status", text=f"{_TOOL_LABELS.get(name, name)}: {arg[:60]}"
                    if name in ("search", "deep_search", "smart_search")
                    else _TOOL_LABELS.get(name, name))
                agent.config.thinking = bool(req.think)
                if req.image:
                    emit("status", text="画像を読んでいます")
                elif req.think:
                    emit("status", text="考えています")
                question = text or "この画像について説明してください。"
                answer, trace = agent.ask(question, image_ref=req.image)
                stopped = bool(getattr(eng, "cancel", False))
                elapsed = round(time.perf_counter() - started, 1)
                tools = [{"name": n, "arg": a} for n, a, _ in trace.tool_calls]
                now = time.time()
                chat_doc["messages"].append({"role": "user", "content": question,
                                             "thumb": req.thumb, "time": now})
                chat_doc["messages"].append({"role": "assistant", "content": answer,
                                             "thinking": trace.thinking, "tools": tools,
                                             "elapsed": elapsed, "stopped": stopped,
                                             "time": now})
                chat_doc["updated"] = now
                _save_chat(chat_doc)
                emit("done", answer=answer, thinking=trace.thinking, tools=tools,
                     elapsed=elapsed, stopped=stopped)
            except Exception as e:  # shown in the window; the log gets the type only
                log.exception("generation failed")
                emit("error", text=f"{type(e).__name__}: {e}")
            finally:
                eng.echo_sink = eng.echo_guard = eng.echo_reset = None
                eng.cancel = False
                agent.on_thinking = agent.on_tool = None
                emit("end")

    threading.Thread(target=work, daemon=True).start()

    async def events():
        while True:
            ev = await queue.get()
            if ev["type"] == "end":
                break
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.post("/api/stop")
def stop(request: Request):
    _check(request)
    eng = STATE["engine"]
    if eng is not None:
        eng.cancel = True
    return {"ok": True}


class OpenIn(BaseModel):
    url: str


@app.post("/api/open")
def open_url(body: OpenIn, request: Request):
    """Links in answers open in the normal browser, not inside the app window."""
    _check(request)
    if not re.match(r"^https?://", body.url):
        raise HTTPException(400)
    os.startfile(body.url)  # noqa: S606 — http(s) only, checked above
    return {"ok": True}


# -- lifecycle --------------------------------------------------------------
def _load_engine():
    try:
        STATE["engine"] = build_engine()
        STATE["status"] = "ready"
        log.info("ready: %s (%s)", STATE["engine"].display_name,
                 getattr(STATE["engine"], "backend", "transformers"))
    except BaseException as e:
        STATE["status"] = "error"
        STATE["error"] = f"{type(e).__name__}: {e}"
        log.exception("engine failed to load")


RUN_FILE = os.path.join(DATA, "running.txt")


def _quit():
    eng = STATE["engine"]
    if eng is not None and hasattr(eng, "close"):
        eng.close()
    try:
        os.remove(RUN_FILE)
    except OSError:
        pass
    os._exit(0)


def _watchdog():
    """Exit when the window is gone. The launcher stops us when Edge exits, but
    Edge can hand a window to an already-running instance, and then there is no
    process to wait on; the page's heartbeat stopping is the fallback signal."""
    while True:
        time.sleep(1)
        last = STATE["last_ping"]
        bye = STATE.get("bye")
        now = time.monotonic()
        # The page said goodbye and no page has pinged since: the window is gone.
        # Three seconds' grace covers a reload, whose new page pings at once.
        if bye and (last is None or last < bye) and now - bye > 3:
            log.info("window closed; exiting")
            _quit()
        if (last is None and now - STATE["started"] > 180) or (last and now - last > 45):
            log.info("no heartbeat; exiting")
            _quit()


def _setup(port: int, token: str):
    global _memory
    os.makedirs(DATA, exist_ok=True)
    # Third-party loggers stay at WARNING: the search library logs every request
    # URL at INFO, and a search URL carries the question in its query string.
    logging.basicConfig(filename=os.path.join(DATA, "rula.log"), level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        encoding="utf-8")
    logging.getLogger("rula").setLevel(logging.INFO)
    # .env next to the app, as the API reads it (RULA_GGUF and friends).
    env = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(env):
        with open(env, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

    STATE["token"], STATE["port"] = token, port
    _memory = Memory(os.path.join(DATA, "memory.json"))
    threading.Thread(target=_load_engine, daemon=True).start()
    threading.Thread(target=_watchdog, daemon=True).start()


def _server(port: int):
    import uvicorn
    # 127.0.0.1, never 0.0.0.0: this is a desktop app's private backend.
    return uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                         log_level="warning", log_config=None))


# -- the app without an exe ---------------------------------------------------
# Rula.exe does the steps below in C#. They are here too because a freshly built,
# unsigned exe is what antivirus heuristics quarantine: on this machine
# Malwarebytes / Tencent PC Manager deleted Rula.exe the moment it ran. A
# shortcut to the signed pythonw.exe running this does the same with no exe.
def _alive(url: str) -> bool:
    import urllib.request
    base = url.split("?")[0]
    try:
        with urllib.request.urlopen(base, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _find_edge() -> str | None:
    for root in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.environ.get("LOCALAPPDATA")):
        if root:
            p = os.path.join(root, "Microsoft", "Edge", "Application", "msedge.exe")
            if os.path.isfile(p):
                return p
    return None


# Edge signs a brand-new profile into the Windows account and syncs it. Measured:
# the first Rula profile received the user's 11 extensions — MetaMask among them,
# which opened a window of its own inside the Rula profile, so closing Rula's
# window no longer ended that Edge process. Rula's window needs none of it.
EDGE_FLAGS = ["--window-size=1180,820", "--no-first-run", "--no-default-browser-check",
              "--disable-extensions", "--disable-sync",
              "--disable-features=Translate,msEdgeSidebarV2"]


def _open_window(url: str) -> "subprocess.Popen | None":
    """An Edge app window (no address bar, own taskbar entry) on its own profile,
    so it shares nothing with the user's browsing and is its own process —
    which is what lets closing it stop Rula. Returns None when there is no
    process to wait on (no Edge, or the window went to a running instance)."""
    edge = _find_edge()
    if edge is None:
        os.startfile(url)
        return None
    profile = os.path.join(os.environ.get("LOCALAPPDATA") or DATA, "Rula", "EdgeProfile")
    proc = subprocess.Popen([edge, f"--app={url}", f"--user-data-dir={profile}", *EDGE_FLAGS])
    try:
        proc.wait(timeout=1.5)
        return None
    except subprocess.TimeoutExpired:
        return proc


def run_app():
    """Start Rula and open its window; stop everything when the window closes."""
    os.makedirs(DATA, exist_ok=True)
    if os.path.isfile(RUN_FILE):
        with open(RUN_FILE, encoding="utf-8") as f:
            url = f.read().strip()
        if _alive(url):
            _open_window(url)  # already running: just another window onto it
            return
    port = _free_port()
    token = secrets.token_hex(24)
    url = f"http://127.0.0.1:{port}/?t={token}"
    _setup(port, token)
    server = _server(port)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if _alive(url):
            break
        time.sleep(0.05)
    with open(RUN_FILE, "w", encoding="utf-8") as f:
        f.write(url)
    window = _open_window(url)
    if window is not None:
        window.wait()
        _quit()
    threading.Event().wait()  # the watchdog ends it when the heartbeat stops


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, help="backend only, for Rula.exe")
    p.add_argument("--token")
    args = p.parse_args()
    if args.port is None:
        run_app()
        return
    _setup(args.port, args.token)
    _server(args.port).run()
