"""The desktop app's backend: serves the window and runs the Agent.

Started by Rula.exe (launcher/RulaLauncher.cs) as

    pythonw -m src.rula.gui --port N --token T

It is the same Agent the CLI and the API use — search, calc, dates, images, MTP —
behind a page shown in an Edge app window. Binds to 127.0.0.1 only, and every
API call must carry the token the launcher generated, so neither another program
on this machine nor a web page that guesses the port can drive it.

Chats are the user's own, on their own machine, so they are kept — like any chat
app — in %APPDATA%\\Rula\\chats (or %APPDATA%\\RulaCoder\\chats for the dedicated
Coder app). The log file holds status and errors only, never message text.
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
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..agent import Agent, AgentConfig, Turn
from ..coder_agent import CoderAgent
from ..engine import build_engine
from ..memory import Memory
from ..tools import default_registry
from ..tools.coder import CoderTools
from ..version import __version__

log = logging.getLogger("rula.gui")

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
APP_MODE = "coder" if os.environ.get("RULA_APP_MODE", "").lower() == "coder" else "general"
APP_DATA_ROOT = os.environ.get("APPDATA") or os.path.expanduser("~")
DATA = os.path.join(APP_DATA_ROOT, "RulaCoder" if APP_MODE == "coder" else "Rula")
CHATS = os.path.join(DATA, "chats")
RUN_FILE = os.path.join(APP_DATA_ROOT, "Rula", "running.txt")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
STATE = {"status": "loading", "error": None, "engine": None, "token": "", "port": 0,
         "last_ping": None, "started": time.monotonic(), "switching": False}
GEN_LOCK = threading.Lock()  # one GPU: one answer at a time
_agents: dict[str, Agent] = {}
_memory: Memory | None = None
_pending_interactions: dict[str, "PendingInteraction"] = {}
_interaction_lock = threading.Lock()

@dataclass
class PendingInteraction:
    kind: str
    event: threading.Event = field(default_factory=threading.Event)
    response: str | None = None


def _tool_words() -> list[str]:
    from ..tools.base import ALIASES
    coder_tool_words = {
        "list_files", "find_files", "find_symbol", "read_file", "read_files",
        "file_outline", "search_code", "check_python", "project_info", "project_memory",
        "update_plan",
        "git_status", "git_diff", "ask_user", "write_file", "edit_file",
        "apply_patch", "review_changes", "run_command",
    }
    words = ({n.lower() for n in default_registry().names()}
             | {a.lower() for a in ALIASES} | coder_tool_words)
    return sorted(words, key=len, reverse=True)


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
                "calc": "計算しています", "date": "日付を計算しています",
                "list_files": "作業フォルダを一覧しています", "read_file": "ファイルを読んでいます",
                "find_files": "ファイルを検索しています", "find_symbol": "定義を検索しています",
                "file_outline": "ファイル構造を確認しています", "read_files": "複数ファイルを読んでいます",
                "check_python": "Python構文を検査しています",
                "search_code": "コードを検索しています", "project_info": "プロジェクト情報を確認しています",
                "project_memory": "プロジェクト記憶を確認しています",
                "update_plan": "作業計画を更新しています", "git_status": "Git状態を確認しています",
                "git_diff": "Git差分を確認しています",
                "write_file": "ファイルを保存しています", "edit_file": "ファイルを編集しています",
                "apply_patch": "変更を適用しています", "review_changes": "変更差分を確認しています",
                "run_command": "コマンドを実行しています", "ask_user": "質問しています"}


def _coder_activity_detail(name: str, argument: str) -> str:
    value = argument.strip()
    if name in ("write_file", "edit_file"):
        return value.split(" :: ", 1)[0][:240]
    if name == "apply_patch":
        paths = re.findall(r"(?m)^@@ ([^\r\n]+)", value)
        return "、".join(paths[:5]) + (f" ほか{len(paths) - 5}件" if len(paths) > 5 else "")
    if name in ("run_command",):
        command = re.sub(
            r"(?i)(--?(?:password|token|secret|api[-_]?key)(?:=|\s+))"
            r"(\"[^\"]*\"|'[^']*'|[^\s;]+)",
            r"\1[伏せ字]", value)
        command = re.sub(
            r"(?i)(authorization\s*:\s*bearer\s+)[^\s\"']+",
            r"\1[伏せ字]", command)
        command = re.sub(
            r"(?i)\b(password|token|secret|api[-_]?key)(\s*=\s*)"
            r"(\"[^\"]*\"|'[^']*'|[^\s;]+)",
            r"\1\2[伏せ字]", command)
        return command[:240]
    if name in ("read_file", "find_files", "file_outline", "check_python", "git_diff"):
        return value.split(" :: ", 1)[0][:240]
    if name in ("search_code", "find_symbol"):
        return value[:120]
    if name == "read_files":
        return "、".join(value.splitlines()[:5])[:240]
    return value[:120] if name in ("list_files",) else ""


def _coder_activity_result(name: str, result: str) -> str:
    if name == "run_command":
        if result.startswith((
                "サーバー起動完了: ", "既存のローカルサーバー応答を確認: ")):
            return re.sub(r"\s+", " ", result).strip()[:240]
        if result.startswith("サーバー起動失敗: "):
            result = re.sub(
                r"(?i)(authorization\s*:\s*bearer\s+)[^\s\"']+",
                r"\1[伏せ字]", result)
            result = re.sub(
                r"(?i)\b(password|token|secret|api[-_]?key)(\s*[:=]\s*)"
                r"(\"[^\"]*\"|'[^']*'|\S+)",
                r"\1\2[伏せ字]", result)
            return re.sub(r"\s+", " ", result).strip()[:240]
        match = re.match(r"終了コード: (-?\d+)\n?(.*)", result, re.S)
        if match:
            output = re.sub(
                r"(?i)(authorization\s*:\s*bearer\s+)[^\s\"']+",
                r"\1[伏せ字]", match.group(2))
            output = re.sub(
                r"(?i)\b(password|token|secret|api[-_]?key)(\s*[:=]\s*)"
                r"(\"[^\"]*\"|'[^']*'|\S+)",
                r"\1\2[伏せ字]", output)
            output = re.sub(r"\s+", " ", output).strip()
            return f"終了コード {match.group(1)}" + (
                f" · {output[:240]}" if output and output != "(出力なし)" else "")
    if result.startswith((
            "ツール呼び出し形式エラー:", "ファイルが見つかりません",
            "ファイルを読めません", "フォルダが見つかりません",
            "Gitを実行できません", "Gitが見つかりません",
            "Gitの読み取り処理", "置換を中止しました",
            "適用を中止しました", "調査上限:")):
        return re.sub(r"\s+", " ", result).strip()[:240]
    if name in ("write_file", "edit_file", "apply_patch"):
        return result[:240]
    if result.startswith(("保存しました:", "変更しました:")):
        return result[:240]
    if result.startswith(("調査上限:", "ツール呼び出し形式エラー:", "ファイルが見つかりません")):
        return result[:240]
    if name in ("read_file", "search_code", "project_info", "git_diff", "git_status"):
        if "該当箇所はありません" in result or "ファイルはありません" in result:
            return result[:160]
        lines = len(result.splitlines())
        return f"結果を取得 ({lines}行・{len(result)}文字)"
    return re.sub(r"\s+", " ", result).strip()[:240]


def _coder_tool_succeeded(name: str, result: str) -> bool:
    if name in ("write_file", "edit_file"):
        return result.startswith(("保存しました:", "変更なし:"))
    if name == "apply_patch":
        return result.startswith(("変更しました:", "変更なし:"))
    if name == "run_command":
        if result.startswith((
                "サーバー起動完了: ", "既存のローカルサーバー応答を確認: ")):
            return True
        match = re.match(r"終了コード: (-?\d+)(?:\s|$)", result)
        return bool(match and match.group(1) == "0")
    if result.startswith((
            "ツール呼び出し形式エラー:", "形式エラー:", "ファイルが見つかりません",
            "ファイルを読めません", "フォルダが見つかりません",
            "ファイルではなくフォルダです:", "ファイルが大きすぎる",
            "置換を中止しました:", "適用を中止しました",
            "Python構文エラー:", "Gitを実行できません",
            "Gitが見つかりません", "Gitの読み取り処理",
            "調査上限:", "開始行がファイルの範囲外です",
    )):
        return False
    return True


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
            "version": __version__,
            "app_mode": APP_MODE,
            "switching": STATE.get("switching", False),
            "model": getattr(eng, "display_name", None),
            "size": getattr(eng, "size", None),
            "backend": getattr(eng, "backend", None),
            "mtp": bool(getattr(eng, "mtp", False)),
            "vision": bool(getattr(eng, "supports_vision", False)),
            "thinking": bool(getattr(eng, "supports_thinking", False)),
            "vram_free": _vram_free() if eng else None}


@app.post("/api/ping")
def ping(request: Request):
    _check(request)
    STATE["last_ping"] = time.monotonic()
    return {"ok": True, "switching": STATE.get("switching", False)}


@app.post("/api/switch")
def switch_app(request: Request, t: str = ""):
    """Tell the current app window to close so the other Rula app can take over."""
    host = (request.headers.get("host") or "").lower()
    if host not in (f"127.0.0.1:{STATE['port']}", f"localhost:{STATE['port']}"):
        raise HTTPException(403, "forbidden host")
    if not secrets.compare_digest(t, STATE["token"]):
        raise HTTPException(403, "bad token")
    STATE["switching"] = True
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
    profile: str = "general"
    coder_mode: str = "chat"
    coder_effort: str = "medium"
    workspace: str | None = None
    auto_approved: bool = False


def _profile_for_app(profile: str) -> str:
    return "coder" if APP_MODE == "coder" else profile


def _agent_for(chat: dict) -> Agent:
    agent = _agents.get(chat["id"])
    agent_type = CoderAgent if chat.get("profile") == "coder" else Agent
    if agent is None or type(agent) is not agent_type:
        agent = agent_type(
            STATE["engine"],
            AgentConfig(candid=True, stream=True, preview=True, use_imagegen=False,
                        max_new_tokens=1024, max_continuations=2),
            default_registry(), _memory)
        agent.history = [Turn(m["role"], m["content"]) for m in chat["messages"]]
        if len(_agents) >= 20:
            _agents.pop(next(iter(_agents)))
        _agents[chat["id"]] = agent
    return agent


@app.post("/api/workspace/select")
def select_workspace(request: Request):
    _check(request)
    if os.name != "nt":
        raise HTTPException(501, "この環境ではフォルダ選択ダイアログを利用できません。")
    script = r"""
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

[ComImport]
[Guid("D57C7288-D4AD-4768-BE02-9D969532D960")]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IRulaFileOpenDialog {
    [PreserveSig] int Show(IntPtr parent);
    [PreserveSig] int SetFileTypes(uint count, IntPtr types);
    [PreserveSig] int SetFileTypeIndex(uint index);
    [PreserveSig] int GetFileTypeIndex(out uint index);
    [PreserveSig] int Advise(IntPtr events, out uint cookie);
    [PreserveSig] int Unadvise(uint cookie);
    [PreserveSig] int SetOptions(uint options);
    [PreserveSig] int GetOptions(out uint options);
    [PreserveSig] int SetDefaultFolder(IntPtr item);
    [PreserveSig] int SetFolder(IntPtr item);
    [PreserveSig] int GetFolder(out IntPtr item);
    [PreserveSig] int GetCurrentSelection(out IntPtr item);
    [PreserveSig] int SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
    [PreserveSig] int GetFileName(out IntPtr name);
    [PreserveSig] int SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
    [PreserveSig] int SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
    [PreserveSig] int SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
    [PreserveSig] int GetResult(out IntPtr item);
    [PreserveSig] int AddPlace(IntPtr item, uint alignment);
    [PreserveSig] int SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string extension);
    [PreserveSig] int Close(int result);
    [PreserveSig] int SetClientGuid(ref Guid guid);
    [PreserveSig] int ClearClientData();
    [PreserveSig] int SetFilter(IntPtr filter);
    [PreserveSig] int GetResults(out IntPtr items);
    [PreserveSig] int GetSelectedItems(out IntPtr items);
}

[ComImport]
[Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE")]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IRulaShellItem {
    [PreserveSig] int BindToHandler(IntPtr bindContext, ref Guid handler, ref Guid iid, out IntPtr result);
    [PreserveSig] int GetParent(out IntPtr parent);
    [PreserveSig] int GetDisplayName(uint nameType, out IntPtr name);
    [PreserveSig] int GetAttributes(uint mask, out uint attributes);
    [PreserveSig] int Compare(IntPtr other, uint hint, out int order);
}

public static class RulaFolderPicker {
    public static string Pick() {
        Type dialogType = Type.GetTypeFromCLSID(
            new Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7"), true);
        IRulaFileOpenDialog dialog = (IRulaFileOpenDialog)Activator.CreateInstance(dialogType);
        IntPtr itemPointer = IntPtr.Zero;
        IRulaShellItem item = null;
        try {
            uint options;
            Marshal.ThrowExceptionForHR(dialog.GetOptions(out options));
            options |= 0x20 | 0x40 | 0x800 | 0x8;
            Marshal.ThrowExceptionForHR(dialog.SetOptions(options));
            int result = dialog.Show(IntPtr.Zero);
            if (result == unchecked((int)0x800704C7)) return null;
            Marshal.ThrowExceptionForHR(result);
            Marshal.ThrowExceptionForHR(dialog.GetResult(out itemPointer));
            item = (IRulaShellItem)Marshal.GetObjectForIUnknown(itemPointer);
            IntPtr namePointer;
            Marshal.ThrowExceptionForHR(item.GetDisplayName(0x80058000, out namePointer));
            try {
                return Marshal.PtrToStringUni(namePointer);
            } finally {
                Marshal.FreeCoTaskMem(namePointer);
            }
        } finally {
            if (item != null) Marshal.ReleaseComObject(item);
            if (itemPointer != IntPtr.Zero) Marshal.Release(itemPointer);
            Marshal.ReleaseComObject(dialog);
        }
    }
}
'@
$selected = [RulaFolderPicker]::Pick()
if ($selected) { [Console]::WriteLine($selected) }
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
        capture_output=True, text=True, timeout=300,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise HTTPException(500, f"フォルダ選択に失敗しました: {result.stderr.strip()}")
    selected = result.stdout.strip()
    if not selected:
        return {"workspace": None}
    path = os.path.realpath(selected)
    if not os.path.isdir(path):
        raise HTTPException(400, "選択したフォルダが見つかりません。")
    return {"workspace": path}


class InteractionIn(BaseModel):
    interaction_id: str
    response: str = ""


@app.post("/api/interaction")
def answer_interaction(body: InteractionIn, request: Request):
    _check(request)
    with _interaction_lock:
        pending = _pending_interactions.get(body.interaction_id)
        if pending is None:
            raise HTTPException(404, "この確認はすでに終了しています。")
        pending.response = body.response[:8000]
        pending.event.set()
    return {"ok": True}


@app.post("/api/chat")
async def chat(req: ChatIn, request: Request):
    _check(request)
    req.profile = _profile_for_app(req.profile)
    if STATE["status"] != "ready":
        raise HTTPException(503, "モデルの準備ができていません。")
    text = req.message.strip()[:8000]
    if not text and not req.image:
        raise HTTPException(400, "empty")
    if req.image and not req.image.startswith("data:image/"):
        raise HTTPException(400, "image must be a data URI")
    if req.profile not in ("general", "fast", "quality", "coder"):
        raise HTTPException(400, "unknown profile")
    if req.coder_mode not in ("plan", "chat", "auto"):
        raise HTTPException(400, "unknown coder mode")
    if req.coder_effort not in ("small", "medium", "large"):
        raise HTTPException(400, "unknown coder effort")
    if req.profile == "coder":
        if not req.workspace:
            raise HTTPException(400, "Coderを使うには作業フォルダを選んでください。")
        if req.image:
            raise HTTPException(400, "Coderモードでは画像を使えません。")
        if req.coder_mode == "auto" and not req.auto_approved:
            raise HTTPException(403, "オートモードの作業フォルダ確認が必要です。")
        try:
            coder_tools = CoderTools(
                req.workspace, req.coder_mode, auto_approved=req.auto_approved,
                request_text=text, memory_root=os.path.join(DATA, "project-memory"))
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"作業フォルダを開けません: {exc}") from exc
    else:
        coder_tools = None

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    activity_log: list[dict[str, str | int]] = []

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
    chat_doc["profile"] = req.profile
    chat_doc["coder_mode"] = req.coder_mode
    chat_doc["coder_effort"] = req.coder_effort
    chat_doc["coder_plan"] = []
    if coder_tools:
        workspace = str(coder_tools.workspace)
        previous_workspace = chat_doc.get("coder_workspace")
        if (previous_workspace
                and os.path.normcase(os.path.realpath(previous_workspace))
                != os.path.normcase(workspace)):
            raise HTTPException(
                409, "このチャットは別の作業フォルダを使っています。新しいチャットを開始してください。")
        chat_doc["coder_workspace"] = workspace

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
                def tool_started(name: str, argument: str) -> None:
                    detail = _coder_activity_detail(name, argument)
                    label = _TOOL_LABELS.get(name, name)
                    emit("status", text=f"{label}: {detail}" if detail else label)
                    entry: dict[str, str | int] = {
                        "index": len(activity_log), "name": name, "phase": "running",
                        "label": label, "detail": detail,
                    }
                    activity_log.append(entry)
                    emit("activity", entry=dict(entry))

                def tool_finished(name: str, argument: str, result: str) -> None:
                    for entry in reversed(activity_log):
                        if (entry.get("name") == name
                                and entry.get("detail") == _coder_activity_detail(name, argument)
                                and entry.get("phase") == "running"):
                            entry["phase"] = (
                                "done" if _coder_tool_succeeded(name, result) else "error")
                            entry["result"] = _coder_activity_result(name, result)
                            emit("activity", entry=dict(entry))
                            break

                agent.on_tool = tool_started
                agent.on_tool_result = tool_finished
                agent.on_progress = (
                    lambda message: emit("status", text=message)
                    if req.profile == "coder" else None)
                if isinstance(agent, CoderAgent):
                    memory_context = coder_tools.project_memory.prompt() if coder_tools else ""
                    agent.configure_request(
                        req.coder_mode, req.coder_effort,
                        coder_tools.registry() if coder_tools else default_registry(),
                        request_text=text, memory_context=memory_context)
                else:
                    agent.config.extra_system = None
                    agent.config.coder_auto = False
                    agent.config.max_tool_calls = 3
                    agent.config.max_new_tokens = (
                        1024 if req.profile == "quality" else
                        512 if req.profile == "fast" else 1024)
                    agent.config.max_continuations = 2
                    agent.config.temperature = (
                        0.5 if req.profile == "quality" else 0.7)
                    agent.config.stream = True
                    agent.config.preview = True
                    agent.tools = default_registry()
                if coder_tools:
                    def interact(kind: str, **data) -> str:
                        interaction_id = uuid.uuid4().hex
                        pending = PendingInteraction(kind=kind)
                        with _interaction_lock:
                            _pending_interactions[interaction_id] = pending
                        emit("interaction", interaction_id=interaction_id, kind=kind, **data)
                        try:
                            if not pending.event.wait(timeout=600):
                                return "ユーザーの応答がなくタイムアウトしました。"
                            return pending.response or (
                                "approved" if kind == "command" else "")
                        finally:
                            with _interaction_lock:
                                _pending_interactions.pop(interaction_id, None)

                    coder_tools.interact = interact
                    def publish_plan(tasks: list[dict[str, str]]) -> None:
                        chat_doc["coder_plan"] = tasks
                        emit("plan", tasks=tasks)

                    coder_tools.on_plan = publish_plan
                if not isinstance(agent, CoderAgent):
                    agent.config.thinking = bool(req.think or req.profile == "quality")
                agent.config.think_cap = (
                    2000 if req.profile == "coder" and req.coder_effort == "large"
                    else None)
                if req.image:
                    emit("status", text="画像を読んでいます")
                elif req.think and req.profile != "coder":
                    emit("status", text="考えています")
                elif req.profile == "fast":
                    emit("status", text="超高速モード")
                elif req.profile == "quality":
                    emit("status", text="高性能モードで回答を準備しています")
                elif req.profile == "coder":
                    emit("status", text={"plan": "作業フォルダを調査しています",
                                         "chat": "コードを確認しています",
                                         "auto": "依頼を確認し、作業手順を組み立てています"}[req.coder_mode])
                question = text or "この画像について説明してください。"
                answer, trace = agent.ask(question, image_ref=req.image)
                stopped = bool(getattr(eng, "cancel", False))
                elapsed = round(time.perf_counter() - started, 1)
                tools = [{"name": n, "arg": a} for n, a, _ in trace.tool_calls]
                files_changed = coder_tools.changed_files() if coder_tools else []
                if (coder_tools and req.coder_mode in ("auto", "plan")
                        and not coder_tools.memory_cleared):
                    checks = [
                        f"{entry.get('detail', '')}: {entry.get('result', '')}"
                        for entry in activity_log
                        if entry.get("name") in ("run_command", "check_python")
                        and entry.get("phase") == "done"
                    ]
                    try:
                        coder_tools.project_memory.record_task(
                            question, files_changed, checks,
                            plan=chat_doc.get("coder_plan", []), interrupted=stopped)
                    except (OSError, ValueError) as exc:
                        log.exception("could not save project memory")
                        answer += f"\n\nプロジェクト記憶の保存に失敗しました: {exc}"
                now = time.time()
                chat_doc["messages"].append({"role": "user", "content": question,
                                             "thumb": req.thumb, "time": now})
                chat_doc["messages"].append({"role": "assistant", "content": answer,
                                             "thinking": trace.thinking, "tools": tools,
                                             "files_changed": files_changed,
                                             "activity_log": activity_log,
                                             "coder_plan": chat_doc.get("coder_plan", []),
                                             "elapsed": elapsed, "stopped": stopped,
                                             "time": now})
                chat_doc["updated"] = now
                _save_chat(chat_doc)
                emit("done", answer=answer, thinking=trace.thinking, tools=tools,
                     files_changed=files_changed,
                     activity_log=activity_log,
                     coder_plan=chat_doc.get("coder_plan", []),
                     elapsed=elapsed, stopped=stopped)
            except Exception as e:  # shown in the window; the log gets the type only
                log.exception("generation failed")
                emit("error", text=f"{type(e).__name__}: {e}")
            finally:
                eng.echo_sink = eng.echo_guard = eng.echo_reset = None
                eng.cancel = False
                agent.on_thinking = agent.on_tool = None
                agent.on_tool_result = None
                agent.on_progress = None
                agent.config.coder_auto = False
                agent.config.extra_system = None
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
    with _interaction_lock:
        for pending in _pending_interactions.values():
            pending.response = (
                "denied" if pending.kind == "command"
                else "ユーザーが処理を停止しました。作業を中止してください。")
            pending.event.set()
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
        # Closing the window says goodbye (/api/bye) and quits at once, above —
        # that is the normal path. This heartbeat timeout is only the backstop
        # for a crash or kill that sent no goodbye, so it is generous: a
        # minimized or backgrounded window has its ping timer throttled by the
        # browser to about once a minute, and a short timeout quit the backend
        # while the window was merely in the background.
        if (last is None and now - STATE["started"] > 180) or (last and now - last > 300):
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
    profile = os.path.join(
        os.environ.get("LOCALAPPDATA") or APP_DATA_ROOT, "Rula",
        "RulaCoderEdgeProfile" if APP_MODE == "coder" else "EdgeProfile")
    proc = subprocess.Popen([edge, f"--app={url}", f"--user-data-dir={profile}", *EDGE_FLAGS])
    try:
        proc.wait(timeout=1.5)
        return None
    except subprocess.TimeoutExpired:
        return proc


def _app_process_handles():
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CreateEventW.argtypes = (
        wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateEventW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.SetEvent.argtypes = (wintypes.HANDLE,)
    ctypes.set_last_error(0)
    mutex = kernel.CreateMutexW(None, True, "Local\\RulaDesktopApp")
    already_running = ctypes.get_last_error() == 183
    event = kernel.CreateEventW(None, False, False, "Local\\RulaDesktopAppSwitch")
    if not mutex or not event:
        if mutex:
            kernel.CloseHandle(mutex)
        if event:
            kernel.CloseHandle(event)
        raise OSError(ctypes.get_last_error(), "Rulaアプリの起動制御を作成できません。")
    return {
        "kernel": kernel, "mutex": mutex, "event": event,
        "owner": not already_running,
    }


def _release_app_process_handles(handles) -> None:
    if not handles:
        return
    if handles["owner"]:
        handles["kernel"].ReleaseMutex(handles["mutex"])
    handles["kernel"].CloseHandle(handles["event"])
    handles["kernel"].CloseHandle(handles["mutex"])


def _read_running_app() -> tuple[str, str] | None:
    try:
        with open(RUN_FILE, encoding="utf-8") as source:
            value = source.read().strip()
    except OSError:
        return None
    if "|" in value:
        mode, url = value.split("|", 1)
    else:
        mode, url = "general", value
    return (mode, url) if mode in ("general", "coder") else None


def _ask_switch_app(active_mode: str) -> bool:
    if os.name != "nt":
        return False
    import ctypes
    active_name = "RulaCoder" if active_mode == "coder" else "Rula"
    app_name = "RulaCoder" if APP_MODE == "coder" else "Rula"
    message = (f"{active_name} が起動中です。\n\n"
               f"「はい」: {active_name} を終了して {app_name} を起動\n"
               f"「いいえ」: {active_name} を使い続ける")
    return ctypes.windll.user32.MessageBoxW(
        None, message, app_name + " の起動確認", 0x04 | 0x20) == 6


def _request_app_switch(url: str) -> None:
    import urllib.request
    endpoint = url.split("?", 1)[0].rsplit("/", 1)[0] + "/api/switch"
    token = url.split("?", 1)[1] if "?" in url else ""
    request = urllib.request.Request(
        endpoint + "?" + token, method="POST", data=b"")
    try:
        with urllib.request.urlopen(request, timeout=2):
            pass
    except OSError:
        log.warning("アプリ切替通知を送れませんでした。")


def run_app():
    """Start Rula and open its window; stop everything when the window closes."""
    handles = _app_process_handles()
    try:
        if handles and not handles["owner"]:
            running = _read_running_app()
            if running is None:
                wait_result = handles["kernel"].WaitForSingleObject(handles["mutex"], 0)
                if wait_result not in (0, 0x80):
                    log.error("起動中のRulaアプリを確認できません。")
                    return
                handles["owner"] = True
            else:
                active_mode, active_url = running
                if active_mode == APP_MODE:
                    if _alive(active_url):
                        _open_window(active_url)
                    return
                if not _ask_switch_app(active_mode):
                    return
                handles["kernel"].SetEvent(handles["event"])
                wait_result = handles["kernel"].WaitForSingleObject(handles["mutex"], 30000)
                if wait_result not in (0, 0x80):
                    log.error("起動中のRulaアプリを終了できませんでした。")
                    return
                handles["owner"] = True

        os.makedirs(DATA, exist_ok=True)
        if os.path.isfile(RUN_FILE):
            running = _read_running_app()
            if running and running[0] == APP_MODE and _alive(running[1]):
                _open_window(running[1])
                return
        port = _free_port()
        token = secrets.token_hex(24)
        url = f"http://127.0.0.1:{port}/?t={token}"
        _setup(port, token)
        with open(RUN_FILE, "w", encoding="utf-8") as f:
            f.write(APP_MODE + "|" + url)
        server = _server(port)
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(200):
            if handles and handles["kernel"].WaitForSingleObject(handles["event"], 0) == 0:
                _quit()
            if _alive(url):
                break
            time.sleep(0.05)
        window = _open_window(url)
        switching = False
        switch_deadline = 0.0
        while True:
            if handles and handles["kernel"].WaitForSingleObject(handles["event"], 0) == 0:
                switching = True
                switch_deadline = time.monotonic() + 15
                _request_app_switch(url)
            if window is not None and window.poll() is not None:
                _quit()
            if switching and time.monotonic() >= switch_deadline:
                if window is not None and window.poll() is None:
                    subprocess.run(
                        ["taskkill", "/PID", str(window.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=5, check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                _quit()
            time.sleep(0.25)
    finally:
        _release_app_process_handles(handles)


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
