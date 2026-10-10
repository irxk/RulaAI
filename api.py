"""HTTP API for Rula.

Binds to localhost only. Reaching it from outside is Cloudflare Tunnel's job —
the tunnel connects outwards from this machine, so nothing has to be opened on
the router and the server is never directly exposed to the internet.

One GPU means one generation at a time; requests queue behind a lock rather than
competing for VRAM.
"""
import asyncio
import json
import logging
import os
import re
import hashlib
import secrets
import sys
import time
from collections import OrderedDict, defaultdict, deque

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from make_api_key import hash_key
from src.rula.agent import Agent, AgentConfig, Trace
from src.rula.engine import Engine, build_engine
from src.rula.memory import Memory
from src.rula.planner import wants_depth
from src.rula.tools import default_registry

log = logging.getLogger("rula.api")

# What OpenAI-compatible clients see in `model`. A stable public name: the field is
# never validated on the way in, so a client pinning an older value keeps working,
# and /health reports which weights and backend actually loaded.
MODEL_ALIAS = "rula-1.2-4b"
# The same weights with reasoning switched on. Offered as a second model because a
# model name is the only knob many OpenAI-compatible clients expose — several send
# no custom fields at all, and picking from a dropdown is the whole interface.
THINK_ALIAS = f"{MODEL_ALIAS}-think"
MAX_MESSAGE_CHARS = 4000
MAX_SYSTEM_CHARS = 4000
# Default answer budget. Length is governed by the persona, not by this cap —
# capping at 384 to force brevity produced thin answers, while the same prompt
# with a looser cap lands around 450 characters on its own. This is a ceiling to
# keep a runaway answer inside a client's timeout, not a target.
BRIEF_TOKENS = 768
# Raised only for turns that explicitly ask for depth.
DETAILED_TOKENS = 1536
# Reasoning is spent from the same budget as the answer, so this has to cover the
# reasoning *and* leave the reply intact.
#
# The earlier sizing was wrong, and wrong in the direction that hurts: it converted
# 6,500-9,600 characters of reasoning to "roughly 2,500-3,200 tokens", about
# 2.6-3.0 characters per token. Measured against this tokenizer on Japanese prose
# it is 1.77, so the same reasoning is 3,700-5,400 tokens — and at THINK_CHAR_CAP's
# 12,000 characters, 6,780.
#
# At 4096 the brief budget was 768+4096=4,864, and reasoning ran past the whole
# budget on two of four measured turns. One came back with empty content, so the
# reply had to be rebuilt by _write_up() from the notes — the path a raw <tool>
# call escaped through.
#
# 6144 puts the brief budget at 6,912 and the detailed one at 7,680, covering
# reasoning to ~10,900 characters with the answer's own room left over. It does not
# make the model reason longer — reasoning ends when it ends, or at THINK_CHAR_CAP
# — it stops the answer being cut off when it does. The cost is latency on the
# turns that use it: ~7,700 tokens at the measured 58 tok/s is about 130s, inside
# GENERATION_TIMEOUT and the bot's 330s read timeout.
#
# num_ctx has to stay ahead of this: 7,680 plus a ~4,000 token prompt is 11,680,
# which is why ollama_backend.NUM_CTX is 16384.
THINK_TOKENS = 6144
MAX_CONVERSATIONS = 200
MAX_HISTORY_TURNS = 12
RATE_LIMIT = 20          # requests per window
RATE_WINDOW = 60.0       # seconds
GENERATION_TIMEOUT = 300.0

app = FastAPI(title="Rula API", version="1.0",
              description="ローカルで動く日本語アシスタントのHTTP API")
# Registered under both /v1 and the root. Whether a client's base_url ends in /v1
# differs by tool, and it is the most common way to get a 404 here, so both work.
router = APIRouter()
_bearer = HTTPBearer(auto_error=False)

_engine: Engine | None = None
_lock = asyncio.Lock()
_agents: OrderedDict[str, Agent] = OrderedDict()
_calls: dict[str, deque[float]] = defaultdict(deque)

# How reasoning is wrapped when it is shown in the answer itself. A horizontal rule
# under it, because without a visible break the reasoning reads as the opening of
# the answer rather than as something separate.
DEFAULT_THINK_FORMAT = "💭 {}\n\n---\n\n"
# Written into the format to say where the reasoning goes. Several spellings, since
# which one a caller reaches for is a coin toss.
_THINK_PLACEHOLDERS = ("{}", "{thinking}", "{think}", "{reasoning}")
_THINK_OFF = ("", "off", "false", "no", "none", "hidden", "0")
_THINK_ON = ("on", "true", "yes", "1", "inline", "show", "stream")


def _think_format(value) -> str | None:
    """Resolve `think-stream` into a wrapper, or None to keep reasoning out of the
    answer.

    Deliberately liberal: a bool, a toggle word and a format string all mean
    something sensible here, and guessing wrong on the caller's behalf just makes
    the setting look broken.
    """
    if value is None or value is False:
        return None
    if value is True:
        return DEFAULT_THINK_FORMAT
    text = str(value)
    keyword = text.strip().lower()
    if keyword in _THINK_OFF:
        return None
    if keyword in _THINK_ON:
        return DEFAULT_THINK_FORMAT
    # Returned unstripped, deliberately. The surrounding whitespace is the layout:
    # stripping "…{}\n\n---\n\n" ate the blank line under the rule and ran the
    # answer straight into it.
    return text[:200]


def _split_think_format(fmt: str) -> tuple[str, str]:
    """The wrapper, as the text before the reasoning and the text after it.

    Split rather than substituted because streaming has to send the opening before
    the reasoning exists. Both paths go through here, so a streamed answer and a
    blocking one come out character for character identical.
    """
    for placeholder in _THINK_PLACEHOLDERS:
        if placeholder in fmt:
            before, _, after = fmt.partition(placeholder)
            return before, after
    # No placeholder: the string is a label, not a template. "【思考】" on its own is
    # the shape people actually type, and it should still produce something sane.
    return f"{fmt}\n", "\n\n---\n\n"


def _render_thinking(fmt: str, thinking: str) -> str:
    """The reasoning, wrapped, ready to sit in front of the answer."""
    before, after = _split_think_format(fmt)
    return before + thinking + after


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    conversation_id: str | None = Field(
        None, max_length=64,
        description="同じ値を渡すと会話が継続します。省略すると毎回新規。")
    reset: bool = Field(False, description="この会話の履歴を消してから答えます。")
    system: str | None = Field(
        None, max_length=MAX_SYSTEM_CHARS,
        description="追加のシステム指示。Rulaの既定の人格に追加されます。")
    think: bool = Field(
        False,
        description="考えてから答えます。推論・計算・比較検討に強くなりますが、"
                    "10秒ほど余分にかかります。")
    think_stream: str | bool | None = Field(
        None, alias="think-stream",
        description='思考の過程を answer 本文にも入れます。true / "on" で既定の書式、'
                    '文字列を渡すとその書式を使います(例: "<思考>{}</思考>\\n\\n")。'
                    "指定すると think は自動的に有効になります。")

    model_config = {"populate_by_name": True}


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    searched: list[str] = Field(default_factory=list,
                                description="この回答で実行した検索クエリ")
    elapsed_ms: int
    thinking: str | None = Field(
        None, description="think=true のときの思考過程。think-stream を指定しない限り "
                          "answer には含まれません。")


# --- OpenAI-compatible shapes -------------------------------------------------
# Most clients (openai-python, LangChain, LibreChat, SillyTavern) speak this and
# nothing else, so they get served here rather than asked to adapt.

class OpenAIMessage(BaseModel):
    role: str
    # Either a plain string or the "content parts" array that newer clients send:
    # [{"type": "text", "text": "..."}]. Rejecting the array is a 422 on the very
    # first message from several popular tools.
    content: str | list[dict] | None = None

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if not self.content:
            return ""
        parts = []
        for part in self.content:
            if not isinstance(part, dict):
                continue
            # "text" is the chat format, "input_text" the Responses one.
            if part.get("type") in ("text", "input_text"):
                parts.append(str(part.get("text") or ""))
        return "\n".join(p for p in parts if p)

    def image(self) -> str | None:
        """The first attached image, as a URL or a data: URI."""
        if not isinstance(self.content, list):
            return None
        for part in self.content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("image_url", "input_image"):
                url = part.get("image_url") or part.get("url")
                if isinstance(url, dict):
                    url = url.get("url")
                if isinstance(url, str) and url:
                    return url
        return None


class CompletionRequest(BaseModel):
    model: str = MODEL_ALIAS
    messages: list[OpenAIMessage]
    stream: bool = False
    # Three ways in, because no single one reaches every client: OpenAI's own field,
    # a plain boolean for clients that pass arbitrary JSON through, and the "-think"
    # model name for the many that let you choose nothing else.
    reasoning_effort: str | None = None
    think: bool | None = None
    # Shows the reasoning inside `content`, which is the only field many clients
    # render at all — a Discord bot or a minimal frontend drops reasoning_content
    # on the floor, so without this the thinking is invisible to them.
    think_stream: str | bool | None = Field(None, alias="think-stream")
    # Accepted and ignored: generation settings are fixed server-side, and
    # rejecting unknown fields would break clients that always send them.
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    user: str | None = None

    model_config = {"extra": "allow", "populate_by_name": True}

    def wants_thinking(self) -> bool:
        # Asking to see the reasoning is asking for reasoning. Requiring both flags
        # would just mean an empty block and a confused caller.
        if _think_format(self.think_stream) is not None:
            return True
        if self.think is not None:
            return self.think
        if self.reasoning_effort is not None:
            # "none" and "minimal" are the documented ways to ask for no reasoning;
            # low/medium/high all mean yes. Depth itself isn't adjustable here — the
            # model has one reasoning mode — so anything else is treated as on.
            return self.reasoning_effort.lower() not in ("none", "minimal", "off")
        name = self.model.lower()
        return name.endswith("-think") or name.endswith(":think")


# Turned away before the model sees them. A backstop, not the control: the weights
# are an "uncensored" fine-tune, so persona.HARD_LIMITS is what actually decides,
# and this catches only the plainest phrasings of the two categories where a
# generated refusal arriving late is worst.
#
# Each pattern needs the subject *and* a request to be told how, within a short
# distance. Asking what sarin is, reporting on an attack, or writing fiction that
# mentions one all still go through — refusing those would be both wrong and the
# fastest way to have the whole screen switched off.
_REFUSE_PATTERNS = (
    ("weapon", re.compile(
        # The generic words are here because the specific ones were not enough:
        # asked for "大量に人を傷つける化学剤の製造手順" the model did not refuse,
        # it asked which substance so it could continue.
        r"(爆弾|爆発物|起爆装置|時限発火|サリン|VXガス|神経ガス|毒ガス|生物兵器|"
        r"炭疽菌|ボツリヌス毒素|核兵器|ダーティボム|リシン|"
        r"化学兵器|化学剤|神経剤|猛毒|大量に人を(傷つけ|殺)|"
        r"殺傷できる|致死性の|致命的な(気体|ガス|毒))"
        r"[^。．\n]{0,50}"
        # "レシピ" is deliberately absent. It is an everyday cooking word, and
        # "爆弾みたいな辛さのカレーのレシピ" tripped the screen on it.
        r"(作り方|作る方法|作れる|製造方法|製造手順|合成方法|調合|精製方法|"
        r"手順を教え|自作する|組み立て方|段取り|用意する)")),
    ("minor", re.compile(
        r"(小学生|中学生|児童|幼女|幼児|未成年|[0-9０-９]{1,2}\s*歳の(子|少女|少年))"
        r"[^。．\n]{0,50}"
        r"(性行為|セックス|裸|全裸|陰部|性器|エロ|官能|猥褻|わいせつ|"
        r"性的な(描写|場面|シーン)|アダルト)")),
)

REFUSAL_TEXT = "その内容には答えられません。"


def _refused_category(message: str) -> str | None:
    """Which hard limit the message plainly asks to cross, if any."""
    for name, pattern in _REFUSE_PATTERNS:
        if pattern.search(message):
            return name
    return None


def _configured_key() -> tuple[str, str] | None:
    salt = os.environ.get("RULA_API_KEY_SALT")
    digest = os.environ.get("RULA_API_KEY_HASH")
    return (salt, digest) if salt and digest else None


async def require_key(request: Request,
                      creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    configured = _configured_key()
    if configured is None:
        raise HTTPException(503, "APIキーが未設定です。make_api_key.py を実行してください。")

    presented = creds.credentials if creds else request.headers.get("X-API-Key", "")
    salt, digest = configured
    # Constant-time: a plain == leaks how much of the key matched via timing.
    if not presented or not secrets.compare_digest(hash_key(presented, salt), digest):
        raise HTTPException(401, "APIキーが正しくありません。")

    _enforce_rate_limit(presented)
    return presented


def _enforce_rate_limit(key: str):
    now = time.monotonic()
    calls = _calls[hashlib.sha256(key.encode()).hexdigest()[:16]]
    while calls and now - calls[0] > RATE_WINDOW:
        calls.popleft()
    if len(calls) >= RATE_LIMIT:
        raise HTTPException(429, f"リクエストが多すぎます({RATE_LIMIT}件/{int(RATE_WINDOW)}秒)。")
    calls.append(now)


def _agent_for(conversation_id: str) -> Agent:
    if conversation_id in _agents:
        _agents.move_to_end(conversation_id)
        return _agents[conversation_id]
    if len(_agents) >= MAX_CONVERSATIONS:
        _agents.popitem(last=False)
    # No EmbeddingMemory here. It persists what was said, and this server proxies
    # other people's conversations — the same reason message contents are never
    # logged. In-history summaries (Agent._compress_history) cover long chats.
    _agents[conversation_id] = Agent(
        _engine,
        AgentConfig(thinking=False, candid=True, use_tools=True,
                    use_vision=True, use_imagegen=False, stream=False,
                    max_new_tokens=BRIEF_TOKENS, max_continuations=0),
        default_registry(),
        Memory(),
    )
    return _agents[conversation_id]


@app.on_event("startup")
def _startup():
    global _engine
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if _configured_key() is None:
        log.warning("APIキーが未設定です。make_api_key.py を実行してください。")
    log.info("モデルを読み込んでいます…")
    _engine = build_engine()
    log.info("準備完了: %s (%s)", _engine.display_name,
             getattr(_engine, "backend", "transformers"))


@app.get("/")
@app.get("/v1")
def root():
    """Returns the routes rather than a bare 404. Clients probe both / and /v1
    while working out a base_url, and a 404 there tells them nothing."""
    return {
        "service": "Rula API",
        "model": MODEL_ALIAS,
        "base_url": "どちらでも可: https://<ホスト>/v1  または  https://<ホスト>",
        "endpoints": {
            "openai_compatible": "POST /v1/chat/completions",
            "models": "GET /v1/models",
            "native": "POST /v1/chat",
            "health": "GET /health",
            "docs": "GET /docs",
        },
        "auth": "Authorization: Bearer <APIキー>",
        "thinking": {
            "model": f'model を "{THINK_ALIAS}" にする',
            "openai_field": 'reasoning_effort: "low" / "medium" / "high"',
            "native_field": "POST /v1/chat に think: true",
            "note": "思考過程は reasoning_content(/v1/chat では thinking)に入ります。"
                    "回答は10秒ほど遅くなります。",
            "show_thinking": '"think-stream": true で思考を回答本文にも表示。'
                             '書式を変えるなら "think-stream": "<思考>{}</思考>\\n\\n"。'
                             "指定すると思考は自動的に有効になります。",
        },
    }


@app.get("/health")
@app.get("/v1/health")
def health():
    return {
        "status": "ok" if _engine else "loading",
        "model": MODEL_ALIAS,
        # The alias above is a public name; these two say what is really running,
        # which is the first thing to check when speed or answers look wrong.
        "weights": _engine.display_name if _engine else None,
        "backend": getattr(_engine, "backend", "transformers") if _engine else None,
        # False means think:true still answers, but without the model's own
        # reasoning — worth seeing here rather than inferring it from timings.
        "thinking": getattr(_engine, "supports_thinking", False) if _engine else None,
    }


async def _generate(conversation_id: str, message: str, reset: bool,
                    image: str | None = None, system: str | None = None,
                    think: bool = False, on_thinking=None):
    """Run one turn. Shared by the native and OpenAI-compatible endpoints."""
    if _engine is None:
        raise HTTPException(503, "モデルを読み込み中です。")
    refused = _refused_category(message)
    if refused is not None:
        # Answered rather than errored: a 400 makes clients show a transport
        # failure, which reads as the service being broken rather than as a
        # refusal. Returned before the lock, so it never waits on the GPU.
        # The category is logged, not the message — see the chat handler.
        log.info("refused conv=%s category=%s", conversation_id[:8], refused)
        return REFUSAL_TEXT, Trace()
    try:
        async with _lock:  # a single GPU: serialise rather than contend
            agent = _agent_for(conversation_id)
            # Set per call and cleared in the finally below: agents are reused
            # across requests, and a stale callback would push this turn's
            # reasoning into a response that has already been sent.
            agent.on_thinking = on_thinking if think else None
            # Added to Rula's own prompt rather than replacing it, so a caller's
            # instructions do not silently drop the Japanese-only rule, the search
            # behaviour or the refusal handling underneath.
            agent.config.extra_system = _system_note(system) if system else None
            # Per request, not per conversation: the same client can ask for
            # reasoning on a hard question and not on the follow-up.
            agent.config.thinking = think
            # Length follows the request: brief unless this turn asked for depth.
            detailed = wants_depth(message)
            agent.config.max_new_tokens = DETAILED_TOKENS if detailed else BRIEF_TOKENS
            if think:
                # The reasoning is drawn from this same budget, so the answer only
                # gets what reasoning leaves behind.
                agent.config.max_new_tokens += THINK_TOKENS
            # Continuations multiply an already-large budget. Measured on the worst
            # case here — a question that searches, asks for depth and thinks — the
            # turn takes ~63s; two more passes on top is how that becomes a timeout.
            agent.config.max_continuations = (1 if think else 2) if detailed else 0
            if reset:
                agent.reset()
            _trim(agent)
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(agent.ask, message, image), GENERATION_TIMEOUT)
            finally:
                agent.on_thinking = None
    except asyncio.TimeoutError:
        raise HTTPException(504, "生成がタイムアウトしました。")
    except HTTPException:
        raise
    except Exception:
        log.exception("生成に失敗しました")
        raise HTTPException(500, "生成に失敗しました。")


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _key: str = Depends(require_key)):
    if _engine is None:
        raise HTTPException(503, "モデルを読み込み中です。")

    conversation_id = req.conversation_id or secrets.token_urlsafe(12)
    started = time.monotonic()
    fmt = _think_format(req.think_stream)
    answer, trace = await _generate(conversation_id, req.message, req.reset,
                                    system=req.system,
                                    think=req.think or fmt is not None)
    if fmt and trace.thinking:
        answer = _render_thinking(fmt, trace.thinking) + answer

    # The query is logged, not the message: this proxies someone's conversation.
    searched = [arg for name, arg, _ in trace.tool_calls if "search" in name]
    log.info("chat conv=%s %.1fs think=%s searched=%s", conversation_id[:8],
             time.monotonic() - started, bool(trace.thinking), searched or "-")
    # `thinking` stays populated either way: this endpoint returns fields, not a
    # rendered message, so having it twice costs nothing and losing it would.
    return ChatResponse(answer=answer, conversation_id=conversation_id,
                        searched=searched, thinking=trace.thinking,
                        elapsed_ms=int((time.monotonic() - started) * 1000))


@router.get("/models")
async def list_models(_key: str = Depends(require_key)):
    return {"object": "list", "data": [
        {"id": MODEL_ALIAS, "object": "model", "created": 0, "owned_by": "rula"},
        {"id": THINK_ALIAS, "object": "model", "created": 0, "owned_by": "rula"},
    ]}


@router.post("/chat/completions")
async def chat_completions(req: CompletionRequest, _key: str = Depends(require_key)):
    """OpenAI-compatible entry point.

    Conversation state lives here, not in the client: the prompt already carries
    history, so only the newest user turn is passed on and the `user` field (or a
    hash of the thread) keys the stored conversation.
    """
    if _engine is None:
        raise HTTPException(503, "モデルを読み込み中です。")

    latest = next((m for m in reversed(req.messages) if m.role == "user"), None)
    if latest is None:
        raise HTTPException(400, "user ロールのメッセージが必要です。")
    image = latest.image()
    message = latest.text().strip()[:MAX_MESSAGE_CHARS]
    if not message:
        if not image:
            raise HTTPException(400, "テキストか画像のどちらかが必要です。")
        message = "この画像について説明してください。"

    # Clients that resend the whole thread get a stable id from its opening turn,
    # so a continuing conversation lands on the same agent.
    conversation_id = req.user or _thread_id(req.messages)

    think = req.wants_thinking()
    # Reasoning goes in exactly one place. Asked to show it in the answer, it is
    # moved rather than copied — a client that renders both fields would otherwise
    # print the whole thing twice.
    fmt = _think_format(req.think_stream)
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _live_chunks(conversation_id, message, image,
                         _collect_system(req.messages), think, fmt,
                         req.model, created),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    answer, trace = await _generate(conversation_id, message, reset=False, image=image,
                                    system=_collect_system(req.messages), think=think)
    shown, aside = (None, trace.thinking) if fmt is None else (trace.thinking, None)
    if shown:
        answer = _render_thinking(fmt, shown) + answer
    return _completion_body(answer, conversation_id, created, message,
                            req.model, aside)


def _system_note(system: str) -> str:
    """Frame the caller's instructions so they outrank the defaults they conflict
    with — but say what they do not reach.

    The unqualified version of this ("prefer these where they conflict") handed a
    caller the whole prompt: persona.HARD_LIMITS was above it and therefore was
    something to be overridden. The limits are now appended after this block as
    well, and this says so, so an instruction to ignore them contradicts the text
    on both sides of it.
    """
    return ("\n呼び出し元からの指示(話し方・形式・役割については既定より優先する。"
            "ただし後述の禁止事項は免除されない):\n"
            f"---\n{system.strip()[:MAX_SYSTEM_CHARS]}\n---")


def _collect_system(messages: list[OpenAIMessage]) -> str | None:
    """All system turns, in order. Clients often send several."""
    parts = [m.text().strip() for m in messages
             if m.role in ("system", "developer") and m.text().strip()]
    return "\n\n".join(parts) if parts else None


def _thread_id(messages: list[OpenAIMessage]) -> str:
    first = next((m.text() for m in messages if m.role == "user" and m.text()), "")
    return "thread_" + hashlib.sha256(first.encode()).hexdigest()[:16]


def _completion_body(answer: str, conversation_id: str, created: int, prompt: str,
                     model: str = MODEL_ALIAS, thinking: str | None = None) -> dict:
    message = {"role": "assistant", "content": answer}
    if thinking:
        # Beside the answer, not inside it. "reasoning_content" is what DeepSeek and
        # vLLM settled on and what clients that render a collapsible "thinking"
        # section look for; anything else ignores the field and shows the answer.
        message["reasoning_content"] = thinking
    return {
        "id": f"chatcmpl-{conversation_id[:24]}",
        "object": "chat.completion",
        "created": created,
        # Echoed back, so a client that asked for "-think" sees it confirmed.
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }],
        # Character counts, not real tokens: clients display them, none depend on
        # the exact figure, and counting properly would mean another tokenise pass.
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(answer),
            "total_tokens": len(prompt) + len(answer),
        },
    }


async def _live_chunks(conversation_id: str, message: str, image: str | None,
                       system: str | None, think: bool, fmt: str | None,
                       model: str, created: int):
    """Stream the turn while it is still being generated.

    Generation runs as a task, not awaited up front, so reasoning can be forwarded
    the moment the model produces it — which is the difference between a client
    showing "考え中…" for ten seconds and showing nothing at all. The reasoning is
    the part worth watching: it is where nearly all the time goes, and it starts
    arriving about half a second in.

    The answer still goes out at the end, in one piece. Any single pass's answer
    text can be thrown away — it may turn out to be a tool call, or be replaced by
    a retry — and a chunk already sent cannot be recalled.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    finished = object()

    def on_thinking(chunk: str):
        # Runs on the generation thread; the queue must be touched on the loop's.
        loop.call_soon_threadsafe(queue.put_nowait, chunk)

    task = asyncio.ensure_future(
        _generate(conversation_id, message, reset=False, image=image,
                  system=system, think=think, on_thinking=on_thinking))
    task.add_done_callback(
        lambda _: loop.call_soon_threadsafe(queue.put_nowait, finished))

    head = {"id": f"chatcmpl-{conversation_id[:24]}", "object": "chat.completion.chunk",
            "created": created, "model": model}
    yield _sse({**head, "choices": [
        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

    # Where the reasoning goes decides which field carries it, but not the timing:
    # either way it is sent as it arrives.
    before, after = _split_think_format(fmt) if fmt else ("", "")
    field = "content" if fmt else "reasoning_content"
    opened = False

    while True:
        chunk = await queue.get()
        if chunk is finished:
            break
        if not opened:
            opened = True
            if before:
                yield _sse({**head, "choices": [
                    {"index": 0, "delta": {field: before}, "finish_reason": None}]})
        yield _sse({**head, "choices": [
            {"index": 0, "delta": {field: chunk}, "finish_reason": None}]})
    if opened and after:
        yield _sse({**head, "choices": [
            {"index": 0, "delta": {field: after}, "finish_reason": None}]})

    try:
        answer, _trace = await task
    except HTTPException as e:
        # The status line went out with the first chunk, so an error can only be
        # said in the body. Silence would look like a hung stream.
        yield _sse({**head, "choices": [
            {"index": 0, "delta": {"content": f"\n\n[エラー] {e.detail}"},
             "finish_reason": "stop"}]})
        yield "data: [DONE]\n\n"
        return

    for i in range(0, len(answer), 256):
        yield _sse({**head, "choices": [
            {"index": 0, "delta": {"content": answer[i:i + 256]},
             "finish_reason": None}]})
    yield _sse({**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    yield "data: [DONE]\n\n"


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.delete("/conversations/{conversation_id}")
async def forget(conversation_id: str, _key: str = Depends(require_key)):
    existed = _agents.pop(conversation_id, None) is not None
    return {"deleted": existed}


def _trim(agent: Agent):
    if len(agent.history) > MAX_HISTORY_TURNS * 2:
        del agent.history[:-MAX_HISTORY_TURNS * 2]


# Both spellings. include_router must come after every @router route is defined.
app.include_router(router, prefix="/v1")
app.include_router(router)


def main():
    import uvicorn
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_env()
    port = int(os.environ.get("RULA_API_PORT", "8000"))
    # 127.0.0.1, never 0.0.0.0: the tunnel reaches it locally, and binding wider
    # would expose the server to anything else on the network.
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


def _load_env():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


if __name__ == "__main__":
    main()
