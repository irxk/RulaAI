"""Ollama backend, used when the Rula model is registered there.

Ollama ships its own llama.cpp builds with runtime CPU dispatch, which is exactly
what the hand-installed llama-cpp-python wheels could not do on this machine: the
CUDA ones assume AVX512 and die on an illegal instruction on a 12th-gen i5. Same
engine underneath, without the build problem — so this is the fast path.

Selected automatically by build_engine(); nothing else in the codebase changes.
"""
import json
import logging
import os
import time
import urllib.error
import urllib.request

log = logging.getLogger("rula.ollama")

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not HOST.startswith("http"):
    HOST = f"http://{HOST}"

# The rebranded build, created by setup_ollama.py from Modelfile.rula. Only the
# name and self-identification differ from the upstream weights.
MODEL = os.environ.get("RULA_OLLAMA_MODEL", "rula-1.2-4b")
_MODEL_NAMES = {
    "rula-1.2-4b": "Rula1.2-4b",
    "rula-2.0-9b": "Rula2.0-9b",
}
DISPLAY_NAME = _MODEL_NAMES.get(MODEL, MODEL)

# Room for the whole turn, reasoning included. Ollama's own default is 4096.
#
# This has to cover generation, not just the prompt, because Ollama does not shift
# the cache when the window fills — it stops, with done_reason "length", mid
# sentence. Measured: at num_ctx 1024 a 200-line answer ended at exactly 1024
# total tokens, unfinished.
#
# Measured budget for one reasoning turn (字→token at 1.77, this tokenizer on
# Japanese): system prompt 1,188 · user input up to 2,254 · search results ~1,352
# · reasoning at THINK_CHAR_CAP 6,763 · answer 845 = 12,402 tokens. 8,192 cut two
# of three grounded reasoning turns short; 16,384 cut none, and answers that had
# been truncated came back whole (983 → 1,761 characters on the same question).
#
# The ceiling is VRAM, not appetite: on an 8GB card at Q6_K, 32,768 spills the
# compute buffer and generation drops from 42 to 24 tok/s. 16,384 costs +187MB
# over 8,192 and nothing in speed. Overridable because that ceiling is per-card.
NUM_CTX = int(os.environ.get("RULA_NUM_CTX", "16384"))

# A last resort against reasoning that will not terminate, not a routine limit.
# Set from measured traces: grounded turns legitimately think at length — 6,534,
# 7,009 and 9,584 characters when answering from search results, the last of those
# producing a 1,577-character answer that was worth the wait. A question it cannot
# settle ran to 12,988. There is no wide gap between those, so this sits above the
# longest good trace and below the runaway, and nothing else routinely reaches it.
#
# Two earlier attempts at something cleverer failed, and are recorded so they are
# not tried again:
#   - Sampling is not the cause. The runaway reproduces 4 times out of 4 at the
#     model's own temperature 1.0 / top_p 0.95 exactly as at ours.
#   - Novelty does not separate them. Measuring the share of each window's text
#     that had not appeared earlier, a legitimate search turn bottomed out at 0.18
#     while the runaway held 0.29 — every threshold cuts good reasoning first.
# What remains is verbatim repetition, which _cycling does catch, and length.
THINK_CHAR_CAP = int(os.environ.get("RULA_THINK_CHAR_CAP", "12000"))


def _post(path: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        f"{HOST}{path}", method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(path: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(f"{HOST}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cycling(tail: str, lo: int = 30, hi: int = 500) -> bool:
    """Whether the text ends in the same block repeated three times over.

    Three rather than two: two identical blocks happen legitimately — a repeated
    formula, a restated constraint — while three in a row is the model stuck.
    """
    for period in range(lo, hi + 1):
        if len(tail) < 3 * period:
            break
        if tail[-period:] == tail[-2 * period:-period] == tail[-3 * period:-2 * period]:
            return True
    return False


def installed() -> list[str]:
    """Model names Ollama has locally, or an empty list if it isn't running."""
    try:
        return [m["name"] for m in _get("/api/tags").get("models", [])]
    except Exception:
        return []


def available(model: str = MODEL) -> bool:
    """Whether Ollama is up and the Rula model is registered."""
    # Names come back as "model:tag"; a bare name means the :latest tag.
    return any(n == model or n.split(":")[0] == model for n in installed())


def capabilities(model: str = MODEL) -> list[str]:
    """What the registered model can do — "completion", "tools", "thinking"."""
    try:
        return _post("/api/show", {"model": model}, timeout=15).get("capabilities", [])
    except Exception:
        return []


class OllamaEngine:
    """Same surface as Engine: generate(messages, cfg) -> str."""

    def __init__(self, model: str = MODEL, display_name: str = DISPLAY_NAME,
                 num_ctx: int = NUM_CTX, keep_alive: str = "30m"):
        self.model = model
        self.display_name = display_name
        self.device = "cuda"
        self.quantized = True
        self.hit_token_limit = False  # set by generate(); see Engine.generate
        self.last_reasoning: str | None = None  # ditto; the reasoning for one turn
        self.backend = "ollama"
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self._supports_think = True  # cleared on the first model that rejects it
        # Fail during construction rather than on the first request, so that
        # build_engine() can fall back to the transformers path while starting up.
        if not available(model):
            raise RuntimeError(f"Ollama に {model} が登録されていません。")
        # Asked once, at startup: whether the weights were trained to reason. Agent
        # picks its thinking strategy from this, and getting it wrong is expensive
        # in both directions — a wasted reasoning pass, or a prompted one that this
        # model's template silently swallows.
        self.supports_thinking = "thinking" in capabilities(model)

    def generate(self, messages, cfg=None) -> str:
        from .engine import GenConfig
        cfg = cfg or GenConfig()

        chat = [{"role": m["role"], "content": m["content"]} for m in messages]
        if cfg.prefill:
            # A trailing assistant turn is continued rather than replied to. This is
            # what makes the decline-retry and the continuation of a cut-off answer
            # work: the model finishes text it has apparently already started.
            chat.append({"role": "assistant", "content": cfg.prefill})

        payload = {
            "model": self.model,
            "messages": chat,
            # Stream when: someone is listening, the model is going to reason,
            # or a quality monitor wants real-time checking.
            "stream": (cfg.on_token is not None or bool(cfg.think)
                       or cfg.abort_check is not None),
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "repeat_penalty": cfg.repetition_penalty,
                "num_ctx": self.num_ctx,
                # top_k and presence_penalty are deliberately absent: the model
                # publishes its own values for those and Rula has no opinion, so
                # leaving them out keeps the author's defaults.
                **({"stop": cfg.stop} if cfg.stop else {}),
            },
        }
        # Off unless the turn asked for it. Left on, the model reasons before every
        # reply — around 1,000 tokens and ~18s that most turns have no use for. The
        # flag is always sent explicitly: the model's own default is to reason, and
        # its reasoning lands in a separate "thinking" field, so omitting this
        # returns an empty "content" and a blank turn.
        if self._supports_think:
            payload["think"] = bool(cfg.think)

        body = (self._chat_streaming(payload, cfg.on_token, cfg.abort_check)
                if payload["stream"] else self._chat(payload))
        message = body.get("message") or {}
        text = (message.get("content") or "").strip()
        # Kept off to the side rather than prepended to the answer: callers that
        # want to show the working ask for it, and the rest must not have it
        # appear in the reply.
        self.last_reasoning = (message.get("thinking") or "").strip() or None
        self.hit_token_limit = body.get("done_reason") == "length"
        return text

    def _chat_streaming(self, payload: dict, on_token,
                         abort_check=None) -> dict:
        """Consume the NDJSON stream, forwarding each piece as it lands.

        Returns the same shape as the blocking call, assembled from the deltas, so
        the caller cannot tell which path ran. Reasoning arrives here token by token
        — measured at ~500 pieces starting 0.5s in — which is the whole point: it is
        the part worth watching, since it is where the time goes.
        """
        req = urllib.request.Request(
            f"{HOST}/api/chat", method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        content, thinking, done_reason = [], [], None
        think_len, since_check = 0, 0
        try:
            resp = urllib.request.urlopen(req, timeout=600.0)
        except urllib.error.HTTPError as e:
            return self._retry_without_think(payload, e)
        with resp:
            for raw in resp:  # one JSON object per line
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                message = chunk.get("message") or {}
                for field, sink in (("thinking", thinking), ("content", content)):
                    piece = message.get(field)
                    if piece:
                        sink.append(piece)
                        if field == "thinking":
                            think_len += len(piece)
                        # A callback that raises must not kill the generation: the
                        # answer is still wanted even if nobody can display it.
                        if on_token is not None:
                            try:
                                on_token(field, piece)
                            except Exception:
                                pass
                if chunk.get("done"):
                    done_reason = chunk.get("done_reason")
                    break
                # Content quality check — catches refusals and loops in the
                # answer itself, so the turn aborts early instead of spending
                # the full token budget on something that will be discarded.
                if content and abort_check is not None:
                    try:
                        if abort_check("".join(content)):
                            log.info("ストリーミング品質チェックにより生成を中断")
                            done_reason = "quality_abort"
                            break
                    except Exception:
                        pass
                # Only while still reasoning: once the answer has started, the
                # thinking is over and there is nothing left to cut short.
                if content:
                    continue
                if think_len >= THINK_CHAR_CAP:
                    # Logged because it is otherwise invisible: the turn still
                    # answers, just from reasoning that was cut off, and knowing
                    # how often that happens is how the cap gets tuned.
                    log.warning("思考が上限に達したため打ち切りました (%d字)。"
                                "回答は途中までの検討から書き起こします。", think_len)
                    done_reason = "think_cap"
                    break
                # Cheaper to catch than the cap, when it applies: a verbatim cycle
                # is already lost, and waiting out the rest of the budget only
                # spends time. Checked periodically because joining the pieces is
                # not free, and a loop stays a loop for another few hundred tokens.
                since_check += 1
                if since_check >= 60:
                    since_check = 0
                    if _cycling("".join(thinking)[-1500:]):
                        log.warning("思考が同じ内容を繰り返しているため打ち切りました "
                                    "(%d字)。", think_len)
                        done_reason = "think_loop"
                        break
        # Leaving the `with` closes the connection, which is how Ollama is told to
        # stop generating; without that the model would run to its budget anyway
        # and the cap would save nothing but the waiting.
        return {"message": {"content": "".join(content),
                            "thinking": "".join(thinking)},
                "done_reason": done_reason}

    def _retry_without_think(self, payload: dict, error) -> dict:
        detail = error.read().decode("utf-8", "replace")
        if "think" in detail and self._supports_think:
            self._supports_think = False
            payload.pop("think", None)
            return self._chat_streaming(payload, lambda *_: None)
        raise RuntimeError(f"Ollama がエラーを返しました: {detail[:300]}") from error

    def _chat(self, payload: dict) -> dict:
        try:
            return _post("/api/chat", payload)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            # Older builds, and models without the capability, reject the flag
            # outright. Drop it and retry rather than failing the request.
            if "think" in detail and self._supports_think:
                self._supports_think = False
                payload.pop("think", None)
                return _post("/api/chat", payload)
            raise RuntimeError(f"Ollama がエラーを返しました: {detail[:300]}") from e

    def yield_gpu(self, needed_gb: float = 6.0):
        """Free the card for the vision model.

        Ollama owns its own allocation, so the only lever is keep_alive: 0. The
        next generate() loads the model back — from the OS page cache, so it costs
        a couple of seconds rather than a fresh read from disk.
        """
        from contextlib import contextmanager

        @contextmanager
        def _swap():
            self.unload()
            try:
                yield
            finally:
                pass  # reloaded lazily by the next generate()
        return _swap()

    def unload(self, timeout: float = 20.0):
        """Drop the model from VRAM and wait until it is actually gone.

        The wait is the point. Unloading is asynchronous, and vision.py checks free
        VRAM with nvidia-smi the moment yield_gpu() enters — returning before the
        card is clear makes it refuse to load with "GPUの空きが不足しています".
        """
        try:
            _post("/api/generate", {"model": self.model, "keep_alive": 0}, timeout=30)
        except Exception:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._resident():
                return
            time.sleep(0.3)

    def close(self):
        """Release this model when its owning app is closing."""
        self.unload()

    def _resident(self) -> bool:
        try:
            running = _get("/api/ps").get("models", [])
        except Exception:
            return False  # can't tell; don't block on a guess
        stem = self.model.split(":")[0]
        return any((m.get("name") or "").split(":")[0] == stem for m in running)
