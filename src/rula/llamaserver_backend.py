"""llama.cpp's own server, run directly.

Ollama 0.40 already runs llama-server underneath — lib/ollama/llama-server.exe,
started with `-c 16384 -np 1 --flash-attn auto -b 1024 -ub 1024` — and proxies
every request to it. Running the same server ourselves removes that layer and
puts its flags in our hands. Speed per token is the same engine either way
(measured: 64.9 vs 65.9 tok/s); what running it directly buys is:

  - MTP speculative decoding (--spec-type draft-mtp): +30-45% decode speed,
    with the main model still deciding every token. See MTP below.
  - Images through the chat model itself (--mmproj). Qwen3.5 reads images
    natively; the projector is 644MB. The alternative was a separate 4.4GB
    vision model that needed 6GB free and evicted the chat model on every image.
  - A q8_0 KV cache (-201 MiB), which Ollama sets server-wide or not at all.

The server binds to 127.0.0.1 only, like the API itself.

Selected by build_engine() when the binary and weights are present.
"""
import base64
import http.client
import io
import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request

from .ollama_backend import THINK_CHAR_CAP, _cycling

log = logging.getLogger("rula.llamaserver")

# Where setup_llama.ps1 puts things, inside the app folder.
_APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNTIME = os.path.join(_APP, "runtime", "llama.cpp")
_GGUF = os.path.join(_APP, "models", "gguf")
# The development layout this was first built in: the folder that holds dist/.
_DEV = os.path.abspath(os.path.join(_APP, "..", ".."))


def _first(*paths: str) -> str:
    """The first of these that exists, else the first (for the error message)."""
    return next((p for p in paths if os.path.isfile(p)), paths[0])


SERVER_EXE = os.environ.get("RULA_LLAMA_SERVER") or _first(
    os.path.join(_RUNTIME, "llama-server.exe"),
    os.path.join(_DEV, "_llamacpp", "b11172", "llama-server.exe"))
# 9B first, chosen for quality. Unaided (no search), on ten knowledge and
# reasoning questions read and marked by hand: 9B 7/10, 4B 4/10 — the 4B put
# 北岳 in 北海道 and made 3 + 6 come to 7. Cost: 61 tok/s against 87-105, and
# ~5.9GB of VRAM against ~3.6GB. RULA_GGUF picks another file.
#
# Each size has an MTP build — the same HauhauCS weights with unsloth's MTP layer
# (blk.32) grafted on by scripts/graft_mtp.py (see NOTICE) — preferred over the
# plain file when present.
MODEL_PATH = os.environ.get("RULA_GGUF") or _first(
    os.path.join(_GGUF, "rula-9b-mtp-q4_k_m.gguf"),
    os.path.join(_DEV, "_mtp", "rula-9b-mtp-q4_k_m.gguf"),
    os.path.join(_GGUF, "rula-9b-q4_k_m.gguf"),
    os.path.join(_DEV, "_9b", "rula-9b-q4.gguf"),
    os.path.join(_GGUF, "rula-4b-mtp-q4_k_m.gguf"),
    os.path.join(_DEV, "_mtp", "rula-4b-mtp-q4_k_m.gguf"),
    os.path.join(_GGUF, "rula-4b-q4_k_m.gguf"),
    os.path.join(_DEV, "_q4", "rula-q4_k_m.gguf"))
# The projector has to match the model's width (2,560 for the 4B, 4,096 for the
# 9B); a mismatched one stops the server from starting. The plain 4B file's name
# carries no size, hence the default.
SIZE = "9b" if "9b" in os.path.basename(MODEL_PATH).lower() else "4b"
# Optional. Without it the server is text-only and Agent falls back to vision.py.
MMPROJ_PATH = os.environ.get("RULA_MMPROJ") or _first(
    os.path.join(_GGUF, f"mmproj-rula-{SIZE}-bf16.gguf"),
    os.path.join(_DEV, "_mmproj", f"mmproj-qwen35-{SIZE}-bf16.gguf"))
DISPLAY_NAME = os.environ.get("RULA_DISPLAY_NAME",
                              "Rula2.0-9b" if SIZE == "9b" else "Rula1.2-4b")
PORT = int(os.environ.get("RULA_LLAMA_PORT", "8081"))
HOST = f"http://127.0.0.1:{PORT}"
# Same budget as the Ollama path; see the measurement notes on NUM_CTX there.
NUM_CTX = int(os.environ.get("RULA_NUM_CTX", "16384"))
# q8_0 rather than the f16 Ollama runs. Measured on the 4B at 16,384 context:
# 3,198 MiB against 3,399 (-201 MiB), with prefill (2,752 vs 2,745 tok/s) and
# decode (64.6 vs 64.9 tok/s) unchanged. Only 8 of this hybrid model's 32 layers
# keep a KV cache at all, which is why the saving is not larger.
KV_TYPE = os.environ.get("RULA_KV_TYPE", "q8_0")
# Speculative decoding with the model's own MTP layer: it guesses the next two
# tokens and the main model checks them in one pass. Measured at temperature 0.7,
# free-form / grounded:
#   9B: 42.5 / 41.9 -> 62.8 / 61.0 tok/s, +489 MiB of VRAM
#   4B: 68.8 / 67.5 -> 93.5 / 87.3 tok/s, +413 MiB
# n=2 drafted tokens beat n=1 (9B 57.0/56.1, +275 MiB — the choice if VRAM is
# short) and matched n=3. The main model still decides every token: at
# temperature 0, wherever output with and without MTP differed, the main model's
# own top two candidates were within 0.023 of each other — rounding in the
# batched check picking the runner-up, never a clear winner overturned.
# A model without an MTP layer fails to start with this on; see _start.
MTP = os.environ.get("RULA_MTP", "1") == "1"
# Reuse the cached prefix of the previous request (the conversation so far).
CACHE_PROMPT = os.environ.get("RULA_CACHE_PROMPT", "1") == "1"
MTP_DRAFT = int(os.environ.get("RULA_MTP_DRAFT", "2"))

# The model's own sampling defaults, which Ollama applied from the Modelfile.
# Rula overrides temperature/top_p/repeat_penalty per request and has no opinion
# on these two, so they are sent as the author set them.
_MODEL_DEFAULTS = {"top_k": 20, "presence_penalty": 1.5}


def available() -> bool:
    """Whether the server is already up, or can be started from local files."""
    return _healthy() or (os.path.isfile(SERVER_EXE) and os.path.isfile(MODEL_PATH))


def _healthy(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{HOST}/health", timeout=timeout) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def _kill_with_parent(proc: subprocess.Popen):
    """Tie the server's lifetime to this process (Windows).

    run_api.ps1 stops the API with Stop-Process -Force, which runs no Python
    cleanup at all. Left alone, llama-server would survive it holding 3.2GB of
    VRAM. A job object with KILL_ON_JOB_CLOSE has the OS end it instead, however
    this process exits.
    """
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class _Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _Basic), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    info = _Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(wintypes.HANDLE(job), 9,  # ExtendedLimitInformation
                                ctypes.byref(info), ctypes.sizeof(info))
    handle = k32.OpenProcess(0x0101, False, proc.pid)  # TERMINATE | SET_QUOTA
    if handle and k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(handle)):
        _JOBS.append(job)  # the handle must outlive this call; closing it kills the server
    else:
        log.warning("llama-server を親プロセスに紐付けられませんでした (err %d)",
                    ctypes.get_last_error())
    if handle:
        k32.CloseHandle(wintypes.HANDLE(handle))


_JOBS: list = []


def _warn_if_vram_tight(floor_mib: int = 400):
    """Say so when the card is nearly full.

    On Windows the driver does not fail an allocation that does not fit: it moves
    it to system memory, and generation slows to a fraction without any error. The
    9B with MTP leaves 300-700 MiB free next to a typical desktop, so another app
    taking more is enough. A log line is the only place this would ever show.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10,
                             # no console flashing up behind the desktop app
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        free = int(out.strip().splitlines()[0])
    except Exception:
        return
    if free < floor_mib:
        log.warning("VRAM の空きが %d MiB しかありません。足りなくなると生成が極端に遅く"
                    "なります。GPU を使う他のアプリを閉じるか、RULA_MTP_DRAFT=1 / "
                    "RULA_MTP=0 / RULA_GGUF で 4B に切り替えてください。", free)


def server_args(model: str = MODEL_PATH, mmproj: str | None = MMPROJ_PATH,
                num_ctx: int = NUM_CTX, kv_type: str = KV_TYPE, port: int = PORT,
                mtp: bool = False) -> list[str]:
    args = [SERVER_EXE, "-m", model,
            "--host", "127.0.0.1", "--port", str(port),
            "-c", str(num_ctx), "-np", "1", "-ngl", "999",
            "--flash-attn", "on", "-b", "1024", "-ub", "1024",
            "--cache-type-k", kv_type, "--cache-type-v", kv_type,
            "--jinja", "--no-webui"]
    if mtp:
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(MTP_DRAFT)]
    if mmproj and os.path.isfile(mmproj):
        args += ["--mmproj", mmproj]
        # The projector stays on the CPU unless asked otherwise. On the GPU it is
        # held for as long as the server runs, for images that are occasional:
        #   4B: +760 MiB, an image in 1.1s instead of 6.2s.
        #   9B: +816 MiB, 4.2s instead of 8.7s — and with MTP that no longer fits
        #       an 8GB card next to a desktop. The process's shared (system)
        #       memory rose from 214 to 528 MiB: spilled, which nvidia-smi does
        #       not show. Leave this off for the 9B on 8GB.
        if os.environ.get("RULA_MMPROJ_GPU", "0") != "1":
            args.append("--no-mmproj-offload")
    return args


class LlamaServerEngine:
    """Same surface as OllamaEngine: generate(messages, cfg) -> str."""

    # GenConfig.stream prints the answer as it arrives; see _echo.
    echoes_stream = True

    def __init__(self, display_name: str = DISPLAY_NAME, start: bool = True,
                 load_timeout: float = 180.0):
        self.display_name = display_name
        self.device = "cuda"
        self.quantized = True
        self.backend = "llama-server"
        self.hit_token_limit = False  # set by generate(); see Engine.generate
        self.last_reasoning: str | None = None  # the reasoning for one turn
        self.last_timings: dict | None = None  # the server's own prefill/decode timings
        self.mtp: bool | None = None  # whether MTP drafting is on; None = server not ours
        # Where a streamed answer goes (GenConfig.stream). The terminal by default;
        # the desktop app points it at its window.
        self.echo_sink = None
        self.echo_guard = None  # see _echo
        self.echo_reset = None  # called when a new pass (not a continuation) starts showing
        # Set from another thread to stop the generation in progress (a Stop
        # button). Every pass checks it, so the turn winds down instead of
        # running its remaining retries and continuations.
        self.cancel = False
        self._proc: subprocess.Popen | None = None
        if not _healthy():
            if not start:
                raise RuntimeError(f"llama-server が {HOST} で応答しません。")
            self._start(load_timeout)
        props = self._get("/props")
        # A server that was already running may hold different weights — one left
        # over from a test, say. Answering as Rula from the wrong model is worse
        # than refusing to start.
        loaded = os.path.basename(props.get("model_path") or "")
        if loaded and loaded != os.path.basename(MODEL_PATH):
            raise RuntimeError(f"{HOST} の llama-server は別のモデル ({loaded}) を読み込んでいます。")
        # Read from the server rather than assumed: a model without a reasoning
        # channel, or a server started without --mmproj, must not be driven as
        # if it had one.
        template = props.get("chat_template") or ""
        self.supports_thinking = "enable_thinking" in template or "<think>" in template
        self.supports_vision = bool((props.get("modalities") or {}).get("vision"))

    # -- process ------------------------------------------------------------
    def _start(self, timeout: float):
        if not os.path.isfile(SERVER_EXE):
            raise RuntimeError(f"llama-server が見つかりません: {SERVER_EXE}")
        if not os.path.isfile(MODEL_PATH):
            raise RuntimeError(f"モデルが見つかりません: {MODEL_PATH}")
        for mtp in ((True, False) if MTP else (False,)):
            log.info("llama-server を起動します: %s%s", os.path.basename(MODEL_PATH),
                     " (MTP)" if mtp else "")
            # Its log goes nowhere: at this verbosity it would interleave with the
            # API's own output, and it never contains message text worth keeping.
            self._proc = subprocess.Popen(
                server_args(mtp=mtp), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            _kill_with_parent(self._proc)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    break
                if _healthy():
                    self.mtp = mtp
                    _warn_if_vram_tight()
                    return
                time.sleep(0.5)
            else:
                self.close()
                raise RuntimeError("llama-server の起動がタイムアウトしました。")
            # Exited during load. With MTP on, the usual cause is weights that have
            # no MTP layer ("failed to create MTP context"); start again without.
            if mtp:
                log.warning("MTP なしで起動し直します (このモデルには MTP 層が無いとみられます)。")
        raise RuntimeError(f"llama-server が終了しました (code {self._proc.returncode})。")

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    # -- generation ---------------------------------------------------------
    def generate(self, messages, cfg=None) -> str:
        from .engine import GenConfig
        cfg = cfg or GenConfig()

        chat = [{"role": m["role"], "content": m["content"]} for m in messages]
        if cfg.prefill:
            # A trailing assistant turn is continued, not replied to — what the
            # decline retry and the continuation of a cut-off answer rely on.
            chat.append({"role": "assistant", "content": cfg.prefill})

        payload = {
            "messages": chat,
            "stream": True,
            "max_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "repeat_penalty": cfg.repetition_penalty,
            **_MODEL_DEFAULTS,
            "cache_prompt": CACHE_PROMPT,
            # Always explicit: the template's default is to reason, and an
            # unrequested reasoning pass costs ~1,000 tokens on every reply.
            "chat_template_kwargs": {"enable_thinking": bool(cfg.think and self.supports_thinking)},
            "timings_per_token": False,
        }
        if cfg.stop:
            payload["stop"] = cfg.stop

        self._think_cap = cfg.think_cap or THINK_CHAR_CAP
        content, thinking, finish = self._request(
            payload, cfg.on_token, cfg.abort_check,
            echo=cfg.prefill if cfg.stream else None)
        self.last_reasoning = "".join(thinking).strip() or None
        self.hit_token_limit = finish == "length"
        text = "".join(content)
        # llama-server returns the prefill along with its continuation; Ollama
        # returns the continuation only, and callers append it to the prefill.
        # Measured: prefill "【結論】私が一番好きなのは" came back as
        # "【結論】私が一番好きなのは、「バナナ」です…".
        # The chat template may trim the prefill's trailing whitespace before
        # continuing it, so both forms are checked.
        for head in (cfg.prefill, cfg.prefill.rstrip()) if cfg.prefill else ():
            if head and text.startswith(head):
                text = text[len(head):]
                break
        return text.strip()

    def _request(self, payload: dict, on_token, abort_check, echo: str | None = None):
        """_stream, restarting the server once if it has died underneath us.

        A CUDA error or a driver reset ends the server process; without this,
        the API would answer every request after that with an error until someone
        restarted it by hand.
        """
        try:
            return self._stream(payload, on_token, abort_check, echo)
        except (urllib.error.URLError, ConnectionError, http.client.HTTPException):
            if self._proc is None or self._proc.poll() is None:
                raise  # not ours, or still running: a different problem
            log.warning("llama-server が停止していたため起動し直します (code %s)。",
                        self._proc.returncode)
            self._proc = None
            self._start(180.0)
            return self._stream(payload, on_token, abort_check, echo)

    def _echo(self, content: list[str], prefill: str, printed: int | None) -> int | None:
        """Print the answer as it arrives (GenConfig.stream, the CLI's --stream).

        The server sends a prefill back before continuing it, and when the
        prefill is the answer so far — the continuation of a cut-off reply —
        printing it again would repeat everything already on screen. Nothing is
        printed until it is clear whether the text opens with the prefill.

        echo_guard, when set, sees the new text before anything is shown and
        answers True (show), False (not this pass — it is turning into a tool
        call) or None (too early to tell). The engine knows nothing of tools;
        the caller that streams to a screen does.

        Returns how much of the text has been printed (None: not decided yet,
        -1: this pass is not to be shown).
        """
        if printed == -1:
            return -1
        joined = "".join(content)
        if printed is None:
            if prefill and len(joined) < len(prefill) and prefill.startswith(joined):
                return None
            start = len(prefill) if prefill and joined.startswith(prefill) else 0
            if self.echo_guard is not None:
                verdict = self.echo_guard(joined[start:])
                if verdict is None:
                    return None
                if verdict is False:
                    return -1
            # A fresh pass, not the continuation of the text already shown:
            # whatever an earlier pass streamed is no longer the answer.
            if start == 0 and self.echo_reset is not None:
                try:
                    self.echo_reset()
                except Exception:
                    pass
            printed = start
        if len(joined) > printed:
            if self.echo_sink is not None:
                try:
                    self.echo_sink(joined[printed:])
                except Exception:
                    pass  # a closed window must not end the generation
            else:
                import sys
                sys.stdout.write(joined[printed:])
                sys.stdout.flush()
        return len(joined)

    def _stream(self, payload: dict, on_token, abort_check, echo: str | None = None):
        """Read the SSE stream, forwarding pieces and enforcing the same caps as
        the Ollama path: reasoning length, reasoning loops, answer loops."""
        req = urllib.request.Request(
            f"{HOST}/v1/chat/completions", method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        content, thinking, finish = [], [], None
        think_len, since_check = 0, 0
        printed = None  # see _echo
        try:
            resp = urllib.request.urlopen(req, timeout=600.0)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"llama-server がエラーを返しました: {detail[:300]}") from e
        with resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if self.cancel:
                    finish = "cancelled"
                    break
                chunk = json.loads(data)
                if "timings" in chunk:
                    self.last_timings = chunk["timings"]
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                for field, kind, sink in (("reasoning_content", "thinking", thinking),
                                          ("content", "content", content)):
                    piece = delta.get(field)
                    if piece:
                        sink.append(piece)
                        if kind == "thinking":
                            think_len += len(piece)
                        if on_token is not None:
                            try:
                                on_token(kind, piece)
                            except Exception:
                                pass
                        if echo is not None and kind == "content":
                            printed = self._echo(content, echo, printed)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                    # The usage/timings chunk can follow; keep reading to [DONE].
                    continue
                if content:
                    if abort_check is not None:
                        try:
                            if abort_check("".join(content)):
                                log.info("ストリーミング品質チェックにより生成を中断")
                                finish = "quality_abort"
                                break
                        except Exception:
                            pass
                    continue
                if think_len >= getattr(self, "_think_cap", THINK_CHAR_CAP):
                    log.warning("思考が上限に達したため打ち切りました (%d字)。", think_len)
                    finish = "think_cap"
                    break
                since_check += 1
                if since_check >= 60:
                    since_check = 0
                    if _cycling("".join(thinking)[-1500:]):
                        log.warning("思考が同じ内容を繰り返しているため打ち切りました (%d字)。",
                                    think_len)
                        finish = "think_loop"
                        break
        # Closing the connection is what tells the server to stop generating.
        return content, thinking, finish

    # -- images -------------------------------------------------------------
    def describe_image(self, ref: str, question: str | None = None,
                       max_new_tokens: int = 320) -> str:
        """Read an image with the chat model itself. No second model, no swap."""
        from .vision import DEFAULT_QUESTION, _open_image
        try:
            image = _open_image(ref)
        except Exception as e:
            return f"画像を読み込めませんでした: {type(e).__name__}: {e}"
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": question or DEFAULT_QUESTION},
            ]}],
            "stream": True, "max_tokens": max_new_tokens, "temperature": 0.2,
            **_MODEL_DEFAULTS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            content, _, _ = self._request(payload, None, None)
        except Exception as e:
            return f"画像の解析に失敗しました: {type(e).__name__}: {e}"
        return "".join(content).strip() or "画像の内容を説明できませんでした。"

    def yield_gpu(self, needed_gb: float = 6.0):
        """Free the card for the separate vision model, then take it back.

        Only reached when this server has no projector: with --mmproj, images go
        through the chat model and Agent never calls this. Without one, Agent
        falls back to vision.py's 4.4GB model, which needs 6GB free — so a
        server we started is stopped for the duration and started again after.
        """
        from contextlib import contextmanager, nullcontext
        if self.supports_vision or self._proc is None:
            return nullcontext()

        @contextmanager
        def _swap():
            self.close()
            try:
                yield
            finally:
                self._start(180.0)
        return _swap()

    # -- helpers ------------------------------------------------------------
    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{HOST}{path}", timeout=10) as r:
            return json.loads(r.read())
