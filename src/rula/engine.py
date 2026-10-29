"""Model loading and text generation for Rula.

Wraps a pretrained Hugging Face causal LM (default Qwen2.5-1.5B-Instruct) plus an
optional LoRA adapter, and exposes chat-style generation with optional streaming.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

import gc
import json
import logging
import os
from contextlib import contextmanager

# torch and transformers are imported inside Engine, not here. The Ollama path —
# which is what actually runs — needs neither, and importing them at module level
# made every caller of GenConfig pay for them: ~1GB of RSS, seconds of startup,
# and a hard crash whenever an unrelated package broke their dependency chain
# (measured: an httpx downgrade by googletrans took the whole API down this way).


@contextmanager
def quiet_loading():
    """Silence the weight-loading progress bar.

    It streams internal parameter names ("model.visual.patch_embed.proj.weight")
    into the middle of the conversation, which is noise to someone chatting.
    """
    from transformers.utils import logging as hf_logging
    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        hf_logging.enable_progress_bar()
        hf_logging.set_verbosity(level)

# The fallback is the pre-quantised build, not the fp16 original: a machine that
# runs this bot has 8GB of VRAM, and pointing it at a 16GB download it cannot load
# is a worse failure than any saving in fidelity.
MODELS = {
    "4b": {"name": "Rula1.1-4b", "dir": os.path.join("models", "Rula1.1-4b"),
           "upstream": "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit", "quantize": True},
}
_PREFERENCE = ("4b",)


def default_model() -> str:
    for key in _PREFERENCE:
        if os.path.isdir(MODELS[key]["dir"]):
            return MODELS[key]["dir"]
    return MODELS[_PREFERENCE[-1]]["upstream"]


def _spec_for(model_id: str) -> dict | None:
    target = os.path.abspath(model_id)
    for spec in MODELS.values():
        if target.startswith(os.path.abspath(spec["dir"])) or model_id == spec["upstream"]:
            return spec
    return None


def display_name(model_id: str) -> str:
    """What to show the user. A local build is Rula's own model, so name it that."""
    spec = _spec_for(model_id)
    return spec["name"] if spec and os.path.isdir(model_id) else model_id


def wants_quantization(model_id: str) -> bool:
    """Whether we must apply 4-bit ourselves. A checkpoint that was saved already
    quantised carries its own config, and re-applying one would fail."""
    if _already_quantized(model_id):
        return False
    spec = _spec_for(model_id)
    return bool(spec and spec["quantize"])


def resolve_weights(model_id: str) -> str:
    """Follow a source.json pointer to where the weights actually live.

    Some builds only register a model (attribution files plus a pointer) because
    their weights cannot be re-saved locally; see scripts/build_rula_model.py.
    """
    pointer = os.path.join(model_id, "source.json")
    if os.path.isfile(pointer):
        try:
            with open(pointer, encoding="utf-8") as f:
                spec = json.load(f)
            # local_dir when the weights already exist elsewhere on this machine,
            # repo_id when they live in the shared Hugging Face cache.
            return spec.get("local_dir") or spec["repo_id"]
        except (OSError, ValueError, KeyError):
            pass
    return model_id


def _already_quantized(model_id: str) -> bool:
    pointer = os.path.join(model_id, "source.json")
    if os.path.isfile(pointer):
        try:
            with open(pointer, encoding="utf-8") as f:
                spec = json.load(f)
        except (OSError, ValueError):
            return True
        if "local_dir" in spec:
            # Points at real files; judge them the same way as any other directory.
            return _already_quantized(spec["local_dir"])
        return bool(spec.get("prequantized", True))
    config_path = os.path.join(model_id, "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, encoding="utf-8") as f:
            return "quantization_config" in json.load(f)
    except (OSError, ValueError):
        return False


DEFAULT_MODEL = default_model()


@dataclass
class GenConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    stop: list[str] = field(default_factory=list)
    # Text forced into the start of the assistant turn. Small models won't reliably
    # open a <think> block on request, but they will continue one that's already open.
    prefill: str = ""
    # Print tokens as they arrive. A long write-up takes over a minute at 16 tok/s,
    # which feels like a hang if nothing appears until it is done.
    stream: bool = False
    # Let the model reason before answering, using the reasoning it was trained for.
    # Only honoured by backends that advertise it (see Engine.supports_thinking);
    # elsewhere it is ignored and Agent falls back to prompting for a <think> block.
    # The reasoning is never part of the returned text — it is left on the engine as
    # `last_reasoning`, the same side-channel `hit_token_limit` uses.
    think: bool = False
    # Called with ("thinking" | "content", chunk) as each piece arrives, on backends
    # that can stream. generate() still returns the whole text afterwards, so this
    # is additive: a caller that ignores it sees no difference.
    on_token: "Callable[[str, str], None] | None" = None
    # Called with accumulated content so far. Returns True to abort generation.
    abort_check: "Callable[[str], bool] | None" = None
    # Stop reasoning after this many characters (None: the backend's own cap).
    think_cap: int | None = None
    response_format: dict | None = None


def build_engine(model_id: str | None = None, adapter: str | None = None,
                 device: str | None = None, quantize: bool | None = None,
                 prefer_fast: bool = True):
    """Build the Main Rula engine, using TabbyAPI by default."""
    load_env_file()
    backend = os.environ.get("RULA_MAIN_BACKEND", "tabbyapi").strip().lower()
    legacy_backend = os.environ.get("RULA_BACKEND", "").strip().lower()
    if legacy_backend and legacy_backend != backend:
        logging.getLogger("rula.engine").warning(
            "RULA_BACKEND=%s は旧設定のため無視します。RULA_MAIN_BACKEND=%s を使用します。",
            legacy_backend,
            backend,
        )
    if adapter:
        raise RuntimeError(
            "LoRA adapter指定はこの推論APIでは使用できません。"
        )
    if device is not None or quantize is not None:
        raise RuntimeError(
            "device/quantize指定はこの推論APIでは使用できません。"
            "モデルと実行設定は推論サーバー側で選択してください。"
        )
    if backend == "ollama":
        from .ollama_backend import OllamaEngine
        model = model_id or os.environ.get(
            "RULA_MAIN_OLLAMA_MODEL", "llama3.1:8b"
        ).strip()
        return OllamaEngine(
            model=model,
            display_name=(
                "Llama 3.1 8B Instruct"
                if model == "llama3.1:8b" else model
            ),
            num_ctx=int(os.environ.get("RULA_MAIN_NUM_CTX", "8192")),
            keep_alive=os.environ.get("RULA_MAIN_KEEP_ALIVE", "0"),
        )
    if backend == "strata":
        return _build_strata_engine(model_id)
    if backend == "tabbyapi":
        from .tabbyapi_backend import TabbyAPIEngine
        return TabbyAPIEngine(model_id)
    raise RuntimeError(
        "RULA_MAIN_BACKEND は tabbyapi、ollama、または strata を指定してください。"
    )


def _build_strata_engine(model_id: str | None = None):
    if model_id:
        requested = os.environ.get("RULA_STRATA_MODEL", "").strip()
        if requested and requested != model_id:
            raise RuntimeError(
                "model_id と RULA_STRATA_MODEL が異なります。"
                "Strataで使うモデルIDを RULA_STRATA_MODEL に一つだけ指定してください。"
            )
        os.environ.setdefault("RULA_STRATA_MODEL", model_id)
    from .strata_backend import StrataEngine
    return StrataEngine()


def build_coder_engine():
    """Build the dedicated Coder model, independently of the main model."""
    load_env_file()
    backend = os.environ.get("RULA_CODER_BACKEND", "exllamav2").strip().lower()
    if backend == "ollama":
        from .ollama_backend import OllamaEngine
        model = os.environ.get(
            "RULA_CODER_OLLAMA_MODEL", "qwen2.5-coder:7b"
        ).strip()
        return OllamaEngine(
            model=model,
            display_name=(
                "Qwen2.5-Coder-7B-Instruct"
                if model == "qwen2.5-coder:7b" else model
            ),
            num_ctx=int(os.environ.get("RULA_CODER_NUM_CTX", "8192")),
            keep_alive=os.environ.get("RULA_CODER_KEEP_ALIVE", "0"),
        )
    if backend == "strata":
        return _build_strata_engine()
    if backend == "exllamav2":
        from .coder_v2_backend import ExLlamaV2Engine
        return ExLlamaV2Engine()
    if backend == "tabbyapi":
        from .tabbyapi_backend import TabbyAPIEngine
        model = (
            os.environ.get("RULA_CODER_TABBYAPI_MODEL", "").strip()
            or "Qwen2.5-Coder-7B-Instruct-exl3"
        )
        return TabbyAPIEngine(
            model=model,
            model_env="RULA_CODER_TABBYAPI_MODEL",
            settings_prefix="RULA_CODER_TABBYAPI",
            display_name=os.environ.get(
                "RULA_CODER_TABBYAPI_DISPLAY_NAME",
                "Qwen2.5-Coder-7B-Instruct",
            ).strip(),
        )
    raise RuntimeError(
        "RULA_CODER_BACKEND は exllamav2、tabbyapi、ollama、または strata "
        "を指定してください。"
    )


def load_env_file(path: str | None = None) -> None:
    """Load Rula's local .env settings without overriding process environment."""
    env_path = path or os.path.join(os.getcwd(), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class Engine:
    # These weights have no reasoning mode of their own, so GenConfig.think is
    # ignored here and Agent prompts for a <think> block instead.
    supports_thinking = False
    # GenConfig.stream prints the answer as it is generated (TextStreamer).
    echoes_stream = True

    def __init__(self, model_id: str | None = None, adapter: str | None = None,
                 device: str | None = None, quantize: bool | None = None):
        import torch
        from transformers import AutoTokenizer
        model_id = model_id or default_model()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.display_name = display_name(model_id)
        prequantized = _already_quantized(model_id)
        if quantize is None:
            quantize = wants_quantization(model_id)
        self.quantized = prequantized or (bool(quantize) and self.device == "cuda")
        self.hit_token_limit = False  # set by generate(); see its docstring
        self.last_reasoning: str | None = None  # never set here; see supports_thinking
        self._weights = resolve_weights(model_id)
        self._adapter = adapter

        self.tokenizer = AutoTokenizer.from_pretrained(self._weights)
        self._load_kwargs = {"device_map": self.device}
        if not prequantized:
            # A pre-quantised checkpoint brings its own dtypes; forcing one errors.
            self._load_kwargs["dtype"] = torch.bfloat16
        if self.quantized and not prequantized:
            from transformers import BitsAndBytesConfig
            # NF4 with double quantisation: ~5.5GB for a 7B, which is what lets it
            # run at all on an 8GB card.
            self._load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self._load_model()

    def _load_model(self):
        from transformers import AutoModelForCausalLM
        with quiet_loading():
            self.model = AutoModelForCausalLM.from_pretrained(
                self._weights, **self._load_kwargs)
        if self._adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self._adapter)
            self.model = self.model.merge_and_unload()
        self.model.eval()

    @contextmanager
    def yield_gpu(self, needed_gb: float = 6.5):
        """Free the card for another model, then take it back — unless there is
        room to spare.

        Measured on an 8GB RTX 4060, where the vision model needs 4.43GB: leaving
        the 4-bit 4B resident frees 4.79GB, which nominally fits — but loading the
        pair took 22s and then 37s, against 10.6s and 7.3s when the chat model steps
        aside. Allocating into the last few hundred MB costs far more than the swap
        saves, so the threshold demands real headroom rather than a bare fit.
        Bigger cards skip the swap; 8GB ones always take turns.

        A plain model can be parked on the CPU and moved back. bitsandbytes layers
        cannot, so a quantised model is dropped entirely and re-read from the local
        cache afterwards — slower, but it is the only option that works.
        """
        import torch
        if self.device != "cuda":
            yield
            return
        if torch.cuda.mem_get_info()[0] / 1e9 >= needed_gb:
            yield  # genuinely spacious; swapping would only cost time
            return
        if self.quantized:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            try:
                yield
            finally:
                self._load_model()
            return
        self.model.to("cpu")
        torch.cuda.empty_cache()
        try:
            yield
        finally:
            self.model.to("cuda")
            torch.cuda.empty_cache()

    def _prepare(self, messages, prefill: str = ""):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text + prefill, return_tensors="pt").to(self.model.device)

    def _gen_kwargs(self, cfg: GenConfig):
        return dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            # Pre-allocating the cache instead of growing it measured ~9% faster
            # decoding (13.7 -> 14.9 tok/s) at no cost in output.
            cache_implementation="static",
        )

    def generate(self, messages, cfg: GenConfig | None = None) -> str:
        """Generate a reply. `hit_token_limit` afterwards says whether the model
        was still mid-sentence when it ran out of budget, so the caller can decide
        to continue rather than shipping a truncated answer."""
        import torch
        from transformers import TextStreamer
        cfg = cfg or GenConfig()
        inputs = self._prepare(messages, cfg.prefill)
        prompt_len = inputs["input_ids"].shape[1]
        streamer = (TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                    if cfg.stream else None)
        with torch.no_grad():
            out = self.model.generate(**inputs, **self._gen_kwargs(cfg), streamer=streamer)
        produced = out[0, prompt_len:]
        text = self.tokenizer.decode(produced, skip_special_tokens=True)

        stopped_early = _first_stop(text, cfg.stop) is not None
        ended_on_eos = bool(produced.numel()) and produced[-1].item() in self._eos_ids
        self.hit_token_limit = (produced.numel() >= cfg.max_new_tokens
                                and not ended_on_eos and not stopped_early)
        return _apply_stops(text.strip(), cfg.stop)

    @property
    def _eos_ids(self) -> set[int]:
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        return set(eos) if isinstance(eos, (list, tuple)) else {eos}


def _first_stop(text: str, stops: list[str]) -> int | None:
    positions = [text.find(s) for s in stops if s and s in text]
    return min(positions) if positions else None


def _apply_stops(text: str, stops: list[str]) -> str:
    cut = _first_stop(text, stops)
    return text[:cut].strip() if cut is not None else text
