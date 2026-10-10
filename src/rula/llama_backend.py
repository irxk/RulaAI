"""llama.cpp backend, used when a GGUF model is available.

Roughly three to four times faster than the bitsandbytes 4-bit path for the same
model, which is what makes longer answers viable inside a caller's timeout.
Selected automatically by Engine; nothing else in the codebase changes.
"""
import os

GGUF_DIR = os.path.join("models", "gguf")
GGUF_REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
GGUF_FILE = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"


def _enable_cuda_dlls():
    """Put the CUDA runtime on the DLL search path.

    The llama-cpp-python wheel ships ggml-cuda.dll but not cudart/cublas, so
    importing it fails with a bare "could not find module". PyTorch bundles the
    same runtime, and pointing at that is enough.
    """
    if os.name != "nt":
        return
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
    except Exception:
        pass  # non-CUDA install; llama.cpp will fall back to CPU or fail loudly


def available() -> bool:
    """Whether llama.cpp can actually run here, weights included."""
    if find_gguf() is None:
        return False
    _enable_cuda_dlls()
    try:
        import llama_cpp  # noqa: F401
        return True
    except Exception:
        return False


def find_gguf() -> str | None:
    """A .gguf placed locally, or one already in the Hugging Face cache."""
    if os.path.isdir(GGUF_DIR):
        for name in sorted(os.listdir(GGUF_DIR)):
            if name.endswith(".gguf"):
                return os.path.join(GGUF_DIR, name)
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(GGUF_REPO, GGUF_FILE, local_files_only=True)
    except Exception:
        return None


class LlamaEngine:
    """Same surface as Engine: generate(messages, cfg) -> str."""

    # No reasoning parser here: a <think> block would come back inside the answer
    # rather than beside it, so Agent prompts for one and strips it instead.
    supports_thinking = False

    def __init__(self, model_path: str | None = None, display_name: str = "Rula1.1-4b",
                 n_ctx: int = 8192):
        _enable_cuda_dlls()
        from llama_cpp import Llama

        self.model_path = model_path or find_gguf()
        self.display_name = display_name
        self.device = "cuda"
        self.quantized = True
        self.hit_token_limit = False
        self.last_reasoning: str | None = None  # never set; see supports_thinking
        self.backend = "llama.cpp"
        self._llama = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,   # every layer on the GPU; a 4B Q4 fits in 8GB
            verbose=False,
        )

    def generate(self, messages, cfg=None) -> str:
        from .engine import GenConfig
        cfg = cfg or GenConfig()

        # Prefill is expressed by appending an assistant turn that the model is
        # meant to continue, which llama.cpp's chat handler supports directly.
        chat = [{"role": m["role"], "content": m["content"]} for m in messages]
        if cfg.prefill:
            chat.append({"role": "assistant", "content": cfg.prefill})

        out = self._llama.create_chat_completion(
            messages=chat,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repeat_penalty=cfg.repetition_penalty,
            stop=cfg.stop or None,
        )
        choice = out["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        self.hit_token_limit = choice.get("finish_reason") == "length"
        return text

    def yield_gpu(self, needed_gb: float = 6.0):
        """Free the card for the vision model, then reload.

        llama.cpp holds its weights outside the torch allocator, so the only way
        to release them is to drop the model and read it back.
        """
        from contextlib import contextmanager

        @contextmanager
        def _swap():
            path, name, ctx = self.model_path, self.display_name, self._llama.n_ctx()
            del self._llama
            self._llama = None
            import gc
            gc.collect()
            try:
                yield
            finally:
                from llama_cpp import Llama
                self._llama = Llama(model_path=path, n_ctx=ctx,
                                    n_gpu_layers=-1, verbose=False)
        return _swap()
