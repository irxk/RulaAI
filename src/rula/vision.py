"""Image understanding for Rula.

An 8GB card cannot hold the chat model and a vision model at once, so the vision
model is loaded on demand and released as soon as it has described the image. The
description is then handed to the normal chat flow, which keeps one model resident
and one conversation history rather than splitting Rula in two.
"""
import gc
import os
import re
from contextlib import nullcontext as _nullcontext

from .engine import quiet_loading, resolve_weights

VISION_MODEL_NAME = "Rula1.0-vision-2b"
LOCAL_VISION_DIR = os.path.join("models", VISION_MODEL_NAME)
# Built by scripts/build_rula_model.py; until then fall back to the upstream weights.
UPSTREAM_VISION_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


def default_vision_model() -> str:
    return LOCAL_VISION_DIR if os.path.isdir(LOCAL_VISION_DIR) else UPSTREAM_VISION_MODEL


VISION_MODEL = default_vision_model()

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
# Quoted or bare paths, and http(s) URLs, ending in an image extension.
IMAGE_REF_RE = re.compile(
    r"""(?:"([^"]+?\.(?:png|jpe?g|webp|bmp|gif))"|'([^']+?\.(?:png|jpe?g|webp|bmp|gif))'"""
    r"""|(\S+\.(?:png|jpe?g|webp|bmp|gif)))""",
    re.I,
)

DEFAULT_QUESTION = "この画像に何が写っているか、日本語で具体的に説明してください。文字があれば読み取ってください。"


def find_image_ref(text: str) -> str | None:
    """Pull an image path or URL out of a user message."""
    m = IMAGE_REF_RE.search(text)
    if not m:
        return None
    ref = next(g for g in m.groups() if g)
    if ref.lower().startswith(("http://", "https://")) or os.path.exists(ref):
        return ref
    return None


class VisionReader:
    def __init__(self, model_id: str | None = None, device: str | None = None,
                 keep_loaded: bool = False, gpu_owner=None):
        # Resolved at call time so a directory built after import is still picked up.
        self.model_id = model_id or default_vision_model()
        self.keep_loaded = keep_loaded
        # Engine to displace while we run; both models on the card crashes the driver.
        self.gpu_owner = gpu_owner
        self._model = None
        self._processor = None
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self):
        if self._model is not None:
            return
        _require_vram()
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        # Registered directories hold a pointer into the shared cache, not weights.
        weights = resolve_weights(self.model_id)
        # use_fast is stated explicitly: leaving it unset makes transformers print a
        # warning about which processor it picked, naming its internal classes.
        self._processor = AutoProcessor.from_pretrained(weights, use_fast=True)
        # Deliberately not quantised. 4-bit made this model answer "there is no
        # image" instead of reading one, and excluding the image tower to fix that
        # left it at 4.42GB — the same as bf16. No saving, broken OCR.
        with quiet_loading():
            self._model = AutoModelForImageTextToText.from_pretrained(
                weights, dtype=torch.bfloat16, device_map=self.device)
        self._model.eval()
        # The checkpoint ships sampling defaults, and generate() warns that they are
        # ignored under do_sample=False. Drop them so the run is clean.
        for unused in ("temperature", "top_p", "top_k"):
            if hasattr(self._model.generation_config, unused):
                setattr(self._model.generation_config, unused, None)

    def _unload(self):
        if self._model is None:
            return
        import torch
        del self._model, self._processor
        self._model = self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def describe(self, ref: str, question: str | None = None, max_new_tokens: int = 320) -> str:
        try:
            image = _open_image(ref)
        except Exception as e:
            return f"画像を読み込めませんでした: {type(e).__name__}: {e}"

        try:
            import torch
            borrow = (self.gpu_owner.yield_gpu() if self.gpu_owner is not None
                      and self.device == "cuda" else _nullcontext())
            with borrow:
                self._load()
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question or DEFAULT_QUESTION},
                ]}]
                prompt = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = self._processor(text=[prompt], images=[image], return_tensors="pt")
                inputs = inputs.to(self._model.device)
                with torch.no_grad():
                    out = self._model.generate(**inputs, max_new_tokens=max_new_tokens,
                                               do_sample=False)
                text = self._processor.batch_decode(
                    out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                # Free the card before the chat model moves back onto it.
                if not self.keep_loaded:
                    self._unload()
            return text.strip() or "画像の内容を説明できませんでした。"
        except Exception as e:
            return f"画像の解析に失敗しました: {type(e).__name__}: {e}"
        finally:
            if not self.keep_loaded:
                self._unload()


class NotEnoughVRAM(RuntimeError):
    """Raised instead of letting a doomed load take the process with it."""


# The model itself is 4.43GB, but a load with 5.0GB free still segfaulted: the
# image tower's activations and the CUDA context need considerably more headroom
# than the weights alone. Set from the measurement, not from the file size.
_REQUIRED_VRAM_GB = 6.0


def _require_vram():
    """Refuse to load when the card is too full.

    Running out of VRAM here does not raise — the driver dies and the process
    segfaults, which on a long-running server means every other request dies too.
    Checking first turns that into an error the caller can report.

    nvidia-smi is the source of truth: torch.cuda.mem_get_info has been observed
    reporting 7.45GB free on this machine while nvidia-smi showed 4.1GB, and the
    load then crashed.
    """
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
        used, total = (int(x) for x in out.strip().splitlines()[0].split(","))
    except Exception:
        return  # can't tell; let the load proceed rather than block on a guess
    free_gb = (total - used) / 1024
    if free_gb < _REQUIRED_VRAM_GB:
        raise NotEnoughVRAM(
            f"GPUの空きが不足しています(空き {free_gb:.1f}GB / 必要 {_REQUIRED_VRAM_GB:.1f}GB)。"
            "他のアプリを閉じてから再試行してください。")


def _open_image(ref: str):
    import io

    from PIL import Image
    if ref.startswith("data:"):
        # OpenAI clients inline images as data:image/png;base64,... rather than
        # giving a URL the server could fetch.
        import base64
        header, _, payload = ref.partition(",")
        if not payload:
            raise ValueError("データURIに画像本体が含まれていません")
        raw = (base64.b64decode(payload) if "base64" in header
               else payload.encode("latin-1"))
        image = Image.open(io.BytesIO(raw))
    elif ref.lower().startswith(("http://", "https://")):
        import urllib.request
        req = urllib.request.Request(ref, headers={"User-Agent": "Mozilla/5.0 (Rula)"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        image = Image.open(io.BytesIO(data))
    else:
        image = Image.open(ref)
    image = image.convert("RGB")
    # Cap resolution: the model tiles large images into many visual tokens, which
    # is the difference between a few seconds and a stall on a small card.
    max_side = 1024
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    return image
