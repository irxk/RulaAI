"""Loopback-only OpenAI-compatible server for the RulaCoder EXL2 model."""
import gc
import importlib
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_DLL_DIRECTORY = None
if os.name == "nt":
    _DLL_DIRECTORY = os.add_dll_directory(
        str(Path(torch.__file__).resolve().parent / "lib")
    )
try:
    importlib.import_module("exllamav2_ext")
except ImportError as exc:
    raise RuntimeError(
        "ExLlamaV2 CUDA wheel is missing or incompatible. "
        "Run install-coder-v2.ps1."
    ) from exc

log = logging.getLogger("rula.coder_v2_server")


def _default_model_dir() -> str:
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if not local_app_data:
        return ""
    return os.path.join(
        local_app_data,
        "Programs",
        "TabbyAPI",
        "models",
        "Qwen2.5-Coder-7B-Instruct-4.0bpw-exl2",
    )


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=1024, gt=0, le=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    stream: bool = False
    response_format: dict | None = None


class ExLlamaV2Runtime:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.model = None
        self.cache = None
        self.tokenizer = None
        self.generator = None
        self.model_id = os.environ.get(
            "RULA_CODER_V2_MODEL",
            "Qwen2.5-Coder-7B-Instruct-4.0bpw-exl2",
        ).strip() or "Qwen2.5-Coder-7B-Instruct-4.0bpw-exl2"
        self.model_dir = os.environ.get(
            "RULA_CODER_V2_MODEL_DIR", _default_model_dir()
        ).strip()
        self.max_seq_len = int(os.environ.get(
            "RULA_CODER_V2_MAX_SEQ_LEN", "8192"
        ))

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        with self.lock:
            if self.loaded:
                return
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "ExLlamaV2 Coderには利用可能なNVIDIA CUDA GPUが必要です。"
                )
            if not self.model_dir or not Path(self.model_dir, "config.json").is_file():
                raise FileNotFoundError(
                    f"EXL2モデルのconfig.jsonが見つかりません: {self.model_dir}"
                )
            if not 1 <= self.max_seq_len <= 32768:
                raise ValueError("RULA_CODER_V2_MAX_SEQ_LEN は1〜32768が必要です。")

            from exllamav2 import (
                ExLlamaV2,
                ExLlamaV2Cache_Q4,
                ExLlamaV2Config,
                ExLlamaV2Tokenizer,
            )
            from exllamav2.generator import ExLlamaV2StreamingGenerator

            config = ExLlamaV2Config()
            config.model_dir = self.model_dir
            config.prepare()
            config.max_seq_len = self.max_seq_len
            model = ExLlamaV2(config)
            cache = ExLlamaV2Cache_Q4(
                model, max_seq_len=self.max_seq_len, lazy=True
            )
            try:
                model.load_autosplit(cache)
                tokenizer = ExLlamaV2Tokenizer(config)
                generator = ExLlamaV2StreamingGenerator(
                    model, cache, tokenizer
                )
                generator.set_stop_conditions([
                    tokenizer.single_id("<|im_end|>"),
                    tokenizer.eos_token_id,
                ])
            except Exception:
                model.unload()
                raise

            self.model = model
            self.cache = cache
            self.tokenizer = tokenizer
            self.generator = generator

    def unload(self) -> None:
        with self.lock:
            model = self.model
            if model is not None:
                model.unload()
            self.generator = None
            self.tokenizer = None
            self.cache = None
            self.model = None
            if model is not None:
                gc.collect()
                torch.cuda.empty_cache()

    @staticmethod
    def _format_messages(messages: list[ChatMessage]) -> str:
        prompt = "".join(
            f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
            for message in messages
        )
        return prompt + "<|im_start|>assistant\n"

    def generate(self, request: ChatCompletionRequest) -> tuple[str, str]:
        if request.model != self.model_id:
            raise HTTPException(status_code=404, detail="Unknown model ID")
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not supported by this Coder endpoint.",
            )
        if request.response_format is not None:
            raise HTTPException(
                status_code=400,
                detail="Structured response formats are not supported by ExLlamaV2.",
            )

        with self.lock:
            if not self.loaded or self.generator is None or self.tokenizer is None:
                raise HTTPException(
                    status_code=409, detail="The EXL2 model is not loaded."
                )
            prompt = self._format_messages(request.messages)
            input_ids = self.tokenizer.encode(
                prompt, add_bos=False, encode_special_tokens=True
            )
            context_tokens = input_ids.shape[-1]
            output_limit = min(
                request.max_tokens, self.max_seq_len - context_tokens
            )
            if output_limit < 1:
                raise HTTPException(
                    status_code=400,
                    detail="Prompt exceeds the configured model context length.",
                )

            from exllamav2.generator import ExLlamaV2Sampler

            settings = ExLlamaV2Sampler.Settings(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=50,
                token_repetition_penalty=1.05,
            )
            stops: list[int | str] = [
                self.tokenizer.single_id("<|im_end|>"),
                self.tokenizer.eos_token_id,
            ]
            if isinstance(request.stop, str):
                stops.append(request.stop)
            elif request.stop:
                stops.extend(request.stop)
            self.generator.set_stop_conditions(stops)
            self.generator.begin_stream(input_ids, settings)

            pieces: list[str] = []
            finished = False
            for _ in range(output_limit):
                text, finished, _ = self.generator.stream()
                if text:
                    pieces.append(text)
                if finished:
                    break
            return "".join(pieces), "stop" if finished else "length"


runtime = ExLlamaV2Runtime()
server = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.dll_directory = _DLL_DIRECTORY
    yield
    runtime.unload()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health")
def health() -> dict:
    return {
        "service": "rula-coder-exllamav2",
        "status": "ok",
        "model_loaded": runtime.loaded,
    }


@app.get("/v1/models")
def models() -> dict:
    return {
        "data": [{
            "id": runtime.model_id,
            "object": "model",
            "owned_by": "rula",
        }]
    }


@app.post("/v1/model/load")
def load_model() -> dict:
    try:
        runtime.load()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        log.exception("Could not load EXL2 model")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "loaded", "model": runtime.model_id}


@app.post("/v1/model/unload")
def unload_model() -> dict:
    runtime.unload()
    return {"status": "unloaded"}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict:
    content, finish_reason = runtime.generate(request)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": runtime.model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
    }


@app.post("/v1/shutdown")
def shutdown() -> dict:
    if server is not None:
        server.should_exit = True
    return {"status": "shutting_down"}


def main() -> None:
    import uvicorn

    global server
    port = int(os.environ.get("RULA_CODER_V2_SERVER_PORT", "5101"))
    if not 1 <= port <= 65535:
        raise ValueError("RULA_CODER_V2_SERVER_PORT must be between 1 and 65535.")
    server = uvicorn.Server(uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
    ))
    server.run()


if __name__ == "__main__":
    main()
