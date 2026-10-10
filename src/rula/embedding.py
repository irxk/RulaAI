"""Vector memory using sentence-transformers for semantic retrieval.

The keyword-based Memory catches explicit self-introductions, but misses everything
else the user said — their interests, what was discussed, context from earlier turns.
Embedding similarity retrieves by meaning, so "前に話したPythonの件" pulls up the
turn where Python was discussed, even if the word "Python" doesn't appear in the
stored fact.

The embedding model runs on CPU (< 500MB RAM) to avoid competing with the LLM on GPU.
Falls back gracefully when sentence-transformers is not installed.
"""
import json
import logging
import os
from datetime import datetime

log = logging.getLogger("rula.embedding")

_DEFAULT_MODEL = "intfloat/multilingual-e5-small"
_MAX_ENTRIES = 200


class EmbeddingMemory:
    def __init__(self, path: str, model_name: str = _DEFAULT_MODEL):
        self.path = path
        self._model_name = model_name
        self._model = None
        self._entries: list[dict] = []
        if os.path.exists(path):
            self._load()

    @staticmethod
    def is_available() -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name, device="cpu")
        log.info("Embeddingモデル読み込み完了: %s", self._model_name)

    def _encode(self, text: str, is_query: bool = False):
        self._ensure_model()
        import numpy as np
        prefix = "query: " if is_query else "passage: "
        vec = self._model.encode(prefix + text, normalize_embeddings=True)
        return np.asarray(vec, dtype="float32")

    def add(self, text: str, category: str = "topic") -> bool:
        text = text.strip()
        if not text:
            return False
        for e in self._entries:
            if e["text"] == text:
                return False
        try:
            embedding = self._encode(text).tolist()
        except Exception as exc:
            log.warning("Embedding生成に失敗: %s", exc)
            return False
        self._entries.append({
            "text": text,
            "category": category,
            "ts": datetime.now().isoformat(),
            "emb": embedding,
        })
        if len(self._entries) > _MAX_ENTRIES:
            self._entries = self._entries[-_MAX_ENTRIES:]
        self._save()
        return True

    def search(self, query: str, top_k: int = 3,
               threshold: float = 0.35) -> list[tuple[float, str]]:
        if not self._entries or not query.strip():
            return []
        try:
            import numpy as np
            q = self._encode(query, is_query=True)
        except Exception:
            return []
        scored = []
        for entry in self._entries:
            emb = np.asarray(entry["emb"], dtype="float32")
            sim = float(np.dot(q, emb))
            if sim >= threshold:
                scored.append((sim, entry["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def as_prompt(self, query: str) -> str:
        results = self.search(query)
        if not results:
            return ""
        lines = "\n".join(f"- {text}" for _, text in results)
        return f"\n過去の会話で分かっていること:\n{lines}"

    def _save(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False)

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self._entries = [e for e in data
                             if isinstance(e, dict) and "emb" in e and "text" in e]
        except (OSError, ValueError) as exc:
            log.warning("Embeddingメモリ読み込み失敗: %s", exc)
            self._entries = []

    def __len__(self):
        return len(self._entries)
