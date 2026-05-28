"""Embedding model interfaces: sentence-transformers + OpenAI + Mistral.

Code-review-2026-05-21 fixes applied:
- V1/V14: cache key / model.name now include revision SHA and full org path
- V13: EmbeddingCache._key() uses JSON-encoded payload (delimiter-collision-free)
- V18: dimension property falls back to encode-probe when sentence-transformers
       deprecated `get_sentence_embedding_dimension()` returns None
- V9:  Mistral retry no longer honors server `Retry-After` (bounded by backoff
       only), avoiding multi-hour stalls
- V10: OpenAI timeout 60s → 300s to avoid regressing legacy batch callers
"""

from __future__ import annotations

import hashlib
import json as _json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embedding vectors. Returns shape (n, d)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


class SentenceTransformerEmbedder(EmbeddingModel):
    """Multilingual sentence embeddings via sentence-transformers."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", **kwargs):
        from sentence_transformers import SentenceTransformer
        self._model_name = model_name
        # V1: capture revision (if any) BEFORE forwarding kwargs so .name can
        # encode it. The SentenceTransformer ctor itself also accepts it.
        self._revision = kwargs.get("revision", None)
        self._model = SentenceTransformer(model_name, **kwargs)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    @property
    def name(self) -> str:
        # V14: keep the full repo path so `intfloat/e5-large` and
        # `sentence-transformers/e5-large` do not collide.
        # V1: append revision short SHA so revision bumps produce a new cache key.
        base = f"st_{self._model_name.replace('/', '__')}"
        if self._revision:
            return f"{base}@{self._revision[:8]}"
        return f"{base}@unpinned"

    @property
    def dimension(self) -> int:
        # V18: get_sentence_embedding_dimension() is deprecated in
        # sentence-transformers ≥5.5 and returns None for some
        # trust_remote_code custom modules. Fall back to a single-token
        # encode probe so callers get an int.
        try:
            d = self._model.get_sentence_embedding_dimension()
            if d is not None:
                return int(d)
        except Exception:
            pass
        # Fallback: probe shape from a 1-text encode.
        probe = self._model.encode(["probe"], show_progress_bar=False, normalize_embeddings=False)
        return int(np.asarray(probe).shape[-1])


class OpenAIEmbedder(EmbeddingModel):
    """OpenAI text embeddings."""

    def __init__(self, model: str = "text-embedding-3-small"):
        import openai
        from dotenv import load_dotenv
        load_dotenv()
        # V10: 60s previously regressed legacy batch callers whose ≥100-text
        # batches occasionally cross 60s on server-side. 300s covers them
        # while still bounded; SDK exponential-backoff retry handles 429/5xx.
        self._client = openai.OpenAI(max_retries=5, timeout=300.0)
        self._model = model
        self._dim = 1536 if "small" in model else 3072

    def encode(self, texts: list[str]) -> np.ndarray:
        from tqdm import tqdm
        results = []
        batch_size = 100
        for i in tqdm(range(0, len(texts), batch_size), desc=f"OpenAI {self._model}"):
            batch = texts[i:i + batch_size]
            resp = self._client.embeddings.create(input=batch, model=self._model)
            results.extend([d.embedding for d in resp.data])
        arr = np.array(results, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-8)

    @property
    def name(self) -> str:
        return f"openai_{self._model}"

    @property
    def dimension(self) -> int:
        return self._dim


class MistralEmbedder(EmbeddingModel):
    """Mistral API embeddings (Codestral Embed)."""

    def __init__(self, model: str = "codestral-embed-2505"):
        from dotenv import load_dotenv
        load_dotenv()
        import os
        self._api_key = os.environ["MISTRAL_API_KEY"]
        self._model = model
        self._dim = 1024
        self._session = self._make_session()

    @staticmethod
    def _make_session():
        """Session with retry/backoff for 429 + 5xx.

        V9: ``respect_retry_after_header=False`` — a server-sent ``Retry-After``
        of e.g. 3600s would otherwise block encode() for hours per attempt.
        We rely on the exponential backoff only: 1s, 2s, 4s, 8s, 16s
        (31s total worst case), bounded by total=5.
        """
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST"]),
            respect_retry_after_header=False,  # V9 fix
            raise_on_status=False,
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def encode(self, texts: list[str]) -> np.ndarray:
        from tqdm import tqdm
        results = []
        batch_size = 50
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Mistral {self._model}"):
            batch = texts[i:i + batch_size]
            resp = self._session.post(
                "https://api.mistral.ai/v1/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": batch},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            results.extend([d["embedding"] for d in data])
        arr = np.array(results, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-8)

    def close(self) -> None:
        """Release the underlying requests Session (file descriptors)."""
        try:
            self._session.close()
        except Exception:
            pass

    @property
    def name(self) -> str:
        return f"mistral_{self._model}"

    @property
    def dimension(self) -> int:
        return self._dim


# --- Embedding cache ---

class EmbeddingCache:
    """Cache embeddings to disk as .npz files."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model_name: str, texts: list[str]) -> str:
        # V13: JSON-encode the (model_name, texts) tuple before hashing so
        # texts containing `|` (e.g. `s1 | s2`) cannot collide with split
        # variants. JSON length-prefixes each string implicitly via quoting.
        payload = _json.dumps(
            {"model": model_name, "texts": texts},
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        h = hashlib.sha256(payload).hexdigest()[:16]
        # Filesystem-safe: model_name now may contain '__' (from SentenceTransformerEmbedder.name)
        # and '@<sha>' — those are safe on POSIX. Replace any remaining '/' just in case.
        safe = model_name.replace("/", "__")
        return f"{safe}_{h}"

    def get(self, model_name: str, texts: list[str]) -> np.ndarray | None:
        path = self.cache_dir / f"{self._key(model_name, texts)}.npz"
        if path.exists():
            return np.load(path)["embeddings"]
        return None

    def put(self, model_name: str, texts: list[str], embeddings: np.ndarray):
        path = self.cache_dir / f"{self._key(model_name, texts)}.npz"
        np.savez_compressed(path, embeddings=embeddings)

    def get_or_compute(self, model: EmbeddingModel, texts: list[str]) -> np.ndarray:
        cached = self.get(model.name, texts)
        if cached is not None:
            print(f"  Cache hit for {model.name} ({len(texts)} texts)")
            return cached
        print(f"  Computing embeddings with {model.name} ({len(texts)} texts)...")
        embeddings = model.encode(texts)
        self.put(model.name, texts, embeddings)
        return embeddings
