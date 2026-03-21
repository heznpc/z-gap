"""Embedding model interfaces: sentence-transformers + OpenAI."""

from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
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

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    @property
    def name(self) -> str:
        return f"st_{self._model_name.split('/')[-1]}"

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


class OpenAIEmbedder(EmbeddingModel):
    """OpenAI text embeddings."""

    def __init__(self, model: str = "text-embedding-3-small"):
        import openai
        from dotenv import load_dotenv
        load_dotenv()
        self._client = openai.OpenAI()
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

    def encode(self, texts: list[str]) -> np.ndarray:
        import requests
        from tqdm import tqdm
        results = []
        batch_size = 50
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Mistral {self._model}"):
            batch = texts[i:i + batch_size]
            resp = requests.post(
                "https://api.mistral.ai/v1/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": batch},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            results.extend([d["embedding"] for d in data])
        arr = np.array(results, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-8)

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
        h = hashlib.sha256(f"{model_name}:{'|'.join(texts)}".encode()).hexdigest()[:16]
        return f"{model_name}_{h}"

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
