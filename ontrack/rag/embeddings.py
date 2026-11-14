"""Sentence-transformers embedder (small, CPU-friendly). Vectors are L2-normalised so inner product = cosine."""
from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")
