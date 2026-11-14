"""Build, save and load a FAISS index over knowledge-base chunks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_documents, load_documents
from .embeddings import Embedder


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embed_model: str):
        import faiss

        self.chunks = chunks
        self.embed_model = embed_model
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        scores, ids = self.index.search(qvec.reshape(1, -1).astype("float32"), k)
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def save(self, out_dir: str | Path) -> None:
        import faiss

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "index.faiss"))
        with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (out / "meta.json").write_text(json.dumps({"embed_model": self.embed_model, "n_chunks": len(self.chunks)}, indent=2))

    @classmethod
    def load(cls, in_dir: str | Path) -> "VectorIndex":
        import faiss

        d = Path(in_dir)
        meta = json.loads((d / "meta.json").read_text())
        chunks = [Chunk(**json.loads(l)) for l in open(d / "chunks.jsonl", encoding="utf-8")]
        obj = cls.__new__(cls)
        obj.chunks, obj.embed_model = chunks, meta["embed_model"]
        obj.index = faiss.read_index(str(d / "index.faiss"))
        return obj


def build_index(kb_dir, embedder: Embedder, size: int = 900, overlap: int = 150) -> VectorIndex:
    chunks = chunk_documents(load_documents(kb_dir), size=size, overlap=overlap)
    vecs = embedder.encode([c.text for c in chunks])
    return VectorIndex(chunks, vecs, embedder.model_name)
