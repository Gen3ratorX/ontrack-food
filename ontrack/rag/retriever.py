"""Retriever: embeds a query, searches FAISS, filters by a similarity floor, formats cited context."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import CONFIG
from .embeddings import Embedder
from .index import VectorIndex, build_index
from .lexical import BM25


@dataclass
class Retrieved:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    title: str
    url: str


class Retriever:
    """Hybrid retriever: dense (FAISS cosine) + BM25, fused with reciprocal rank fusion (RRF).

    A chunk is kept if its dense score >= min_score OR it has a lexical match among the BM25 top ranks.
    Set hybrid=False for dense-only retrieval (useful as an ablation).
    """

    def __init__(self, index: VectorIndex, embedder: Embedder, top_k: int = 4, min_score: float = 0.25,
                 hybrid: bool = True, rrf_k: int = 60):
        self.index, self.embedder, self.top_k, self.min_score = index, embedder, top_k, min_score
        self.hybrid, self.rrf_k = hybrid, rrf_k
        self.bm25 = BM25([c.text for c in index.chunks]) if hybrid else None

    @classmethod
    def from_config(cls, cfg=CONFIG, rebuild: bool = False) -> "Retriever":
        embedder = Embedder(cfg.embedding_model)
        if rebuild or not (cfg.index_dir / "index.faiss").exists():
            idx = build_index(cfg.kb_dir, embedder, cfg.chunk_size, cfg.chunk_overlap)
            idx.save(cfg.index_dir)
        else:
            idx = VectorIndex.load(cfg.index_dir)
            if idx.embed_model != cfg.embedding_model:
                raise ValueError(f"Index built with {idx.embed_model}, config uses {cfg.embedding_model}; rebuild it.")
        return cls(idx, embedder, cfg.top_k, cfg.min_score)

    def _ranked(self, query: str) -> list[tuple]:
        n = len(self.index.chunks)
        dense = self.index.search(self.embedder.encode([query])[0], n)
        dense_score = {ch.chunk_id: s for ch, s in dense}
        if not self.hybrid:
            return [(ch, s, s >= self.min_score) for ch, s in dense]
        lex = self.bm25.scores(query)
        lex_rank = {self.index.chunks[i].chunk_id: r for r, i in enumerate(sorted(range(n), key=lambda i: -lex[i])) if lex[i] > 0}
        fused = []
        for r, (ch, s) in enumerate(dense):
            rrf = 1 / (self.rrf_k + r + 1)
            if ch.chunk_id in lex_rank:
                rrf += 1 / (self.rrf_k + lex_rank[ch.chunk_id] + 1)
            keep = s >= self.min_score or lex_rank.get(ch.chunk_id, 10**9) < 10
            fused.append((ch, dense_score[ch.chunk_id], keep, rrf))
        fused.sort(key=lambda x: -x[3])
        return [(ch, s, keep) for ch, s, keep, _ in fused]

    def search(self, query: str, k: int | None = None, dedupe_docs: bool = False) -> list[Retrieved]:
        k = k or self.top_k
        out, seen = [], set()
        for ch, s, keep in self._ranked(query):
            if not keep:
                continue
            if dedupe_docs and ch.doc_id in seen:
                continue
            seen.add(ch.doc_id)
            out.append(Retrieved(ch.chunk_id, ch.doc_id, ch.text, s, ch.meta.get("title", ""), ch.meta.get("url", "")))
            if len(out) >= k:
                break
        return out


def format_context(hits: list[Retrieved]) -> str:
    """Numbered context block the model can cite as [S1], [S2]..."""
    if not hits:
        return "(No relevant passages were found in the OnTrack knowledge base.)"
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[S{i}] Source: {h.title} ({h.url})\n{h.text}")
    return "\n\n".join(parts)
