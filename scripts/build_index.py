"""Build the FAISS index over knowledge_base/docs.  python scripts/build_index.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ontrack.config import CONFIG  # noqa: E402
from ontrack.rag.embeddings import Embedder  # noqa: E402
from ontrack.rag.index import build_index  # noqa: E402

if __name__ == "__main__":
    emb = Embedder(CONFIG.embedding_model)
    idx = build_index(CONFIG.kb_dir, emb, CONFIG.chunk_size, CONFIG.chunk_overlap)
    idx.save(CONFIG.index_dir)
    docs = sorted({c.doc_id for c in idx.chunks})
    print(f"Indexed {len(idx.chunks)} chunks from {len(docs)} documents with {CONFIG.embedding_model} -> {CONFIG.index_dir}")
