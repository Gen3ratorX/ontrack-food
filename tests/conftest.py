import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def retriever(tmp_path_factory):
    from ontrack.config import CONFIG
    from ontrack.rag.embeddings import Embedder
    from ontrack.rag.index import build_index
    from ontrack.rag.retriever import Retriever

    emb = Embedder(CONFIG.embedding_model)
    idx = build_index(CONFIG.kb_dir, emb, CONFIG.chunk_size, CONFIG.chunk_overlap)
    d = tmp_path_factory.mktemp("index")
    idx.save(d)
    from ontrack.rag.index import VectorIndex

    return Retriever(VectorIndex.load(d), emb, top_k=5, min_score=CONFIG.min_score)
