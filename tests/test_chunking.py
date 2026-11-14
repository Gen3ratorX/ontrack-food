from ontrack.config import CONFIG
from ontrack.rag.chunking import chunk_documents, load_documents, parse_front_matter


def test_front_matter_parsed():
    meta, body = parse_front_matter("---\nid: x\nurl: https://a.b/c:d\n---\n# T\nhello")
    assert meta == {"id": "x", "url": "https://a.b/c:d"}
    assert body.startswith("# T")


def test_every_doc_has_citation_metadata():
    docs = load_documents(CONFIG.kb_dir)
    assert len(docs) >= 15
    for d in docs:
        for key in ("id", "title", "publisher", "url", "licence"):
            assert d.meta.get(key), f"{d.doc_id} missing {key}"


def test_non_indexed_docs_skipped():
    ids = {d.doc_id for d in load_documents(CONFIG.kb_dir)}
    assert "afrimedqa_dataset_card" not in ids


def test_chunks_respect_size_and_keep_source():
    chunks = chunk_documents(load_documents(CONFIG.kb_dir), size=900, overlap=150)
    assert len(chunks) > 40
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        assert c.meta["url"] and c.meta["title"]
        assert len(c.text) < 900 * 1.6 + 300  # title prefix + soft limit


def test_gi_values_always_cite_a_source():
    """Every GI table row in the KB must name its source (no unsourced GI numbers)."""
    import re

    for d in load_documents(CONFIG.kb_dir):
        if not d.doc_id.startswith("gi_"):
            continue
        for line in d.text.splitlines():
            if line.startswith("|") and re.search(r"\|\s*\d", line) and "---" not in line:
                assert re.search(r"(19|20)\d\d", line), f"GI row without source year in {d.doc_id}: {line}"
