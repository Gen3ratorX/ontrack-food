"""Load markdown knowledge-base documents (with simple front-matter) and split them into chunks.

Chunking is heading-aware: each chunk keeps the document title and the nearest section heading so
retrieved text is understandable on its own, and carries the source citation metadata.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Parse a '---' delimited block of `key: value` lines. Returns (meta, body)."""
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    meta = {}
    for line in raw[3:end].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    body = raw[end + 4 :].lstrip("\n")
    return meta, body


def load_documents(kb_dir: str | Path) -> list[Document]:
    docs = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        if meta.get("index", "true").lower() == "false":
            continue
        doc_id = meta.get("id", path.stem)
        meta.setdefault("title", path.stem)
        meta["file"] = path.name
        docs.append(Document(doc_id=doc_id, text=body, meta=meta))
    return docs


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _sections(body: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, text) sections."""
    sections: list[tuple[str, str]] = []
    heading, buf = "", []
    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph/line boundaries into pieces of at most ~size characters."""
    if len(text) <= size:
        return [text]
    units = [u for u in re.split(r"\n(?=\S)", text) if u.strip()]  # lines / list items / table rows
    pieces, cur = [], ""
    for u in units:
        if len(u) > size:  # very long line: hard split on sentences
            for s in re.split(r"(?<=[.;])\s+", u):
                if len(cur) + len(s) + 1 > size and cur:
                    pieces.append(cur.strip())
                    cur = cur[-overlap:] if overlap else ""
                cur += " " + s
            continue
        if len(cur) + len(u) + 1 > size and cur:
            pieces.append(cur.strip())
            cur = cur[-overlap:] if overlap else ""
        cur += "\n" + u
    if cur.strip():
        pieces.append(cur.strip())
    return pieces


def chunk_documents(docs: list[Document], size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        n = 0
        dsize = int(d.meta.get("chunk_size", size))  # per-document override (e.g. one food per chunk)
        dover = min(overlap, dsize // 4)
        for heading, text in _sections(d.text):
            # Keep markdown tables intact with their header row where possible
            for piece in _split_text(text, dsize, dover):
                prefix = f"{d.meta.get('title', d.doc_id)}"
                if heading:
                    prefix += f" — {heading}"
                meta = {k: d.meta.get(k, "") for k in ("title", "publisher", "url", "licence", "topic", "file")}
                meta["section"] = heading
                chunks.append(Chunk(chunk_id=f"{d.doc_id}#{n}", doc_id=d.doc_id, text=f"{prefix}\n{piece}", meta=meta))
                n += 1
    return chunks
