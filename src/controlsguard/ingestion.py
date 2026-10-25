"""Document ingestion: load markdown policies, clean, parse structured fields, chunk.

Chunking is structure-aware: each `## <ID> <Title>` section is one chunk (a control or a risk is the
natural unit of retrieval). Oversized sections are split on sentence boundaries with overlap. Every
chunk is prefixed with a contextual header ("Doc title > Section") so it is self-describing once
retrieved, which helps both lexical and semantic search.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .textutil import count_tokens, sentences

ENTITY_RE = re.compile(r"^(?P<id>[CR]-[A-Z]+(?:-[A-Z]+)*(?:-\d+)?)\s+(?P<title>.+)$")
FIELD_RE = re.compile(r"^-\s+(?P<key>[A-Za-z ]+):\s*(?P<val>.+)$")
LIST_FIELDS = {"risks", "systems", "related", "affects"}


@dataclass
class Document:
    id: str
    title: str
    body: str
    path: str


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    section: str
    entity_id: str  # e.g. C-AC-01 / R-UNAUTH, or "" for free text
    text: str
    fields: dict = field(default_factory=dict)  # parsed "- Key: value" lines (lower-case keys)

    @property
    def header(self) -> str:
        return f"{self.doc_title} > {self.section}"


def clean_text(text: str) -> str:
    """Normalise unicode, strip control characters and zero-width chars, collapse blank runs."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(data_dir: str | Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(data_dir).glob("*.md")):
        raw = clean_text(path.read_text(encoding="utf-8"))
        m = re.match(r"#\s+(.+)", raw)
        title = m.group(1).strip() if m else path.stem.replace("-", " ").title()
        docs.append(Document(id=path.stem, title=title, body=raw, path=str(path)))
    return docs


def parse_fields(lines: list[str]) -> tuple[dict, list[str]]:
    fields: dict = {}
    rest: list[str] = []
    for line in lines:
        m = FIELD_RE.match(line.strip())
        if m:
            key = m.group("key").strip().lower()
            val = m.group("val").strip()
            fields[key] = [v.strip() for v in val.split(",") if v.strip()] if key in LIST_FIELDS else val
        else:
            rest.append(line)
    return fields, rest


def _split_sections(body: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for line in body.splitlines():
        if line.startswith("## "):
            if current:
                sections.append(current)
            current = (line[3:].strip(), [])
        elif current is not None:
            current[1].append(line)
    if current:
        sections.append(current)
    return sections


def _window_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split on sentence boundaries into windows of <= max_tokens, carrying `overlap_tokens` forward."""
    sents = sentences(text)
    windows: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for s in sents:
        t = count_tokens(s)
        if cur and cur_tokens + t > max_tokens:
            windows.append(" ".join(cur))
            carry: list[str] = []
            carried = 0
            for prev in reversed(cur):
                pt = count_tokens(prev)
                if carried + pt > overlap_tokens:
                    break
                carry.insert(0, prev)
                carried += pt
            cur, cur_tokens = carry, carried
        cur.append(s)
        cur_tokens += t
    if cur:
        windows.append(" ".join(cur))
    return windows or [text]


def chunk_document(doc: Document, max_tokens: int = 220, overlap_tokens: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec_title, lines in _split_sections(doc.body):
        m = ENTITY_RE.match(sec_title)
        entity_id = m.group("id") if m else ""
        fields, rest = parse_fields(lines)
        prose = clean_text("\n".join(rest))
        field_text = "\n".join(l.strip() for l in lines if FIELD_RE.match(l.strip()))
        base = f"{doc.title} > {sec_title}"
        pieces = _window_split(prose, max_tokens, overlap_tokens) if count_tokens(prose) > max_tokens else [prose]
        slug = entity_id or re.sub(r"\W+", "-", sec_title.lower())
        for i, piece in enumerate(pieces):
            cid = f"{doc.id}:{slug}" + (f"#{i}" if len(pieces) > 1 else "")
            text = f"{base}\n{field_text}\n{piece}".strip() if i == 0 else f"{base}\n{piece}"
            chunks.append(
                Chunk(id=cid, doc_id=doc.id, doc_title=doc.title, section=sec_title,
                      entity_id=entity_id, text=text, fields=fields)
            )
    return chunks


def ingest(data_dir: str | Path, max_tokens: int = 220, overlap_tokens: int = 40) -> tuple[list[Document], list[Chunk]]:
    docs = load_documents(data_dir)
    chunks = [c for d in docs for c in chunk_document(d, max_tokens, overlap_tokens)]
    return docs, chunks
