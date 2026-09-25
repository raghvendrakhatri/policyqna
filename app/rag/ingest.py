"""Load a document, chunk it, embed and store."""

import os
import re
import sys

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import BATCH, CHUNK_CHARS, CHUNK_OVERLAP, env
from .db import IN_COLLECTION, sql, store


# A contents page lists every policy name, so it scores well against almost any
# question while holding no policy text - it crowds real answers out of the
# top-k. These two patterns spot the heading-only lines such a page is made of.
NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
CAPS_HEADING = re.compile(r"^[A-Z][A-Z 0-9’'&/(),.-]{3,}$")


def looks_like_contents(text: str, threshold: float = 0.8, min_lines: int = 4) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < min_lines:
        return False
    headings = sum(
        1
        for line in lines
        if len(line) <= 60 and (NUMBERED_HEADING.match(line) or CAPS_HEADING.match(line))
    )
    return headings / len(lines) >= threshold


def load_document(path: str) -> list[Document]:
    """One Document per page, so page numbers survive into chunk metadata."""
    source = os.path.basename(path)
    if path.lower().endswith(".pdf"):
        pages = [(i, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages, 1)]
        skipped = [i for i, text in pages if looks_like_contents(text)]
        if skipped:
            print(f"Skipping {len(skipped)} contents page(s): {skipped}")
        return [
            Document(page_content=text, metadata={"source": source, "page": i})
            for i, text in pages
            if i not in set(skipped)
        ]
    with open(path, encoding="utf-8") as fh:
        return [Document(page_content=fh.read(), metadata={"source": source})]


def chunk_document(path: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = [c for c in splitter.split_documents(load_document(path)) if c.page_content.strip()]
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def chunk_id(chunk: Document) -> str:
    """Deterministic, so a re-run can tell what is already stored."""
    return f"{chunk.metadata['source']}#{chunk.metadata['chunk_index']}"


def cmd_ingest(path: str, replace: bool) -> int:
    if not os.path.exists(path):
        sys.exit(f"No such file: {path}")

    chunks = chunk_document(path)
    if not chunks:
        sys.exit(f"No extractable text in {path} (scanned PDF? it would need OCR).")
    source = chunks[0].metadata["source"]

    vectors = store()
    if replace:
        sql(
            "DELETE FROM langchain_pg_embedding"
            f" WHERE {IN_COLLECTION} AND cmetadata->>'source' = :source",
            source=source,
        )
        todo = chunks
    else:
        ids = [chunk_id(c) for c in chunks]
        done = {d.id for d in vectors.get_by_ids(ids)}
        todo = [c for c in chunks if chunk_id(c) not in done]
        if not todo:
            print(f"'{source}' is already indexed ({len(chunks)} chunks). Use --replace to rebuild.")
            return 0
        if done:
            print(f"Resuming '{source}': {len(done)} stored, {len(todo)} to go.")

    print(f"Embedding {len(todo)} chunks via {env('EMBED_MODEL')} ...")
    stored = 0
    try:
        for start in range(0, len(todo), BATCH):
            batch = todo[start:start + BATCH]
            vectors.add_documents(batch, ids=[chunk_id(c) for c in batch])
            stored += len(batch)
            print(f"\r  {stored}/{len(todo)} stored", end="", flush=True)
        print()
    except KeyboardInterrupt:
        print(f"\nStopped after {stored}/{len(todo)}. Re-run to resume.", file=sys.stderr)
        raise

    print(f"Indexed {len(chunks)} chunks as '{source}'.")
    return 0
