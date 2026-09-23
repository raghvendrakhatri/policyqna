"""The knowledge/ folder: files whose text is sent with every question."""

import os
import sys

from .config import KNOWLEDGE_DIR, KNOWLEDGE_SKIP, KNOWLEDGE_SUFFIXES, KNOWLEDGE_WARN_CHARS


def knowledge_files() -> list[str]:
    if not os.path.isdir(KNOWLEDGE_DIR):
        return []
    return sorted(
        os.path.join(KNOWLEDGE_DIR, name)
        for name in os.listdir(KNOWLEDGE_DIR)
        if name.endswith(KNOWLEDGE_SUFFIXES)
        and not name.startswith(".")
        and name.lower() not in KNOWLEDGE_SKIP
    )


def read_knowledge() -> dict[str, str]:
    """Each knowledge file by name, so an answer can be credited to one."""
    found = {}
    for path in knowledge_files():
        with open(path, encoding="utf-8") as fh:
            text = fh.read().strip()
        if text:
            found[f"knowledge/{os.path.basename(path)}"] = text
    return found


def load_knowledge() -> str:
    """Every knowledge file, concatenated and headed by its filename."""
    parts = [f"[{os.path.basename(name)}]\n{text}" for name, text in read_knowledge().items()]
    joined = "\n\n".join(parts)
    if len(joined) > KNOWLEDGE_WARN_CHARS:
        print(
            f"Warning: knowledge/ is {len(joined)} chars and is sent with every question."
            " Consider ingesting the longer files instead.",
            file=sys.stderr,
        )
    return joined
