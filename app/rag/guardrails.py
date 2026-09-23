"""Input safety, prompt-injection filter, output cleanup.

`clean()` also lives here because the stray-reasoning-tag regex it uses is the
same backstop as the safety layer: both exist for models that misbehave.
"""

import os
import re
import sys

from langchain_ollama import ChatOllama

from .config import KEEP_ALIVE, env


# Prompt-injection patterns. A user question should never try to reset the
# system prompt or exfiltrate it; anything matching is refused before it
# reaches retrieval, so a poisoned question cannot leak the profile either.
INJECTION = re.compile(
    r"ignore (all |the |any )?(previous|above|prior|earlier) (instructions|prompt|rules|context)"
    r"|disregard (the |all )?(system|previous|above)"
    r"|you are now\b|act as (a |an )?\w+"
    r"|forget (everything|all|your) (you|prior|previous|instructions)"
    r"|reveal (the |your )?(system prompt|instructions|prompt)"
    r"|print (the |your )?(system prompt|instructions)",
    re.I,
)
MAX_QUESTION_CHARS = 2000
REFUSAL = "I can only answer questions about the indexed policy documents."
UNSAFE_REFUSAL = "That request was flagged as unsafe and won't be answered."
UNSAFE_OUTPUT = "The generated answer was flagged as unsafe and has been suppressed."

# CHAT_MODEL is a non-thinking instruct model, so nothing should reach these.
# They are the backstop for a thinking model: note that on such a model
# `reasoning=False` does NOT stop it thinking, it merges the thinking into the
# reply - pass `reasoning=True` to route it into its own field instead.
REASONING = re.compile(r"<(reasoning|analysis|think|thinking)>.*?</\1>", re.S | re.I)
STRAY_CLOSE = re.compile(r"^.*?</(reasoning|analysis|think|thinking)>", re.S | re.I)


# Optional LlamaGuard-style safety classifier served by the same Ollama instance.
# Set SAFETY_MODEL=llama-guard3:1b in .env to enable; unset to skip. The 1B
# variant is small enough to run per call without evicting the chat model.
_safety_llm: "ChatOllama | None" = None


def safety_llm() -> "ChatOllama | None":
    global _safety_llm
    name = os.getenv("SAFETY_MODEL")
    if not name:
        return None
    if _safety_llm is None:
        _safety_llm = ChatOllama(
            model=name,
            base_url=env("OLLAMA_BASE_URL"),
            temperature=0,
            keep_alive=KEEP_ALIVE,
        )
    return _safety_llm


def check_safety(text: str, role: str) -> str | None:
    """None when safe (or classifier disabled), else the unsafe categories.

    LlamaGuard replies "safe" or "unsafe\\nS1,S3,..." where the S-codes name
    Meta's harm categories. `role` picks which side is being evaluated: "human"
    for input, "ai" for output - the same message text scores differently.
    """
    model = safety_llm()
    if model is None or not text.strip():
        return None
    try:
        verdict = model.invoke([(role, text)]).content.strip().lower()
    except Exception as exc:  # noqa: BLE001 - a classifier failure must not block answers
        print(f"Safety check skipped: {exc}", file=sys.stderr)
        return None
    if verdict.startswith("safe"):
        return None
    lines = verdict.splitlines()
    return lines[1].strip() if len(lines) > 1 else "unsafe"


def guard_input(question: str) -> str | None:
    """Refusal message when the question should not reach the model, else None."""
    if not question.strip():
        return "Please ask a question."
    if len(question) > MAX_QUESTION_CHARS:
        return f"Question is too long ({len(question)} chars). Please shorten it."
    if INJECTION.search(question):
        return REFUSAL
    unsafe = check_safety(question, "human")
    if unsafe:
        print(f"Input flagged unsafe: {unsafe}", file=sys.stderr)
        return UNSAFE_REFUSAL
    return None


def clean(text: str) -> str:
    return STRAY_CLOSE.sub("", REASONING.sub("", text)).strip()
