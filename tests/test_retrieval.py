"""Answer quality against the real index and the real model.

Slow - every case is a model call - and it needs Ollama and Postgres up with
both documents ingested. Skipped automatically when they are not, so the fast
suite stays runnable anywhere.

    uv run --dev pytest tests/test_retrieval.py -v
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import rag  # noqa: E402

CASES = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "evals.yaml")))


def ready() -> str:
    """Why the suite cannot run, or an empty string when it can."""
    missing = [n for n in rag.REQUIRED_ENV if not os.getenv(n)]
    if missing:
        return f"missing env: {', '.join(missing)}"
    try:
        if not rag.indexed():
            return "nothing ingested"
    except Exception as exc:  # noqa: BLE001 - any failure here means "not ready"
        return f"no database: {type(exc).__name__}"
    if not rag.ollama_reachable(os.getenv("OLLAMA_BASE_URL", "")):
        return "ollama not reachable"
    return ""


pytestmark = pytest.mark.skipif(bool(ready()), reason=ready() or "ready")


@pytest.fixture(scope="module")
def chain():
    """One chain for the whole module, so the models are loaded once."""
    return rag.build_chain(rag.TOP_K)


@pytest.mark.parametrize("case", CASES, ids=[c["question"][:45] for c in CASES])
def test_answer(chain, case):
    answer = rag.clean(chain.invoke(case["question"]))
    lowered = answer.lower()
    for wanted in case.get("expect", []):
        assert wanted.lower() in lowered, (
            f"missing {wanted!r} ({case['why']})\n  asked : {case['question']}\n"
            f"  got   : {answer}"
        )
    for unwanted in case.get("reject", []):
        assert unwanted.lower() not in lowered, (
            f"found {unwanted!r}, which is wrong ({case['why']})\n"
            f"  asked : {case['question']}\n  got   : {answer}"
        )
