"""Reading the agent's question router's reply. Runs without Ollama."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from rag.guardrails import parse_route  # noqa: E402


@pytest.mark.parametrize("verdict, expected", [
    ("policy", "policy"),
    ("hrms", "hrms"),
    ("HRMS.", "hrms"),
    ("**no**", "no"),
    ("<think>hmm</think>hrms", "hrms"),
    ("", "policy"),            # nothing usable: search, as before routing
    ("maybe", "policy"),
])
def test_parse_route(verdict, expected):
    assert parse_route(verdict) == expected
