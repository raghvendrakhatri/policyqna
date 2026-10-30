"""The dates the agent spells out for the model. Runs without Ollama."""

import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from rag.agent import date_hints, resolve_dates  # noqa: E402

TODAY = date(2026, 10, 9)  # a Friday


@pytest.mark.parametrize("text, expected", [
    ("apply leave for 15 october", ["2026-10-15"]),
    ("15th of Oct", ["2026-10-15"]),
    ("wfh on oct 15, 2027", ["2027-10-15"]),
    ("leave from oct 20th to oct 22nd", ["2026-10-20", "2026-10-22"]),
    ("apply wfh for 5 oct", ["2026-10-05"]),       # recent past stays this year
    ("leave on 2 jan", ["2027-01-02"]),            # long past rolls to next year
    ("wfh tomorrow", ["2026-10-10"]),
    ("leave day after tomorrow", ["2026-10-11"]),
    ("leave next monday", ["2026-10-12"]),
    ("wfh on friday", ["2026-10-16"]),             # a weekday is never today
    ("cancel leave on 31 feb", []),
    ("apply leave", []),
])
def test_resolve_dates(text, expected):
    assert [d.isoformat() for _, d in resolve_dates(text, TODAY)] == expected


def test_hint_names_what_was_written():
    assert date_hints("apply leave for 15 october", TODAY) == (
        "\n\nDates in this message:\n- 15 october = 2026-10-15 (Thursday)")


def test_no_hint_without_a_date():
    assert date_hints("apply leave", TODAY) == ""
    assert date_hints("apply leave on 2026-10-15", TODAY) == ""
