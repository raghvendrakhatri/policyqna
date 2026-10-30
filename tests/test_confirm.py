"""What the employee is shown before a write is sent. Runs without Ollama."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from langchain_core.messages import ToolMessage  # noqa: E402

from rag.agent import shown_args  # noqa: E402

CIA = "2b4f4456-15ef-4afe-94b3-c3f3cbcac484"
TEAM = ToolMessage(
    content=json.dumps({"data": [
        {"id": CIA, "full_name": "Cia Sharma", "official_email": "cia.sharma@mailinator.com"},
    ]}),
    name="get_team_members", tool_call_id="1",
)
ARGS = {"access_token": "secret", "leave_type": "el-1", "start_date": "2026-10-26",
        "end_date": "2026-10-26", "people_to_notify": [CIA], "is_half_day": False}


def test_people_to_notify_are_shown_by_name():
    shown = shown_args(ARGS, [TEAM])
    assert shown["people_to_notify"] == ["Cia Sharma (cia.sharma@mailinator.com)"]
    assert "access_token" not in shown
    assert ARGS["people_to_notify"] == [CIA]  # what is sent keeps the id


def test_an_id_never_looked_up_is_shown_as_is():
    assert shown_args(ARGS, [])["people_to_notify"] == [CIA]
    error = ToolMessage(content='{"error": "Session expired"}',
                        name="get_team_members", tool_call_id="2")
    assert shown_args(ARGS, [error])["people_to_notify"] == [CIA]
