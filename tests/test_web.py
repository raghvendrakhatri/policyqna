"""The web API's pause-and-resume around the agent, with the model stubbed out.

Runs without Ollama, Postgres or a network: the agent's graph is replaced by a
script of states, so only the web layer's own decisions are under test.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from rag import web  # noqa: E402
from rag.agent import LOGIN_DECLINED, NOT_LOGGED_IN, WRITE_DECLINED  # noqa: E402


class Interrupt:
    def __init__(self, *requests):
        self.value = {"action_requests": [{"name": n, "args": a} for n, a in requests]}


def paused(*requests) -> dict:
    return {"messages": [], "__interrupt__": [Interrupt(*requests)]}


def done(text: str, tools: tuple[str, ...] = ()) -> dict:
    messages = [HumanMessage("q")] + [ToolMessage("{}", name=t, tool_call_id=t) for t in tools]
    return {"messages": messages + [AIMessage(text)]}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web, "screen", lambda agent, q: (None, "", False))
    monkeypatch.setattr(web, "checked", lambda answer: None)
    web.SESSIONS.clear()
    c = TestClient(web.app)
    c.get("/")
    return c


def script(client, *states) -> list:
    """Make the session's agent return these states in turn; returns the
    payloads it was resumed with."""
    session = next(iter(web.SESSIONS.values()))
    queue, seen = list(states), []

    def step(payload, config):
        seen.append(payload)
        return queue.pop(0)

    session.agent.step = step
    return seen


def test_login_pauses_then_resumes_with_credentials(client):
    session = next(iter(web.SESSIONS.values()))
    seen = script(client, paused(("login", {})), done("Logged in.", ("login",)))
    r = client.post("/api/chat", json={"message": "log me in"}).json()
    assert r["kind"] == "approval"
    assert r["actions"] == [{"index": 0, "name": "login", "args": {}}]

    r = client.post("/api/chat/decide", json={"decisions": [
        {"index": 0, "approve": True, "email": "a@b.c", "password": "pw", "subdomain": ""}]}).json()
    assert r == {"kind": "answer", "text": "Logged in.", "tools": ["login"],
                 "logged_in": False, "profile": ""}
    assert seen[1].resume == {"decisions": [{"type": "approve"}]}
    assert session.agent.context.credentials == ("a@b.c", "pw", web.SUBDOMAIN)
    assert session.memory.history == [("log me in", "Logged in.")]


def test_login_without_password_is_declined(client):
    seen = script(client, paused(("login", {})), done("ok"))
    client.post("/api/chat", json={"message": "log me in"})
    client.post("/api/chat/decide", json={"decisions": [{"index": 0, "approve": True, "email": "a@b.c"}]})
    assert seen[1].resume == {"decisions": [{"type": "reject", "message": LOGIN_DECLINED}]}


def test_write_is_shown_without_token_and_rejected(client):
    session = next(iter(web.SESSIONS.values()))
    session.agent.context.access_token = "tok"
    args = {"start_date": "2026-10-19", "end_date": "2026-10-19", "access_token": "x"}
    seen = script(client, paused(("apply_wfh", args)), done("Not submitted."))
    r = client.post("/api/chat", json={"message": "wfh monday"}).json()
    assert r["actions"][0]["args"] == {"start_date": "2026-10-19", "end_date": "2026-10-19"}
    client.post("/api/chat/decide", json={"decisions": [{"index": 0, "approve": False}]})
    assert seen[1].resume == {"decisions": [{"type": "reject", "message": WRITE_DECLINED}]}


def test_write_with_no_session_is_settled_without_asking(client):
    args = {"start_date": "2026-10-19", "end_date": "2026-10-19"}
    seen = script(client, paused(("apply_wfh", args)), done("Log in first."))
    r = client.post("/api/chat", json={"message": "wfh monday"}).json()
    assert r["kind"] == "answer"
    assert seen[1].resume == {"decisions": [{"type": "reject", "message": NOT_LOGGED_IN}]}


def test_decide_with_nothing_waiting(client):
    assert client.post("/api/chat/decide", json={"decisions": []}).status_code == 409


def test_direct_tools_need_login_and_the_allowlist(client):
    assert client.post("/api/tool/get_leave_balance", json={}).status_code == 401
    assert client.post("/api/tool/submit_feedback_form", json={}).status_code == 404


def test_ingest_only_reads_data_dir(client):
    r = client.post("/api/ingest", json={"file": "../.env"})
    assert r.status_code == 400


FORM = """Fields,Sub-Category,Q2 2026,Asha Rao,Ravi Kumar
,,EMAIL,asha@example.com,ravi@example.com
Team Skills,Team Satisfaction,Score,,
,,Remarks,,
,Collaboration,Score,,
,,Remarks,,
,Section Remarks,,,
Kudos,,,,
"""


@pytest.fixture
def form(monkeypatch, tmp_path):
    from rag import tools

    monkeypatch.setattr(tools, "FEEDBACK_DIR", str(tmp_path))
    (tmp_path / "blank.csv").write_text(FORM, encoding="utf-8")
    return tmp_path


def test_feedback_form_outline(client, form):
    r = client.get("/api/feedback/form", params={"file": "blank.csv"}).json()
    assert r["colleagues"] == [{"name": "Asha Rao", "filled": False},
                               {"name": "Ravi Kumar", "filled": False}]
    assert r["sections"] == [
        {"name": "Team Skills", "kudos": False, "aspects": ["Team Satisfaction", "Collaboration"]},
        {"name": "Kudos", "kudos": True, "aspects": []},
    ]


def test_feedback_only_from_its_folder(client, form):
    assert client.get("/api/feedback/form", params={"file": "../blank.csv"}).status_code == 400
    assert client.post("/api/feedback/save", json={"file": "/etc/passwd", "answers": {}}).status_code == 400


def test_feedback_draft_passes_the_chosen_pronoun(client, form, monkeypatch):
    seen = {}

    def build(name, shared, sections, model, pronoun=None):
        seen.update(name=name, shared=shared, pronoun=pronoun)
        return {"Team Skills": {"score": 4, "section_remark": "r", "subcat_remarks": {}}}

    monkeypatch.setattr(web, "build_answers", build)
    monkeypatch.setattr(web, "feedback_model", lambda: None)
    r = client.post("/api/feedback/draft", json={
        "file": "blank.csv", "name": "Asha Rao", "scores": {"Team Skills": 4, "Kudos": None},
        "note": "great", "kudos": "", "pronoun": "they"})
    assert r.status_code == 200
    assert seen == {"name": "Asha Rao", "pronoun": "they",
                    "shared": {"scores": {"Team Skills": 4}, "note": "great", "kudos": ""}}
    bad = client.post("/api/feedback/draft", json={
        "file": "blank.csv", "name": "Asha Rao", "scores": {"Team Skills": 9}})
    assert bad.status_code == 400


def test_feedback_save_writes_a_copy(client, form):
    answers = {"Asha Rao": {
        "Team Skills": {"score": 4, "section_remark": "Solid teammate.",
                        "subcat_remarks": {"Team Satisfaction": "They help.",
                                           "Collaboration": "They pair well."}},
        "Kudos": {"remark": "Thanks!"}}}
    r = client.post("/api/feedback/save", json={"file": "blank.csv", "answers": answers}).json()
    assert r["file"].startswith("blank_filled_") and r["colleagues"] == ["Asha Rao"]
    assert (form / "blank.csv").read_text(encoding="utf-8") == FORM
    rows = (form / r["file"]).read_text(encoding="utf-8").splitlines()
    assert rows[2].endswith(",4,") and rows[4].endswith(",4,")
    assert rows[3].endswith("They help.,") and rows[6].endswith("Solid teammate.,")
    assert rows[7].endswith("Thanks!,")
    outline = client.get("/api/feedback/form", params={"file": r["file"]}).json()
    assert [c["filled"] for c in outline["colleagues"]] == [True, False]


def test_feedback_upload_needs_login(client, form):
    r = client.post("/api/feedback/submit", json={"file": "blank.csv"})
    assert r.status_code == 401


def test_backdated_write_from_a_past_month_is_settled_without_asking(client):
    session = next(iter(web.SESSIONS.values()))
    session.agent.context.access_token = "tok"
    args = {"start_date": "2000-01-03", "end_date": "2000-01-03"}  # a Monday, long gone
    seen = script(client, paused(("apply_leave", args)), done("Too far back."))
    r = client.post("/api/chat", json={"message": "leave on 3 jan 2000"}).json()
    assert r["kind"] == "answer"
    assert "past month" in seen[1].resume["decisions"][0]["message"]
