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
