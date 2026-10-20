"""Web UI: `uv run python app/main.py web`, then open http://127.0.0.1:8000.

The same agent, chain and tools as the CLI, behind a small JSON API and one
static page. Where the CLI stops at the terminal to ask (log in, confirm a
write), a request here returns the paused calls instead, and the page resumes
the run with the employee's decisions.

One session per browser cookie, held in RAM like the CLI's chat: restarting
the server is how everyone forgets, and is logged out.
"""

import contextlib
import io
import os
import threading
import uuid
from dataclasses import dataclass, field

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from .agent import (
    LOGIN_DECLINED,
    STEPS_EXHAUSTED,
    WRITE_DECLINED,
    WRITE_TOOLS,
    Agent,
    checked,
    pending,
    screen,
)
from .chain import Memory, Provenance, answer, build_chain, build_summariser, clean
from .cli import health, pa_score
from .config import DATA_DIR, TOP_K
from .db import counts_by_source, indexed, store
from .ingest import cmd_ingest
from .knowledge import knowledge_files, load_knowledge
from .profile import profile_from
from .tools import SUBDOMAIN, TOOLS, get_user_details, list_feedback_files, password_login

STATIC = os.path.join(os.path.dirname(__file__), "static")
COOKIE = "policyqa_sid"

# What the Actions tab may call directly, without the model. The writes are
# confirmed in the page before the request is sent, as the agent's are.
DIRECT_TOOLS = {
    t.name: t for t in TOOLS
    if t.name not in ("login", "save_feedback_form", "submit_feedback_form",
                      "get_feedback_form", "search_policy", "list_feedback_files")
}


@dataclass
class Session:
    memory: Memory
    agent: Agent
    lock: threading.Lock = field(default_factory=threading.Lock)
    # A run waiting on the employee: {"config", "question", "requests"}.
    paused: dict | None = None

    @property
    def token(self) -> str | None:
        return self.agent.context.access_token

    @property
    def profile(self) -> str:
        return self.agent.context.profile


def new_session() -> Session:
    memory = Memory(summariser=build_summariser())
    return Session(memory=memory, agent=Agent(None, "", memory))


SESSIONS: dict[str, Session] = {}
app = FastAPI(title="policyqa")


def session_for(request: Request) -> Session:
    sid = request.cookies.get(COOKIE)
    if not sid or sid not in SESSIONS:
        raise HTTPException(401, "No session. Reload the page.")
    return SESSIONS[sid]


def logged_in(session: Session) -> str:
    if not session.token:
        raise HTTPException(401, "Log in to the HRMS first.")
    return session.token


@app.exception_handler(Exception)
def failed(_request: Request, exc: Exception) -> JSONResponse:
    # Postgres and Ollama errors arrive in several shapes; the page only needs
    # the message. SystemExit is how the CLI helpers say what is wrong.
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


@app.get("/")
def index(request: Request) -> FileResponse:
    response = FileResponse(os.path.join(STATIC, "index.html"))
    if request.cookies.get(COOKIE) not in SESSIONS:
        sid = uuid.uuid4().hex
        SESSIONS[sid] = new_session()
        response.set_cookie(COOKIE, sid, httponly=True, samesite="strict")
    return response


@app.get("/api/session")
def session_info(request: Request) -> dict:
    session = session_for(request)
    return {"logged_in": bool(session.token), "profile": session.profile,
            "subdomain": SUBDOMAIN, "tools": sorted(session.agent.tools),
            "waiting": bool(session.paused)}


# --- login ---------------------------------------------------------------

def sign_in(session: Session, email: str, password: str, subdomain: str) -> None:
    token = password_login(email, password, subdomain or SUBDOMAIN)
    if isinstance(token, dict):
        raise HTTPException(401, f"Login failed: {token['error']}. {token.get('detail', '')}")
    details = get_user_details.invoke({"access_token": token})
    session.agent.context.access_token = token
    session.agent.context.profile = details.get("profile", "") \
        if not details.get("error") else ""


@app.post("/api/login")
def login(request: Request, email: str = Body(), password: str = Body(),
          subdomain: str = Body("")) -> dict:
    session = session_for(request)
    with session.lock:
        sign_in(session, email.strip(), password, subdomain.strip())
    return session_info(request)


@app.post("/api/logout")
def logout(request: Request) -> dict:
    sid = request.cookies.get(COOKIE)
    if sid in SESSIONS:
        SESSIONS[sid] = new_session()
    return {"logged_in": False}


# --- agent chat ----------------------------------------------------------

def drive(session: Session, payload, config: dict, question: str) -> dict:
    """Run the agent until it answers or pauses on a call the employee must
    decide. Calls `auto_decision` settles are resumed without asking."""
    agent = session.agent
    try:
        while True:
            state = agent.step(payload, config)
            requests = pending(state)
            if not requests:
                break
            auto = [agent.auto_decision(r["name"], r["args"]) for r in requests]
            if all(auto):
                payload = Command(resume={"decisions": auto})
                continue
            session.paused = {"config": config, "question": question,
                              "requests": requests, "auto": auto}
            return {"kind": "approval", "actions": [
                {"index": i, "name": r["name"],
                 "args": {k: v for k, v in r["args"].items() if k != "access_token"}}
                for i, r in enumerate(requests) if not auto[i]
            ]}
    except GraphRecursionError:
        return {"kind": "answer", "text": STEPS_EXHAUSTED, "tools": [],
                "logged_in": bool(session.token)}
    text = clean(state["messages"][-1].content)
    # The tools this question called: every ToolMessage after its own message.
    messages = state["messages"]
    start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    tools = [m.name for m in messages[start:] if isinstance(m, ToolMessage)]
    unsafe = checked(text)
    if unsafe:
        return {"kind": "refused", "title": "Unsafe output", "text": unsafe}
    # A run that ran out of steps answered nothing; see cli.agent_loop.
    if text and text != STEPS_EXHAUSTED:
        session.memory.remember(question, text)
    return {"kind": "answer", "text": text, "tools": tools,
            "logged_in": bool(session.token), "profile": session.profile}


@app.post("/api/chat")
def chat(request: Request, message: str = Body(embed=True)) -> dict:
    session = session_for(request)
    question = message.strip()
    if not question:
        raise HTTPException(400, "Type a question.")
    with session.lock:
        # A new message abandons a run still waiting on a decision.
        session.paused = None
        refusal, title, prefetch = screen(session.agent, question)
        if refusal:
            return {"kind": "refused", "title": title, "text": refusal}
        payload, config = session.agent.begin(question, prefetch)
        return drive(session, payload, config, question)


@app.post("/api/chat/decide")
def decide(request: Request, decisions: list[dict] = Body(embed=True)) -> dict:
    """`decisions`: one {"index", "approve", "email"?, "password"?, "subdomain"?}
    per action the page was shown."""
    session = session_for(request)
    with session.lock:
        paused = session.paused
        if not paused:
            raise HTTPException(409, "Nothing is waiting for a decision.")
        session.paused = None
        chosen = {d.get("index"): d for d in decisions}
        resume = []
        for i, req in enumerate(paused["requests"]):
            if paused["auto"][i]:
                resume.append(paused["auto"][i])
                continue
            choice = chosen.get(i) or {}
            if req["name"] == "login":
                if choice.get("approve") and choice.get("email") and choice.get("password"):
                    # Read by the login tool from the run context, never a message.
                    session.agent.context.credentials = (
                        choice["email"].strip(), choice["password"],
                        (choice.get("subdomain") or "").strip() or SUBDOMAIN)
                    resume.append({"type": "approve"})
                else:
                    resume.append({"type": "reject", "message": LOGIN_DECLINED})
            elif choice.get("approve"):
                resume.append({"type": "approve"})
            else:
                resume.append({"type": "reject", "message": WRITE_DECLINED})
        return drive(session, Command(resume={"decisions": resume}),
                     paused["config"], paused["question"])


@app.post("/api/chat/clear")
def clear_chat(request: Request) -> dict:
    session = session_for(request)
    with session.lock:
        session.paused = None
        session.memory.history.clear()
        session.memory.summary = ""
    return {"ok": True}


# --- policy-only Q&A -----------------------------------------------------

@app.post("/api/ask")
def ask(request: Request, question: str = Body(), k: int = Body(TOP_K)) -> dict:
    """The `ask` command: retrieval and the chain, no tools, with sources."""
    session = session_for(request)
    if not indexed():
        raise HTTPException(400, "Nothing indexed yet - ingest a document in Admin.")
    provenance = Provenance()
    chain = build_chain(k, session.profile, provenance)
    reply = answer(chain, question.strip(), provenance, session.profile)
    if reply.refused:
        return {"kind": "refused", "title": reply.refused, "text": reply.text}
    return {"kind": "answer", "text": reply.text, "sources": reply.sources}


# --- direct HRMS actions -------------------------------------------------

@app.post("/api/tool/{name}")
def call_tool(request: Request, name: str, args: dict = Body(default_factory=dict)) -> dict:
    session = session_for(request)
    tool = DIRECT_TOOLS.get(name)
    if tool is None:
        raise HTTPException(404, f"No action {name!r}.")
    args = {k: v for k, v in args.items() if k != "access_token"}
    if "access_token" in tool.args_schema.model_fields:
        args["access_token"] = logged_in(session)
    result = tool.invoke(args)
    return {"write": name in WRITE_TOOLS, "result": result}


@app.get("/api/feedback-files")
def feedback_files() -> dict:
    return list_feedback_files.invoke({})


# --- admin ---------------------------------------------------------------

@app.get("/api/health")
def check() -> dict:
    return {"lines": [{"level": level, "message": message} for level, message in health()]}


@app.get("/api/stats")
def stats() -> dict:
    rows = counts_by_source()
    files = knowledge_files()
    return {
        "indexed": None if rows is None else [{"source": s, "chunks": c} for s, c in rows],
        "knowledge": [{"file": os.path.basename(p), "bytes": os.path.getsize(p)} for p in files],
        "knowledge_chars": len(load_knowledge()) if files else 0,
    }


def data_files() -> list[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(n for n in os.listdir(DATA_DIR)
                  if n.lower().endswith((".pdf", ".md", ".txt")))


@app.get("/api/data-files")
def list_data_files() -> dict:
    return {"files": data_files()}


@app.post("/api/ingest")
def ingest(file: str = Body(), replace: bool = Body(False)) -> dict:
    """Only files already in data/: the page cannot point it anywhere else."""
    if file not in data_files():
        raise HTTPException(400, f"{file!r} is not in data/.")
    out = io.StringIO()
    # cmd_ingest reports by printing and fails by sys.exit; keep both.
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            cmd_ingest(os.path.join(DATA_DIR, file), replace)
            ok = True
        except SystemExit as exc:
            print(exc)
            ok = False
    # The carriage-return progress line is only noise in a log.
    log = "\n".join(line.split("\r")[-1] for line in out.getvalue().splitlines())
    return {"ok": ok, "log": log.strip()}


@app.post("/api/reset")
def reset() -> dict:
    store().delete_collection()
    return {"ok": True}


@app.post("/api/pa")
def pa(client_weight: float = Body(), team_weight: float = Body(),
       criteria: list[dict] = Body()) -> dict:
    try:
        return pa_score(criteria, client_weight, team_weight)
    except (KeyError, TypeError) as exc:
        raise HTTPException(400, f"Each criterion needs a weight: {exc}") from exc


def serve(host: str, port: int) -> int:
    import uvicorn

    print(f"policyqa web UI on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
