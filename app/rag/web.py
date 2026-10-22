"""Web UI: `uv run python app/main.py web`, then open http://127.0.0.1:8000.

The same agent, chain and tools as the CLI, behind a small JSON API and one
static page. Where the CLI stops at the terminal to ask (log in, confirm a
write), a request here returns the paused calls instead, and the page resumes
the run with the employee's decisions.

One session per browser cookie, held in RAM like the CLI's chat: restarting
the server is how everyone forgets, and is logged out.
"""

import contextlib
import csv
import io
import os
import threading
import uuid
from dataclasses import dataclass, field
from functools import cache

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
    shown_args,
)
from .chain import Memory, Provenance, answer, build_chain, build_summariser, clean
from .cli import health, pa_score
from .config import DATA_DIR, TOP_K
from .db import counts_by_source, indexed, store
from .feedback import (
    FIXED_COLS,
    KUDOS,
    PRONOUNS,
    SCORE_MAX,
    SCORE_MIN,
    apply_to_rows,
    build_answers,
    output_path,
    parse_csv,
)
from .ingest import cmd_ingest
from .knowledge import knowledge_files, load_knowledge
from .models import llm
from . import tools
from .tools import (
    SUBDOMAIN,
    TOOLS,
    feedback_files,
    get_user_details,
    password_login,
    save_feedback_form,
    submit_feedback_form,
)

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
                 "args": shown_args(r["args"], state["messages"])}
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


# --- feedback forms ------------------------------------------------------
# The `feedback` command in steps: load a CSV, draft one colleague at a time
# (so the page can show progress through a slow local model), let the
# reviewer edit the drafts, then write a `_filled_` copy and upload it.

def feedback_path(file: str) -> str:
    """Only a CSV already in the feedback folder, matched by exact name."""
    if file not in feedback_files():
        raise HTTPException(400, f"{file!r} is not in {tools.FEEDBACK_DIR}/.")
    return os.path.join(tools.FEEDBACK_DIR, file)


def read_form(file: str) -> tuple[str, list[list[str]], list[str], list[dict]]:
    path = feedback_path(file)
    try:
        return (path, *parse_csv(path))
    except SystemExit as exc:  # parse_csv's way of saying it is not a form
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/feedback/files")
def list_feedback() -> dict:
    return {"folder": os.path.abspath(tools.FEEDBACK_DIR),
            "files": [{"file": n, "filled": "_filled_" in n} for n in feedback_files()]}


@app.post("/api/feedback/download")
def download_feedback(request: Request, form: str = Body("", embed=True)) -> dict:
    """A fresh blank form from the HRMS into the feedback folder."""
    session = session_for(request)
    result = save_feedback_form.invoke({"access_token": logged_in(session), "request": form})
    if result.get("saved_to"):
        result["file"] = os.path.basename(result["saved_to"])
    return result


@app.get("/api/feedback/form")
def feedback_form(file: str) -> dict:
    _, rows, headers, sections = read_form(file)
    # Only the answer cells count: the rows above the first section hold each
    # colleague's email and the like.
    answer_rows = [r for s in sections
                   for r in [s["section_remarks_row"]]
                   + [x for a in s["subcats"] for x in (a["score_row"], a["remarks_row"])]
                   if r is not None]
    colleagues = []
    for i, name in enumerate(headers):
        col = FIXED_COLS + i
        filled = any(len(rows[r]) > col and rows[r][col].strip() for r in answer_rows)
        colleagues.append({"name": name, "filled": filled})
    return {
        "file": file,
        "colleagues": colleagues,
        "sections": [{"name": s["name"], "kudos": s["name"] == KUDOS,
                      "aspects": [a["name"] for a in s["subcats"] if a["name"]]}
                     for s in sections],
        "score_range": [SCORE_MIN, SCORE_MAX],
    }


@cache
def feedback_model():
    return llm()


@app.post("/api/feedback/draft")
def draft_feedback(file: str = Body(), name: str = Body(), scores: dict = Body(),
                   note: str = Body(""), kudos: str = Body(""),
                   pronoun: str = Body("")) -> dict:
    """One colleague's answers, phrased by the model from the reviewer's
    scores and note. Nothing is written."""
    _, _, headers, sections = read_form(file)
    if name not in headers:
        raise HTTPException(400, f"{name!r} is not on this form.")
    clean_scores = {}
    for section, value in scores.items():
        if value in (None, ""):
            continue
        if not (isinstance(value, int) and SCORE_MIN <= value <= SCORE_MAX):
            raise HTTPException(400, f"{section}: score must be {SCORE_MIN}-{SCORE_MAX}.")
        clean_scores[section] = value
    if not clean_scores and not kudos.strip():
        raise HTTPException(400, "Give at least one score or a kudos line.")
    shared = {"scores": clean_scores, "note": note, "kudos": kudos.strip()}
    return build_answers(name, shared, sections, feedback_model(),
                         pronoun if pronoun in PRONOUNS else None)


@app.post("/api/feedback/save")
def save_feedback(file: str = Body(), answers: dict[str, dict] = Body()) -> dict:
    """Write the (possibly edited) answers into a new `_filled_` copy. The form
    it came from is never changed."""
    path, rows, headers, sections = read_form(file)
    unknown = [n for n in answers if n not in headers]
    if unknown:
        raise HTTPException(400, f"Not on this form: {', '.join(unknown)}.")
    if not answers:
        raise HTTPException(400, "Nothing to save.")
    for name, entry in answers.items():
        apply_to_rows(rows, headers, name, entry, sections)
    out = output_path(path)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return {"file": os.path.basename(out), "colleagues": list(answers)}


@app.post("/api/feedback/submit")
def submit_feedback(request: Request, file: str = Body(), form: str = Body("")) -> dict:
    """Upload a filled CSV to the HRMS. The page confirms before calling."""
    session = session_for(request)
    feedback_path(file)
    return submit_feedback_form.invoke({"access_token": logged_in(session),
                                        "file": file, "request": form})


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
