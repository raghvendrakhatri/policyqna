"""Agent mode: LangChain's tool-calling agent over the HRMS and policy tools.

`create_agent` runs the loop - model, tool calls, ToolMessages, until the model
answers in plain text. Two middlewares sit on it:

- HumanInTheLoopMiddleware pauses before every write tool. The graph raises an
  interrupt, the employee approves or rejects at the prompt, and the run resumes
  from the checkpoint with that decision.
- `hrms_session` puts the employee's token into each tool call that takes one.
  The token comes from the run context, so the model never sees it.
"""

import json
import re
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from rich import box
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .chain import Memory, looks_like_followup
from .config import (
    AGENT_HRMS_PROMPT,
    AGENT_MAX_STEPS,
    AGENT_NO_LOGIN_PROMPT,
    AGENT_PROFILE_PROMPT,
    AGENT_PROMPT,
    AGENT_EXCERPTS_PROMPT,
    AGENT_KNOWLEDGE_PROMPT,
    AGENT_PREFETCH_K,
    ACTION_RE,
    PERSONAL_RE,
    TOOL_RESULT_MAX_CHARS,
)
from .guardrails import (
    OFF_TOPIC_REFUSAL,
    UNSAFE_OUTPUT,
    check_safety,
    clean,
    guard_input,
    route_question,
)
from .knowledge import load_knowledge
from .models import llm
from .tools import SUBDOMAIN, TOOLS, date_problem, find_policy
from .ui import console, err_console, refusal_panel, waiting


# LangGraph hands the run context to pydantic, which expects None there and
# warns on every tool call, printing the whole context with it.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings",
                        category=UserWarning)

# Tools that change something in the HRMS. Each call is shown to the employee
# and runs only on an explicit yes.
WRITE_TOOLS = ("apply_leave", "cancel_leave", "apply_wfh", "submit_feedback_form")
STEPS_EXHAUSTED = "I couldn't finish that within the allowed number of steps."
NOT_LOGGED_IN = "The employee is not logged in. Call the login tool first, then retry."
LOGIN_DECLINED = "The employee chose not to log in."
WRITE_DECLINED = "The employee declined, so nothing was submitted."


@dataclass
class HRMSContext:
    """The session, shared by every run in a chat. Reaches the middleware and
    the tools, never the messages. The login tool fills both fields in."""

    # repr=False: a repr of the context ends up in warnings and tracebacks.
    access_token: str | None = field(repr=False)
    profile: str = ""
    # (email, password, subdomain) from the login confirmation, until the login
    # tool uses them. Never in a message, so never seen by the model.
    credentials: tuple[str, str, str] | None = field(default=None, repr=False)


@wrap_tool_call
def hrms_session(request, handler):
    """Give the tool the employee's token, trace the call, cap the result.

    A tool that takes the token is invoked here rather than through `handler`:
    ToolNode strips every InjectedToolArg value out of the call's args (so a
    model cannot forge one), and that would strip ours too.
    """
    call, tool = request.tool_call, request.tool
    args = {k: v for k, v in call["args"].items() if k != "access_token"}
    err_console.print(f"[dim]· {call['name']}({describe(args)})[/dim]")
    if tool is None or "access_token" not in tool.args_schema.model_fields:
        with waiting(f"calling {call['name']}"):
            result = handler(request)
    elif not request.runtime.context.access_token:
        result = ToolMessage(content=NOT_LOGGED_IN, name=call["name"],
                             tool_call_id=call["id"], status="error")
    else:
        args["access_token"] = request.runtime.context.access_token
        try:
            with waiting(f"calling {call['name']}"):
                output = tool.invoke(args)
            content, status = json.dumps(output, default=str), "success"
        except Exception as exc:  # noqa: BLE001 - bad args or a tool bug; tell the model
            content, status = f"Error: {type(exc).__name__}: {exc}", "error"
        result = ToolMessage(content=content, name=call["name"],
                             tool_call_id=call["id"], status=status)
    # Every result stays in the context for the rest of the run.
    if isinstance(result, ToolMessage) and isinstance(result.content, str) \
            and len(result.content) > TOOL_RESULT_MAX_CHARS:
        result.content = result.content[:TOOL_RESULT_MAX_CHARS] + "... (truncated)"
    return result


def build_system_prompt(profile: str, logged_in: bool, knowledge: str) -> str:
    system = AGENT_PROMPT.format(
        today=date.today().isoformat(),
        hrms=AGENT_HRMS_PROMPT + ("" if logged_in else AGENT_NO_LOGIN_PROMPT),
    )
    # .replace, not .format: the knowledge and profile are arbitrary text whose
    # braces must stay literal.
    if knowledge:
        system += AGENT_KNOWLEDGE_PROMPT.replace("{knowledge}", knowledge)
    if profile:
        system += AGENT_PROFILE_PROMPT.replace("{profile}", profile)
    return system


class Agent:
    """One chat session's agent.

    Every tool is always bound. Without a token the HRMS tools refuse and the
    model is told to call `login` first, for which the employee types their
    email and password at the confirmation. The system prompt is rebuilt on every model call from the
    session, so it follows a login made mid-chat.

    Each question runs on a fresh thread, seeded with Memory's bounded window
    and summary, so the context stays flat however long the chat runs. The
    checkpointer is still needed within a question: it is what an approval
    resumes from.
    """

    def __init__(self, token: str | None, profile: str = "",
                 memory: Memory | None = None) -> None:
        self.context = HRMSContext(access_token=token, profile=profile)
        self.memory = memory
        self.tools = {t.name: t for t in TOOLS}
        knowledge = load_knowledge()  # read once, not on every model call

        @dynamic_prompt
        def system_prompt(request) -> str:
            session = request.runtime.context
            return build_system_prompt(session.profile, bool(session.access_token),
                                       knowledge)

        self.graph = create_agent(
            model=llm(),
            tools=TOOLS,
            context_schema=HRMSContext,
            checkpointer=InMemorySaver(),
            middleware=[
                system_prompt,
                hrms_session,
                HumanInTheLoopMiddleware(interrupt_on={
                    name: {"allowed_decisions": ["approve", "reject"]}
                    for name in (*WRITE_TOOLS, "login")
                }),
            ],
        )

    @property
    def token(self) -> str | None:
        return self.context.access_token

    def run(self, question: str, prefetch: bool = True) -> str:
        """`prefetch=False` for a reply to the agent's own question ("el-1",
        "yes"): searched alone it matches policy text that says nothing about
        it, and the model reads that as the answer being wrong."""
        payload, config = self.begin(question, prefetch)
        try:
            while True:
                state = self.step(payload, config)
                requests = pending(state)
                if not requests:
                    break
                decisions = [self.decide(r["name"], r["args"]) for r in requests]
                payload = Command(resume={"decisions": decisions})
        except GraphRecursionError:
            return STEPS_EXHAUSTED
        return clean(state["messages"][-1].content)

    def begin(self, question: str, prefetch: bool = True) -> tuple[dict, dict]:
        """The first payload and the config for one question, on a fresh thread."""
        messages: list = []
        if self.memory is not None:
            messages += self.memory.prior_messages() + self.memory.messages()
        content = with_policy(question) if prefetch else question
        messages.append(HumanMessage(content=content + date_hints(question)))
        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            # A step is a model call, then the HITL after_model node, then the
            # tool round: three graph nodes, not two. At two, a login plus
            # three lookups ran out of steps before the model could answer.
            "recursion_limit": AGENT_MAX_STEPS * 3 + 1,
        }
        return {"messages": messages}, config

    def step(self, payload: dict | Command, config: dict) -> dict:
        """Run until the answer or the next paused write. Raises
        GraphRecursionError when the steps run out."""
        with waiting("thinking"):
            return self.graph.invoke(payload, config, context=self.context)

    def auto_decision(self, name: str, args: dict) -> dict | None:
        """A paused call that needs no asking, decided here; None to ask. A
        write with no session is turned back: confirming it would only end in
        an error."""
        if name == "login" and self.token:
            return {"type": "reject", "message": "Already logged in."}
        if name in WRITE_TOOLS and not self.token:
            return {"type": "reject", "message": NOT_LOGGED_IN}
        if name in ("apply_leave", "apply_wfh"):
            problem = date_problem(str(args.get("start_date")), str(args.get("end_date")))
            if problem:
                return {"type": "reject", "message": problem["error"]}
        return None

    def decide(self, name: str, args: dict) -> dict:
        """The employee's answer to one paused call, asked at the terminal."""
        automatic = self.auto_decision(name, args)
        if automatic:
            return automatic
        if name == "login":
            self.context.credentials = ask_credentials()
            if self.context.credentials:
                return {"type": "approve"}
            return {"type": "reject", "message": LOGIN_DECLINED}
        if confirm(name, args):
            return {"type": "approve"}
        return {"type": "reject", "message": WRITE_DECLINED}


def pending(state: dict) -> list[dict]:
    """The write calls a run paused on, each {"name", "args"}; empty when it
    finished. One interrupt per model turn, holding every write call it made."""
    interrupts = state.get("__interrupt__")
    return interrupts[0].value["action_requests"] if interrupts else []


MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
YEAR = r"(?:,?\s+(\d{4}))?"
DATE_RES = (
    # 15 october, 15th oct 2026
    (re.compile(rf"\b{DAY}\s+(?:of\s+)?{MONTH}{YEAR}\b", re.I), lambda m: (m[1], m[2], m[3])),
    # october 15, oct 15th, 2026
    (re.compile(rf"\b{MONTH}\s+{DAY}{YEAR}\b", re.I), lambda m: (m[2], m[1], m[3])),
)
ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
RELATIVE_RE = re.compile(rf"\b(today|tomorrow|day after tomorrow|(?:next\s+)?(?:{'|'.join(WEEKDAYS)}))\b", re.I)


def resolve_dates(text: str, today: date) -> list[tuple[str, date]]:
    """The dates written in `text`, as (what was written, date).

    A day and month without a year is the next such date, except that one up
    to a month back stays in this year: "5 oct" on 9 Oct is a backdated leave,
    not one a year away. A weekday is the next one after today.
    """
    found: list[tuple[str, date]] = []
    for pattern, parts in DATE_RES:
        for m in pattern.finditer(text):
            day, month, year = parts(m)
            try:
                when = date(int(year or today.year), MONTHS[month[:3].lower()], int(day))
            except ValueError:
                continue  # 31 feb
            if not year and when < today - timedelta(days=31):
                when = when.replace(year=when.year + 1)
            found.append((m[0], when))
    for m in RELATIVE_RE.finditer(text):
        word = m[1].lower()
        if word == "today":
            when = today
        elif word == "tomorrow":
            when = today + timedelta(days=1)
        elif word == "day after tomorrow":
            when = today + timedelta(days=2)
        else:
            ahead = (WEEKDAYS.index(word.split()[-1]) - today.weekday()) % 7 or 7
            when = today + timedelta(days=ahead)
        found.append((m[0], when))
    return found


def date_hints(question: str, today: date | None = None) -> str:
    """The question's dates in YYYY-MM-DD, to append to it. A 4B model asked
    for leave "for 15 october" replied "Which date?" rather than convert it."""
    resolved = [(said, when) for said, when in resolve_dates(question, today or date.today())
                if not ISO_RE.fullmatch(said)]
    if not resolved:
        return ""
    lines = "\n".join(f"- {said} = {when.isoformat()} ({when:%A})" for said, when in resolved)
    return f"\n\nDates in this message:\n{lines}"


def with_policy(question: str) -> str:
    """The question with the policy excerpts that match it, if any do."""
    err_console.print(f"[dim]· search_policy(query={question!r})[/dim]")
    with waiting("searching the policy"):
        results = find_policy(question, k=AGENT_PREFETCH_K)
    if not results:
        return question
    excerpts = "\n\n---\n\n".join(f"[{r['source']}]\n{r['text']}" for r in results)
    # .replace, not .format: policy text can hold braces.
    return (AGENT_EXCERPTS_PROMPT.replace("{excerpts}", excerpts)
            .replace("{question}", question))


def describe(args: dict) -> str:
    """Arguments for the trace line, with the token kept off the screen."""
    return ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "access_token")


def ask_credentials() -> tuple[str, str, str] | None:
    """The employee's HRMS email, password and subdomain, typed at the
    terminal; None if they leave the email or password blank or press Ctrl-C.
    The password is not echoed; a blank subdomain is HRMS_SUBDOMAIN's."""
    console.print()
    console.print(Panel(Text("Enter your HRMS email and password, or leave the email "
                             "blank to cancel."),
                        title="Log in to the HRMS", title_align="left",
                        border_style="yellow", box=box.ROUNDED))
    try:
        email = console.input("[bold yellow]Email:[/bold yellow] ").strip()
        if not email:
            return None
        password = console.input("[bold yellow]Password:[/bold yellow] ", password=True)
        if not password:
            return None
        subdomain = console.input(
            f"[bold yellow]Subdomain[/bold yellow] [dim]({SUBDOMAIN})[/dim]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return email, password, subdomain or SUBDOMAIN


def confirm(name: str, args: dict) -> bool:
    """Show exactly what will be sent and ask. Anything but y/yes is a no."""
    body = Text()
    body.append(f"  action  {name}\n", style="bold")
    for key, value in args.items():
        if key == "access_token":
            continue
        body.append(f"  {key:<16} ", style="dim")
        body.append(f"{json.dumps(value) if isinstance(value, list) else value}\n")
    console.print()
    console.print(Panel(body, title="About to submit via HRMS", title_align="left",
                        border_style="yellow", box=box.ROUNDED))
    return ask_yes()


def ask_yes() -> bool:
    try:
        answer = console.input("[bold yellow]Confirm? [y/N][/bold yellow] ")
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in ("y", "yes")


def screen(agent: Agent, question: str) -> tuple[str | None, str, bool]:
    """Guardrails and routing before the agent runs: (refusal, panel title,
    prefetch). `refusal` is None when the question goes ahead."""
    refusal = guard_input(question)
    if refusal:
        return refusal, "Refused", False
    # "apply wfh for me", "my balance": about the employee, so in scope. Without
    # a login the agent itself says so, which beats an "out of scope" refusal.
    action = bool(ACTION_RE.search(question))
    personal = action or bool(PERSONAL_RE.search(question))
    # "yes", "tomorrow", "do it": an answer to the agent's own question. Out of
    # context the classifier rejects it, and searched alone it matches nothing.
    reply = bool(agent.memory and agent.memory.history) and looks_like_followup(question)
    if reply:
        return None, "", False
    with waiting("checking the question"):
        route = route_question(question)
    if route == "no" and not personal:
        return OFF_TOPIC_REFUSAL, "Out of scope", False
    # The policy is searched only for a policy question. Excerpts in front
    # of an HRMS request only give the model things to ask about - WFH
    # quotas per band made it ask which WFH "type" to apply for. It can
    # still call search_policy itself. A personal question the classifier
    # rejected is searched, as every question was before routing. An
    # action with a date ("apply wfh for 5 oct") is never searched, however
    # it is routed; without one ("how many leaves can I take") the route decides.
    return None, "", route != "hrms" and not (action and date_hints(question))


def checked(answer: str) -> str | None:
    """None when the answer is safe to show, else the refusal to show instead."""
    unsafe = check_safety(answer, "ai") if answer else None
    if unsafe:
        err_console.print(f"[red]Output flagged unsafe: {unsafe}[/red]")
        return UNSAFE_OUTPUT
    return None


def ask_agent(agent: Agent, question: str, profile: str = "") -> str:
    """The agent-mode counterpart of chain.ask: same guardrails, same panels."""
    refusal, title, prefetch = screen(agent, question)
    if refusal:
        console.print()
        console.print(refusal_panel(refusal, title=title))
        return refusal
    answer = agent.run(question, prefetch=prefetch)
    unsafe = checked(answer)
    if unsafe:
        console.print()
        console.print(refusal_panel(unsafe, title="Unsafe output"))
        return unsafe
    console.print()
    if answer:
        console.print(Panel(Markdown(answer), border_style="cyan", box=box.ROUNDED,
                            title="Answer", title_align="left"))
    else:
        console.print(Panel(Text("(empty answer)", style="dim italic"),
                            border_style="dim", box=box.ROUNDED))
    return answer
