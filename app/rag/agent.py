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
import uuid
from dataclasses import dataclass
from datetime import date

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
    PERSONAL_RE,
    TOOL_RESULT_MAX_CHARS,
)
from .guardrails import (
    OFF_TOPIC_REFUSAL,
    UNSAFE_OUTPUT,
    check_safety,
    check_topicality,
    clean,
    guard_input,
)
from .knowledge import load_knowledge
from .models import llm
from .tools import TOOLS, find_policy
from .ui import console, err_console, refusal_panel, waiting


# Tools that change something in the HRMS. Each call is shown to the employee
# and runs only on an explicit yes.
WRITE_TOOLS = ("apply_leave", "apply_wfh")
STEPS_EXHAUSTED = "I couldn't finish that within the allowed number of steps."
NOT_LOGGED_IN = "The employee is not logged in. Call the login tool first, then retry."


@dataclass
class HRMSContext:
    """The session, shared by every run in a chat. Reaches the middleware and
    the tools, never the messages. The login tool fills both fields in."""

    access_token: str | None
    profile: str = ""


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
    model is told to call `login` first, which the employee approves before the
    browser opens. The system prompt is rebuilt on every model call from the
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

    def run(self, question: str) -> str:
        messages: list = []
        if self.memory is not None:
            messages += self.memory.prior_messages() + self.memory.messages()
        messages.append(HumanMessage(content=with_policy(question)))
        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            # Each step is a model call or a tool round; bound it.
            "recursion_limit": AGENT_MAX_STEPS * 2 + 1,
        }
        payload: dict | Command = {"messages": messages}
        try:
            while True:
                with waiting("thinking"):
                    state = self.graph.invoke(payload, config, context=self.context)
                interrupts = state.get("__interrupt__")
                if not interrupts:
                    break
                # One interrupt per model turn, holding every write call it made.
                requests = interrupts[0].value["action_requests"]
                decisions = [self.decide(r["name"], r["args"]) for r in requests]
                payload = Command(resume={"decisions": decisions})
        except GraphRecursionError:
            return STEPS_EXHAUSTED
        return clean(state["messages"][-1].content)

    def decide(self, name: str, args: dict) -> dict:
        """The employee's answer to one paused call. A write with no session is
        turned back without asking: confirming it would only end in an error."""
        if name == "login" and self.token:
            return {"type": "reject", "message": "Already logged in."}
        if name in WRITE_TOOLS and not self.token:
            return {"type": "reject", "message": NOT_LOGGED_IN}
        if confirm(name, args):
            return {"type": "approve"}
        if name == "login":
            return {"type": "reject", "message": "The employee chose not to log in."}
        return {"type": "reject",
                "message": "The employee declined, so nothing was submitted."}


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


def confirm(name: str, args: dict) -> bool:
    """Show exactly what will be sent and ask. Anything but y/yes is a no."""
    if name == "login":
        console.print()
        console.print(Panel(Text("Opens the HRMS login page in your browser."),
                            title="Log in to the HRMS?", title_align="left",
                            border_style="yellow", box=box.ROUNDED))
        return ask_yes()
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


def ask_agent(agent: Agent, question: str, profile: str = "") -> str:
    """The agent-mode counterpart of chain.ask: same guardrails, same panels."""
    refusal = guard_input(question)
    if refusal:
        console.print()
        console.print(refusal_panel(refusal))
        return refusal
    # "apply wfh for me", "my balance": about the employee, so in scope. Without
    # a login the agent itself says so, which beats an "out of scope" refusal.
    personal = bool(PERSONAL_RE.search(question))
    # "yes", "tomorrow", "do it": an answer to the agent's own question. Out of
    # context the classifier rejects it; the conversation is what makes it fit.
    reply = bool(agent.memory and agent.memory.history) and looks_like_followup(question)
    if not (personal or reply):
        with waiting("checking scope"):
            on_topic = check_topicality(question)
        if not on_topic:
            console.print()
            console.print(refusal_panel(OFF_TOPIC_REFUSAL, title="Out of scope"))
            return OFF_TOPIC_REFUSAL
    answer = agent.run(question)
    unsafe = check_safety(answer, "ai") if answer else None
    if unsafe:
        err_console.print(f"[red]Output flagged unsafe: {unsafe}[/red]")
        console.print()
        console.print(refusal_panel(UNSAFE_OUTPUT, title="Unsafe output"))
        return UNSAFE_OUTPUT
    console.print()
    if answer:
        console.print(Panel(Markdown(answer), border_style="cyan", box=box.ROUNDED,
                            title="Answer", title_align="left"))
    else:
        console.print(Panel(Text("(empty answer)", style="dim italic"),
                            border_style="dim", box=box.ROUNDED))
    return answer
