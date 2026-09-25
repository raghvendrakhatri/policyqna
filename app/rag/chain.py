"""Retrieval + generation, plus in-session memory and answer provenance.

`build_chain` composes retrieval and generation. `Memory` and `Provenance` are
per-session objects the chain reads on every invoke. `credit` and `ask` sit at
the boundary between the chain and the CLI.
"""

import re

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from rich import box
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .config import (
    CONTEXTUALIZE_PROMPT,
    KNOWLEDGE_PROMPT,
    MEMORY_SUMMARISE_PROMPT,
    MEMORY_SUMMARY_CHARS,
    MEMORY_WINDOW,
    MIN_SUB_K,
    MMR_FETCH_K,
    MMR_LAMBDA,
    PROFILE_PROMPT,
    SPLIT_PROMPT,
    SYSTEM_PROMPT,
)
from .db import store
from .guardrails import (
    REFUSAL,
    UNSAFE_OUTPUT,
    check_safety,
    clean,
    guard_input,
)
from .knowledge import load_knowledge, read_knowledge
from .models import llm
from .ui import console, err_console, refusal_panel, status, waiting


# A compound question averages into one vector that matches none of its topics,
# so it is split and each part retrieved separately.
COMPOUND = re.compile(r"\band\b|\s&\s|;", re.I)


def looks_compound(question: str) -> bool:
    return bool(COMPOUND.search(question)) or question.count("?") > 1


# A question that opens with one of these needs the previous turn to make sense:
# "and for Gold?", "what about tier 3?". Anything else already stands alone -
# rewriting it just gives the LLM a chance to inject the user's identity into
# what should be a topical query.
FOLLOWUP_OPENERS = re.compile(
    r"^\s*(and|but|also|what about|how about|and (for|in|about)|for|in|"
    r"can (i|you|it|they)|do (i|they|you|we)|does (it|he|she|that)|is (it|that)|"
    r"why|then|so)\b",
    re.I,
)


def looks_like_followup(question: str) -> bool:
    """True when the question genuinely refers back to the previous turn.

    A short question, or one opening with a linking word, still needs the
    conversation to make sense. A longer question that already names its topic
    ("what is the reimbursement policy") stands alone and must not be rewritten.
    """
    stripped = question.strip()
    if len(stripped.split()) <= 3:
        return True
    return bool(FOLLOWUP_OPENERS.match(stripped))


class NoContext(Exception):
    """Raised from gather() when retrieval returns nothing usable."""


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def build_chain(k: int, profile: str = "", provenance: "Provenance | None" = None,
                memory: "Memory | None" = None):
    """`provenance`, if given, records what each answer could have drawn on, so
    the caller can credit it. Built from the texts rather than asked of the
    model, which would sooner or later invent a page number.

    `memory`, if given, injects the running summary + rolling window into every
    call. RunnableLambda re-reads the memory object each invoke, so mutations
    after this returns are picked up."""
    model = llm()  # one model object, so both calls hit the same loaded weights
    vectors = store()

    split = (
        ChatPromptTemplate.from_messages([("system", SPLIT_PROMPT), ("human", "{question}")])
        | model
        | StrOutputParser()
    )
    knowledge = load_knowledge()
    if provenance is not None:
        provenance.profile = profile
        provenance.knowledge = read_knowledge()
    system = SYSTEM_PROMPT + (KNOWLEDGE_PROMPT if knowledge else "")
    system += PROFILE_PROMPT if profile else ""
    # MessagesPlaceholders keep summary + history out of the template escaping
    # path, so an HRMS field or a summariser output containing `{` won't blow
    # up prompt rendering the way an interpolated string would.
    prompt_messages: list = [("system", system)]
    if memory is not None:
        prompt_messages.append(MessagesPlaceholder("prior_messages", optional=True))
        prompt_messages.append(MessagesPlaceholder("history", optional=True))
    prompt_messages.append(("human", "Context:\n{context}\n\nQuestion: {question}"))
    answer_prompt = ChatPromptTemplate.from_messages(prompt_messages)
    # partial, not a chain input: both are fixed for the run, and substituting
    # once keeps braces inside a formula or a profile out of the templating.
    if knowledge:
        answer_prompt = answer_prompt.partial(knowledge=knowledge)
    if profile:
        answer_prompt = answer_prompt.partial(profile=profile)

    def subquestions(question: str) -> list[str]:
        if not looks_compound(question):
            return [question]
        parts = [line.strip(" -*\t") for line in clean(split.invoke(question)).splitlines()]
        return [p for p in parts if p] or [question]

    def gather(question: str) -> list[Document]:
        """Retrieve per sub-question, then merge - a union, not an average."""
        subs = subquestions(question)
        per_sub = k if len(subs) == 1 else max(MIN_SUB_K, k // len(subs))
        if len(subs) > 1:
            status(f"answering {len(subs)} parts, {per_sub} chunks each")
        retriever = vectors.as_retriever(
            search_type="mmr",  # diverse chunks, not k near-duplicates
            search_kwargs={
                "k": per_sub,
                "fetch_k": max(MMR_FETCH_K, per_sub * 5),
                "lambda_mult": MMR_LAMBDA,
            },
        )
        seen: set[str] = set()
        docs: list[Document] = []
        for sub in subs:
            for doc in retriever.invoke(sub):
                if doc.id not in seen:
                    seen.add(doc.id)
                    docs.append(doc)
        if provenance is not None:
            provenance.docs = docs  # this question's chunks, not the last one's
        # Retrieval guardrail: if nothing came back, the model would hallucinate
        # against an empty context. Raising here lets ask() print the refusal
        # instead of asking the model to answer from thin air.
        if not docs and not knowledge and not profile:
            raise NoContext()
        return docs

    inputs: dict = {
        "context": RunnableLambda(gather) | format_docs,
        "question": RunnablePassthrough(),
    }
    if memory is not None:
        inputs["prior_messages"] = RunnableLambda(lambda _: memory.prior_messages())
        inputs["history"] = RunnableLambda(lambda _: memory.messages())
    return inputs | answer_prompt | model | StrOutputParser()


def build_contextualizer():
    """Rewrites a follow-up into a standalone question. A separate ChatOllama
    object, but the same model name, so Ollama serves it from the same weights."""
    return (
        ChatPromptTemplate.from_messages(
            [("system", CONTEXTUALIZE_PROMPT), ("human", "{conversation}")]
        )
        | llm()
        | StrOutputParser()
    )


def build_summariser():
    """Small chain that folds one turn into a running one-line summary."""
    return (
        ChatPromptTemplate.from_messages([
            ("system", MEMORY_SUMMARISE_PROMPT),
            ("human",
             "Existing summary: {summary}\n\nNew turn:\nQ: {question}\nA: {answer}"),
        ])
        | llm()
        | StrOutputParser()
    )


class Memory:
    """In-RAM chat memory: a rolling verbatim window + a running summary.

    The last MEMORY_WINDOW turns reach the model as HumanMessage / AIMessage
    pairs. Everything older has already been folded into `summary`, one line,
    so the prompt-visible budget stays flat regardless of chat length.
    Nothing is persisted; letting the process exit is how you forget.
    """

    def __init__(self, summariser=None) -> None:
        self._summariser = summariser
        self.history: list[tuple[str, str]] = []
        self.summary: str = ""

    def messages(self) -> list:
        """The verbatim window as LangChain messages, newest last."""
        msgs = []
        for question, answer in self.history[-MEMORY_WINDOW:]:
            msgs.append(HumanMessage(content=question))
            msgs.append(AIMessage(content=answer))
        return msgs

    def prior_messages(self) -> list:
        """The summary as a system message, empty when there is none yet."""
        if not self.summary:
            return []
        return [SystemMessage(content=f"Summary of earlier conversation: {self.summary}")]

    def remember(self, question: str, answer: str) -> None:
        self.history.append((question, answer))
        # Once the buffer exceeds the window, the *oldest still-visible* turn is
        # about to fall off. Fold it into the summary before it disappears.
        if self._summariser is None or len(self.history) <= MEMORY_WINDOW:
            return
        oldest_q, oldest_a = self.history[-(MEMORY_WINDOW + 1)]
        try:
            with waiting("updating memory"):
                updated = self._summariser.invoke({
                    "summary": self.summary or "(nothing yet)",
                    "question": oldest_q,
                    "answer": oldest_a,
                })
            updated = clean(updated).strip()
            if updated:
                self.summary = updated[:MEMORY_SUMMARY_CHARS]
        except Exception as exc:  # noqa: BLE001 - a summary failure must not break the chat
            err_console.print(f"[dim]· memory summary skipped: {exc}[/dim]")


WORD = re.compile(r"[a-z]{4,}|\d[\d.]*")
# Retrieval hands the model more than it uses, and the profile and knowledge are
# in every prompt whether they are relevant or not. So a source is credited only
# where the answer's own wording overlaps it.
CITE_MAX = 4
CITE_SHARE = 0.6  # of the best-scoring source


class Provenance:
    """Everything the answer could have come from, so it can be credited.

    Three layers reach the model - the HRMS profile, the knowledge files and the
    retrieved chunks - and until this existed only the third was ever named. An
    answer about who someone is would be credited to whichever policy page
    happened to be retrieved alongside it.
    """

    def __init__(self) -> None:
        self.profile = ""
        self.knowledge: dict[str, str] = {}
        self.docs: list[Document] = []

    def candidates(self) -> list[tuple[str, str]]:
        named: list[tuple[str, str]] = []
        if self.profile:
            # Score the profile per top-level block, not as one blob. A long
            # profile has hundreds of unrelated words, so the sqrt-of-length
            # denominator drowns the one "wfh: remaining 13.0" line that
            # actually matched - and the HRMS never gets credited even when
            # the answer's own figure came from it.
            block: list[str] = []
            for line in self.profile.splitlines():
                # A top-level field starts flush left; a nested one is indented.
                if line and not line[0].isspace() and block:
                    named.append(("the HRMS", "\n".join(block)))
                    block = []
                block.append(line)
            if block:
                named.append(("the HRMS", "\n".join(block)))
        named += list(self.knowledge.items())
        return named + [(doc_label(d), d.page_content) for d in self.docs]


def doc_label(doc: Document) -> str:
    source = doc.metadata.get("source", "?")
    page = doc.metadata.get("page")
    return f"{source} p. {page}" if isinstance(page, int) else source


def terms(text: str) -> set[str]:
    return set(WORD.findall(text.lower().replace(",", "")))


def overlap(wanted: set[str], text: str) -> float:
    """How much of the answer this source accounts for.

    Figures count for much more than words: an answer of "13.0 days left" shares
    ordinary words with any file that discusses leave, but the 13.0 comes from
    exactly one place. Divided by the source's own size, so a long file does not
    out-score a terse one just by containing more words.
    """
    have = terms(text)
    if not have:
        return 0.0
    common = wanted & have
    figures = sum(1 for term in common if term[0].isdigit())
    return (len(common) + 4 * figures) / len(have) ** 0.5


def credit(answer: str, provenance: "Provenance") -> str:
    """The sources whose wording the answer overlaps, best first.

    A heuristic, but it is built from the text rather than asked of the model,
    so it cannot cite a page that does not exist.
    """
    wanted = terms(answer)
    scored = [(overlap(wanted, text), label) for label, text in provenance.candidates()]
    best = max((score for score, _ in scored), default=0)
    if not best:
        return ""
    seen: set[str] = set()
    keep: list[str] = []
    for score, label in sorted(scored, key=lambda p: -p[0]):
        if score < best * CITE_SHARE:
            break
        # Same source can score twice - the HRMS is split into blocks, a doc
        # into pages. Dedup by label so one big source cannot fill CITE_MAX.
        if label in seen:
            continue
        seen.add(label)
        keep.append(label)

    # One entry per document, with its pages gathered: "policy.pdf pp. 44, 46".
    pages: dict[str, list[str]] = {}
    for label in keep[:CITE_MAX]:
        name, _, page = label.partition(" p. ")
        pages.setdefault(name, [])
        if page and page not in pages[name]:  # two chunks can share a page
            pages[name].append(page)
    parts = []
    for name, numbers in pages.items():
        if not numbers:
            parts.append(name)
        elif len(numbers) == 1:
            parts.append(f"{name} p. {numbers[0]}")
        else:
            parts.append(f"{name} pp. {', '.join(sorted(numbers, key=int))}")
    return " - ".join(parts)


def ask(chain, question: str, provenance: "Provenance | None" = None) -> str:
    refusal = guard_input(question)
    if refusal:
        console.print()
        console.print(refusal_panel(refusal))
        console.print()
        return refusal
    try:
        with waiting("thinking"):
            answer = chain.invoke(question)
    except NoContext:
        console.print()
        console.print(refusal_panel(REFUSAL, title="No matching policy"))
        console.print()
        return REFUSAL
    cleaned = clean(answer)
    unsafe = check_safety(cleaned, "ai") if cleaned else None
    if unsafe:
        err_console.print(f"[red]Output flagged unsafe: {unsafe}[/red]")
        console.print()
        console.print(refusal_panel(UNSAFE_OUTPUT, title="Unsafe output"))
        console.print()
        return UNSAFE_OUTPUT
    console.print()
    if cleaned:
        console.print(Panel(Markdown(cleaned), border_style="cyan", box=box.ROUNDED,
                            title="Answer", title_align="left"))
    else:
        console.print(Panel(Text("(empty answer)", style="dim italic"),
                            border_style="dim", box=box.ROUNDED))
    console.print()
    if provenance is not None and cleaned:
        sources = credit(cleaned, provenance)
        if sources:
            console.print(f"[green]Sources[/green] [dim]·[/dim] {sources}\n")
        else:
            # Grounding guardrail: credit() couldn't tie the answer to any
            # retrieved chunk, knowledge file or profile field. Say so rather
            # than silently ship an unsourced answer.
            err_console.print("[yellow]! answer could not be tied to any indexed"
                              " source[/yellow]")
    return cleaned
