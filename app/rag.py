"""Single-file CLI RAG over policy documents, built on LangChain and Ollama.

Everything runs locally: Ollama serves both models, Postgres + pgvector stores
the index. No API keys, no network calls off the machine.

  chat   - ChatOllama  (a non-thinking instruct model: a thinking model spends
           ~90% of its tokens on reasoning that is then discarded)
  embed  - OllamaEmbeddings
  store  - langchain_postgres.PGVector, which owns its own schema
           (langchain_pg_collection / langchain_pg_embedding)

Commands: ingest | ask | chat | discover | profile | pa | stats | reset

Answers draw on three things: knowledge/ (in every prompt), the pgvector
index (retrieved per question), and the HRMS (fetched per session).
"""

import argparse
import contextlib
import json
import os
import re
import sys
import threading
import time

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy import create_engine, text

load_dotenv()

# Every one of these must be set; see .env.example. main() checks them up front,
# so env() below is only a backstop for an import-time or library-side read.
REQUIRED_ENV = (
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_PORT",
    "OLLAMA_BASE_URL",
    "CHAT_MODEL",
    "EMBED_MODEL",
    "NUM_CTX",
)


def env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"Missing env var: {name}. Copy .env.example to .env.")
    return value


MAX_ANSWER_TOKENS = 2048

# Loading a model costs far more than running it (~2 min on a busy 16GB machine),
# and every question needs both models, so keep both resident between questions.
KEEP_ALIVE = 3600  # seconds; OllamaEmbeddings rejects the "1h" string form
# Chunks are CHUNK_CHARS long, so the embedder never needs Ollama's default 4096
# context - the unused buffer is pure resident memory.
EMBED_CTX = 2048

COLLECTION = "policies"
# Facts that are not in any document - formulas, definitions, internal rules.
# Every file here goes into the system prompt on every question, so they can
# never be missed by retrieval the way an ingested chunk can. That only holds
# while the total stays small; past this the context is better spent on chunks.
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge")
KNOWLEDGE_SUFFIXES = (".md", ".txt")
KNOWLEDGE_SKIP = {"readme.md", "readme.txt"}  # notes about the folder, not facts
KNOWLEDGE_WARN_CHARS = 6000

# Who is asking. The profile comes from the HRMS, keyed by the employee's own
# access token - nothing about an employee is stored in this repo. PROFILE_DIR is
# only for working offline; it is gitignored and normally empty.
PROFILE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiles")
# The HRMS is a deployed service reached over the internet, so allow for more
# than a LAN round trip.
HRMS_TIMEOUT = 15
# An HRMS "me" endpoint answers for its own UI, not for us: it can carry every
# permission, menu and lookup table the app needs, which is far more than fits
# in NUM_CTX alongside the retrieved chunks. So the response is pruned hard.
PROFILE_MAX_CHARS = 4000
PROFILE_MAX_VALUE_CHARS = 200
PROFILE_MAX_LIST = 15  # enough for every leave type an employee holds
# Keys never worth spending context on. Credentials must not reach the prompt at
# all; the rest are simply bulk.
PROFILE_SKIP = re.compile(
    # Credentials. An HRMS "me" response can carry real key material, so this
    # errs heavily towards dropping.
    r"token|password|secret|credential|signature|api[_-]?key|\bkeys?\b"
    # Authorisation bulk: roles carry a policy per screen, and none of it helps
    # answer a question about leave.
    r"|permission|privilege|policies|policy_|scope|roles?\b"
    # Assets and markup.
    r"|avatar|photo|picture|image|logo|base64|__|html|css|resume|document"
    # Personal data a policy answer never needs. An HR record holds a lot of it,
    # and none of it should be sent to a model to answer a leave question.
    r"|birth|\bdob\b|gender|marital|blood|nationality|mobile|phone|address"
    # Pay. Out of scope here, and the most damaging field to leak.
    r"|bank|ctc\b|salary|remuneration|payment|pan_|gst_|tan_"
    # Third-party integration state.
    r"|jira|google_|microsoft_|fitbit|slack|calendar_|zoom"
    # Collections an HR record carries that no policy answer needs, and which
    # crowd out the fields that matter - the band sits after them in the record.
    r"|skills|projects|timeline|praise|compliment|issues|education|contacts"
    r"|squad|candidate|interests|favourite|about_me|gem_|milestone|form\b"
    # Scheme configuration. A leave type ships the whole rulebook - accrual,
    # approval chains, sandwich policy, encashment - which is both enormous and
    # already answered by the handbook. The approval chains also carry other
    # employees' records, which must not reach the prompt at all.
    r"|configuration|accrual|approval|assignee|restriction|sandwich|usage_limit"
    r"|encashment|carry_?forward|prior_notice|probation_config|notice_period_config"
    r"|reasons|color|is_description|is_system_generated|quota_unit"
    r"|is_reset|reset_date|is_floater|leave_type_id"
    # Account flags that describe the HRMS account, not the employee.
    r"|is_staff|is_super|is_editable|is_removable|is_default|is_verified"
    r"|is_interviewer|not_joined|old_status|invited"
    # Opaque identifiers. Useful to a frontend, meaningless to the model, and
    # they are most of the payload by volume.
    r"|(^|_)ids?$|uuid",
    re.I,
)
# Envelopes an API wraps its payload in: {"status": ..., "data": {...}}.
PROFILE_ENVELOPE = ("data", "result", "payload", "user", "profile")
# `--token` with no value: prompt instead, so a live credential never reaches
# the shell history or the process list.
TOKEN_PROMPT = "\0prompt"
# `--login`: open the HRMS login page in a real browser and read the session
# back out, so nobody has to go digging in devtools.
TOKEN_LOGIN = "\0login"
LOGIN_TIMEOUT = 300
# How many past turns of a chat are replayed when rewriting a follow-up. Two is
# enough for "and for Gold?" and keeps the rewrite call cheap.
HISTORY_TURNS = 2
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
TOP_K = 8
BATCH = 25

# A compound question averages into one vector that matches none of its topics,
# so it is split and each part retrieved separately. MIN_SUB_K keeps the joined
# context from outgrowing NUM_CTX when a question has many parts.
MIN_SUB_K = 3
MMR_FETCH_K = 30
# MMR's own default (0.5) diversifies so hard it drops neighbouring clauses of
# the very section being asked about; 0.8 keeps relevance in charge.
MMR_LAMBDA = 0.8

SYSTEM_PROMPT = """You answer questions about the policy documents given as context.

- Use only the context and the reference facts. If the answer is in neither, say
  so plainly.
- Quote the policy's wording for anything binding (limits, deadlines, exclusions).
- Give what the policy actually says. Never answer with a pointer to a section
  number or heading when the text itself is in the context.
- Answer the question that was asked, and stop. Do not add related facts that
  were not asked for, however useful they look.
- If the question has several parts, answer every one of them, each under its
  own short heading.
- Answer directly: no preamble, no reasoning, no citations or source references."""

# Appended to SYSTEM_PROMPT only when knowledge/ has files. The text arrives as a
# partial template variable, so braces or percent signs in a formula stay literal.
KNOWLEDGE_PROMPT = """

Reference facts. These are authoritative and always apply, even when the context
below says nothing about them. Use a formula exactly as written here, and show
the substituted numbers when you apply one.

{knowledge}"""


# Appended after KNOWLEDGE_PROMPT when a profile is loaded. Same partial-variable
# treatment, for the same reason: the data is arbitrary text from elsewhere.
PROFILE_PROMPT = """

The person asking is:

{profile}

Answer as them: apply whichever of their details the question actually needs -
their band, their city tier, their balances - and name the detail you used.

Anything asked about them is answered from this profile - their email, their
manager, their joining date. But answer with the part they asked for and nothing
else: "who am I" wants their name and role, not their leave balance, their
account flags or their employer's tax details.

When the question is about them - "my", "I", "do I" - their own figure in this
profile is the answer, not the policy's general rule. Asked what they have left,
give their remaining balance, not the yearly allocation everyone gets. Their
figures are as of the date in the profile, so say so when quoting one.

The exception is where the reference facts above say which source wins. Follow
that, and follow it over this paragraph.

Example
  in:  who am I?
  out: You are <full name>, <position> (<position level>, <band> band).
  and nothing further - not the balances, the work mode, the manager or the
  account flags.

Example
  in:  how many WFH days do I have left?
  out: You have <their remaining balance> left, as of <the profile date>.
  and not the yearly allocation, unless they asked for that too."""

# A follow-up question is rewritten against the conversation before it reaches
# retrieval: "and for Gold?" embeds to nothing on its own.
CONTEXTUALIZE_PROMPT = """Rewrite the follow-up question so that it stands alone.

- Carry the subject over from the conversation. A follow-up opening with "and",
  "what about" or "for X?" never stands alone - name in full the thing it asks
  about, even when that means repeating the previous question almost verbatim.
- Keep the user's own wording for whatever they did say.
- Output the rewritten question and nothing else.

Example
  Q: what is my daily food allowance?
  A: 1,900 per day, as Gold band in a Tier 1 city.
  Follow-up: and in a tier 3 city?
  out: what is my daily food allowance in a tier 3 city?

Example
  Q: how many WFH days do I get?
  A: 30 days a year.
  Follow-up: do they carry forward?
  out: do WFH days carry forward to the next year?"""

SPLIT_PROMPT = """Split the question into the separate questions it literally contains.

- Output one question per line. No numbering, no bullets, no other text.
- Split only where the user actually asked for two things. Never invent a
  question they did not ask, and never break one topic into sub-topics.
- Correct obvious typos and make each line a standalone question.

Example
  in:  what is the notice period policy?
  out: what is the notice period policy?

Example
  in:  what is resignation poliyc and company missiona
  out: what is the resignation policy?
       what is the company mission?"""

# The splitter is only worth a call when the question really has several parts;
# left to itself on a simple question it invents sub-topics that dilute retrieval.
COMPOUND = re.compile(r"\band\b|\s&\s|;", re.I)


def looks_compound(question: str) -> bool:
    return bool(COMPOUND.search(question)) or question.count("?") > 1

# CHAT_MODEL is a non-thinking instruct model, so nothing should reach these.
# They are the backstop for a thinking model: note that on such a model
# `reasoning=False` does NOT stop it thinking, it merges the thinking into the
# reply - pass `reasoning=True` to route it into its own field instead.
REASONING = re.compile(r"<(reasoning|analysis|think|thinking)>.*?</\1>", re.S | re.I)
STRAY_CLOSE = re.compile(r"^.*?</(reasoning|analysis|think|thinking)>", re.S | re.I)


# ----------------------------------------------------------------------- models


def embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=env("EMBED_MODEL"),
        base_url=env("OLLAMA_BASE_URL"),
        num_ctx=EMBED_CTX,
        keep_alive=KEEP_ALIVE,
    )


def llm() -> ChatOllama:
    return ChatOllama(
        model=env("CHAT_MODEL"),
        base_url=env("OLLAMA_BASE_URL"),
        temperature=0,
        keep_alive=KEEP_ALIVE,
        num_ctx=int(env("NUM_CTX")),
        num_predict=MAX_ANSWER_TOKENS,
        validate_model_on_init=True,  # fail with a clear message if it isn't pulled
    )


# --------------------------------------------------------------------------- db


def dsn() -> str:
    """SQLAlchemy URL for the psycopg3 driver, which PGVector requires."""
    return (
        f"postgresql+psycopg://{env('POSTGRES_USER')}:{env('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{env('POSTGRES_PORT')}"
        f"/{env('POSTGRES_DB')}"
    )


def store() -> PGVector:
    return PGVector(
        embeddings=embeddings(),
        connection=dsn(),
        collection_name=COLLECTION,
        use_jsonb=True,
    )


# Scopes a raw statement to this collection, since the tables are shared.
IN_COLLECTION = (
    "collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = :collection)"
)


def sql(statement: str, **params):
    """PGVector has no delete-by-metadata or count, so those go through SQL."""
    params.setdefault("collection", COLLECTION)
    with create_engine(dsn()).begin() as conn:
        result = conn.execute(text(statement), params)
        return result.fetchall() if result.returns_rows else []


# ---------------------------------------------------------------- load + chunk


# A contents page lists every policy name, so it scores well against almost any
# question while holding no policy text - it crowds real answers out of the
# top-k. These two patterns spot the heading-only lines such a page is made of.
NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
CAPS_HEADING = re.compile(r"^[A-Z][A-Z 0-9\u2019'&/(),.-]{3,}$")


def looks_like_contents(text: str, threshold: float = 0.8, min_lines: int = 4) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < min_lines:
        return False
    headings = sum(
        1
        for line in lines
        if len(line) <= 60 and (NUMBERED_HEADING.match(line) or CAPS_HEADING.match(line))
    )
    return headings / len(lines) >= threshold


def load_document(path: str) -> list[Document]:
    """One Document per page, so page numbers survive into chunk metadata."""
    source = os.path.basename(path)
    if path.lower().endswith(".pdf"):
        pages = [(i, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages, 1)]
        skipped = [i for i, text in pages if looks_like_contents(text)]
        if skipped:
            print(f"Skipping {len(skipped)} contents page(s): {skipped}")
        return [
            Document(page_content=text, metadata={"source": source, "page": i})
            for i, text in pages
            if i not in set(skipped)
        ]
    with open(path, encoding="utf-8") as fh:
        return [Document(page_content=fh.read(), metadata={"source": source})]


def chunk_document(path: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = [c for c in splitter.split_documents(load_document(path)) if c.page_content.strip()]
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def chunk_id(chunk: Document) -> str:
    """Deterministic, so a re-run can tell what is already stored."""
    return f"{chunk.metadata['source']}#{chunk.metadata['chunk_index']}"


# ------------------------------------------------------------------- knowledge


def knowledge_files() -> list[str]:
    if not os.path.isdir(KNOWLEDGE_DIR):
        return []
    return sorted(
        os.path.join(KNOWLEDGE_DIR, name)
        for name in os.listdir(KNOWLEDGE_DIR)
        if name.endswith(KNOWLEDGE_SUFFIXES)
        and not name.startswith(".")
        and name.lower() not in KNOWLEDGE_SKIP
    )


def load_knowledge() -> str:
    """Every knowledge file, concatenated and headed by its filename."""
    parts = []
    for path in knowledge_files():
        with open(path, encoding="utf-8") as fh:
            text = fh.read().strip()
        if text:
            parts.append(f"[{os.path.basename(path)}]\n{text}")
    joined = "\n\n".join(parts)
    if len(joined) > KNOWLEDGE_WARN_CHARS:
        print(
            f"Warning: knowledge/ is {len(joined)} chars and is sent with every question."
            " Consider ingesting the longer files instead.",
            file=sys.stderr,
        )
    return joined


NAME_KEYS = ("name", "full_name", "title", "label")


def collapse_named(value: dict, kept: dict, depth: int, in_list: bool):
    """Reduce a nested record to its name, where the name is all it is worth.

    Two cases only:

    - a person, so `reporting_to` holding a colleague's entire HR record - their
      date of birth, their phone number - becomes just "Aayush Sharma";
    - a lookup with nothing left but its name, so `position: {id, name}` becomes
      "Backend Node Engineer".

    Anything else keeps its fields: a leave type of `{name: "Sick", balance: 3}`
    is about the 3, and collapsing it to "Sick" would throw the answer away.
    Never applied to a list item or to a merged source's root.
    """
    if depth == 0 or in_list:
        return None
    person = value.get("full_name")
    if isinstance(person, str) and person.strip():
        return person.strip()
    if kept and set(kept) <= set(NAME_KEYS):
        for key in NAME_KEYS:
            if isinstance(kept.get(key), str) and kept[key].strip():
                return kept[key].strip()
    return None


def all_zero(value) -> bool:
    """A record whose every number is zero, which says nothing.

    A leave type the HRMS has not configured comes back as allocated 0, used 0,
    balance 0. That is absence of data, but it reads as an entitlement of zero
    and the model will quote it over the handbook's actual grant. Dropping it
    lets the handbook answer, which is what it is authoritative for.

    A genuinely spent balance is not this: allocated 6, used 6, balance 0 still
    has a 6 in it, so it stays.
    """
    if not isinstance(value, dict):
        return False
    numbers = [v for v in value.values() if isinstance(v, (int, float))
               and not isinstance(v, bool)]
    return bool(numbers) and not any(numbers)


def prune_profile(value, depth: int = 0, in_list: bool = False, protect=()):
    """Strip an HRMS response down to what is worth prompt space.

    Drops credentials and bulk by key name, empty values, over-long strings and
    over-long lists. Still names no HRMS field: it is all shape and size, so a
    different HRMS needs no change here.
    """
    if isinstance(value, dict):
        kept = {}
        for key, item in value.items():
            if PROFILE_SKIP.search(key):
                continue
            # A merged source sits under its own label but is a root, not a
            # lookup: collapsing it would throw the whole response away.
            child = 0 if depth == 0 and key in protect else depth + 1
            item = prune_profile(item, child, protect=protect)
            if item not in (None, "", [], {}):
                kept[key] = item
        # Decided after pruning, so "is there anything here but a name?" is
        # asked of what survived, not of what the HRMS sent.
        named = collapse_named(value, kept, depth, in_list)
        return kept if named is None else named
    if isinstance(value, list):
        items = [prune_profile(v, depth + 1, in_list=True, protect=protect)
                 for v in value[:PROFILE_MAX_LIST]]
        return [v for v in items if v not in (None, "", [], {}) and not all_zero(v)]
    if isinstance(value, str) and len(value) > PROFILE_MAX_VALUE_CHARS:
        return value[:PROFILE_MAX_VALUE_CHARS] + "..."
    return value


def unwrap_profile(data: dict) -> dict:
    """Step past a {"status": ..., "message": ..., "data": {...}} envelope.

    Only when the wrapper holds nothing else of substance, so a response whose
    real fields sit at the top level is left alone.
    """
    while isinstance(data, dict):
        inner = next((k for k in PROFILE_ENVELOPE if isinstance(data.get(k), dict)), None)
        if not inner or len(data) > 4:
            return data
        data = data[inner]
    return data


def profile_from(data: dict) -> str:
    """Prune, optionally narrow to chosen fields, render, and cap the result."""
    data = unwrap_profile(data)
    wanted = [f.strip() for f in os.getenv("HRMS_PROFILE_FIELDS", "").split(",") if f.strip()]
    if wanted:
        data = {k: v for k, v in data.items() if k in wanted}
        missing = [f for f in wanted if f not in data]
        if missing:
            print(f"HRMS_PROFILE_FIELDS not in the response: {', '.join(missing)}",
                  file=sys.stderr)
    # Each merged source's label names a root that must survive pruning whole.
    labels = {split_source(s)[0] for s in profile_sources()}
    return fit_profile(prune_profile(data, protect=labels))


def fit_profile(data: dict, cap: int = PROFILE_MAX_CHARS) -> str:
    """Render within the budget, taking the space from whatever is largest.

    Cutting the tail would make the answer depend on the order sources happen to
    be configured in - one long list of leave types could push a band off the
    end. Instead every block gets an equal share, whatever is under its share
    keeps all of it, and only the blocks that are over get trimmed.
    """
    blocks = {key: render_profile({key: value}) for key, value in data.items()}
    if sum(len(b) for b in blocks.values()) <= cap:
        return "\n".join(blocks.values())

    over = dict(blocks)
    budget = cap
    while over:
        share = budget // len(over)
        small = {k: v for k, v in over.items() if len(v) <= share}
        if not small:
            break
        budget -= sum(len(v) for v in small.values())
        over = {k: v for k, v in over.items() if k not in small}
    share = budget // len(over) if over else 0

    trimmed = []
    for key, block in blocks.items():
        if key not in over:
            trimmed.append(block)
            continue
        kept, used = [], 0
        for line in block.splitlines():
            if used + len(line) > share:
                kept.append("  (...)")
                break
            kept.append(line)
            used += len(line) + 1
        trimmed.append("\n".join(kept))
        print(f"Profile: '{key}' was too long for the context and was trimmed.",
              file=sys.stderr)
    return "\n".join(trimmed)


def render_profile(data: dict, indent: int = 0) -> str:
    """JSON as an indented labelled list. Models read that far better than they
    read braces, and it stays generic, so a new HRMS field needs no code here."""
    pad = "  " * indent
    lines = []
    for key, value in data.items():
        label = key.replace("_", " ")
        if isinstance(value, dict):
            lines.append(f"{pad}- {label}:")
            lines.append(render_profile(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{pad}- {label}:")
            for item in value:
                lines.append(f"{pad}  - {flatten(item)}")
        else:
            lines.append(f"{pad}- {label}: {flatten(value)}")
    return "\n".join(lines)


def flatten(value) -> str:
    """One list item on one line: a dict becomes 'Quality: weight 0.6, ...'."""
    if isinstance(value, dict):
        name = value.get("name")
        rest = {k: v for k, v in value.items() if k != "name"}
        body = ", ".join(f"{k.replace('_', ' ')} {flatten(v)}" for k, v in rest.items())
        return f"{name}: {body}" if name else body
    return "none recorded" if value is None else str(value)


def profile_sources() -> list[str]:
    """Where the employee's details live. One HRMS service rarely holds all of
    it - identity in one, leave balances in another, payroll in a third."""
    raw = os.getenv("HRMS_PROFILE_SOURCES") or os.getenv("HRMS_PROFILE_PATH", "api/me")
    return [s.strip() for s in raw.split(",") if s.strip()]


def split_source(source: str) -> tuple[str, str, list[str]]:
    """`leave=https://.../balance|balance_days,band` -> label, url, fields.

    The label groups the fields in the prompt. The optional `|fields` list keeps
    only those keys from that response - worth pinning when a record is large
    and the field that matters sits at the end of it.
    """
    source, _, raw_fields = source.partition("|")
    fields = [f.strip() for f in raw_fields.split("+") if f.strip()]
    label, _, target = source.partition("=")
    if not target:  # no label given; name it after the last useful path segment
        target = label
        label = [p for p in target.rstrip("/").split("/") if p and "{" not in p][-1]
    return label.strip().replace("_", " "), target.strip(), fields


def absolute(target: str) -> str:
    if target.startswith("http://") or target.startswith("https://"):
        return target
    base = os.getenv("HRMS_BASE_URL")
    if not base:
        sys.exit("Set HRMS_BASE_URL in .env, or give each source a full URL.")
    return base.rstrip("/") + "/" + target.lstrip("/")


def scalars(data, into: dict) -> dict:
    """Flatten every scalar by key name, so a later URL can use {some_id}."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                scalars(value, into)
            elif value not in (None, ""):
                into.setdefault(key, value)
    elif isinstance(data, list):
        for item in data:
            scalars(item, into)
    return into


def fill(url: str, known: dict) -> str | None:
    """Substitute {attendance_employee_id} and friends from what is known so
    far. Returns None when a placeholder cannot be filled."""
    for name in re.findall(r"\{(\w+)\}", url):
        if name not in known:
            print(f"Skipping {url}: no '{name}' in the earlier responses.", file=sys.stderr)
            return None
        url = url.replace("{" + name + "}", str(known[name]))
    return url


def fetch_profile(token: str) -> dict:
    """Read every configured source with the employee's own token and merge.

    The first source's fields sit at the top level; each later one is nested
    under its label. Whatever each returns is used as-is - no field is named
    here, so an HRMS that renames one needs no change.
    """
    merged: dict = {}
    known: dict = {}
    for index, source in enumerate(profile_sources()):
        label, target, fields = split_source(source)
        url = fill(absolute(target), known)
        if url is None:
            continue
        data = unwrap_profile(fetch_json(url, token))
        scalars(data, known)  # before narrowing: a later URL may need a dropped id
        if fields and isinstance(data, dict):
            missing = [f for f in fields if f not in data]
            if missing:
                print(f"{label}: no {', '.join(missing)} in {url}", file=sys.stderr)
            data = {k: v for k, v in data.items() if k in fields}
        if index == 0 and isinstance(data, dict):
            merged.update(data)
        else:
            merged[label] = data
    return merged


def fetch_json(url: str, token: str) -> dict:
    import urllib.error
    import urllib.request

    if not url.startswith("https://") and "//127.0.0.1" not in url and "//localhost" not in url:
        sys.exit(f"Refusing to send the HRMS token over plain HTTP to {url}. Use https://.")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    # Some HRMS services want a header of their own alongside the session, such
    # as an x-client-id. HRMS_HEADERS carries them as "Name: value" pairs.
    for pair in os.getenv("HRMS_HEADERS", "").split(","):
        name, sep, value = pair.partition(":")
        if sep and name.strip():
            headers[name.strip()] = value.strip()
    request = urllib.request.Request(url, headers=headers)
    host = url.split("/")[2] if "//" in url else url
    try:
        with waiting(f"reading {host}"), urllib.request.urlopen(
            request, timeout=HRMS_TIMEOUT
        ) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # The body usually says exactly what is wrong, and every HRMS picks its
        # own code for a bad session - 401, 403, and 422 are all in use - so
        # quote the server rather than guessing from the number alone.
        try:
            detail = exc.read(500).decode("utf-8", "replace").strip()
        except Exception:  # noqa: BLE001 - a body is a nicety, not a requirement
            detail = ""
        hint = "\nLog in again with --login." if exc.code in (401, 403, 422) else ""
        # The token itself is never printed.
        sys.exit(f"HRMS returned {exc.code} for {url}.\n{detail}{hint}")
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(f"Cannot reach the HRMS at {url}: {exc}")
    except json.JSONDecodeError:
        sys.exit(f"HRMS did not return JSON from {url}. Is that source right?")


def read_profile_file(name: str) -> dict:
    """A saved response, for working with no HRMS reachable. PROFILE_DIR is
    gitignored: real employee data must not be committed."""
    path = os.path.join(PROFILE_DIR, f"{name.lower()}.json")
    if not os.path.exists(path):
        sys.exit(f"No profile file at {path}. Use --token to read from the HRMS.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def resolve_token(token: str | None) -> str | None:
    """Turn the --login / --prompt-token sentinels into an actual session."""
    if token == TOKEN_LOGIN:
        import hrms_login

        login_url = os.getenv("HRMS_LOGIN_URL") or os.getenv("HRMS_BASE_URL")
        if not login_url:
            sys.exit("Set HRMS_LOGIN_URL (or HRMS_BASE_URL) in .env to log in.")
        token = hrms_login.capture_token(
            login_url,
            api_hint=os.getenv("HRMS_API_HINT", ""),
            timeout=LOGIN_TIMEOUT,
        )
    if token == TOKEN_PROMPT:
        import getpass

        token = getpass.getpass("HRMS token (not echoed): ")
    # An empty token means HRMS_TOKEN is set but blank, or --token was passed
    # with nothing. Falling back to a generic answer there would look
    # personalised and quietly not be, so say so instead.
    if token is not None and not token.strip():
        sys.exit("--token (or HRMS_TOKEN) is empty. Omit it entirely for a generic answer.")
    return token


def profile_text(token: str | None, name: str | None) -> str:
    token = resolve_token(token)
    if token:
        return profile_from(fetch_profile(token))
    if name:
        return profile_from(read_profile_file(name))
    return ""


# -------------------------------------------------------------------- commands


def cmd_ingest(path: str, replace: bool) -> int:
    if not os.path.exists(path):
        sys.exit(f"No such file: {path}")

    chunks = chunk_document(path)
    if not chunks:
        sys.exit(f"No extractable text in {path} (scanned PDF? it would need OCR).")
    source = chunks[0].metadata["source"]

    vectors = store()
    if replace:
        sql(
            "DELETE FROM langchain_pg_embedding"
            f" WHERE {IN_COLLECTION} AND cmetadata->>'source' = :source",
            source=source,
        )
        todo = chunks
    else:
        ids = [chunk_id(c) for c in chunks]
        done = {d.id for d in vectors.get_by_ids(ids)}
        todo = [c for c in chunks if chunk_id(c) not in done]
        if not todo:
            print(f"'{source}' is already indexed ({len(chunks)} chunks). Use --replace to rebuild.")
            return 0
        if done:
            print(f"Resuming '{source}': {len(done)} stored, {len(todo)} to go.")

    print(f"Embedding {len(todo)} chunks via {env('EMBED_MODEL')} ...")
    stored = 0
    try:
        for start in range(0, len(todo), BATCH):
            batch = todo[start:start + BATCH]
            vectors.add_documents(batch, ids=[chunk_id(c) for c in batch])
            stored += len(batch)
            print(f"\r  {stored}/{len(todo)} stored", end="", flush=True)
        print()
    except KeyboardInterrupt:
        print(f"\nStopped after {stored}/{len(todo)}. Re-run to resume.", file=sys.stderr)
        raise

    print(f"Indexed {len(chunks)} chunks as '{source}'.")
    return 0


# ---------------------------------------------------------------------- waiting


SPINNER_FRAMES = "|/-\\"
SPINNER_INTERVAL = 0.12
# Whatever is spinning right now, so a progress note can retitle it instead of
# printing over the top of it.
_SPINNER: "Spinner | None" = None


class Spinner:
    """A frame, a label and the seconds so far, on one rewritten line.

    Everything here runs locally, so a question can sit for a minute while
    Ollama loads a model. Without this the terminal looks hung.
    """

    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._width = 0

    def _spin(self) -> None:
        start = time.monotonic()
        for tick in range(sys.maxsize):
            if self._stop.wait(SPINNER_INTERVAL):
                break
            frame = SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]
            line = f"{frame} {self.label}... {time.monotonic() - start:.0f}s"
            self._width = max(self._width, len(line))
            sys.stderr.write("\r" + line.ljust(self._width))
            sys.stderr.flush()
        sys.stderr.write("\r" + " " * self._width + "\r")
        sys.stderr.flush()

    def __enter__(self) -> "Spinner":
        global _SPINNER
        _SPINNER = self
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        global _SPINNER
        self._stop.set()
        self._thread.join(timeout=1)
        _SPINNER = None


@contextlib.contextmanager
def waiting(label: str):
    """No spinner when stderr is redirected: it would only litter the file."""
    if not sys.stderr.isatty():
        yield
        return
    with Spinner(label):
        yield


def status(message: str) -> None:
    """A progress note. Retitles the spinner if one is running, so the two do
    not fight over the same line."""
    if _SPINNER is not None:
        _SPINNER.label = message
    else:
        print(message, file=sys.stderr)


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def clean(text: str) -> str:
    return STRAY_CLOSE.sub("", REASONING.sub("", text)).strip()


def build_chain(k: int, profile: str = "", retrieved: list | None = None):
    """`retrieved`, if given, is refilled with the chunks used by each question,
    so the caller can cite them. Taken from the chunks themselves rather than
    asked of the model, which would sooner or later invent a page number."""
    model = llm()  # one model object, so both calls hit the same loaded weights
    vectors = store()

    split = (
        ChatPromptTemplate.from_messages([("system", SPLIT_PROMPT), ("human", "{question}")])
        | model
        | StrOutputParser()
    )
    knowledge = load_knowledge()
    system = SYSTEM_PROMPT + (KNOWLEDGE_PROMPT if knowledge else "")
    system += PROFILE_PROMPT if profile else ""
    answer_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "Context:\n{context}\n\nQuestion: {question}")]
    )
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
        if retrieved is not None:
            retrieved.clear()  # this question's chunks, not the last one's
            retrieved.extend(docs)
        return docs

    return (
        {"context": RunnableLambda(gather) | format_docs, "question": RunnablePassthrough()}
        | answer_prompt
        | model
        | StrOutputParser()
    )


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


WORD = re.compile(r"[a-z]{4,}|\d[\d.]*")
# Retrieval hands the model more than it uses, so citing everything retrieved
# would credit pages the answer never drew on. Keep the chunks the answer
# actually overlaps, and at most this many.
CITE_MAX = 4


def terms(text: str) -> set[str]:
    return set(WORD.findall(text.lower().replace(",", "")))


def used_docs(answer: str, docs: list[Document]) -> list[Document]:
    """The retrieved chunks the answer appears to have come from.

    Scored by shared wording, which is a heuristic - but a citation naming a
    page the answer never used is worse than a slightly short list.
    """
    wanted = terms(answer)
    if not wanted:
        return docs[:CITE_MAX]
    scored = [(len(wanted & terms(d.page_content)), d) for d in docs]
    best = max((score for score, _ in scored), default=0)
    if not best:
        return docs[:CITE_MAX]
    keep = [d for score, d in sorted(scored, key=lambda p: -p[0]) if score >= best * 0.6]
    return keep[:CITE_MAX]


def cite(docs: list[Document]) -> str:
    """"policy.pdf pp. 44, 46 - perks-and-benefits.md". Pages come from the
    chunk metadata, so a document with none (a Markdown file is one Document)
    is named without them."""
    pages: dict[str, set[int]] = {}
    for doc in docs:
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page")
        pages.setdefault(source, set())
        if isinstance(page, int):
            pages[source].add(page)
    parts = []
    for source in sorted(pages):
        numbers = sorted(pages[source])
        if not numbers:
            parts.append(source)
        elif len(numbers) == 1:
            parts.append(f"{source} p. {numbers[0]}")
        else:
            parts.append(f"{source} pp. {', '.join(str(n) for n in numbers)}")
    return " - ".join(parts)


def ask(chain, question: str, retrieved: list | None = None) -> str:
    with waiting("thinking"):
        answer = chain.invoke(question)
    cleaned = clean(answer)
    print("\n" + (cleaned or "(empty answer)") + "\n")
    if retrieved and cleaned:
        print(f"Sources: {cite(used_docs(cleaned, retrieved))}\n")
    return cleaned


def cmd_ask(question: str, k: int, token: str | None, profile_name: str | None,
            sources: bool) -> int:
    if not indexed():
        sys.exit("Nothing indexed yet - run `ingest` first.")
    retrieved: list | None = [] if sources else None
    chain = build_chain(k, profile_text(token, profile_name), retrieved)
    ask(chain, question, retrieved)
    return 0


def cmd_chat(k: int, token: str | None, profile_name: str | None, sources: bool) -> int:
    if not indexed():
        sys.exit("Nothing indexed yet - run `ingest` first.")
    # Fetched once per session, not per question: a chat would otherwise hammer
    # the HRMS, and the balances should not shift underneath a conversation.
    profile = profile_text(token, profile_name)
    retrieved: list | None = [] if sources else None
    chain = build_chain(k, profile, retrieved)  # built once, so one warm-up
    contextualize = build_contextualizer()
    history: list[tuple[str, str]] = []

    if profile:
        print("Answering for the employee the HRMS returned." if token
              else f"Answering from the saved profile '{profile_name}'.")
    print("Ask about the indexed policies. Ctrl-C or empty line to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0

        standalone = question
        if history:
            recent = "\n".join(
                f"Q: {q}\nA: {a}" for q, a in history[-HISTORY_TURNS:]
            )
            with waiting("reading your follow-up"):
                rewritten = contextualize.invoke(
                    {"conversation": f"{recent}\n\nFollow-up: {question}"}
                )
            standalone = clean(rewritten).strip() or question
            if standalone != question:
                print(f"(reading that as: {standalone})", file=sys.stderr)

        answer = ask(chain, standalone, retrieved)
        history.append((standalone, answer))


def counts_by_source() -> list[tuple[str, int]] | None:
    """None when the store has never been created."""
    exists = sql("SELECT to_regclass('public.langchain_pg_embedding')")[0][0]
    if exists is None:
        return None
    return [
        (row[0], row[1])
        for row in sql(
            "SELECT cmetadata->>'source', count(*) FROM langchain_pg_embedding"
            f" WHERE {IN_COLLECTION} GROUP BY 1 ORDER BY 1"
        )
    ]


def indexed() -> bool:
    return bool(counts_by_source())


def cmd_stats() -> int:
    rows = counts_by_source()
    if rows is None:
        print("No index yet - run `ingest` first.")
        return 0
    print("Index is empty." if not rows else "")
    for source, count in rows:
        print(f"{source:<40} {count:>6} chunks")
    files = knowledge_files()
    if files:
        print(f"\nknowledge/ (sent with every question, {len(load_knowledge())} chars):")
        for path in files:
            print(f"  {os.path.basename(path)}")
    return 0


def cmd_reset() -> int:
    store().delete_collection()
    print(f"Dropped the '{COLLECTION}' collection.")
    return 0


# ------------------------------------------------------------------------ main


def cmd_discover() -> int:
    """Find the HRMS endpoint that returns the employee's own record."""
    import hrms_login

    login_url = os.getenv("HRMS_LOGIN_URL") or os.getenv("HRMS_BASE_URL")
    if not login_url:
        sys.exit("Set HRMS_LOGIN_URL (or HRMS_BASE_URL) in .env first.")
    rows = hrms_login.discover(login_url, api_hint=os.getenv("HRMS_API_HINT", ""))
    if not rows:
        print("\nNo JSON API calls seen. Is HRMS_API_HINT too narrow?")
        return 0
    print(f"\n{len(rows)} endpoint(s) seen:\n")
    for method, url, status, keys in rows:
        print(f"{method} {status}  {url}\n    keys: {keys}\n")
    print("Pick the one holding band, city tier and balances, then set")
    print("HRMS_BASE_URL and HRMS_PROFILE_PATH in .env to its two halves.")
    return 0


# The scale feedback is given on, which is what finalScore is normalised by.
PA_SCALE = 5.0


def pa_score(criteria: list[dict], client_weight: float, team_weight: float,
             scale: float = PA_SCALE) -> dict:
    """The Performance Allowance score, computed rather than reasoned about.

    A model doing this arithmetic in tokens has been right every time so far,
    which is not the same as being reliable. Here it is once, in code:

        score      = client avg x client weight + team avg x team weight
                     (only the sources that have feedback)
        weightedPa = score x the criterion's own weight
        finalScore = sum(weightedPa) / scale x 100
    """
    rows = []
    for item in criteria:
        parts = []
        if item.get("client_avg") is not None:
            parts.append(("client", item["client_avg"] * client_weight))
        if item.get("team_avg") is not None:
            parts.append(("team", item["team_avg"] * team_weight))
        score = sum(value for _, value in parts)
        weighted = score * item["weight"]
        rows.append({"name": item.get("name", "?"), "weight": item["weight"],
                     "sources": [n for n, _ in parts], "score": score,
                     "weighted": weighted})
    total = sum(row["weighted"] for row in rows)
    return {"criteria": rows, "total": total, "final_score": total / scale * 100}


def cmd_pa(path: str) -> int:
    """Show the working, so the number can be checked rather than trusted."""
    with open(path, encoding="utf-8") as fh:
        data = unwrap_profile(json.load(fh))
    try:
        result = pa_score(data["criteria"], data["client_weight"], data["team_weight"])
    except (KeyError, TypeError) as exc:
        sys.exit(f"{path} needs client_weight, team_weight and criteria: {exc}")

    print(f"\nclient weight {data['client_weight']}, team weight {data['team_weight']}\n")
    for row in result["criteria"]:
        sources = "+".join(row["sources"]) or "no feedback"
        print(f"  {row['name']:<24} {sources:<14} score {row['score']:.4g}"
              f" x weight {row['weight']} = {row['weighted']:.4g}")
    print(f"\n  sum of weightedPa = {result['total']:.4g}")
    print(f"  finalScore = ({result['total']:.4g} / {PA_SCALE}) x 100 ="
          f" {result['final_score']:.4g}\n")
    return 0


def cmd_profile(token: str | None, name: str | None) -> int:
    """Show what the HRMS returns, and how much of the context it would cost."""
    raw = fetch_profile(resolve_token(token)) if token else read_profile_file(name or "")
    raw = unwrap_profile(raw)
    pruned = prune_profile(raw)
    print(f"\nTop-level fields ({len(raw)} returned, {len(pruned)} after pruning):\n")
    for key in raw:
        size = len(json.dumps(raw[key]))
        mark = " " if key in pruned else "-"  # '-' was dropped as bulk or secret
        print(f" {mark} {key:<32} {size:>7} chars")
    text = profile_from(raw)
    print(f"\nRendered for the prompt: {len(text)} chars, roughly"
          f" {len(text) // 4} tokens of NUM_CTX.\n")
    print(text)
    print("\nToo big? Set HRMS_PROFILE_FIELDS to a comma-separated list of the"
          " fields above.")
    return 0


def chosen_token(args: argparse.Namespace) -> str | None:
    """--login beats --prompt-token beats whatever was passed or is in the env."""
    if args.login:
        return TOKEN_LOGIN
    return TOKEN_PROMPT if args.prompt_token else args.token


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    """Personalise the answer. --login is the everyday path; --as reads a saved
    response for working offline."""
    # Deliberately not `--token` with an optional value: argparse would swallow
    # the question after a bare `--token` and treat it as the token.
    parser.add_argument(
        "--token",
        default=os.getenv("HRMS_TOKEN"),
        help="the employee's own HRMS token (or set HRMS_TOKEN in .env, which is gitignored)",
    )
    parser.add_argument(
        "--prompt-token",
        action="store_true",
        help="ask for the token on a hidden prompt, so it misses the shell history",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="open the HRMS login page in a browser and read the token from your session",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="leave off the documents and pages the answer was drawn from",
    )
    parser.add_argument("--as", dest="profile", help="a saved profile in profiles/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rag", description="Ask questions about local policy documents, answered by Ollama."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="chunk, embed and store a document")
    p.add_argument("path", nargs="?", default="data/policy.pdf")
    p.add_argument("--replace", action="store_true", help="rebuild instead of resuming")

    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question", nargs="+")
    p.add_argument("-k", type=int, default=TOP_K, help=f"chunks to retrieve (default {TOP_K})")
    add_profile_args(p)

    p = sub.add_parser("chat", help="interactive question loop")
    p.add_argument("-k", type=int, default=TOP_K)
    add_profile_args(p)

    sub.add_parser("discover", help="find the HRMS endpoint holding your record")

    p = sub.add_parser("pa", help="compute a PA score from a JSON file of feedback")
    p.add_argument("path", help="JSON with client_weight, team_weight and criteria")

    p = sub.add_parser("profile", help="show what the HRMS returns about you")
    add_profile_args(p)

    sub.add_parser("stats", help="show what is indexed")
    sub.add_parser("reset", help="drop the collection")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    missing = [n for n in REQUIRED_ENV if not os.getenv(n)]
    if missing:
        sys.exit(f"Missing env vars: {', '.join(missing)}. Copy .env.example to .env.")

    try:
        if args.command == "ingest":
            return cmd_ingest(args.path, args.replace)
        if args.command == "ask":
            return cmd_ask(" ".join(args.question), args.k, chosen_token(args),
                           args.profile, not args.no_sources)
        if args.command == "chat":
            return cmd_chat(args.k, chosen_token(args), args.profile, not args.no_sources)
        if args.command == "discover":
            return cmd_discover()
        if args.command == "pa":
            return cmd_pa(args.path)
        if args.command == "profile":
            return cmd_profile(chosen_token(args), args.profile)
        if args.command == "stats":
            return cmd_stats()
        return cmd_reset()
    except psycopg.OperationalError as exc:
        sys.exit(f"Cannot reach Postgres: {exc}\nIs it up? `docker compose up -d postgres`")
    except Exception as exc:  # noqa: BLE001 - Ollama errors arrive in several shapes
        url = env("OLLAMA_BASE_URL")
        if not ollama_reachable(url):
            sys.exit(f"Cannot reach Ollama at {url}. Is `ollama serve` running?")
        raise SystemExit(f"{type(exc).__name__}: {exc}") from exc


def ollama_reachable(url: str) -> bool:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False


if __name__ == "__main__":
    sys.exit(main())
