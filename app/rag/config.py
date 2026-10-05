"""Environment lookup, tuning constants, and every prompt template.

Nothing here does I/O or imports LangChain: this module is safe to load first
and is depended on by everything else in the package.
"""

import os
import re
import sys


# Every one of these must be set; see .env.example. cli.main() checks them up
# front, so env() below is only a backstop for an import-time or library-side read.
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
_PKG_DIR = os.path.dirname(os.path.dirname(__file__))  # .../app
_REPO_DIR = os.path.dirname(_PKG_DIR)                  # repo root
KNOWLEDGE_DIR = os.path.join(_REPO_DIR, "knowledge")
KNOWLEDGE_SUFFIXES = (".md", ".txt")
KNOWLEDGE_SKIP = {"readme.md", "readme.txt"}  # notes about the folder, not facts
KNOWLEDGE_WARN_CHARS = 6000

# Who is asking. The profile comes from the HRMS, keyed by the employee's own
# access token - nothing about an employee is stored in this repo. PROFILE_DIR is
# only for working offline; it is gitignored and normally empty.
PROFILE_DIR = os.path.join(_REPO_DIR, "profiles")
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
# In-session memory: the last MEMORY_WINDOW turns reach the answer prompt
# verbatim; everything older is folded into a running one-line summary, so the
# prompt stays flat regardless of how long the chat runs. Nothing is written to
# disk - Ctrl-C wipes it.
MEMORY_WINDOW = 5
MEMORY_SUMMARY_CHARS = 400
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

# Below this normalised cosine score (0..1) no indexed chunk is close enough to
# the question for an answer to be grounded. Tuned on in-domain questions
# scoring ~0.45+ and off-topic ones ("capital of France") scoring <0.25.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))
# A question with "my", "I", "me" may be answered from the HRMS profile alone,
# so a low retrieval score is not grounds to refuse it.
PERSONAL_RE = re.compile(
    r"\b(my|mine|me|i|i'm|im|i've|ive|i'd|id|myself)\b",
    re.I,
)


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

But if the profile has no specific figure for what they asked, do NOT refuse.
Fall back to the general policy in the context above and apply it to them -
picking their band's row, their city tier's row, or noting that the policy is
the same for everyone. The question is answered from the profile when it can
be, and from the general policy otherwise; "not defined for you specifically"
is never the answer when the policy applies to everyone.

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
- Do NOT add the asker's identity, name, role, band, city tier or any other
  personal detail. Those are applied downstream when the answer is composed.
  A rewrite is about the topic, never about who is asking.
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

MEMORY_SUMMARISE_PROMPT = """Update a running summary of a conversation with the new turn.

- Output one line, at most 400 characters. Never a preamble, never a reply.
- Preserve names, dates, figures, entitlements, bands, cities, and preferences
  the user stated. Those are the parts a later question is likely to lean on.
- Drop pleasantries, repeats, and things now covered by a newer turn.
- If the existing summary already covers the new turn, restate it as-is.

Example
  Existing: user is Gold band, Tier 1, WFH balance 13.0 days as of 2026-09-23.
  New Q: what is my daily food allowance?
  New A: 1,900 - Gold band, Tier 1.
  out: user is Gold band, Tier 1, WFH balance 13.0 days as of 2026-09-23; daily food allowance 1,900."""


TOPICALITY_PROMPT = """You are a classifier for a company HR/policy assistant.

Decide if the question is in scope. Reply with exactly one word: yes or no.

In scope: company policies, employee benefits, leave, work-from-home,
reimbursement, performance appraisal, HR processes, workplace rules, and
anything about the asker's own HRMS record (balances, band, manager, joining
date, city tier, etc.).

Out of scope: general knowledge (geography, history, trivia, sports), coding
or technical help, creative writing, math/calculations unrelated to policy,
current events, other companies, personal opinions, chit-chat.

If the question is ambiguous but could plausibly be about company policy or
the asker's record, answer yes.

Always in scope (answer yes):
- "who am I", "what is my name", "what is my role / band / designation"
- "who is my manager", "when did I join", "what is my email"
- "how many leaves / WFH days do I have left"
- any question using "my", "I", "me" about work, role, HR, pay, or benefits."""


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
