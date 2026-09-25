# policyqa

Ask questions about your policy documents in plain English and get answers from
the document itself. Everything runs on your own machine — no API keys, and
nothing leaves the laptop.

Built on LangChain and Ollama. `app/rag.py` holds the pipeline;
`app/hrms_login.py` is the optional browser login.

## What it does

1. **Ingest** — reads a PDF, cuts it into small pieces, turns each piece into
   numbers (an "embedding"), and saves them in Postgres.
2. **Ask** — turns your question into numbers the same way, finds the pieces of
   the document that are closest to it, and hands those to the model to answer.

The model only sees the pieces that were pulled from your document, so it answers
from the policy instead of from memory.

## Before you start

You need two things:

- **Docker running** — the database lives in a container.
- **Ollama running** — `ollama serve`, with the two models below pulled.

## Setup

Run these once:

```bash
cp .env.example .env
docker compose up -d postgres
uv sync

ollama pull qwen3:4b
ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:latest
```

`.env.example` already holds working values for a standard local setup, so nothing
in it needs editing. Check it all works:

```bash
uv run python app/main.py
```

That prints the Postgres version, confirms Ollama answers, and warns you if
either model in `.env` hasn't been pulled.

## How to use it

### Step 1: load your document

```bash
uv run python app/rag.py ingest data/policy.pdf
```

This reads the PDF and saves it to the database. It takes a few minutes and shows
progress as it goes. You only need to do this once per document.

If it stops partway (you press Ctrl-C), just run the same command again. It picks
up where it left off and does not redo finished work.

If you edit or replace the PDF, rebuild it from scratch:

```bash
uv run python app/rag.py ingest data/policy.pdf --replace
```

`ingest` takes Markdown and plain text too, not just PDFs — useful for content
that started life somewhere else, like a spreadsheet you've converted:

```bash
uv run python app/rag.py ingest data/perks-and-benefits.md
```

Markdown chunks on blank lines first, so `##` sections and the tables under them
generally stay whole. Worth checking after ingesting a table-heavy file: a table
split across two chunks loses its header row, and the model can no longer tell
which column is which.

### Step 2: ask questions

One question at a time:

```bash
uv run python app/rag.py ask "how many sick leaves do I get?"
```

Or open a back-and-forth session, where you can keep typing questions:

```bash
uv run python app/rag.py chat
```

Press Ctrl-C, or hit Enter on an empty line, to leave the chat. `chat` builds the
retrieval chain once, so only the first question pays the model warm-up cost.

`chat` remembers the conversation, so follow-ups work:

```
> what is my daily food allowance?
1,900 — Gold band, Tier 1 city.

> and in a tier 3 city?
(reading that as: what is my daily food allowance in a tier 3 city?)
1,000.
```

A follow-up is rewritten into a standalone question *before* retrieval, because
"and in a tier 3 city?" embeds to nothing on its own — the rewrite is printed so
you can see what was actually asked. `ask` is single-shot and has no history.

If an answer seems to miss something, ask again and pull in more of the document
with `-k` (the default is 8 pieces):

```bash
uv run python app/rag.py ask -k 15 "what is the notice period?"
```

Raising `-k` stuffs more text into the prompt. If you push it far, raise `NUM_CTX`
in `.env` to match, or the model will silently drop the overflow.

### Answering for a specific employee

Without a profile, answers are generic: *"Gold band gets 30 WFH days a year."*
With one, they are about you:

```bash
export HRMS_BASE_URL=https://hrms.example.com     # in .env, which is gitignored
uv run python app/rag.py ask --login "how many WFH days do I have left?"
```

`--login` opens a browser on your HRMS login page. You sign in there exactly as
you always do — SSO, MFA, whatever it takes — and the browser closes as soon as
your session exists. Nothing types your password and nothing stores it.

The profile is fetched live from the HRMS with the employee's own access token —
nothing about any employee is stored in this repo. It is then injected into the
prompt the way `knowledge/` is, so the model applies it without being asked:

```
> what is my daily food allowance?
Your daily food allowance is 1,900 — Gold band, Tier 1 city.

> what is my PA score?
Quality:  4.5×0.7 + 4.0×0.3 = 4.35 → 4.35 × 0.6 = 2.61
Delivery: 3.0×0.3 = 0.90         → 0.90 × 0.4 = 0.36
finalScore = (2.97 / 5) × 100 = 59.4
```

Nothing was typed into that second question — the criteria, weights and feedback
averages all came from the HRMS.

**How the session is obtained.** No change is needed on the HRMS side, because
this calls the same API its own frontend calls, with the same session that
frontend was issued. `--login` reads it back in one of two ways:

1. It watches the requests the HRMS page makes and takes the `Authorization`
   header off one — this is the session the API actually accepts, wherever the
   frontend chose to keep it. `HRMS_API_HINT` (default `/api/`) restricts this
   to your own endpoints, so a bearer meant for analytics is never picked up.
2. Failing that, it scans local and session storage for a JWT, including one
   nested inside a JSON blob.

`--login` needs the optional browser extra:

```bash
uv sync --extra browser
uv run playwright install chromium
```

Two alternatives if you would rather not run a browser: `--prompt-token` reads
one on a hidden prompt (devtools → Application → Local Storage to find it), and
`HRMS_TOKEN` in `.env` persists one. Avoid `--token <value>` — it lands in your
shell history and is visible in `ps`.

**Several services, one profile.** An HRMS rarely keeps everything in one place:
identity in one service, leave balances in another, payroll in a third — often
on different hosts, and reached by an id only the first response knows.
`HRMS_PROFILE_SOURCES` takes them comma separated and merges the results:

```bash
HRMS_PROFILE_SOURCES=api/users/me,wfh=https://att.example.com/api/employees/{attendance_employee_id}/balance
```

- `label=url` names the group the fields appear under; a bare entry is named
  after its last path segment
- a full URL is used as-is; a bare path is joined to `HRMS_BASE_URL`
- `{field}` is filled from any earlier response, so a second call can use an id
  the first returned
- the first source's fields sit at the top level, later ones nest under their
  label
- a source whose `{field}` cannot be filled is skipped with a warning rather
  than failing the run

Whatever each returns is used as-is: no field name is hardcoded anywhere in this
app, so an HRMS that renames or adds a field needs no change here.

The login page, the API and the profile endpoint may all be on different hosts.
Set each one, and widen `HRMS_API_HINT` to whatever covers them:

```bash
HRMS_LOGIN_URL=https://sp18.autoscal.com/       # where you sign in
HRMS_BASE_URL=https://auth.autoscal.com         # base for any bare path
HRMS_API_HINT=.autoscal.com/api                 # matches every host above
HRMS_PROFILE_SOURCES=api/users/me,wfh=https://attendance.autoscal.com/api/wfh/balance
```

Not sure of the path? `discover` will tell you — see below.

Sessions are short-lived, so expect to log in once per working session. Plain
HTTP to a remote host is refused; only `https://` and localhost are allowed.

**Finding the endpoint.** If you do not know which API call returns your own
record:

```bash
uv run python app/rag.py discover
```

A browser opens on the login page. Sign in, visit your profile and leave pages
so their API calls are made, then close the window. It prints each JSON endpoint
it saw with its top-level keys — never the values, so nobody's salary lands in
your scrollback — and you add whichever ones you need to
`HRMS_PROFILE_SOURCES`.

**Working offline.** `--as <name>` reads a saved response from `profiles/`
instead of calling the HRMS — useful on a plane or when demoing with no network.
That directory is gitignored: real employee data must never be committed.

### Where an answer came from

Every answer ends with the documents and pages behind it:

```
> what is the notice period?
...the employee shall be relieved on the working day prior to the holiday.

Sources: policy.pdf p. 83
```

These come from the retrieved chunks' own metadata, never from the model, so a
page number cannot be invented. Only the chunks whose wording the answer
actually overlaps are listed. `--no-sources` turns it off.

### Checking a PA score

The Performance Allowance formula is code, not something the model works out in
tokens:

```bash
uv run python app/rag.py pa feedback.json
```

with `client_weight`, `team_weight` and a `criteria` list in the file. It prints
each step, so the number can be checked rather than trusted.

### Checking and clearing

See which documents are loaded and how many pieces each one became:

```bash
uv run python app/rag.py stats
```

Delete everything and start over:

```bash
uv run python app/rag.py reset
```

### Facts that aren't in the document

Some things the model needs aren't written in any policy — a calculation
formula, an internal definition, an abbreviation. Put each one in its own `.md`
or `.txt` file under `knowledge/`:

```
knowledge/
  glossary.md          # PA — Performance Allowance
  pa-calculation.md    # the formula itself
```

Every file there is sent to the model with **every** question, on top of the
pieces pulled from your document. That is the point: a formula can't be missed
the way a retrieved chunk can, and the model has it even when the question
doesn't sound like a match for it.

Edit a file and the next `ask` picks it up — there is nothing to re-ingest.
`stats` shows which files are loaded and how much room they take.

The trade-off is context: this text is in the prompt for every question, whether
or not it's relevant. Keep the folder to a few thousand characters — `rag.py`
warns past 6000. Anything longer is a document, so `ingest` it instead.
`README.md` in that folder is skipped.

The dividing line in practice: a table of entitlements is a document, because a
question about it names the thing it's asking for and retrieval finds it. What
stays here is what a question *doesn't* say — the formula behind a number, or
that "Sr1" means the Silver band. Without those in every prompt, a question
phrased in the abbreviation never reaches the table that answers it.

## Tests

```bash
uv run --dev pytest tests/test_profile.py -q   # fast: no model, no database
uv run --dev pytest tests/ -q                  # adds the retrieval evals
```

`tests/test_profile.py` is about what may reach the prompt: no credentials, no
colleague's personal data, no source silently dropped. Every case is a bug that
happened against the live HRMS.

`tests/evals.yaml` holds questions with their expected answers, and
`test_retrieval.py` runs each one against the real index and model. They are
slow, and they skip themselves when Ollama or Postgres is not up. Add a case
whenever an answer comes back wrong: that is what stops it coming back.

## If answers are slow

Everything runs on your CPU/GPU, so speed depends on your machine, not on a rate
limit. Ingest is the slow part — the embedding model is called once per chunk.

A first question after a pause is slower because Ollama has to load the model back
into memory. Keep it resident with `OLLAMA_KEEP_ALIVE=30m ollama serve`. A spinner
shows the seconds while you wait, so a long load does not look like a hang.

## Other errors you might see

| Message | What it means |
|---|---|
| `Cannot reach Postgres` | Docker isn't running. Start it: `docker compose up -d postgres` |
| `Cannot reach Ollama` | `ollama serve` isn't running, or `OLLAMA_BASE_URL` is wrong |
| `model ... not found` | The model isn't pulled. Run the `ollama pull` commands above |
| `No extractable text` | The PDF is a scan (just images), so there's no text to read |

---

## How it works under the hood

| Half | Model | Called through |
|---|---|---|
| Answers | `qwen3:4b` | `ChatOllama` |
| Embeddings | `hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:latest` (1024-d) | `OllamaEmbeddings` |

The pipeline is plain LangChain: `RecursiveCharacterTextSplitter` cuts the pages,
`PGVector` (from `langchain-postgres`) stores and searches them, and an LCEL chain
wires retriever -> prompt -> model -> string:

```python
{"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm() | StrOutputParser()
```

### Retrieval quality

Three things stop the obvious failure modes:

**Contents pages are dropped at ingest.** A table-of-contents page names every
policy in the handbook, so it scores well against almost any question while
containing no policy text — it pushed real answers out of the top-k and left the
model quoting section numbers instead of content. `looks_like_contents()` drops a
page when 80%+ of its lines are bare headings; on the sample handbook that is
5 pages of 87.

**Compound questions are split before retrieval.** One question gets one
embedding, so "what is the resignation policy and the company mission" averages
into a vector that matches neither topic well, and whole parts of the answer come
back as "not specified". `subquestions()` splits it with a short LLM call,
retrieves for each part, and unions the results. The split call is gated behind
`looks_compound()` — a simple question skips it entirely, because left to itself
the splitter invents sub-topics that dilute retrieval.

**MMR keeps the context varied**, at `lambda_mult=0.8`. MMR's own default of 0.5
diversifies so aggressively that it drops neighbouring clauses of the exact
section being asked about.

### Always-on knowledge

`knowledge/` is the escape hatch for facts no document contains. `load_knowledge()`
concatenates the folder at chain-build time and `KNOWLEDGE_PROMPT` appends it to
the system message; retrieval is untouched. It is passed as a *partial* template
variable rather than interpolated into the prompt string, so braces and percent
signs inside a formula stay literal instead of being read as template syntax.

It deliberately bypasses the vector store. An ingested chunk competes for the
top-k and can lose; a formula that silently doesn't reach the model produces a
confidently wrong number rather than a visible miss. The cost is a fixed slice of
`NUM_CTX` on every question, which is why `KNOWLEDGE_WARN_CHARS` exists.

`PGVector` owns its own schema — `langchain_pg_collection` and
`langchain_pg_embedding`, with `source`, `page` and `chunk_index` kept in the
`cmetadata` JSONB column. Chunks get deterministic ids (`policy.pdf#12`), which is
what makes a re-run resumable. `PGVector` has no delete-by-metadata or count, so
`--replace` and `stats` go through the small `sql()` helper.

Pages are loaded with `pypdf` directly rather than `PyPDFLoader`, because
`langchain-community` is being sunset.

Answers come back with no citations and no source list, because the prompt tells
the model not to add them.

Qwen3 is a thinking model, and `reasoning=True` is what keeps the thinking *out*
of the answer: Ollama then returns it in a separate field and `StrOutputParser`
only ever sees the reply. Setting `reasoning=False` does not stop the model
thinking — it merges the thinking into the content, so answers start with
"Okay, the user is asking...". The two `<think>` regexes remain as a backstop for
models that tag their reasoning inline. Thinking tokens are drawn from
`MAX_ANSWER_TOKENS`, which is why it is set well above what an answer needs.

## Settings

Model and connection settings are read from `.env` with no fallbacks — every name
in `REQUIRED_ENV` (`POSTGRES_*`, `OLLAMA_BASE_URL`, `CHAT_MODEL`, `EMBED_MODEL`,
`NUM_CTX`) must be set, and the app exits naming the ones that are missing rather
than quietly using a built-in default. `POSTGRES_HOST` is the one exception: it
falls back to `localhost`, because docker compose sets it to `postgres`.

Tuning knobs are at the top of `app/rag.py`:
`MAX_ANSWER_TOKENS`, `COLLECTION`, `CHUNK_CHARS`, `CHUNK_OVERLAP`, `TOP_K`, `BATCH`.

If you change `EMBED_MODEL`, run `reset` before ingesting again — old vectors from
a different model are not comparable, and a different dimension will error outright.

## Running in Docker

The `app` service is wired to reach Ollama on the host via
`host.docker.internal`, so keep `ollama serve` running outside the container:

```bash
docker compose run --rm app uv run python app/rag.py ask "what is the notice period?"
```
