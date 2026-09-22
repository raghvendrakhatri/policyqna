# policyqa

Ask questions about your policy documents in plain English and get answers from
the document itself.

All the code is in one file: `app/rag.py`.

## What it does

1. **Ingest** — reads a PDF, cuts it into small pieces, turns each piece into
   numbers (an "embedding"), and saves them in Postgres.
2. **Ask** — turns your question into numbers the same way, finds the pieces of
   the document that are closest to it, and hands those to the model to answer.

The model only sees the pieces that were pulled from your document, so it answers
from the policy instead of from memory.

## Before you start

You need three things:

- **Docker running** — the database lives in a container.
- **An Amazon Bedrock API key** — make one in the AWS console under Bedrock -> API keys.
- **Model access turned on** in the Bedrock console, for both models listed below,
  in the region you pick.

## Setup

Run these once:

```bash
cp .env.example .env      # then open .env and paste in your Bedrock API key
docker compose up -d postgres
uv sync
```

In `.env` you need to fill in:

- `AWS_BEARER_TOKEN_BEDROCK` — your Bedrock API key
- `AWS_REGION` — e.g. `us-east-1` or `ap-south-1`

## How to use it

### Step 1: load your document

```bash
uv run python app/rag.py ingest data/policy.pdf
```

This reads the PDF and saves it to the database. It takes a few minutes and shows
progress as it goes. You only need to do this once per document.

If it stops partway (you press Ctrl-C, or AWS rate-limits you), just run the same
command again. It picks up where it left off and does not redo finished work.

If you edit or replace the PDF, rebuild it from scratch:

```bash
uv run python app/rag.py ingest data/policy.pdf --replace
```

### Step 2: ask questions

One question at a time:

```bash
uv run python app/rag.py ask "how many sick leaves do I get?"
```

Or open a back-and-forth session, where you can keep typing questions:

```bash
uv run python app/rag.py chat
```

Press Ctrl-C, or hit Enter on an empty line, to leave the chat.

If an answer seems to miss something, ask again and pull in more of the document
with `-k` (the default is 8 pieces):

```bash
uv run python app/rag.py ask -k 15 "what is the notice period?"
```

### Checking and clearing

See which documents are loaded and how many pieces each one became:

```bash
uv run python app/rag.py stats
```

Delete everything and start over:

```bash
uv run python app/rag.py reset
```

## If you get a "too many requests" error

AWS limits how fast you can call it, and the limit is per account and per region.
Some regions are stricter than others — `ap-south-1` is tighter than `us-east-1`.

Run the ingest more slowly:

```bash
EMBED_WORKERS=1 uv run python app/rag.py ingest data/policy.pdf
```

Nothing already saved gets redone, so this is safe to run as many times as needed.
If it still happens, try switching `AWS_REGION` to `us-east-1` in your `.env`.

## Other errors you might see

| Message | What it means |
|---|---|
| `Cannot reach Postgres` | Docker isn't running. Start it: `docker compose up -d postgres` |
| `Missing AWS_BEARER_TOKEN_BEDROCK` | Your `.env` has no Bedrock key in it |
| `AccessDeniedException` | Model access isn't enabled for your region in the Bedrock console |
| `No extractable text` | The PDF is a scan (just images), so there's no text to read |

---

## How it works under the hood

One Bedrock API key covers both halves:

| Half | Model | Called through |
|---|---|---|
| Answers | `openai.gpt-oss-120b-1:0` | OpenAI SDK -> `bedrock-runtime.<region>.amazonaws.com/openai/v1` |
| Embeddings | `amazon.titan-embed-text-v2:0` (1024-d) | boto3 `InvokeModel` |

Bedrock's OpenAI-compatible endpoint only implements Chat Completions and
Responses — there is no `/v1/embeddings`, and every Bedrock embedding model is
Invoke-only. So the embedding half goes through boto3, which reads the same
`AWS_BEARER_TOKEN_BEDROCK` from the environment.

Answers come back with no citations and no source list, because the prompt tells
the model not to add them. `gpt-oss` is a reasoning model and sometimes puts its
thinking inline in the reply as `<reasoning>...</reasoning>`; `answer()` strips
that out before printing.

`REASONING_EFFORT` is set to `low`, which suits straightforward lookups. Raise it
to `medium` or `high` for questions that need piecing several sections together.

## Settings

All at the top of `app/rag.py`: `EMBED_MODEL`, `EMBED_DIM`, `EMBED_WORKERS`,
`CHAT_MODEL`, `REASONING_EFFORT`, `MAX_ANSWER_TOKENS`, `CHUNK_CHARS`,
`CHUNK_OVERLAP`, `TOP_K`, `BATCH`.

If you change `EMBED_MODEL`, you must also change `EMBED_DIM` to match that
model's size, then run `reset` before ingesting again.
