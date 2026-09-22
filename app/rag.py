"""Single-file CLI RAG over policy documents, running on Amazon Bedrock.

One Bedrock API key (AWS_BEARER_TOKEN_BEDROCK) covers both halves:
  chat  - OpenAI SDK against Bedrock's OpenAI-compatible endpoint
  embed - boto3 InvokeModel, because that endpoint has no /v1/embeddings
          (every Bedrock embedding model is Invoke-only)

Commands: ingest | ask | chat | stats | reset
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import boto3
import openai
import psycopg
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

EMBED_MODEL = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "2"))

CHAT_MODEL = "openai.gpt-oss-120b-1:0"
REASONING_EFFORT = "low"
MAX_ANSWER_TOKENS = 1024

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
TOP_K = 8
BATCH = 25

SYSTEM_PROMPT = """You answer questions about the policy documents given as context.

- Use only the context. If the answer is not there, say so plainly.
- Quote the policy's wording for anything binding (limits, deadlines, exclusions).
- Answer directly: no preamble, no reasoning, no citations or source references."""

# gpt-oss sometimes emits its analysis channel inline instead of in
# reasoning_content; the second pattern catches a truncated opening tag.
REASONING = re.compile(r"<(reasoning|analysis|think|thinking)>.*?</\1>", re.S | re.I)
STRAY_CLOSE = re.compile(r"^.*?</(reasoning|analysis|think|thinking)>", re.S | re.I)


# ---------------------------------------------------------------------- bedrock


def region() -> str:
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"


def chat_client() -> OpenAI:
    """Needs the Bedrock API key, never an OpenAI one."""
    return OpenAI(
        api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
        base_url=f"https://bedrock-runtime.{region()}.amazonaws.com/openai/v1",
    )


def embed_client():
    """boto3 reads AWS_BEARER_TOKEN_BEDROCK itself.

    `adaptive` retry mode rate-limits the client once Bedrock starts throttling,
    which matters because Titan embeds one string per call.
    """
    return boto3.client(
        "bedrock-runtime",
        region_name=region(),
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=60),
    )


def embed(bedrock, texts: list[str]) -> list[list[float]]:
    def one(text: str) -> list[float]:
        response = bedrock.invoke_model(
            modelId=EMBED_MODEL,
            body=json.dumps({"inputText": text, "dimensions": EMBED_DIM, "normalize": True}),
        )
        return json.loads(response["body"].read())["embedding"]

    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        return list(pool.map(one, texts))


def answer(chat, question: str, context: list[str]) -> str:
    response = chat.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Context:\n" + "\n\n---\n\n".join(context) + f"\n\nQuestion: {question}",
            },
        ],
        reasoning_effort=REASONING_EFFORT,
        max_completion_tokens=MAX_ANSWER_TOKENS,
    )
    text = response.choices[0].message.content or ""
    return STRAY_CLOSE.sub("", REASONING.sub("", text)).strip()


# --------------------------------------------------------------------------- db


def connect() -> psycopg.Connection:
    return psycopg.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT"),
        connect_timeout=5,
    )


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""CREATE TABLE IF NOT EXISTS chunks (
                    id          bigserial PRIMARY KEY,
                    source      text    NOT NULL,
                    page        integer,
                    chunk_index integer NOT NULL,
                    content     text    NOT NULL,
                    embedding   vector({EMBED_DIM}) NOT NULL
                )"""
        )
        cur.execute(
            """CREATE INDEX IF NOT EXISTS chunks_embedding_idx
               ON chunks USING hnsw (embedding vector_cosine_ops)"""
        )
    conn.commit()


def as_vector(values: list[float]) -> str:
    """pgvector's text form, so we don't need an adapter dependency."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


# ---------------------------------------------------------------- load + chunk


@dataclass
class Chunk:
    source: str
    page: int | None
    index: int
    content: str


def read_pages(path: str) -> list[tuple[int | None, str]]:
    if path.lower().endswith(".pdf"):
        pages = PdfReader(path).pages
        return [(i, page.extract_text() or "") for i, page in enumerate(pages, 1)]
    with open(path, encoding="utf-8") as fh:
        return [(None, fh.read())]


def split(text: str) -> list[str]:
    """Pack paragraphs up to CHUNK_CHARS, hard-wrapping any that overflow."""
    text = "\n".join(line.rstrip() for line in text.splitlines())  # so "\n\n" splits reliably
    pieces: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        while len(para) > CHUNK_CHARS:
            cut = para.rfind(" ", 0, CHUNK_CHARS)
            cut = cut if cut > CHUNK_CHARS // 2 else CHUNK_CHARS
            pieces.append(para[:cut].strip())
            para = para[max(0, cut - CHUNK_OVERLAP):].strip()
        if para:
            pieces.append(para)

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 2 > CHUNK_CHARS:
            chunks.append(buf)
            buf = buf[-CHUNK_OVERLAP:].strip()
        buf = f"{buf}\n\n{piece}".strip() if buf else piece
    if buf:
        chunks.append(buf)
    return chunks


def chunk_document(path: str) -> list[Chunk]:
    source = os.path.basename(path)
    out: list[Chunk] = []
    for page, text in read_pages(path):
        for body in split(text):
            out.append(Chunk(source, page, len(out), body))
    return out


# -------------------------------------------------------------------- commands


def cmd_ingest(bedrock, path: str, replace: bool) -> int:
    if not os.path.exists(path):
        sys.exit(f"No such file: {path}")

    chunks = chunk_document(path)
    if not chunks:
        sys.exit(f"No extractable text in {path} (scanned PDF? it would need OCR).")
    source = chunks[0].source

    with connect() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            if replace:
                cur.execute("DELETE FROM chunks WHERE source = %s", (source,))
                conn.commit()
                done = set()
            else:
                cur.execute("SELECT chunk_index FROM chunks WHERE source = %s", (source,))
                done = {row[0] for row in cur.fetchall()}

        todo = [c for c in chunks if c.index not in done]
        if not todo:
            print(f"'{source}' is already indexed ({len(chunks)} chunks). Use --replace to rebuild.")
            return 0
        if done:
            print(f"Resuming '{source}': {len(done)} stored, {len(todo)} to go.")
        print(f"Embedding {len(todo)} chunks via {EMBED_MODEL} ({EMBED_WORKERS} workers) ...")

        # Commit per batch so a throttle or Ctrl-C keeps the work already paid for.
        stored = 0
        try:
            for start in range(0, len(todo), BATCH):
                batch = todo[start:start + BATCH]
                vectors = embed(bedrock, [c.content for c in batch])
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks (source, page, chunk_index, content, embedding)"
                        " VALUES (%s, %s, %s, %s, %s)",
                        [
                            (c.source, c.page, c.index, c.content, as_vector(v))
                            for c, v in zip(batch, vectors)
                        ],
                    )
                conn.commit()
                stored += len(batch)
                print(f"\r  {stored}/{len(todo)} stored", end="", flush=True)
            print()
        except (ClientError, BotoCoreError, KeyboardInterrupt):
            print(f"\nStopped after {stored}/{len(todo)}. Re-run to resume.", file=sys.stderr)
            raise

    print(f"Indexed {len(done) + stored} chunks as '{source}'.")
    return 0


def retrieve(bedrock, question: str, k: int) -> list[str]:
    vector = as_vector(embed(bedrock, [question])[0])
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT content FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
            (vector, k),
        )
        return [row[0] for row in cur.fetchall()]


def cmd_ask(chat, bedrock, question: str, k: int) -> int:
    context = retrieve(bedrock, question, k)
    if not context:
        sys.exit("Nothing indexed yet - run `ingest` first.")
    print("\n" + (answer(chat, question, context) or "(empty answer)") + "\n")
    return 0


def cmd_chat(chat, bedrock, k: int) -> int:
    print("Ask about the indexed policies. Ctrl-C or empty line to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0
        cmd_ask(chat, bedrock, question, k)


def cmd_stats() -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.chunks')")
        if cur.fetchone()[0] is None:
            print("No index yet - run `ingest` first.")
            return 0
        cur.execute("SELECT source, count(*) FROM chunks GROUP BY source ORDER BY source")
        rows = cur.fetchall()

    print("Index is empty." if not rows else "")
    for source, count in rows:
        print(f"{source:<40} {count:>6} chunks")
    return 0


def cmd_reset() -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS chunks")
        conn.commit()
    print("Dropped the chunks table.")
    return 0


# ------------------------------------------------------------------------ main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rag", description="Ask questions about local policy documents, answered on Bedrock."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="chunk, embed and store a document")
    p.add_argument("path", nargs="?", default="data/policy.pdf")
    p.add_argument("--replace", action="store_true", help="rebuild instead of resuming")

    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question", nargs="+")
    p.add_argument("-k", type=int, default=TOP_K, help=f"chunks to retrieve (default {TOP_K})")

    p = sub.add_parser("chat", help="interactive question loop")
    p.add_argument("-k", type=int, default=TOP_K)

    sub.add_parser("stats", help="show what is indexed")
    sub.add_parser("reset", help="drop the chunks table")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    missing = [
        n for n in ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_PORT")
        if not os.getenv(n)
    ]
    if missing:
        sys.exit(f"Missing env vars: {', '.join(missing)}. Copy .env.example to .env.")

    chat = bedrock = None
    if args.command in ("ingest", "ask", "chat"):
        if not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
            sys.exit("Missing AWS_BEARER_TOKEN_BEDROCK (your Bedrock API key). Add it to .env.")
        bedrock, chat = embed_client(), chat_client()

    try:
        if args.command == "ingest":
            return cmd_ingest(bedrock, args.path, args.replace)
        if args.command == "ask":
            return cmd_ask(chat, bedrock, " ".join(args.question), args.k)
        if args.command == "chat":
            return cmd_chat(chat, bedrock, args.k)
        if args.command == "stats":
            return cmd_stats()
        return cmd_reset()
    except psycopg.OperationalError as exc:
        sys.exit(f"Cannot reach Postgres: {exc}\nIs it up? `docker compose up -d postgres`")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "?")
        hint = {
            "ThrottlingException": f"Rate-limited, not a permissions problem. Retry with "
                                   f"EMBED_WORKERS=1 (currently {EMBED_WORKERS}).",
            "AccessDeniedException": f"Enable {EMBED_MODEL} and {CHAT_MODEL} in {region()}.",
        }.get(code, "")
        sys.exit(f"Bedrock error ({code}): {exc}" + (f"\n{hint}" if hint else ""))
    except (BotoCoreError, openai.APIError) as exc:
        sys.exit(f"Bedrock call failed: {exc}")


if __name__ == "__main__":
    sys.exit(main())
