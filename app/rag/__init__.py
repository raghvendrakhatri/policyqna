"""Single-CLI RAG over policy documents, built on LangChain and Ollama.

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

The pipeline is spread across sibling modules; this file re-exports the
symbols the tests and app/main.py reach in via `import rag`.
"""

from dotenv import load_dotenv

load_dotenv()

from .chain import (  # noqa: E402
    Memory,
    NoContext,
    Provenance,
    ask,
    build_chain,
    build_contextualizer,
    build_summariser,
    credit,
    doc_label,
    format_docs,
    looks_compound,
    looks_like_followup,
)
from .cli import (  # noqa: E402
    add_profile_args,
    chosen_token,
    cmd_ask,
    cmd_chat,
    cmd_discover,
    cmd_ingest,
    cmd_pa,
    cmd_profile,
    cmd_reset,
    cmd_stats,
    main,
    ollama_reachable,
    pa_score,
    parse_args,
)
from .config import (  # noqa: E402
    COLLECTION,
    KNOWLEDGE_DIR,
    MEMORY_WINDOW,
    PROFILE_DIR,
    PROFILE_MAX_CHARS,
    REQUIRED_ENV,
    TOP_K,
    env,
)
from .db import counts_by_source, dsn, indexed, sql, store  # noqa: E402
from .guardrails import (  # noqa: E402
    REFUSAL,
    check_safety,
    clean,
    guard_input,
    safety_llm,
)
from .ingest import chunk_document, chunk_id, load_document, looks_like_contents  # noqa: E402
from .knowledge import knowledge_files, load_knowledge, read_knowledge  # noqa: E402
from .models import embeddings, llm  # noqa: E402
from .profile import (  # noqa: E402
    absolute,
    all_zero,
    collapse_named,
    fetch_json,
    fetch_profile,
    fill,
    fit_profile,
    flatten,
    profile_from,
    profile_sources,
    profile_text,
    prune_profile,
    read_profile_file,
    render_profile,
    resolve_token,
    scalars,
    split_source,
    unwrap_profile,
)
from .ui import chat_banner, console, err_console, refusal_panel, status, waiting  # noqa: E402
