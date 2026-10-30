# PolicyQA Architecture

```
                         ┌──────────────────────────────┐
                         │          Employee            │
                         │   (terminal on own laptop)   │
                         └──────────────┬───────────────┘
                                        │ CLI: ingest · ask · chat · pa · …
            ┌───────────────────────────▼────────────────────────────┐
            │                    PolicyQA (Python)                   │
            │                       app/rag/*                        │
            └──────┬──────────────────┬──────────────────┬───────────┘
                   │ SQL (psycopg3)   │ HTTP :11434      │ HTTPS + user's bearer token
                   ▼                  ▼                  ▼  (optional)
        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
        │ Postgres 16      │ │ Ollama           │ │ HRMS APIs            │
        │ + pgvector       │ │ chat · embed ·   │ │ (identity, leave,    │
        │ (Docker)         │ │ safety models    │ │  attendance, …)      │
        └──────────────────┘ └──────────────────┘ └──────────────────────┘
             local                local                 remote (company)

        Local files read at runtime:  data/*.pdf|md  ·  knowledge/*.md  ·  profiles/*.json
```

---

## Runtime components

| Component | What it is | Runs where | Port |
|---|---|---|---|
| **PolicyQA CLI** | Python 3.12 app (`app/rag/`), run with `uv run python app/main.py` | Host | — |
| **Ollama** | Local model server (v0.34.4). Serves the chat, embedding and safety models | Host (not in Docker, so it can use the GPU / Metal) | `11434` |
| **Postgres + pgvector** | Postgres 16 with the pgvector extension (`pgvector/pgvector:pg16`) | Docker container `local-rag-postgres` | host `5434` → `5432` |
| **Chromium** (optional) | Browser opened by Playwright for `--login` and `discover` | Host, only on demand | — |
| **HRMS APIs** (optional) | Company HR system, called with the employee's own token | Remote, over HTTPS only | 443 |

---

## Models (all served by Ollama)

| Role | Model | Size | Quantisation | Key settings |
|---|---|---|---|---|
| **Chat**: answers, question splitting, follow-up rewriting, memory summary | `qwen3:4b-instruct` (non-thinking) | 4.0B params · 2.5 GB | Q4_K_M | `temperature 0` · `num_ctx 8192` (`NUM_CTX`) · max answer 2048 tokens · `keep_alive 3600s` |
| **Embeddings**: turns chunks and questions into vectors | `hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF` | 596M params · 0.64 GB | Q8_0 | **1024 dimensions** · last-token pooling · `num_ctx 2048` · `keep_alive 3600s` |
| **Safety** (optional): checks questions and answers for harm | `llama-guard3:1b` | 1.5B params · 1.6 GB | Q8_0 | `temperature 0` · replies `safe` or `unsafe` + S-code (S1–S13) |

- The model names are set in `.env` (`CHAT_MODEL`, `EMBED_MODEL`, `SAFETY_MODEL`). Swapping a model needs no code change.
- The chat model supports a 256K context. The app deliberately uses 8K to save RAM.
- `SAFETY_MODEL` must be a LlamaGuard model. Setting it to the chat model makes Ollama reload the model on every question and flags every question as unsafe.

---

## Vector store (Postgres + pgvector)

| Item | Value |
|---|---|
| Collection | `policies` (one collection for every document) |
| Tables | `langchain_pg_collection`, `langchain_pg_embedding` (created by LangChain's `PGVector`) |
| Row per chunk | `id` (`policy.pdf#12`) · `embedding vector(1024)` · `document` (chunk text) · `cmetadata` jsonb (`source`, `page`, `chunk_index`) |
| Similarity | Cosine distance (`<=>`) |
| Search | top 30 by cosine → **MMR** re-rank to 8 (λ = 0.8) |
| Index | No vector index; exact scan (fine for the current 153 chunks) |
| Storage | Docker volume `postgres_data` |

---

## RAG settings

| Setting | Value | Meaning |
|---|---|---|
| Chunk size / overlap | 1200 / 200 chars | `RecursiveCharacterTextSplitter` |
| Chunks per question (`k`) | 8 | `-k` to change; compound questions get at least 3 per part |
| MMR candidate pool | 30 | `fetch_k` |
| Embed batch size | 25 chunks per call | during `ingest` |
| Chat memory | last 5 turns + one-line summary | RAM only |
| Profile budget | 4000 chars | HRMS data in the prompt |
| Question length cap | 2000 chars | input guardrail |
| Sources shown | up to 4 | computed from text overlap, not asked of the model |

---

## Libraries

| Library | Version | Used for |
|---|---|---|
| `langchain` / `langchain-core` | 1.4.2 / 1.6.4 | LCEL chains, prompt templates, retriever |
| `langchain-ollama` | 1.1.0 | `ChatOllama`, `OllamaEmbeddings` |
| `langchain-postgres` | 0.0.18 | `PGVector` store and schema |
| `langchain-text-splitters` | 1.1.2 | Chunking |
| `psycopg` (v3) + `SQLAlchemy` | 3.3.5 + 2.0.54 | Postgres driver, raw SQL |
| `pgvector` (Python) | 0.3.6 | `vector` column type |
| `pypdf` | 6.19.0 | PDF text extraction |
| `rich` | 15.0.0 | Terminal panels, spinners, tables |
| `python-dotenv` | 1.2.3 | Loads `.env` |
| `playwright` (optional) | 1.63.0 | Browser login (`uv sync --extra browser`) |
| `uv` | — | Dependencies and virtualenv (`uv.lock`) |
| Docker Compose | — | Runs Postgres (and optionally the app) |

---

## Resource needs

| Resource | Usage |
|---|---|
| RAM while answering | ~3.5 GB (chat + embed), ~5 GB with LlamaGuard |
| Disk | ~4.7 GB of models + a small Postgres volume |
| First answer after idle | ~2 min (model load); keep models loaded with `OLLAMA_KEEP_ALIVE=30m ollama serve` |
| Later answers | a few seconds |
| Network | None for the core flow; HTTPS to the HRMS only with `--login` / `--token` |
