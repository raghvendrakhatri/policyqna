# PolicyQA Flows

## 1. Ask / chat flow

```
 you type a question
        │
        ▼
 1. Startup ─────────── check .env, check something is indexed
        │
        ▼
 2. Who is asking? ──── none      → generic answers
        │               --as NAME → read profiles/NAME.json
        │               --login   → browser opens → you sign in → token captured
        │                           → fetch your HRMS record → strip private fields
        ▼
 3. Input check ─────── empty / too long / injection / unsafe?  ──► ✋ refused
        │
        ▼
 4. Find chunks ─────── split if two questions → embed → pgvector search (MMR)
        │               nothing found?  ──► ✋ "No matching policy"
        ▼
 5. Build prompt ────── rules + knowledge/ + your profile + chat memory + chunks + question
        │
        ▼
 6. Answer ──────────── qwen3:4b-instruct writes the answer
        │
        ▼
 7. Output check ────── strip <think> tags, safety check  ──► ✋ suppressed
        │
        ▼
 8. Show ────────────── answer + "Sources: policy.pdf pp. 83, 84"
                        (chat: remember this turn for follow-ups)
```

## 2. Ingest flow

```
 ingest data/policy.pdf
        │
        ▼
 1. Read ───────────── PDF → one piece per page (keeps page numbers)
        │              .md / .txt → the whole file
        ▼
 2. Clean ──────────── skip table-of-contents pages
        │
        ▼
 3. Chunk ──────────── 1200-char pieces, 200-char overlap, id = "policy.pdf#12"
        │
        ▼
 4. Already stored? ── yes → skip those (so a stopped run resumes)
        │              --replace → delete this file's old chunks first
        ▼
 5. Embed + save ───── 25 chunks at a time → Qwen3 embedding → 1024 numbers each
        │              → saved in Postgres (pgvector)
        ▼
 6. Done ───────────── "Indexed 153 chunks" → check with `stats`
```
