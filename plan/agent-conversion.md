# PolicyQA → Agent conversion plan

Convert the current RAG chatbot into an agent that can take actions on behalf
of the logged-in user. v1 scope: apply leave, apply WFH, read leave balance,
list pending requests.

## Current shape

- One-way RAG: question → retrieval → LLM → cited answer (`app/rag/chain.py`).
- HRMS token captured via `--login` (Playwright) or `--token`
  (`app/rag/hrms_login.py`).
- HRMS reads via `profile.py::fetch_profile` with the token as a bearer.
- No tool-calling, no action-taking, no confirmation UI.

## Target shape

A bounded tool-use loop on top of the existing RAG chain. The LLM decides to
either answer from policy (existing) or call a tool (`apply_leave`,
`apply_wfh`, `get_leave_balance`, `list_pending_requests`) that hits HRMS
write APIs with the user's bearer token.

## Confirmed decisions

- HRMS write endpoints exist; staging env available — all dev + E2E hits
  staging, never prod.
- v1 scope: `apply_leave`, `apply_wfh`, `get_leave_balance`,
  `list_pending_requests`.
- Leave: half-day, multi-day, type selection (CL/SL/EL/comp-off/RH),
  reason text, with policy enforcement.
- Model: `qwen3:4b-instruct` supports native tool calling (verified via
  Ollama template — Qwen `<tool_call>`/`<tool_response>` protocol). No model
  swap needed. LangChain `ChatOllama.bind_tools([...])` wires straight into it.
- UX: inline y/N confirmation per write; one automatic retry on write
  failure, then surface the error.
- Auth: capture refresh token at login; auto-refresh access token on 401.

## Design

### 1. `app/rag/tools.py` — tool registry

Each tool declares:
- JSON Schema (name, description, params) consumed by `bind_tools`.
- Python callable receiving validated args + session context (token, profile).
- `write: bool` flag — write tools require inline confirmation.
- Optional pre-flight hook for policy checks (see §4).

| Tool | Kind | Notes |
|---|---|---|
| `get_leave_balance` | read | wraps existing profile fetch, no confirmation |
| `list_pending_requests` | read | GET pending leave/WFH |
| `apply_leave` | write | half/full day, date range, type, reason |
| `apply_wfh` | write | date range, reason |

### 2. `app/rag/hrms_client.py` — write client

- GET/POST/PUT with bearer, `HRMS_HEADERS`, HTTPS-only guard (reuses logic
  from `profile.py::fetch_json`).
- `TokenStore` holds `access_token` + `refresh_token`. On 401 → one silent
  refresh via refresh endpoint, retry original request once. Second 401 →
  surface + re-login.
- Returns `(status, body)` so the tool layer can react.

### 3. `hrms_login.py` changes

Alongside the access token, also capture the refresh token (from the login
response body / `set-cookie` / web storage). New config:
`HRMS_REFRESH_URL`, `HRMS_REFRESH_API_HINT`.

### 4. Leave-policy pre-flight (`app/rag/leave_policy.py`)

Enforced client-side before confirmation + submit so the user sees a clear
reason instead of an HRMS 400:

- `>2 consecutive leaves` → start date must be `≥20 days` away.
- `RH (Restricted Holiday)` → requested date must be one of the configured
  RH dates for the year.
- Half-day: validate not combined with multi-day in one request.
- Not in the past; not overlapping an already-approved leave (uses
  `list_pending_requests` + approved history if available).
- Balance check against `get_leave_balance` for the chosen type.

Rules live as data (`knowledge/leave-rules.yaml`), not hard-coded, so a
policy change does not need a code change.

> **Open:** full rule set beyond the two examples above — pending from user.

### 5. Agent loop in `chain.py`

- New `build_agent()` runs a bounded tool-use loop (max 5 iterations).
- Pure Q&A still goes through the current RAG chain; the agent engages when
  the LLM selects a tool or the user's intent is clearly actionable.
- Single implementation: `ChatOllama.bind_tools([...])` + loop that reads
  `AIMessage.tool_calls`, dispatches, feeds a `ToolMessage` back until the
  model returns a plain `AIMessage` with no tool calls.
- Keep existing `clean()` — strips any stray `<think>` blocks the Qwen
  template may emit around tool calls.
- Read `tool_calls` from the structured field, never regex the text.

### 6. Confirmation & dry-run

Inline y/N for every write tool, with a diff-style preview:

```
About to submit via HRMS:
  action:    apply_leave
  type:      Casual Leave
  dates:     2026-10-10  →  2026-10-11  (2 days, full)
  reason:    "family function"
Confirm? [y/N]
```

`--dry-run` flag on the `agent` command → runs policy checks + prints the
preview but never POSTs.

### 7. Guardrails additions (`guardrails.py`)

- Reject tool calls when no token loaded (can't act without identity).
- Reject tool calls when `guard_input` flagged injection on the originating
  turn — a poisoned turn must not reach a write.
- Rate-limit: ≤5 write attempts per session (configurable).
- Audit log: every tool call (name, args redacted of PII, HTTP status,
  timestamp) appended to `~/.policyqa/audit.log`. No tokens, no reason text,
  no PII — just enough to prove what was attempted.

### 8. CLI

- New subcommand: `python app/main.py agent` — same flags as `chat` plus
  `--dry-run`, `--max-writes N`.
- `chat` stays read-only (no tools). `ask` stays read-only.
- Banner shows: `agent mode · writes: enabled · dry-run: off · staging`.

### 9. Failure handling

- HRMS 4xx/5xx on write → show error body (no token) → one automatic retry
  with same payload → if still failing, surface and stop. Never silently
  second-guess the payload.
- 401 handled transparently by the token refresh in `hrms_client.py` (not
  counted as the retry).

### 10. Tests (`tests/`)

- Unit: policy pre-flight (notice period, RH date, balance, overlap).
- Unit: tool arg schema validation.
- Integration (gated behind `HRMS_STAGING=1`): happy-path leave + WFH submit
  against staging, with a tag/reason that makes test submissions obvious.

## Topicality guardrail (pre-work — fixes an existing bug)

The chatbot currently answers off-topic questions ("capital of France", "write
me Python code") because `NoContext` only raises when retrieved docs *and*
knowledge *and* profile are all empty — but `knowledge/` and `profile` are
truthy on every logged-in session, so the guard never fires and the model
falls back on parametric knowledge.

Two complementary fixes, both landed before the agent work:

- **Relevance threshold on retrieval.** Replace the pass-through `if not docs`
  check with a scored pre-check: one `similarity_search_with_relevance_scores`
  call on the original question. If the top score is below
  `RELEVANCE_THRESHOLD` *and* the question has no personal pronoun (nothing to
  answer from the HRMS profile), raise `NoContext`. Catches off-topic
  questions that happen to score low on retrieval (most of them).

- **Topicality classifier.** A yes/no LLM call before retrieval, using the
  already-resident chat model (`num_predict=4`, temperature=0). Prompt
  describes in-scope vs out-of-scope. Fails open on error, opt-out via
  `TOPICALITY_CHECK=0`. Catches off-topic questions that spuriously match
  some chunk (e.g., "write me Python code" pulling an IT-usage chunk).

Both live in `app/rag/guardrails.py` + `app/rag/config.py`; retrieval change
lives in `app/rag/chain.py::gather`. The agent inherits the same guardrails
unchanged.

## Implementation order

1. Refresh-token plumbing in `hrms_login.py` + `TokenStore`.
2. `hrms_client.py` (reads first, then POST/PUT).
3. `tools.py` — read tools, then write tools.
4. `leave_policy.py` + `knowledge/leave-rules.yaml`.
5. `build_agent()` loop in `chain.py`.
6. `agent` CLI subcommand + confirmation UI.
7. Guardrail additions + audit log.
8. Tests (unit + staging-gated integration).

## Still blocking

- **A.** HRMS endpoint shapes for `apply_leave`, `apply_wfh`,
  `get_leave_balance`, `list_pending_requests`, and the refresh-token
  endpoint — or approval to run `discover` against staging to capture them.
- **B.** Full leave-policy rule set beyond the two examples
  (>2 leaves → 20-day notice; RH on RH date only).
- **D.** Where RH dates live (static YAML, HRMS endpoint, policy PDF?).
- **E.** Staging base URL + a test employee token for E2E.
