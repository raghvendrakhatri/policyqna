# Agent Pipeline

How one question flows through agent mode (`app/rag/agent.py` → `ask_agent`).

```mermaid
flowchart TD
    Q([Employee question]) --> G

    subgraph S1["1 · Screen (screen)"]
        G{guard_input<br/>empty / too long /<br/>injection / unsafe?}
        G -- fail --> R1[[Refused panel]]
        G -- ok --> F{Follow-up reply?<br/>'yes', 'el-1'}
        F -- yes --> NP[prefetch = false]
        F -- no --> RT{route_question<br/>policy / hrms / no}
        RT -- "no & not personal" --> R2[[Out of scope panel]]
        RT -- policy --> P1[prefetch = true]
        RT -- "hrms / action+date" --> NP
    end

    subgraph S2["2 · Build input (begin)"]
        M[Memory window + summary]
        P1 --> W[with_policy<br/>find_policy top-K excerpts]
        W --> H
        NP --> H
        M --> H[HumanMessage<br/>+ date_hints YYYY-MM-DD]
    end

    subgraph S3["3 · Agent loop (LangGraph create_agent, fresh thread)"]
        H --> SP[dynamic_prompt<br/>system prompt: today, HRMS/no-login,<br/>knowledge, profile]
        SP --> LLM[LLM call]
        LLM -- tool calls --> HITL{Write tool<br/>or login?}
        HITL -- no --> MW
        HITL -- yes: interrupt --> D{decide}
        D --> AD{auto_decision<br/>no token / already logged in /<br/>bad dates?}
        AD -- reject --> MW
        AD -- ask --> C[Confirm panel y/N<br/>or ask_credentials]
        C -- approve/reject --> MW
        MW[hrms_session middleware<br/>inject token · trace · cap 8000 chars] --> T[(Tools: login, search_policy,<br/>leave / WFH / holidays / team /<br/>feedback)]
        T -- ToolMessage --> LLM
        LLM -- plain text --> A[Answer]
        LLM -. "> 5 steps" .-> X[[Steps exhausted]]
    end

    subgraph S4["4 · Output"]
        A --> CL[clean: strip reasoning]
        CL --> SF{check_safety<br/>output}
        SF -- unsafe --> R3[[Unsafe output panel]]
        SF -- safe --> OUT([Answer panel])
    end
```

## Steps

| # | Step | Where | Notes |
|---|------|-------|-------|
| 1 | Input guard | `guard_input` | Empty, length, injection regex, safety LLM |
| 2 | Route | `route_question` | `policy` / `hrms` / `no`; follow-ups skip routing; fails open to `policy` |
| 3 | Prefetch | `with_policy` | Policy excerpts only on `policy` route (k = `AGENT_PREFETCH_K`) |
| 4 | Seed messages | `Agent.begin` | Memory + question + resolved dates; new `thread_id` per question |
| 5 | System prompt | `dynamic_prompt` | Rebuilt every model call, so a mid-chat login takes effect |
| 6 | Tool loop | `create_agent` | Up to `AGENT_MAX_STEPS` (5) model→HITL→tool rounds |
| 7 | Approval | `HumanInTheLoopMiddleware` + `decide` | `apply_leave`, `cancel_leave`, `apply_wfh`, `submit_feedback_form`, `login`; resumes via `Command(resume=…)` |
| 8 | Token injection | `hrms_session` | Token from run context, never in messages; results truncated at 8000 chars |
| 9 | Output guard | `clean` + `checked` | Strip reasoning tags, safety check, render panel |
