# CRM Chat Assistant

**Ask questions about your CRM in plain English. Get answers grounded in data.**

![Tests](https://img.shields.io/badge/tests-1,149_passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-purple)

<p align="center">
  <img src="docs/demo-screenshot.png" alt="CRM Chat Assistant Demo" width="800">
</p>

---

## The Problem

LLMs hallucinate. They make up data, invent statistics, and confidently cite sources that don't exist.

**This system can't hallucinate.** Every claim links to actual data with evidence tags. If the data doesn't exist, it says so.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                             SUPERVISOR                                      │
│                    Heuristics-first classification                          │
│                    (LLM fallback only when needed)                          │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
        ┌─────────┬─────────┬─────┴─────┬─────────┬─────────┐
        ▼         ▼         ▼           ▼         ▼         ▼
    ┌───────┐ ┌───────┐ ┌───────┐ ┌─────────┐ ┌───────┐ ┌───────┐
    │ FETCH │ │COMPARE│ │ TREND │ │ PLANNER │ │EXPORT │ │HEALTH │
    │       │ │       │ │       │ │         │ │       │ │       │
    │  SQL  │ │ A vs B│ │ Time  │ │ Multi-  │ │ CSV/  │ │Account│
    │ Query │ │       │ │ Series│ │ Agent   │ │ PDF   │ │ Score │
    └───┬───┘ └───┬───┘ └───┬───┘ └────┬────┘ └───┬───┘ └───┬───┘
        │         │         │          │          │         │
        └─────────┴─────────┴────┬─────┴──────────┴─────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              ANSWER                                         │
│                                                                             │
│   • Synthesize with evidence tags [E1], [E2]                               │
│   • Loop back if more data needed (max 2)                                   │
│   • Validate → Repair → Fallback (never crashes)                           │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
               ┌─────────┐                ┌──────────┐
               │ ACTION  │                │ FOLLOWUP │
               └─────────┘                └──────────┘
```

---

## What Makes This Production-Grade

### Multi-Agent Orchestration

6 specialized agents. Each optimized for its task:

| Query | Agent | What Happens |
|-------|-------|--------------|
| "Show Q1 deals" | **Fetch** | SQL generation → DuckDB |
| "Q1 vs Q2 revenue" | **Compare** | Parallel queries → Delta analysis |
| "Revenue trend" | **Trend** | Time-series → Growth metrics |
| "Deals and compare regions" | **Planner** | Decompose → Fan-out → Aggregate |
| "Export to CSV" | **Export** | Query → File generation |
| "Acme health score" | **Health** | Multi-factor scoring |

**Planner** handles complex queries by decomposing and routing to multiple agents:

```
"Show deals and compare Q1 vs Q2"
              │
    ┌─────────┴─────────┐
    ▼                   ▼
  FETCH              COMPARE
    │                   │
    └─────────┬─────────┘
              ▼
          AGGREGATE → ANSWER
```

### Heuristics-First Classification

**90% of queries classified without LLM** — fast and cheap:

```
"export deals"     →  EXPORT   (keyword)
"Q1 vs Q2"         →  COMPARE  (keyword)
"yes"              →  CLARIFY  (too short)
"hmm not sure"     →  LLM      (ambiguous)
```

### Contract-Enforced Outputs

Every LLM output: **Validate → Repair → Fallback**

```
LLM Output → Validate → [fail] → Repair → [fail] → Fallback
                ↓
            [pass] → Return
```

**The system never crashes on bad LLM output.**

### Evidence-Grounded Responses

Every claim cites its source:

```
The deal is in Negotiation [E1] valued at $50,000 [E2].

Evidence:
- E1: opportunities.stage = "Negotiation"
- E2: opportunities.value = 50000
```

### Data Refinement Loops

Answer can request more data (max 2 iterations):

```
Fetch → Answer: "Need more context"
           └─→ Fetch (refined) → Answer: "Complete picture..."
```

### SQL Safety Guard

All SQL validated via `sqlglot`:
- **Blocked**: INSERT, UPDATE, DELETE, DROP
- **Auto-added**: LIMIT 1000

---

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| **Orchestration** | LangGraph | Stateful workflows, checkpointing |
| **SQL Gen** | Claude | Better structured output |
| **Synthesis** | GPT-4 | Natural language strength |
| **Database** | DuckDB | Fast analytical queries |
| **Backend** | FastAPI | Async, type-safe |
| **Frontend** | React + TypeScript | Modern, maintainable |

---

## Quality

| Metric | Value |
|--------|-------|
| **Total Tests** | **1,149** |
| Backend (pytest) | 420 |
| Frontend (Vitest) | 562 |
| E2E (Playwright) | 167 |
| **Faithfulness** | ≥ 0.9 (RAGAS) |
| **p50 Latency** | < 3s |

---

## Quick Start

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
uvicorn backend.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

Open http://localhost:5173

---

## API

```http
POST /api/chat/stream
{"question": "What deals closed this quarter?"}
```

SSE Events: `fetch_start` → `answer_chunk` → `action` → `followup` → `done`

---

## Deep Dive

- [Full Architecture Docs](docs/ARCHITECTURE.md)
- [LangGraph Diagram](docs/LANGGRAPH_DIAGRAM.md)

---

## License

MIT
