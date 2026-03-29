# CRM Agentic Reasoning Engine - Code Map

Quick reference for navigating the codebase. Find any file in seconds.

## API Routes

| Endpoint | File | Description |
|----------|------|-------------|
| `POST /api/chat/stream` | [backend/api/chat.py](../backend/api/chat.py#L20) | Stream chat response (SSE) |
| `GET /api/chat/starter-questions` | [backend/api/chat.py](../backend/api/chat.py#L16) | Get conversation starters |
| `GET /api/data/contacts` | [backend/api/data.py](../backend/api/data.py) | List all contacts |
| `GET /api/data/opportunities` | [backend/api/data.py](../backend/api/data.py) | List all opportunities |
| `GET /api/data/activities` | [backend/api/data.py](../backend/api/data.py) | List all activities |
| `GET /api/data/companies` | [backend/api/data.py](../backend/api/data.py) | List all companies |
| `GET /api/data/history` | [backend/api/data.py](../backend/api/data.py) | List interaction history |
| `GET /api/health` | [backend/main.py](../backend/main.py) | Health check endpoint |

## LangGraph Agents

| Agent | File | Intent | Description |
|-------|------|--------|-------------|
| **Supervisor** | [backend/agent/supervisor/node.py](../backend/agent/supervisor/node.py) | routes all | Intent classifier + router |
| **Fetch** | [backend/agent/fetch/node.py](../backend/agent/fetch/node.py) | DATA_QUERY | SQL generation + execution |
| **Compare** | [backend/agent/compare/node.py](../backend/agent/compare/node.py) | COMPARE | A vs B comparison queries |
| **Trend** | [backend/agent/trend/node.py](../backend/agent/trend/node.py) | TREND | Time-series analysis |
| **Planner** | [backend/agent/planner/node.py](../backend/agent/planner/node.py) | COMPLEX | Multi-query decomposition |
| **Export** | [backend/agent/export/node.py](../backend/agent/export/node.py) | EXPORT | CSV/PDF generation |
| **Health** | [backend/agent/health/node.py](../backend/agent/health/node.py) | HEALTH | Account health scoring |
| **RAG** | [backend/agent/rag/node.py](../backend/agent/rag/node.py) | DOCS | Documentation search |
| **Graph** | [backend/agent/graph_rag/node.py](../backend/agent/graph_rag/node.py) | GRAPH | Neo4j multi-hop queries |
| **Answer** | [backend/agent/answer/node.py](../backend/agent/answer/node.py) | all | Response synthesis |
| **Action** | [backend/agent/action/node.py](../backend/agent/action/node.py) | all | CRM action suggestions |
| **Followup** | [backend/agent/followup/node.py](../backend/agent/followup/node.py) | all | Follow-up questions |

## Intent Classification

| Component | File | Line | Description |
|-----------|------|------|-------------|
| Intent enum | [backend/agent/supervisor/classifier.py](../backend/agent/supervisor/classifier.py#L11) | 11-22 | All intent types |
| Heuristics | [backend/agent/supervisor/classifier.py](../backend/agent/supervisor/classifier.py#L74) | 74-100 | 90% classified without LLM |
| LLM fallback | [backend/agent/supervisor/classifier.py](../backend/agent/supervisor/classifier.py#L64) | 64+ | Ambiguous queries |
| Classifier prompt | [backend/agent/supervisor/classifier.py](../backend/agent/supervisor/classifier.py#L25) | 25-61 | Full classification prompt |

## SQL Safety

| Component | File | Line | Description |
|-----------|------|------|-------------|
| Forbidden statements | [backend/agent/sql/guard.py](../backend/agent/sql/guard.py#L16) | 16-24 | INSERT, UPDATE, DELETE, DROP |
| Forbidden functions | [backend/agent/sql/guard.py](../backend/agent/sql/guard.py#L27) | 27-36 | copy, export, attach, etc. |
| MAX_ROWS limit | [backend/agent/sql/guard.py](../backend/agent/sql/guard.py#L45) | 45 | 1000 row limit |
| Guard result | [backend/agent/sql/guard.py](../backend/agent/sql/guard.py#L48) | 48+ | Validation result model |
| SQL executor | [backend/agent/sql/executor.py](../backend/agent/sql/executor.py) | — | Execute validated SQL |
| DuckDB connection | [backend/agent/sql/connection.py](../backend/agent/sql/connection.py) | — | Database connection pool |

## Contract Validation

| Component | File | Line | Description |
|-----------|------|------|-------------|
| ContractValidator | [backend/agent/validate/contract.py](../backend/agent/validate/contract.py#L36) | 36+ | Generic validate→repair→fallback |
| ContractResult | [backend/agent/validate/contract.py](../backend/agent/validate/contract.py#L24) | 24-33 | Validation result model |
| Answer validation | [backend/agent/validate/answer.py](../backend/agent/validate/answer.py) | — | Answer contract |
| Action validation | [backend/agent/validate/action.py](../backend/agent/validate/action.py) | — | Action contract |
| Followup validation | [backend/agent/validate/followup.py](../backend/agent/validate/followup.py) | — | Followup contract |
| Grounding check | [backend/agent/validate/grounding.py](../backend/agent/validate/grounding.py) | — | Evidence verification |
| Repair logic | [backend/agent/validate/repair.py](../backend/agent/validate/repair.py) | — | Auto-repair prompts |

## RAG Pipeline (LlamaIndex)

| Component | File | Description |
|-----------|------|-------------|
| Indexer | [backend/agent/rag/indexer.py](../backend/agent/rag/indexer.py) | PDF ingestion + vector index |
| Retriever | [backend/agent/rag/retriever.py](../backend/agent/rag/retriever.py) | Semantic search + reranking |
| RAG node | [backend/agent/rag/node.py](../backend/agent/rag/node.py) | LangGraph integration |

## LangGraph Orchestration

| Component | File | Line | Description |
|-----------|------|------|-------------|
| Graph definition | [backend/agent/graph.py](../backend/agent/graph.py#L1) | 1-16 | Architecture docstring |
| Node names | [backend/agent/graph.py](../backend/agent/graph.py#L47) | 47-60 | All node constants |
| Graph builder | [backend/agent/graph.py](../backend/agent/graph.py) | — | StateGraph setup |
| Agent state | [backend/agent/state.py](../backend/agent/state.py) | — | Shared state model |
| Streaming | [backend/agent/streaming.py](../backend/agent/streaming.py) | — | SSE event generator |

## Follow-up Questions

| Component | File | Description |
|-----------|------|-------------|
| Suggester | [backend/agent/followup/suggester.py](../backend/agent/followup/suggester.py) | LLM-based suggestions |
| Entity context | [backend/agent/followup/entity_context.py](../backend/agent/followup/entity_context.py) | Context extraction |
| Question tree | [backend/agent/followup/tree/loader.py](../backend/agent/followup/tree/loader.py) | Starter questions |

## Evaluation System

| Component | File | Description |
|-----------|------|-------------|
| **Answer Eval** | | |
| Text evaluation | [backend/eval/answer/text/runner.py](../backend/eval/answer/text/runner.py) | RAGAS faithfulness |
| Action evaluation | [backend/eval/answer/action/runner.py](../backend/eval/answer/action/runner.py) | Action quality |
| Suppression | [backend/eval/answer/text/suppression.py](../backend/eval/answer/text/suppression.py) | "I don't know" detection |
| **Fetch Eval** | | |
| SQL judge | [backend/eval/fetch/sql_judge.py](../backend/eval/fetch/sql_judge.py) | SQL correctness |
| **Followup Eval** | | |
| Followup judge | [backend/eval/followup/judge.py](../backend/eval/followup/judge.py) | Question quality |
| **Integration** | | |
| Full runner | [backend/eval/integration/runner.py](../backend/eval/integration/runner.py) | End-to-end eval |
| Gate checks | [backend/eval/integration/gate.py](../backend/eval/integration/gate.py) | Pass/fail criteria |

## Frontend Components

| Component | File | Description |
|-----------|------|-------------|
| Main App | [frontend/src/App.tsx](../frontend/src/App.tsx) | Root component |
| Chat Area | [frontend/src/components/ChatArea.tsx](../frontend/src/components/ChatArea.tsx) | Message display |
| Input Bar | [frontend/src/components/InputBar.tsx](../frontend/src/components/InputBar.tsx) | Question input |
| Data Explorer | [frontend/src/components/DataExplorer.tsx](../frontend/src/components/DataExplorer.tsx) | CRM data browser |
| Evidence Card | [frontend/src/components/EvidenceCard.tsx](../frontend/src/components/EvidenceCard.tsx) | Citation display |

## Frontend Hooks

| Hook | File | Description |
|------|------|-------------|
| useChatStream | [frontend/src/hooks/useChatStream.ts](../frontend/src/hooks/useChatStream.ts) | SSE streaming |
| useFocusTrap | [frontend/src/hooks/useFocusTrap.ts](../frontend/src/hooks/useFocusTrap.ts) | Accessibility |
| useDataExplorer | [frontend/src/hooks/useDataExplorer.ts](../frontend/src/hooks/useDataExplorer.ts) | Data fetching |

## Tests

| Category | Path | Count |
|----------|------|-------|
| Backend unit | [tests/backend/](../tests/backend/) | 610 |
| Agent tests | [tests/backend/agent/](../tests/backend/agent/) | ~400 |
| API tests | [tests/backend/api/](../tests/backend/api/) | ~50 |
| Eval tests | [tests/backend/eval/](../tests/backend/eval/) | ~100 |
| Frontend unit | [frontend/src/__tests__/](../frontend/src/__tests__/) | 562 |
| E2E | [frontend/e2e/](../frontend/e2e/) | 167 |

## Quick Commands

```bash
# Run backend tests
pytest tests/ -v

# Run frontend tests
cd frontend && npm test

# Run E2E tests
cd frontend && npm run test:e2e

# Start dev servers
uvicorn backend.main:app --reload  # Backend on :8000
cd frontend && npm run dev          # Frontend on :5173

# Run full eval
python -m backend.eval.integration
```

## Key Files Summary

| What | File |
|------|------|
| FastAPI app | [backend/main.py](../backend/main.py) |
| LangGraph definition | [backend/agent/graph.py](../backend/agent/graph.py) |
| Agent state | [backend/agent/state.py](../backend/agent/state.py) |
| Intent classifier | [backend/agent/supervisor/classifier.py](../backend/agent/supervisor/classifier.py) |
| SQL guard | [backend/agent/sql/guard.py](../backend/agent/sql/guard.py) |
| Contract validator | [backend/agent/validate/contract.py](../backend/agent/validate/contract.py) |
| RAG retriever | [backend/agent/rag/retriever.py](../backend/agent/rag/retriever.py) |
| SSE streaming | [backend/agent/streaming.py](../backend/agent/streaming.py) |
| React app | [frontend/src/App.tsx](../frontend/src/App.tsx) |
| Chat hook | [frontend/src/hooks/useChatStream.ts](../frontend/src/hooks/useChatStream.ts) |
