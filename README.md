# Acme CRM AI Companion

An AI-powered CRM companion application with RAG (Retrieval-Augmented Generation) and intelligent agent capabilities. Built with FastAPI backend and React frontend.

## 🚀 Features

- **Intelligent Chat Interface**: Natural language queries about CRM data
- **RAG Pipeline**: Hybrid search with Qdrant vectors + BM25 sparse retrieval
- **LLM-Powered Routing**: Smart query classification for optimal data retrieval
- **Agent Orchestration**: Multi-step reasoning with tool execution
- **Real-time Progress Tracking**: Visual feedback during query processing
- **Comprehensive CRM Data**: Companies, contacts, activities, opportunities, renewals

## 📁 Project Structure

```
acme-crm-ai-companion/
├── backend/                  # FastAPI backend
│   ├── agent/               # Agent orchestration & tools
│   │   ├── orchestrator.py  # Main agent pipeline
│   │   ├── llm_router.py    # LLM-based query routing
│   │   ├── datastore.py     # DuckDB CRM data store
│   │   ├── tools.py         # Tool functions
│   │   └── schemas.py       # Pydantic models
│   ├── rag/                 # RAG retrieval system
│   │   ├── retrieval.py     # Hybrid search backend
│   │   ├── pipeline.py      # RAG pipeline
│   │   └── models.py        # Document models
│   ├── common/              # Shared utilities
│   │   └── llm_client.py    # OpenAI client wrapper
│   ├── data/                # CRM data files
│   │   ├── csv/             # CRM CSV data
│   │   └── docs/            # Product documentation
│   ├── main.py              # FastAPI app entry
│   ├── routes.py            # API endpoints
│   ├── middleware.py        # Request/response middleware
│   └── config.py            # Configuration
├── frontend/                 # React + TypeScript frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── hooks/           # Custom hooks
│   │   ├── styles/          # CSS styles
│   │   └── types/           # TypeScript types
│   └── package.json
├── tests/                    # E2E and integration tests
│   └── e2e/                 # End-to-end tests
├── .github/workflows/        # CI/CD pipelines
├── requirements.txt          # Python dependencies
└── pyproject.toml           # Python project config
```

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI with uvicorn
- **Data Validation**: Pydantic v2
- **Database**: DuckDB (in-memory SQL for CSV data)
- **Vector Store**: Qdrant
- **Embeddings**: sentence-transformers (BAAI/bge-small-en-v1.5)
- **Reranking**: Cross-encoder (BAAI/bge-reranker-base)
- **LLM**: OpenAI GPT-4.1
- **Retry Logic**: Tenacity

### Frontend
- **Framework**: React 18 + TypeScript 5
- **Build Tool**: Vite 5
- **Testing**: Vitest + React Testing Library
- **Styling**: CSS with design tokens

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenAI API key

### Backend Setup

```bash
# Clone the repository
git clone <repo-url>
cd acme-crm-ai-companion

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your-api-key

# Run the backend
python -m uvicorn backend.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

## 🧪 Testing

### Run All Tests

```bash
# Backend tests (with mock LLM)
MOCK_LLM=1 pytest backend/ -v

# Frontend tests
cd frontend && npm test

# E2E tests
MOCK_LLM=1 pytest tests/e2e/ -v
```

### Test Coverage

| Component | Tests | Coverage |
|-----------|-------|----------|
| Frontend | 79 | ~95% |
| Backend Agent | 45+ | ~90% |
| Backend RAG | 30+ | ~85% |
| Backend Core | 25+ | ~80% |
| E2E | 35+ | Full flow |

## 📡 API Reference

### Chat Endpoint

```http
POST /api/chat
Content-Type: application/json

{
  "question": "What's going on with Acme Manufacturing?",
  "mode": "auto",  // "auto" | "data" | "docs" | "data+docs"
  "days": 30       // Optional: time range for data queries
}
```

**Response:**
```json
{
  "answer": "Acme Manufacturing is an active Enterprise customer...",
  "sources": [
    {"type": "company", "id": "ACME-MFG", "name": "Acme Manufacturing"}
  ],
  "steps": [
    {"id": "route", "label": "Analyzed question", "status": "done"}
  ],
  "meta": {
    "mode_used": "data+docs",
    "latency_ms": 1234,
    "model": "gpt-4.1-mini"
  },
  "follow_ups": ["What opportunities are open?", "Any upcoming renewals?"]
}
```

### Health Check

```http
GET /api/health
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `MOCK_LLM` | Enable mock LLM for testing | `0` |
| `ACME_DEBUG` | Enable debug mode | `false` |
| `ACME_LOG_LEVEL` | Logging level | `INFO` |
| `ACME_CORS_ORIGINS` | Allowed CORS origins | `localhost:5173,localhost:3000` |

### Mock Mode

For testing without an API key:
```bash
export MOCK_LLM=1
pytest backend/ -v
```

## 🔄 CI/CD

The project includes comprehensive GitHub Actions workflows:

- **Frontend CI** (`frontend.yml`): Lint, type check, test, build
- **Backend CI** (`backend.yml`): Lint, type check, agent tests, RAG tests, core tests
- **E2E Tests** (`e2e.yml`): Full API and integration tests
- **RAG Eval** (`rag-eval.yml`): RAG retrieval quality evaluation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  ChatArea   │  │  InputBar   │  │  MessageBlock/DataTables│ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST /api/chat
┌────────────────────────────▼────────────────────────────────────┐
│                     Backend (FastAPI)                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Agent Orchestrator                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │   │
│  │  │  Router  │→ │  Tools   │→ │   RAG    │→ │   LLM   │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  DuckDB     │  │   Qdrant    │  │    OpenAI API           │ │
│  │  (CRM Data) │  │  (Vectors)  │  │    (LLM)                │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Example Queries

| Query Type | Example |
|------------|---------|
| Company Status | "What's going on with Acme Manufacturing?" |
| Renewals | "Which accounts have renewals this quarter?" |
| Pipeline | "Show me the sales pipeline for TechCorp" |
| Activities | "Recent activities for Global Industries" |
| Documentation | "How do I create a new opportunity?" |
| Mixed | "Best practices for managing Acme's pipeline?" |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `npm test` and `pytest`
5. Commit: `git commit -am 'Add my feature'`
6. Push: `git push origin feature/my-feature`
7. Open a Pull Request

## 📄 License

This project is for demonstration purposes.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Qdrant for vector search
- FastAPI for the excellent web framework
- React team for the frontend framework
