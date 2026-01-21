# Acme CRM AI Companion

An AI-powered CRM companion application with intelligent SQL-based data retrieval and agent capabilities. Built with FastAPI backend and React frontend.

## 🚀 Features

- **Intelligent Chat Interface**: Natural language queries about CRM data
- **SQL-Based Data Access**: DuckDB-powered queries with notes included in results
- **LLM-Powered SQL Planning**: Smart query generation for optimal data retrieval
- **Agent Orchestration**: Multi-step reasoning with tool execution
- **Real-time Progress Tracking**: Visual feedback during query processing
- **Comprehensive CRM Data**: Companies, contacts, activities, opportunities, renewals

## 📁 Project Structure

```
acme-crm-ai-companion/
├── backend/                  # FastAPI backend
│   ├── api/                 # API route handlers
│   │   ├── chat.py          # Chat endpoint
│   │   ├── health.py        # Health & info endpoints
│   │   └── data.py          # Data explorer endpoints
│   ├── agent/               # Agent orchestration
│   │   ├── graph.py         # LangGraph agent definition
│   │   ├── nodes.py         # Graph node implementations
│   │   ├── state.py         # Agent state management
│   │   ├── orchestrator.py  # Main agent pipeline
│   │   ├── llm_router.py    # LLM-based query routing
│   │   ├── datastore.py     # DuckDB CRM data store
│   │   ├── fetch/           # Data fetching (SQL planning & execution)
│   │   ├── schemas.py       # Pydantic models
│   │   ├── prompts.py       # System prompts
│   │   ├── streaming.py     # SSE streaming
│   │   ├── tools/           # Tool implementations
│   │   └── eval/            # Evaluation framework
│   ├── data/                # CRM data files
│   │   └── csv/             # CRM CSV data
│   ├── main.py              # FastAPI app entry
│   ├── exceptions.py        # Custom exceptions
│   └── config.py            # Configuration
├── frontend/                 # React + TypeScript frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── hooks/           # Custom hooks
│   │   ├── styles/          # CSS styles
│   │   └── types/           # TypeScript types
│   ├── e2e/                 # Playwright E2E tests
│   └── package.json
├── tests/                    # Test suites
│   └── backend/             # Backend unit & integration tests
├── scripts/                  # Development scripts
│   └── ci.sh                # Local CI runner
├── requirements.txt          # Python dependencies
└── pyproject.toml           # Python project config
```

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI with uvicorn
- **Data Validation**: Pydantic v2
- **Database**: DuckDB (in-memory SQL for CSV data)
- **LLM**: OpenAI GPT-5.2 (answers), GPT-5-nano (routing)
- **Agent Framework**: LangGraph
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

### Local CI (Recommended)

```bash
# Run all checks before pushing
./scripts/ci.sh all

# Backend only
./scripts/ci.sh backend

# Frontend only
./scripts/ci.sh frontend
```

### Individual Test Commands

```bash
# Backend tests
MOCK_LLM=1 pytest tests/backend/ -v

# Frontend unit tests
cd frontend && npm test

# Playwright E2E tests
cd frontend && npx playwright test
```

## 📡 API Reference

### Chat Endpoint

```http
POST /api/chat
Content-Type: application/json

{
  "question": "What's going on with Acme Manufacturing?",
  "mode": "data",  // Currently only "data" mode is supported
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
    "mode_used": "data",
    "latency_ms": 1234,
    "model": "gpt-5.2"
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
MOCK_LLM=1 pytest tests/backend/ -v
```

## 🔄 CI/CD

GitHub Actions workflows:

- **Frontend CI** (`frontend.yml`): Lint, type check, test, build
- **Backend CI** (`backend.yml`): Lint, type check, unit tests
- **E2E Tests** (`e2e.yml`): Playwright browser tests (full stack)

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
│  │  │  Fetch   │→ │   SQL    │→ │  Answer  │→ │Followup │ │   │
│  │  │  Node    │  │ Executor │  │  Chain   │  │ Node    │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐ │
│  │       DuckDB            │  │        OpenAI API           │ │
│  │  (CRM Data + Notes)     │  │          (LLM)              │ │
│  └─────────────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Example Queries

| Query Type | Example |
|------------|---------|
| Company Status | "What's going on with Acme Manufacturing?" |
| Renewals | "Which accounts have renewals this quarter?" |
| Pipeline | "Show me the sales pipeline for TechCorp" |
| Activities | "Recent activities for Global Industries" |
| Account Search | "Show me all Enterprise accounts" |
| Contacts | "Who are the decision makers at Global Industries?" |
| Notes | "What are the notes about Acme Manufacturing?" |

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
- FastAPI for the excellent web framework
- React team for the frontend framework
