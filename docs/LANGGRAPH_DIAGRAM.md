# LangGraph Multi-Agent Architecture

## Complete Flow Diagram

```mermaid
flowchart TB
    subgraph Entry["Entry Point"]
        Q["User Question"]
    end

    subgraph Supervisor["Supervisor Node"]
        CL["Intent Classifier"]
        CL -->|"Heuristics"| H1["Quick Match"]
        CL -->|"Ambiguous"| H2["LLM Fallback<br/>(GPT-4o-mini)"]
    end

    subgraph DataAgents["Specialized Data Agents"]
        FETCH["Fetch<br/>━━━━━━━━━━<br/>Simple SQL queries<br/>'Show all deals'"]
        COMPARE["Compare<br/>━━━━━━━━━━<br/>A vs B analysis<br/>'Q1 vs Q2 revenue'"]
        TREND["Trend<br/>━━━━━━━━━━<br/>Time-series<br/>'Revenue by month'"]
        PLANNER["Planner<br/>━━━━━━━━━━<br/>Multi-step queries<br/>'Show X and compare Y'"]
        EXPORT["Export<br/>━━━━━━━━━━<br/>CSV/PDF/JSON<br/>'Export to CSV'"]
        HEALTH["Health<br/>━━━━━━━━━━<br/>Account scoring<br/>'Acme health score'"]
    end

    subgraph ResponseGen["Response Generation"]
        ANSWER["Answer Node<br/>━━━━━━━━━━<br/>Synthesize response<br/>Evidence tagging"]
        ACTION["Action Node<br/>━━━━━━━━━━<br/>Suggest next steps"]
        FOLLOWUP["Followup Node<br/>━━━━━━━━━━<br/>Generate questions"]
    end

    subgraph DirectResponse["Direct Response"]
        CLARIFY["Clarify<br/>'Could you be more specific?'"]
        HELP["Help<br/>'I can help with CRM data...'"]
    end

    Q --> CL

    CL -->|"data_query"| FETCH
    CL -->|"compare"| COMPARE
    CL -->|"trend"| TREND
    CL -->|"complex"| PLANNER
    CL -->|"export"| EXPORT
    CL -->|"health"| HEALTH
    CL -->|"clarify"| CLARIFY
    CL -->|"help"| HELP

    FETCH --> ANSWER
    COMPARE --> ANSWER
    TREND --> ANSWER
    PLANNER --> ANSWER
    EXPORT --> ANSWER
    HEALTH --> ANSWER

    ANSWER -->|"has_data"| ACTION
    ANSWER -->|"has_data"| FOLLOWUP
    ANSWER -->|"needs_more_data"| FETCH

    CLARIFY --> END1["END"]
    HELP --> END2["END"]
    ACTION --> END3["END"]
    FOLLOWUP --> END4["END"]
```

## State Flow

```mermaid
stateDiagram-v2
    [*] --> Supervisor: question

    state Supervisor {
        [*] --> Classify
        Classify --> Route
    }

    Supervisor --> Fetch: data_query
    Supervisor --> Compare: compare
    Supervisor --> Trend: trend
    Supervisor --> Planner: complex
    Supervisor --> Export: export
    Supervisor --> Health: health
    Supervisor --> Answer: clarify/help

    state DataAgentGroup {
        Fetch
        Compare
        Trend
        Planner
        Export
        Health
    }

    Fetch --> Answer: sql_results
    Compare --> Answer: comparison
    Trend --> Answer: trend_analysis
    Planner --> Answer: aggregated
    Export --> Answer: export
    Health --> Answer: health_analysis

    Answer --> Fetch: needs_more_data
    Answer --> ActionFollowup: has_data
    Answer --> [*]: no_data

    state ActionFollowup {
        [*] --> Action
        [*] --> Followup
        Action --> [*]
        Followup --> [*]
    }

    ActionFollowup --> [*]
```

## Intent Classification

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTENT CLASSIFIER                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: "Compare Q1 vs Q2 revenue"                              │
│                                                                  │
│  1. HEURISTICS (fast, no API call)                              │
│     ├── Length < 4 chars? ──────────────────────> CLARIFY       │
│     ├── Help priority phrases? ─────────────────> HELP          │
│     ├── Export keywords? ───────────────────────> EXPORT        │
│     ├── Multi-part with "and"? ─────────────────> COMPLEX       │
│     ├── Compare keywords (vs, compare)? ────────> COMPARE  ✓    │
│     ├── Trend keywords? ────────────────────────> TREND         │
│     ├── Health keywords? ───────────────────────> HEALTH        │
│     └── Data indicators? ───────────────────────> DATA_QUERY    │
│                                                                  │
│  2. LLM FALLBACK (if no heuristic match)                        │
│     └── GPT-4o-mini classification                              │
│                                                                  │
│  Output: Intent.COMPARE                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Details

### Data Agents

| Agent | Intent | Input | Output | Example |
|-------|--------|-------|--------|---------|
| **Fetch** | `data_query` | Question | `sql_results.data` | "Show all deals" |
| **Compare** | `compare` | Question | `sql_results.comparison` | "Q1 vs Q2 revenue" |
| **Trend** | `trend` | Question | `sql_results.trend_analysis` | "Revenue by month" |
| **Planner** | `complex` | Question | `sql_results.aggregated` | "Show X and compare Y" |
| **Export** | `export` | Question | `sql_results.export` | "Export to CSV" |
| **Health** | `health` | Question | `sql_results.health_analysis` | "Acme health score" |

### Response Agents

| Agent | Purpose | Input | Output |
|-------|---------|-------|--------|
| **Answer** | Synthesize response | `sql_results` | `answer` with evidence tags |
| **Action** | Suggest next steps | `answer` | `suggested_action` |
| **Followup** | Generate questions | `answer` | `follow_up_suggestions` |

## Data Flow Example

```
User: "Compare Q1 vs Q2 revenue"
        │
        ▼
┌─────────────────┐
│   Supervisor    │──> Intent: COMPARE
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Compare Agent  │
│  ┌───────────┐  │
│  │ Extract   │  │──> entity_a: "Q1", entity_b: "Q2"
│  │ Entities  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ SQL for   │  │──> SELECT ... WHERE date IN Q1
│  │ Entity A  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ SQL for   │  │──> SELECT ... WHERE date IN Q2
│  │ Entity B  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Calculate │  │──> diff: +50000, pct: +25%
│  │ Metrics   │  │
│  └───────────┘  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Answer Node    │──> "Q2 revenue increased 25% vs Q1 [E1]"
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌────────┐
│Action │ │Followup│
└───────┘ └────────┘
    │         │
    ▼         ▼
"Export"  "Show Q3?"
```

## File Structure

```
backend/agent/
├── graph.py              # LangGraph orchestration
├── state.py              # AgentState TypedDict
├── supervisor/
│   ├── classifier.py     # Intent classification
│   └── node.py           # Supervisor node
├── fetch/                # Simple SQL queries
├── compare/              # A vs B comparisons
├── trend/                # Time-series analysis
├── planner/              # Multi-step orchestration
├── export/               # File generation
├── health/               # Account health scoring
├── answer/               # Response synthesis
├── action/               # Action suggestions
└── followup/             # Follow-up questions
```
