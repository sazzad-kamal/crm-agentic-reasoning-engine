# =============================================================================
# Acme CRM AI Companion - Backend API
# =============================================================================
# Run the app:
#   python main.py
#
# Call the endpoint with curl:
#   curl -X POST http://localhost:8000/api/chat \
#     -H "Content-Type: application/json" \
#     -d '{"question": "What'\''s going on with Acme Manufacturing in the last 90 days?"}'
# =============================================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# App Setup
# -----------------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    type: str
    id: str
    label: str


class Company(BaseModel):
    company_id: str
    name: str
    plan: str
    status: str
    region: str
    renewal_date: str
    account_owner: str


class Activity(BaseModel):
    activity_id: str
    type: str
    occurred_at: str
    owner: str
    subject: str


class Opportunity(BaseModel):
    opportunity_id: str
    name: str
    stage: str
    value: int
    expected_close_date: str
    type: str


class RawData(BaseModel):
    companies: list[Company]
    activities: list[Activity]
    opportunities: list[Opportunity]


class MetaInfo(BaseModel):
    latency_ms: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    raw_data: RawData
    meta: MetaInfo


# -----------------------------------------------------------------------------
# Endpoint Implementation
# -----------------------------------------------------------------------------


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    # For now, ignore payload.question and always return the fixed mock response.
    return ChatResponse(
        answer=(
            "In the last 90 days, Acme Manufacturing has had several touchpoints "
            "and one active deal in the pipeline. There were 5 logged activities "
            "(a mix of calls and emails), one opportunity currently in the Proposal "
            "stage, and a renewal coming up at the end of March. Nothing in the CRM "
            "suggests major risk, but activity has slowed a bit in the last 30 days, "
            "so it may be a good time to follow up."
        ),
        sources=[
            Source(type="company", id="ACME-MFG", label="Acme Manufacturing"),
            Source(
                type="doc",
                id="opportunities_pipeline_and_forecasts.md",
                label="Opportunities, Pipeline, and Forecasts",
            ),
            Source(
                type="doc",
                id="history_activities_and_calendar.md",
                label="History, Activities, and Calendar",
            ),
        ],
        raw_data=RawData(
            companies=[
                Company(
                    company_id="ACME-MFG",
                    name="Acme Manufacturing",
                    plan="Pro",
                    status="Active",
                    region="North America",
                    renewal_date="2026-03-31",
                    account_owner="jsmith",
                )
            ],
            activities=[
                Activity(
                    activity_id="ACT-001",
                    type="Call",
                    occurred_at="2025-10-05T15:30:00Z",
                    owner="jsmith",
                    subject="Quarterly check‑in – usage review",
                ),
                Activity(
                    activity_id="ACT-002",
                    type="Email",
                    occurred_at="2025-10-20T10:15:00Z",
                    owner="jsmith",
                    subject="Shared dashboard best‑practices article",
                ),
                Activity(
                    activity_id="ACT-003",
                    type="Meeting",
                    occurred_at="2025-11-02T14:00:00Z",
                    owner="jsmith",
                    subject="Renewal prep – discussed contract options",
                ),
                Activity(
                    activity_id="ACT-004",
                    type="Email",
                    occurred_at="2025-11-18T09:40:00Z",
                    owner="jsmith",
                    subject="Follow‑up on analytics performance questions",
                ),
                Activity(
                    activity_id="ACT-005",
                    type="Call",
                    occurred_at="2025-12-01T16:10:00Z",
                    owner="jsmith",
                    subject="Check‑in – confirming report usage and next steps",
                ),
            ],
            opportunities=[
                Opportunity(
                    opportunity_id="OPP-123",
                    name="Acme Manufacturing – Pro to Enterprise upgrade",
                    stage="Proposal",
                    value=15000,
                    expected_close_date="2025-12-15",
                    type="Expansion",
                )
            ],
        ),
        meta=MetaInfo(latency_ms=820),
    )


# -----------------------------------------------------------------------------
# Simple Test (can be called manually from REPL or converted to pytest later)
# -----------------------------------------------------------------------------


def test_chat_mock() -> None:
    """Test the /api/chat endpoint with a mock question."""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"question": "What's going on with Acme Manufacturing in the last 90 days?"},
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    assert "answer" in data, "Response should contain 'answer'"
    assert (
        data["raw_data"]["companies"][0]["company_id"] == "ACME-MFG"
    ), "First company should be ACME-MFG"

    print("All tests passed!")


# -----------------------------------------------------------------------------
# Local Development Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
