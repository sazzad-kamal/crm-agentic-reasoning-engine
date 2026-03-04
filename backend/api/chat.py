"""Chat endpoint for CRM AI Companion."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agent.followup.tree import get_starters
from backend.agent.streaming import stream_agent

router = APIRouter()

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None

@router.get("/chat/starter-questions", summary="Get starter questions")
def get_starter_questions() -> list[str]:
    return get_starters()

@router.post("/chat/stream", summary="Stream a chat response (SSE)")
def chat_stream_endpoint(payload: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_agent(payload.question, session_id=payload.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
