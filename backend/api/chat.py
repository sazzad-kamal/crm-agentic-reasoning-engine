"""Chat endpoint for CRM AI Companion."""

import logging

from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse

from backend.agent.core.schemas import ChatRequest
from backend.agent.output.streaming import stream_agent
from backend.core.config import get_settings, Settings
from backend.core.exceptions import ValidationError

logger = logging.getLogger(__name__)
router = APIRouter()


def _validate_question(question: str) -> None:
    """Validate question is present and within length limit."""
    if not question or not question.strip():
        raise ValidationError("Question cannot be empty", field="question")
    if len(question) > 2000:
        raise ValidationError("Question too long (max 2000 characters)", field="question")


@router.post("/chat/stream", summary="Stream a chat response (SSE)")
async def chat_stream_endpoint(
    payload: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Stream chat response as Server-Sent Events."""
    request_id = getattr(request.state, "request_id", "unknown")
    _validate_question(payload.question)

    logger.info(f"[{request_id}] Streaming: {payload.question[:100]}...")

    return StreamingResponse(
        stream_agent(
            question=payload.question,
            mode=payload.mode or "auto",
            company_id=payload.company_id,
            session_id=payload.session_id,
            user_id=payload.user_id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
