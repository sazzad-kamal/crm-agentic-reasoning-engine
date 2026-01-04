"""Chat endpoint for CRM AI Companion."""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend.agent.core.schemas import ChatRequest
from backend.agent.output.streaming import stream_agent

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat/stream", summary="Stream a chat response (SSE)")
async def chat_stream_endpoint(payload: ChatRequest, request: Request) -> StreamingResponse:
    logger.info(f"[{getattr(request.state, 'request_id', '-')}] Streaming: {payload.question[:100]}...")
    return StreamingResponse(
        stream_agent(payload.question, payload.mode or "auto", payload.company_id, payload.session_id, payload.user_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
