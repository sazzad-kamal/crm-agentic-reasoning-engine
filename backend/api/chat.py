"""
Chat endpoint for CRM AI Companion.

Main endpoint for answering natural language questions about CRM data.
"""

import logging

from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse

from backend.agent.core.schemas import (
    ChatRequest,
    ChatResponse,
    Source,
    Step,
    RawData,
    MetaInfo,
)
from backend.agent.graph import answer_question
from backend.agent.output.streaming import stream_agent
from backend.core.config import get_settings, Settings
from backend.core.exceptions import AgentError, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question about your CRM",
    description="""
    Main chat endpoint - answers questions using CRM data and/or documentation.

    The agent pipeline:
    1. **Router** - Determines if the question needs CRM data, docs, or both
    2. **Data Gathering** - Fetches relevant company, activity, pipeline data
    3. **Docs RAG** - Retrieves relevant documentation if needed
    4. **Answer Synthesis** - Generates a grounded answer using LLM
    5. **Follow-up Suggestions** - Suggests relevant next questions

    ## Example Questions
    - "What's going on with Acme Manufacturing in the last 90 days?"
    - "Which renewals are coming up this month?"
    - "Show me the pipeline for TechCorp"
    - "How do I import contacts?" (docs query)
    """,
    responses={
        200: {"description": "Successful response with answer"},
        400: {"description": "Invalid request"},
        500: {"description": "Agent processing error"},
        503: {"description": "LLM service unavailable"},
    },
)
async def chat_endpoint(
    payload: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """
    Process a natural language question about CRM data.

    Args:
        payload: The chat request containing the question and optional parameters
        request: FastAPI request object (for request ID)
        settings: Application settings

    Returns:
        ChatResponse with answer, sources, steps, and follow-up suggestions
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Validate question
    if not payload.question or not payload.question.strip():
        raise ValidationError("Question cannot be empty", field="question")

    if len(payload.question) > 2000:
        raise ValidationError(
            "Question too long (max 2000 characters)",
            field="question",
        )

    logger.info(
        f"[{request_id}] Processing question: {payload.question[:100]}...",
        extra={
            "request_id": request_id,
            "question_length": len(payload.question),
            "mode": payload.mode,
            "company_id": payload.company_id,
        },
    )

    try:
        # Call the agent
        result = answer_question(
            question=payload.question,
            mode=payload.mode or "auto",
            company_id=payload.company_id,
            session_id=payload.session_id,
            user_id=payload.user_id,
        )

        # Build response
        response = ChatResponse(
            answer=result["answer"],
            sources=[Source(**s) for s in result["sources"]],
            steps=[Step(**s) for s in result["steps"]],
            raw_data=RawData(**result["raw_data"]),
            meta=MetaInfo(**result["meta"]),
            follow_up_suggestions=result.get("follow_up_suggestions", []),
        )

        logger.info(
            f"[{request_id}] Response generated: {len(response.answer)} chars, "
            f"{len(response.sources)} sources, {len(response.follow_up_suggestions)} follow-ups",
            extra={
                "request_id": request_id,
                "answer_length": len(response.answer),
                "source_count": len(response.sources),
                "mode_used": response.meta.mode_used,
            },
        )

        return response

    except Exception as e:
        logger.error(
            f"[{request_id}] Agent error: {e}",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise AgentError(f"Failed to process question: {str(e)}")


@router.post(
    "/chat/stream",
    summary="Stream a chat response (SSE)",
    description="""
    Streaming version of the chat endpoint using Server-Sent Events (SSE).

    Returns real-time progress updates as the agent processes the question:
    - **status**: Progress messages (e.g., "Routing question...")
    - **step**: Step completions with status
    - **sources**: Sources discovered during data/docs fetch
    - **answer_start**: Answer generation begins
    - **answer_chunk**: Incremental answer text
    - **answer_end**: Full answer available
    - **followup**: Follow-up suggestions
    - **done**: Final complete response
    - **error**: Error occurred

    ## Usage (JavaScript)
    ```javascript
    const eventSource = new EventSource('/api/chat/stream?question=...');
    eventSource.addEventListener('status', (e) => console.log(JSON.parse(e.data)));
    eventSource.addEventListener('answer_chunk', (e) => { /* append text */ });
    eventSource.addEventListener('done', (e) => { /* final response */ });
    ```
    """,
    responses={
        200: {"description": "SSE stream of events"},
        400: {"description": "Invalid request"},
    },
)
async def chat_stream_endpoint(
    payload: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """
    Stream chat response as Server-Sent Events.

    Provides real-time feedback as the agent processes the question.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Validate question
    if not payload.question or not payload.question.strip():
        raise ValidationError("Question cannot be empty", field="question")

    if len(payload.question) > 2000:
        raise ValidationError(
            "Question too long (max 2000 characters)",
            field="question",
        )

    logger.info(
        f"[{request_id}] Streaming question: {payload.question[:100]}...",
        extra={
            "request_id": request_id,
            "question_length": len(payload.question),
            "mode": payload.mode,
            "streaming": True,
        },
    )

    return StreamingResponse(
        stream_agent(
            question=payload.question,
            mode=payload.mode or "auto",
            company_id=payload.company_id,
            session_id=payload.session_id,
            user_id=payload.user_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
