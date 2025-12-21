# =============================================================================
# API Routes
# =============================================================================
"""
API route definitions for the CRM AI Companion.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel, Field

from backend.agent.schemas import (
    ChatRequest,
    ChatResponse,
    Source,
    Step,
    RawData,
    MetaInfo,
)
from backend.agent.orchestrator import answer_question

# Handle both package and direct imports
try:
    from backend.config import get_settings, Settings
    from backend.exceptions import AgentError, ValidationError
except ImportError:
    from config import get_settings, Settings
    from exceptions import AgentError, ValidationError

logger = logging.getLogger(__name__)

# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(prefix="/api", tags=["chat"])


# =============================================================================
# Health Check Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    services: dict[str, str] = Field(default_factory=dict, description="Service statuses")


class SystemInfo(BaseModel):
    """System information for diagnostics."""
    app_name: str
    version: str
    debug: bool
    cors_origins: list[str]


# =============================================================================
# Chat Endpoint
# =============================================================================

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
        raise AgentError(
            message=f"Failed to process question: {str(e)}",
            details={"original_error": str(e)},
        )


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API and dependent services are healthy.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns status of the API and its dependencies.
    """
    services = {
        "api": "healthy",
        "agent": "healthy",
    }
    
    # Check if CSV data is available
    try:
        csv_dir = settings.data_dir / "csv"
        if csv_dir.exists() and any(csv_dir.glob("*.csv")):
            services["data"] = "healthy"
        else:
            services["data"] = "missing"
    except Exception:
        services["data"] = "error"
    
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        services=services,
    )


@router.get(
    "/info",
    response_model=SystemInfo,
    summary="System information",
    description="Get information about the API configuration.",
)
async def system_info(
    settings: Settings = Depends(get_settings),
) -> SystemInfo:
    """
    Get system configuration information.
    
    Useful for debugging and verifying deployment configuration.
    """
    return SystemInfo(
        app_name=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        cors_origins=settings.cors_origins_list,
    )
