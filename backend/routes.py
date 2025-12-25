# =============================================================================
# API Routes
# =============================================================================
"""
API route definitions for the CRM AI Companion.
"""

import logging
from pathlib import Path
from typing import Optional, Any, Callable

import pandas as pd
from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse
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
from backend.agent.streaming import stream_agent

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
# Streaming Chat Endpoint
# =============================================================================

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


# =============================================================================
# Data Explorer Endpoints
# =============================================================================

class DataResponse(BaseModel):
    """Response containing CRM data records."""
    data: list[dict[str, Any]] = Field(description="List of data records")
    total: int = Field(description="Total number of records")
    columns: list[str] = Field(description="Column names")


def load_csv_data(csv_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load data from a CSV file using pandas."""
    if not csv_path.exists():
        return [], []
    df = pd.read_csv(csv_path)
    return df.to_dict("records"), df.columns.tolist()


def load_jsonl_data(jsonl_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load data from a JSONL file using pandas."""
    if not jsonl_path.exists():
        return [], []
    df = pd.read_json(jsonl_path, lines=True)
    # Flatten metadata column if present
    if "metadata" in df.columns:
        meta_df = pd.json_normalize(df["metadata"]).add_prefix("metadata_")
        df = pd.concat([df.drop(columns=["metadata"]), meta_df], axis=1)
    return df.to_dict("records"), df.columns.tolist()


def _group_by_key(
    data: list[dict], key_field: str, extract_field: str | None = None
) -> dict[str, list[dict]]:
    """Group records by a key field. Helper for enrichment."""
    result: dict[str, list[dict]] = {}
    for record in data:
        key = record.get(extract_field or key_field) or record.get(key_field, "")
        if key:
            result.setdefault(key, []).append(record)
    return result


def _create_simple_data_endpoint(
    file_name: str,
    is_jsonl: bool = False,
):
    """Factory for simple data endpoints without enrichment."""
    async def endpoint(settings: Settings = Depends(get_settings)) -> DataResponse:
        path = settings.data_dir / "csv" / file_name
        loader = load_jsonl_data if is_jsonl else load_csv_data
        data, columns = loader(path)
        return DataResponse(data=data, total=len(data), columns=columns)
    return endpoint


@router.get(
    "/data/companies",
    response_model=DataResponse,
    summary="Get all companies with related data",
    description="Returns all company records with their private texts (notes, attachments).",
)
async def get_companies(
    settings: Settings = Depends(get_settings),
) -> DataResponse:
    """Get all company data with nested private texts."""
    data, columns = load_csv_data(settings.data_dir / "csv" / "companies.csv")
    private_texts, _ = load_jsonl_data(settings.data_dir / "csv" / "private_texts.jsonl")
    texts_by_company = _group_by_key(private_texts, "company_id", "metadata_company_id")
    
    for company in data:
        company["_private_texts"] = texts_by_company.get(company.get("company_id", ""), [])
    
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get(
    "/data/contacts",
    response_model=DataResponse,
    summary="Get all contacts with related data",
    description="Returns all contact records with their private texts.",
)
async def get_contacts(
    settings: Settings = Depends(get_settings),
) -> DataResponse:
    """Get all contact data with nested private texts."""
    data, columns = load_csv_data(settings.data_dir / "csv" / "contacts.csv")
    private_texts, _ = load_jsonl_data(settings.data_dir / "csv" / "private_texts.jsonl")
    texts_by_contact = _group_by_key(private_texts, "contact_id", "metadata_contact_id")
    
    for contact in data:
        contact["_private_texts"] = texts_by_contact.get(contact.get("contact_id", ""), [])
    
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get(
    "/data/opportunities",
    response_model=DataResponse,
    summary="Get all opportunities with related data",
    description="Returns all opportunity records with descriptions and attachments.",
)
async def get_opportunities(
    settings: Settings = Depends(get_settings),
) -> DataResponse:
    """Get all opportunity data with descriptions and attachments."""
    data, columns = load_csv_data(settings.data_dir / "csv" / "opportunities.csv")
    descriptions, _ = load_csv_data(settings.data_dir / "csv" / "opportunity_descriptions.csv")
    attachments, _ = load_csv_data(settings.data_dir / "csv" / "attachments.csv")
    
    desc_by_opp = _group_by_key(descriptions, "opportunity_id")
    attach_by_opp = _group_by_key(attachments, "opportunity_id")
    
    for opp in data:
        opp_id = opp.get("opportunity_id", "")
        opp["_descriptions"] = desc_by_opp.get(opp_id, [])
        opp["_attachments"] = attach_by_opp.get(opp_id, [])
    
    return DataResponse(data=data, total=len(data), columns=columns)


# =============================================================================
# Simple Data Endpoints (factory-generated)
# =============================================================================

router.get(
    "/data/activities",
    response_model=DataResponse,
    summary="Get all activities",
    description="Returns all activity records from the CRM.",
)(endpoint := _create_simple_data_endpoint("activities.csv"))

router.get(
    "/data/private-texts",
    response_model=DataResponse,
    summary="Get all private texts",
    description="Returns all private text records (attachments, notes, etc.).",
)(_create_simple_data_endpoint("private_texts.jsonl", is_jsonl=True))

router.get(
    "/data/history",
    response_model=DataResponse,
    summary="Get all history",
    description="Returns all history/timeline records from the CRM.",
)(_create_simple_data_endpoint("history.csv"))

router.get(
    "/data/group-members",
    response_model=DataResponse,
    summary="Get all group members",
    description="Returns all group membership records from the CRM.",
)(_create_simple_data_endpoint("group_members.csv"))

router.get(
    "/data/attachments",
    response_model=DataResponse,
    summary="Get all attachments",
    description="Returns all attachment records from the CRM.",
)(_create_simple_data_endpoint("attachments.csv"))

router.get(
    "/data/opportunity-descriptions",
    response_model=DataResponse,
    summary="Get all opportunity descriptions",
    description="Returns all opportunity description records from the CRM.",
)(_create_simple_data_endpoint("opportunity_descriptions.csv"))


@router.get(
    "/data/groups",
    response_model=DataResponse,
    summary="Get all groups with members",
    description="Returns all group records with their members.",
)
async def get_groups(
    settings: Settings = Depends(get_settings),
) -> DataResponse:
    """Get all group data with nested members."""
    data, columns = load_csv_data(settings.data_dir / "csv" / "groups.csv")
    members, _ = load_csv_data(settings.data_dir / "csv" / "group_members.csv")
    members_by_group = _group_by_key(members, "group_id")
    
    for group in data:
        group["_members"] = members_by_group.get(group.get("group_id", ""), [])
    
    return DataResponse(data=data, total=len(data), columns=columns)
