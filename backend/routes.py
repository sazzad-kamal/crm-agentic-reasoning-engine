# =============================================================================
# API Routes
# =============================================================================
"""
API route definitions for the CRM AI Companion.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional, Any

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


# =============================================================================
# Data Explorer Endpoints
# =============================================================================

class DataResponse(BaseModel):
    """Response containing CRM data records."""
    data: list[dict[str, Any]] = Field(description="List of data records")
    total: int = Field(description="Total number of records")
    columns: list[str] = Field(description="Column names")


def load_csv_data(csv_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load data from a CSV file."""
    if not csv_path.exists():
        return [], []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)
        columns = reader.fieldnames or []
    
    return data, columns


def load_jsonl_data(jsonl_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load data from a JSONL file."""
    if not jsonl_path.exists():
        return [], []
    
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # Flatten metadata for display
                if "metadata" in record and isinstance(record["metadata"], dict):
                    for key, value in record["metadata"].items():
                        record[f"metadata_{key}"] = value
                    del record["metadata"]
                data.append(record)
    
    if data:
        columns = list(data[0].keys())
    else:
        columns = []
    
    return data, columns


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
    csv_path = settings.data_dir / "csv" / "companies.csv"
    data, columns = load_csv_data(csv_path)
    
    # Load private texts and group by company_id
    jsonl_path = settings.data_dir / "csv" / "private_texts.jsonl"
    private_texts, _ = load_jsonl_data(jsonl_path)
    texts_by_company: dict[str, list[dict[str, Any]]] = {}
    for pt in private_texts:
        cid = pt.get("metadata_company_id") or pt.get("company_id", "")
        if cid:
            texts_by_company.setdefault(cid, []).append(pt)
    
    # Enrich companies with their private texts
    for company in data:
        company_id = company.get("company_id", "")
        company["_private_texts"] = texts_by_company.get(company_id, [])
    
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
    csv_path = settings.data_dir / "csv" / "contacts.csv"
    data, columns = load_csv_data(csv_path)
    
    # Load private texts and group by contact_id
    jsonl_path = settings.data_dir / "csv" / "private_texts.jsonl"
    private_texts, _ = load_jsonl_data(jsonl_path)
    texts_by_contact: dict[str, list[dict[str, Any]]] = {}
    for pt in private_texts:
        cid = pt.get("metadata_contact_id") or pt.get("contact_id", "")
        if cid:
            texts_by_contact.setdefault(cid, []).append(pt)
    
    # Enrich contacts with their private texts
    for contact in data:
        contact_id = contact.get("contact_id", "")
        contact["_private_texts"] = texts_by_contact.get(contact_id, [])
    
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
    csv_path = settings.data_dir / "csv" / "opportunities.csv"
    data, columns = load_csv_data(csv_path)
    
    # Load opportunity descriptions
    desc_path = settings.data_dir / "csv" / "opportunity_descriptions.csv"
    descriptions, _ = load_csv_data(desc_path)
    desc_by_opp: dict[str, list[dict[str, Any]]] = {}
    for desc in descriptions:
        oid = desc.get("opportunity_id", "")
        if oid:
            desc_by_opp.setdefault(oid, []).append(desc)
    
    # Load attachments related to opportunities
    attach_path = settings.data_dir / "csv" / "attachments.csv"
    attachments, _ = load_csv_data(attach_path)
    attach_by_opp: dict[str, list[dict[str, Any]]] = {}
    for att in attachments:
        oid = att.get("opportunity_id", "")
        if oid:
            attach_by_opp.setdefault(oid, []).append(att)
    
    # Enrich opportunities with descriptions and attachments
    for opp in data:
        opp_id = opp.get("opportunity_id", "")
        opp["_descriptions"] = desc_by_opp.get(opp_id, [])
        opp["_attachments"] = attach_by_opp.get(opp_id, [])
    
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get(
    "/data/activities",
    response_model=DataResponse,
    summary="Get all activities",
    description="Returns all activity records from the CRM.",
)
async def get_activities(
    settings: Settings = Depends(get_settings),
) -> DataResponse:
    """Get all activity data."""
    csv_path = settings.data_dir / "csv" / "activities.csv"
    data, columns = load_csv_data(csv_path)
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get(
    "/data/private-texts",
    response_model=DataResponse,
    summary="Get all private texts",
    description="Returns all private text records (attachments, notes, etc.).",
)
async def get_private_texts(
    settings: Settings = Depends(get_settings),
) -> DataResponse:
    """Get all private text data (attachments, notes, emails)."""
    jsonl_path = settings.data_dir / "csv" / "private_texts.jsonl"
    data, columns = load_jsonl_data(jsonl_path)
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get(
    "/data/history",
    response_model=DataResponse,
    summary="Get all history",
    description="Returns all history/timeline records from the CRM.",
)
async def get_history(
    settings: Settings = Depends(get_settings),
) -> DataResponse:
    """Get all history data."""
    csv_path = settings.data_dir / "csv" / "history.csv"
    data, columns = load_csv_data(csv_path)
    return DataResponse(data=data, total=len(data), columns=columns)


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
    csv_path = settings.data_dir / "csv" / "groups.csv"
    data, columns = load_csv_data(csv_path)
    
    # Load group members
    members_path = settings.data_dir / "csv" / "group_members.csv"
    members, _ = load_csv_data(members_path)
    members_by_group: dict[str, list[dict[str, Any]]] = {}
    for member in members:
        gid = member.get("group_id", "")
        if gid:
            members_by_group.setdefault(gid, []).append(member)
    
    # Enrich groups with their members
    for group in data:
        group_id = group.get("group_id", "")
        group["_members"] = members_by_group.get(group_id, [])
    
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get(
    "/data/group-members",
    response_model=DataResponse,
    summary="Get all group members",
    description="Returns all group membership records from the CRM.",
)
async def get_group_members(
    settings: Settings = Depends(get_settings),
) -> DataResponse:
    """Get all group member data."""
    csv_path = settings.data_dir / "csv" / "group_members.csv"
    data, columns = load_csv_data(csv_path)
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get(
    "/data/attachments",
    response_model=DataResponse,
    summary="Get all attachments",
    description="Returns all attachment records from the CRM.",
)
async def get_attachments(
    settings: Settings = Depends(get_settings),
) -> DataResponse:
    """Get all attachment data."""
    csv_path = settings.data_dir / "csv" / "attachments.csv"
    data, columns = load_csv_data(csv_path)
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get(
    "/data/opportunity-descriptions",
    response_model=DataResponse,
    summary="Get all opportunity descriptions",
    description="Returns all opportunity description records from the CRM.",
)
async def get_opportunity_descriptions(
    settings: Settings = Depends(get_settings),
) -> DataResponse:
    """Get all opportunity description data."""
    csv_path = settings.data_dir / "csv" / "opportunity_descriptions.csv"
    data, columns = load_csv_data(csv_path)
    return DataResponse(data=data, total=len(data), columns=columns)
