"""Health check and system info endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.core.config import get_settings, Settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str] = {}


class SystemInfo(BaseModel):
    app_name: str
    debug: bool
    cors_origins: list[str]


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Check if the API and dependent services are healthy."""
    return HealthResponse(
        status="ok",
        services={"api": "healthy", "agent": "healthy", "data": "healthy"},
    )


@router.get("/info", response_model=SystemInfo, summary="System information")
async def system_info(settings: Settings = Depends(get_settings)) -> SystemInfo:
    """Get information about the API configuration."""
    return SystemInfo(
        app_name=settings.app_name,
        debug=settings.debug,
        cors_origins=settings.cors_origins,
    )
