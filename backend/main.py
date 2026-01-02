# =============================================================================
# Acme CRM AI Companion - Backend API
# =============================================================================
"""
FastAPI backend for the CRM AI Companion.

Run the app:
    uvicorn backend.main:app --reload
    # or
    python -m backend.main

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)

Example curl:
    curl -X POST http://localhost:8000/api/chat \
        -H "Content-Type: application/json" \
        -d '{"question": "What is going on with Acme Manufacturing?"}'
"""

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

# =============================================================================
# Environment Setup
# =============================================================================

# Load environment variables from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# =============================================================================
# Local Imports
# =============================================================================

from backend.config import get_settings
from backend.api.chat import router as chat_router
from backend.api.health import router as health_router
from backend.api.data import router as data_router
from backend.middleware import (
    RequestLoggingMiddleware,
    CacheControlMiddleware,
    RateLimitMiddleware,
)
from backend.exceptions import APIError, ErrorResponse
from backend.startup import setup_logging, lifespan

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Combined API router
router = APIRouter(prefix="/api")
router.include_router(chat_router, tags=["chat"])
router.include_router(health_router, tags=["health"])
router.include_router(data_router, tags=["data"])


# =============================================================================
# Application Factory
# =============================================================================


def create_app() -> FastAPI:
    """
    Application factory for creating the FastAPI app.

    Returns:
        Configured FastAPI application
    """
    settings = get_settings()

    # Create app
    app = FastAPI(
        title=settings.app_name,
        description="""
## Acme CRM AI Companion API

Talk to your CRM data using natural language.

### Features
- **Natural Language Queries** - Ask questions about accounts, activities, pipeline
- **Smart Routing** - Automatically determines if you need CRM data, docs, or both
- **Grounded Answers** - Responses cite specific data sources
- **Follow-up Suggestions** - Get intelligent suggestions for next questions

### Example Questions
- "What's going on with Acme Manufacturing in the last 90 days?"
- "Which renewals are coming up this month?"
- "How do I import contacts into the CRM?"
        """,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ==========================================================================
    # Middleware (order matters - first added = last executed)
    # ==========================================================================

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-Response-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
        ],
    )

    app.add_middleware(CacheControlMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # ==========================================================================
    # Exception Handlers
    # ==========================================================================

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        """Handle custom API errors."""
        request_id = getattr(request.state, "request_id", None)

        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                status_code=exc.status_code,
                message=exc.detail,
                details=exc.details,
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        request_id = getattr(request.state, "request_id", None)

        logger.error(
            f"Unhandled exception: {exc}",
            extra={"request_id": request_id},
            exc_info=True,
        )

        # Don't expose internal errors in production
        message = str(exc) if settings.debug else "Internal server error"

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                status_code=500,
                message=message,
                request_id=request_id,
            ).model_dump(),
        )

    # ==========================================================================
    # Routes
    # ==========================================================================

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """Redirect root to API docs."""
        return RedirectResponse(url="/docs")

    return app


# =============================================================================
# App Instance
# =============================================================================

app = create_app()


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
