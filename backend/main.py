"""FastAPI backend for the CRM AI Companion."""

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

from backend.core.config import get_settings
from backend.api.chat import router as chat_router
from backend.api.health import router as health_router
from backend.api.data import router as data_router
from backend.core.middleware import RequestLoggingMiddleware
from backend.core.exceptions import APIError, ErrorResponse
from backend.core.lifespan import setup_logging, lifespan

setup_logging()
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Talk to your CRM data using natural language.",
        version=settings.app_version,
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    # Exception handlers
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                status_code=exc.status_code,
                message=exc.detail,
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.error(f"Unhandled exception: {exc}", extra={"request_id": request_id}, exc_info=True)

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                status_code=500,
                message=str(exc) if settings.debug else "Internal server error",
                request_id=request_id,
            ).model_dump(),
        )

    # Routes
    router = APIRouter(prefix="/api")
    router.include_router(chat_router, tags=["chat"])
    router.include_router(health_router, tags=["health"])
    router.include_router(data_router, tags=["data"])
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
