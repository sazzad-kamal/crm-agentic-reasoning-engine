"""FastAPI backend for the CRM AI Companion."""

import time
import uuid
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / ".env")

from backend.core.config import get_settings
from backend.core.exceptions import APIError


class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: bool = True
    status_code: int
    message: str
    request_id: str | None = None


from backend.api.chat import router as chat_router
from backend.api.health import router as health_router
from backend.api.data import router as data_router

# Logging setup
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
for lib in ["httpx", "httpcore", "openai", "urllib3"]:
    logging.getLogger(lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _ensure_rag_collections() -> None:
    from qdrant_client import QdrantClient
    from backend.agent.rag.config import QDRANT_PATH, DOCS_COLLECTION, PRIVATE_COLLECTION
    from backend.agent.rag.ingest import ingest_docs, ingest_private_texts

    for name, ingest_fn, label in [
        (DOCS_COLLECTION, ingest_docs, "docs"),
        (PRIVATE_COLLECTION, ingest_private_texts, "private"),
    ]:
        qdrant = QdrantClient(path=str(QDRANT_PATH))
        exists = qdrant.collection_exists(name)
        count = qdrant.get_collection(name).points_count if exists else 0
        qdrant.close()

        if not exists or count == 0:
            logger.info(f"Ingesting {label} collection...")
            ingest_fn()
        else:
            logger.info(f"{label.capitalize()} collection ready ({count} points)")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(f"Starting {settings.app_name}")
    _ensure_rag_collections()
    yield
    logger.info("Shutting down...")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.time()

        response = await call_next(request)
        ms = int((time.time() - start) * 1000)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{ms}ms"

        if settings.log_requests:
            log = logger.warning if response.status_code >= 400 else logger.info
            log(f"[{request_id}] {request.method} {request.url.path} {response.status_code} ({ms}ms)")
        return response


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="Talk to your CRM data using natural language.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.add_middleware(RequestLoggingMiddleware)

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
        logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                status_code=500,
                message=str(exc) if settings.debug else "Internal server error",
                request_id=request_id,
            ).model_dump(),
        )

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
