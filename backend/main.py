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

load_dotenv(Path(__file__).parent.parent / ".env")

from backend.core.config import get_settings
from backend.core.exceptions import APIError, ErrorResponse
from backend.api.chat import router as chat_router
from backend.api.health import router as health_router
from backend.api.data import router as data_router


# Logging setup
def _setup_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for lib in ["httpx", "httpcore", "openai", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


_setup_logging()
logger = logging.getLogger(__name__)


# RAG collection setup
def _ensure_rag_collections() -> None:
    from qdrant_client import QdrantClient
    from backend.agent.rag.config import QDRANT_PATH, DOCS_COLLECTION, PRIVATE_COLLECTION
    from backend.agent.rag.ingest import ingest_docs, ingest_private_texts

    qdrant = QdrantClient(path=str(QDRANT_PATH))
    try:
        for collection, ingest_fn, label in [
            (DOCS_COLLECTION, ingest_docs, "docs"),
            (PRIVATE_COLLECTION, ingest_private_texts, "private"),
        ]:
            if not qdrant.collection_exists(collection):
                logger.info(f"Collection '{collection}' not found, ingesting {label}...")
                qdrant.close()
                ingest_fn()
            else:
                info = qdrant.get_collection(collection)
                if info.points_count == 0:
                    logger.info(f"Collection '{collection}' is empty, ingesting {label}...")
                    qdrant.close()
                    ingest_fn()
                else:
                    logger.info(f"{label.capitalize()} collection ready with {info.points_count} points")
        qdrant.close()
    except Exception as e:
        qdrant.close()
        logger.error(f"Failed to ensure RAG collections: {e}")
        raise


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    _ensure_rag_collections()
    yield
    logger.info("Shutting down...")


# Request logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start_time = time.time()

        if settings.log_requests:
            logger.info(f"[{request_id}] {request.method} {request.url.path}")

        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"[{request_id}] Error after {int((time.time() - start_time) * 1000)}ms: {e}", exc_info=True)
            raise

        elapsed_ms = int((time.time() - start_time) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed_ms}ms"

        if settings.log_requests:
            log_fn = logger.info if response.status_code < 400 else logger.warning
            log_fn(f"[{request_id}] {request.method} {request.url.path} - {response.status_code} ({elapsed_ms}ms)")

        return response


# App factory
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Talk to your CRM data using natural language.",
        version=settings.app_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"],
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
        logger.error(f"Unhandled exception: {exc}", extra={"request_id": request_id}, exc_info=True)
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
