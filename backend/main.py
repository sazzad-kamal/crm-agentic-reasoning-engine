"""FastAPI backend for the CRM AI Companion."""

import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

load_dotenv(Path(__file__).parent.parent / ".env")

from backend.api.chat import router as chat_router
from backend.api.data import router as data_router

# Configuration
APP_NAME = "Acme CRM AI Companion API"
CORS_ORIGINS = os.getenv("ACME_CORS_ORIGINS", "http://localhost:5173").split(",")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
for lib in ["httpx", "httpcore", "openai", "urllib3"]:
    logging.getLogger(lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _ensure_rag_collections() -> None:
    from qdrant_client import QdrantClient

    from backend.agent.fetch.rag.config import QDRANT_PATH, TEXT_COLLECTION
    from backend.agent.fetch.rag.ingest import ingest_texts

    qdrant = QdrantClient(path=str(QDRANT_PATH))
    exists = qdrant.collection_exists(TEXT_COLLECTION)
    count = qdrant.get_collection(TEXT_COLLECTION).points_count if exists else 0
    qdrant.close()

    if not exists or count == 0:
        logger.info("Ingesting text collection...")
        ingest_texts()
    else:
        logger.info(f"Text collection ready ({count} points)")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(f"Starting {APP_NAME}")
    _ensure_rag_collections()
    yield
    logger.info("Shutting down...")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.time()

        response: Response = await call_next(request)
        ms = int((time.time() - start) * 1000)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{ms}ms"

        log = logger.warning if response.status_code >= 400 else logger.info
        log(f"[{request_id}] {request.method} {request.url.path} {response.status_code} ({ms}ms)")
        return response


def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        description="Talk to your CRM data using natural language.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    router = APIRouter(prefix="/api")

    @router.get("/health", tags=["health"])
    async def health() -> dict:
        return {"status": "ok"}

    router.include_router(chat_router, tags=["chat"])
    router.include_router(data_router, tags=["data"])
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
