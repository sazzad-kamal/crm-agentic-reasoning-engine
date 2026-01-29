"""FastAPI backend for the CRM AI Companion."""

import base64
import logging
import os
import secrets
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

load_dotenv(Path(__file__).parent.parent / ".env")

from backend.api.chat import router as chat_router
from backend.api.data import router as data_router

# Configuration
APP_NAME = "Acme CRM AI Companion API"
CORS_ORIGINS = os.getenv("ACME_CORS_ORIGINS", "http://localhost:5173").split(",")
AUTH_USER = os.getenv("AUTH_USER", "")
AUTH_PASS = os.getenv("AUTH_PASS", "")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
for lib in ["httpx", "httpcore", "openai", "urllib3"]:
    logging.getLogger(lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    auth_status = "enabled" if AUTH_USER and AUTH_PASS else "disabled (no AUTH_USER/AUTH_PASS)"
    logger.info(f"Starting {APP_NAME} — auth {auth_status}")
    yield
    logger.info("Shutting down...")


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """HTTP Basic Auth — only active when AUTH_USER and AUTH_PASS are set."""

    REALM = 'Basic realm="Acme CRM AI"'
    EXEMPT_PATHS = {"/api/health"}

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        header = request.headers.get("authorization", "")
        if not header:
            return PlainTextResponse(
                "Authentication required",
                status_code=401,
                headers={"WWW-Authenticate": self.REALM},
            )

        parts = header.split(" ", 1)
        if len(parts) != 2 or parts[0] != "Basic":
            return PlainTextResponse("Invalid auth format", status_code=401)

        try:
            decoded = base64.b64decode(parts[1]).decode()
            user, password = decoded.split(":", 1)
        except Exception:
            return PlainTextResponse("Invalid auth format", status_code=401)

        if secrets.compare_digest(user, AUTH_USER) and secrets.compare_digest(password, AUTH_PASS):
            return await call_next(request)

        return PlainTextResponse(
            "Invalid credentials",
            status_code=401,
            headers={"WWW-Authenticate": self.REALM},
        )


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
        allow_headers=["Content-Type", "Authorization"],
    )
    if AUTH_USER and AUTH_PASS:
        app.add_middleware(BasicAuthMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    router = APIRouter(prefix="/api")

    @router.get("/health", tags=["health"])
    async def health() -> dict:
        return {"status": "ok"}

    router.include_router(chat_router, tags=["chat"])
    router.include_router(data_router, tags=["data"])
    app.include_router(router)

    # Serve frontend static files in production (built into frontend/dist/)
    frontend_dir = Path(__file__).parent.parent / "frontend" / "dist"
    if frontend_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="static")

        @app.get("/{path:path}", include_in_schema=False)
        async def serve_spa(path: str) -> FileResponse:
            """Serve frontend SPA — all non-API routes return index.html."""
            file_path = frontend_dir / path
            if file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(frontend_dir / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse(url="/docs")

    return app


app = create_app()
