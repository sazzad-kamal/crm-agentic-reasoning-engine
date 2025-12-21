# =============================================================================
# Acme CRM AI Companion - Backend API
# =============================================================================
"""
FastAPI backend for the CRM AI Companion.

Run the app:
    python -m backend.main
    # or
    uvicorn backend.main:app --reload

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)

Example curl:
    curl -X POST http://localhost:8000/api/chat \
        -H "Content-Type: application/json" \
        -d '{"question": "What is going on with Acme Manufacturing?"}'
"""

import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# =============================================================================
# Path Setup
# =============================================================================

# Add project root AND backend to path for flexible imports
project_root = Path(__file__).parent.parent
backend_dir = Path(__file__).parent

for path in [str(project_root), str(backend_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# =============================================================================
# Local Imports (after path setup)
# =============================================================================

# Try absolute imports first, fall back to relative for direct execution
try:
    from backend.config import get_settings
    from backend.routes import router
    from backend.middleware import RequestLoggingMiddleware, CacheControlMiddleware
    from backend.exceptions import APIError, ErrorResponse
except ImportError:
    from config import get_settings
    from routes import router
    from middleware import RequestLoggingMiddleware, CacheControlMiddleware
    from exceptions import APIError, ErrorResponse

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()
    
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    settings = get_settings()
    
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"CORS origins: {settings.cors_origins_list}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


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
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"],
    )
    
    # Custom middleware
    app.add_middleware(CacheControlMiddleware)
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
    
    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect root to API docs."""
        from fastapi.responses import RedirectResponse
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
