"""
Application startup utilities.

Contains logging setup, RAG collection initialization, and lifespan management.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from backend.config import get_settings


logger = logging.getLogger(__name__)


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


# =============================================================================
# RAG Collection Setup
# =============================================================================


def ensure_rag_collections_exist() -> None:
    """
    Ensure RAG Qdrant collections exist, create if missing.

    This runs at startup to auto-ingest data if collections don't exist.
    """
    from qdrant_client import QdrantClient
    from backend.agent.rag.config import QDRANT_PATH, DOCS_COLLECTION, PRIVATE_COLLECTION
    from backend.agent.rag.ingest import ingest_docs, ingest_private_texts

    qdrant = QdrantClient(path=str(QDRANT_PATH))

    try:
        _ensure_docs_collection(qdrant, DOCS_COLLECTION, ingest_docs, QDRANT_PATH)
        _ensure_private_collection(qdrant, PRIVATE_COLLECTION, ingest_private_texts, QDRANT_PATH)
        qdrant.close()
    except Exception as e:
        qdrant.close()
        logger.error(f"Failed to ensure RAG collections: {e}")
        raise


def _ensure_docs_collection(qdrant, collection_name: str, ingest_fn, qdrant_path) -> None:
    """Ensure docs collection exists and has data."""
    from qdrant_client import QdrantClient

    if not qdrant.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' not found, ingesting docs...")
        qdrant.close()
        ingest_fn()
    else:
        info = qdrant.get_collection(collection_name)
        if info.points_count == 0:
            logger.info(f"Collection '{collection_name}' is empty, ingesting docs...")
            qdrant.close()
            ingest_fn()
        else:
            logger.info(f"Docs collection ready with {info.points_count} points")


def _ensure_private_collection(qdrant, collection_name: str, ingest_fn, qdrant_path) -> None:
    """Ensure private collection exists and has data."""
    from qdrant_client import QdrantClient

    if not qdrant.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' not found, ingesting private texts...")
        qdrant.close()
        ingest_fn()
    else:
        info = qdrant.get_collection(collection_name)
        if info.points_count == 0:
            logger.info(f"Collection '{collection_name}' is empty, ingesting private texts...")
            qdrant.close()
            ingest_fn()
        else:
            logger.info(f"Private collection ready with {info.points_count} points")


# =============================================================================
# Application Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    settings = get_settings()

    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"CORS origins: {settings.cors_origins_list}")

    # Ensure RAG collections exist
    ensure_rag_collections_exist()

    yield

    # Shutdown
    logger.info("Shutting down...")


__all__ = [
    "setup_logging",
    "ensure_rag_collections_exist",
    "lifespan",
]
