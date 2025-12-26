"""
Constants for the ingest module.
"""

from pathlib import Path

# Paths
_BACKEND_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = _BACKEND_ROOT / "data/docs"

# Chunking Parameters
TARGET_CHUNK_SIZE = 500  # tokens
MAX_CHUNK_SIZE = 700     # tokens
MIN_CHUNK_SIZE = 100     # tokens
CHUNK_OVERLAP = 50       # tokens
