# backend.rag.ingest - Document Ingestion
"""
Ingestion functions for RAG documents.

Modules:
- docs: Markdown document ingestion
- private_text: Private CRM text ingestion  
- text_builder: Private text JSONL builder
- chunking: Text chunking utilities
- constants: Shared constants

Ingestion runs automatically at API startup if collections are missing.
"""

from backend.rag.ingest.docs import ingest_all_docs
from backend.rag.ingest.private_text import ingest_private_texts

__all__ = [
    "ingest_all_docs",
    "ingest_private_texts",
]
