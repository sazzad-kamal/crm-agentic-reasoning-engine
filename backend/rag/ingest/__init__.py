# backend.rag.ingest - Document Ingestion
"""
Ingestion scripts for RAG documents.

Modules:
- docs: Markdown document ingestion
- private_text: Private CRM text ingestion  
- text_builder: Private text JSONL builder
- csv_utils: CSV directory location utilities

These are CLI scripts - import them directly to run:
    python -m backend.rag.ingest.docs
    python -m backend.rag.ingest.private_text
"""

from backend.rag.ingest.csv_utils import find_csv_dir, get_csv_path

__all__ = ["find_csv_dir", "get_csv_path"]
