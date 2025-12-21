"""
Private text ingestion into Qdrant (MVP2).

Ingests private CRM text (history, opportunity notes, attachments) into a
separate Qdrant collection for account-scoped RAG.

Usage:
    python -m backend.rag.ingest.private_text
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)
from sentence_transformers import SentenceTransformer

from backend.rag.ingest.text_builder import find_csv_dir, build_private_texts_jsonl
from backend.rag.models import DocumentChunk
from backend.rag.config import get_config
from backend.rag.utils import estimate_tokens, chunk_text


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (from centralized config)
# =============================================================================

# Backward compatibility exports
QDRANT_PATH = get_config().qdrant_path
PRIVATE_COLLECTION_NAME = get_config().private_collection_name
EMBEDDING_MODEL = get_config().embedding_model
EMBEDDING_DIM = get_config().embedding_dim
TARGET_CHUNK_SIZE = get_config().target_chunk_size
MAX_CHUNK_SIZE = get_config().max_chunk_size
MIN_CHUNK_SIZE = get_config().min_chunk_size


# =============================================================================
# Utility Functions
# =============================================================================

def sanitize_value(value: Any) -> Any:
    """
    Sanitize a value for JSON serialization.
    
    Converts NaN, Inf, and other non-JSON-compliant values to None.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, bool)):
        return value
    if isinstance(value, dict):
        return {k: sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_value(v) for v in value]
    # For other types, try to convert to string
    try:
        return str(value) if value else None
    except Exception:
        return None


# =============================================================================
# Private Text Loading
# =============================================================================

def load_private_texts(jsonl_path: Path) -> list[dict]:
    """Load private texts from JSONL file."""
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def private_doc_to_chunks(doc: dict, chunk_idx_start: int = 0) -> list[DocumentChunk]:
    """
    Convert a private text document to DocumentChunk objects.
    
    Preserves company_id and other metadata for filtering.
    """
    text = doc.get("text", "")
    if not text or estimate_tokens(text) < MIN_CHUNK_SIZE // 2:
        # Combine title + text for very short docs
        text = f"{doc.get('title', '')}\n{text}".strip()
    
    text_chunks = chunk_text(text)
    
    chunks = []
    for i, txt_chunk in enumerate(text_chunks):
        chunk_id = f"{doc['id']}::chunk_{chunk_idx_start + i}"
        
        # Build metadata
        metadata = {
            "company_id": doc.get("company_id", ""),
            "type": doc.get("type", ""),
            "source_id": doc.get("id", ""),
            "contact_id": doc.get("contact_id"),
            "opportunity_id": doc.get("opportunity_id"),
        }
        
        # Add original metadata
        if "metadata" in doc:
            for k, v in doc["metadata"].items():
                metadata[k] = v
        
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc.get("id", ""),
            title=doc.get("title", ""),
            text=txt_chunk,
            metadata=metadata,
        )
        chunks.append(chunk)
    
    return chunks


# =============================================================================
# Ingestion
# =============================================================================

def ingest_private_texts(
    collection_name: str = PRIVATE_COLLECTION_NAME,
    recreate: bool = False,
) -> None:
    """
    Ingest private texts into Qdrant.
    
    Args:
        collection_name: Name for the Qdrant collection
        recreate: If True, always recreate collection even if it exists
    """
    print("=" * 60)
    print("Private Text Ingestion (MVP2)")
    print("=" * 60)
    
    # ---------------------------------------------------------------------------
    # 1. Locate CSV directory
    # ---------------------------------------------------------------------------
    csv_dir = find_csv_dir()
    print(f"CSV directory: {csv_dir}")
    
    # ---------------------------------------------------------------------------
    # 2. Ensure private_texts.jsonl exists
    # ---------------------------------------------------------------------------
    jsonl_path = csv_dir / "private_texts.jsonl"
    if not jsonl_path.exists():
        print(f"\nprivate_texts.jsonl not found, building it...")
        build_private_texts_jsonl(csv_dir, jsonl_path)
    else:
        print(f"\nUsing existing: {jsonl_path}")
    
    # ---------------------------------------------------------------------------
    # 3. Load JSONL documents
    # ---------------------------------------------------------------------------
    print("\nLoading private texts...")
    docs = load_private_texts(jsonl_path)
    print(f"  Loaded {len(docs)} documents")
    
    # ---------------------------------------------------------------------------
    # 4. Chunk documents
    # ---------------------------------------------------------------------------
    print("\nChunking documents...")
    all_chunks = []
    for doc in docs:
        chunks = private_doc_to_chunks(doc, chunk_idx_start=0)
        all_chunks.extend(chunks)
    
    print(f"  Created {len(all_chunks)} chunks from {len(docs)} documents")
    
    # Count by type
    type_counts = {}
    for chunk in all_chunks:
        t = chunk.metadata.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, count in sorted(type_counts.items()):
        print(f"    - {t}: {count} chunks")
    
    # ---------------------------------------------------------------------------
    # 5. Initialize Qdrant
    # ---------------------------------------------------------------------------
    print("\nInitializing Qdrant...")
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    qdrant = QdrantClient(path=str(QDRANT_PATH))
    
    # Check if collection exists
    if qdrant.collection_exists(collection_name):
        if recreate:
            print(f"  Deleting existing collection: {collection_name}")
            qdrant.delete_collection(collection_name)
        else:
            info = qdrant.get_collection(collection_name)
            if info.points_count > 0:
                print(f"  Collection '{collection_name}' exists with {info.points_count} points")
                print("  Use --recreate to rebuild. Skipping ingestion.")
                return
    
    # Create collection
    print(f"  Creating collection: {collection_name}")
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )
    
    # ---------------------------------------------------------------------------
    # 6. Embed and upload
    # ---------------------------------------------------------------------------
    print("\nLoading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Embedding chunks...")
    texts = [c.text for c in all_chunks]
    
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embed_model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
    
    # Build points
    print("\nUploading to Qdrant...")
    points = []
    for i, chunk in enumerate(all_chunks):
        # Build payload with sanitized values to handle NaN/Inf
        payload = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "text": chunk.text[:500],  # Truncated for storage
            "company_id": sanitize_value(chunk.metadata.get("company_id", "")),
            "type": sanitize_value(chunk.metadata.get("type", "")),
            "source_id": sanitize_value(chunk.metadata.get("source_id", "")),
            "contact_id": sanitize_value(chunk.metadata.get("contact_id")),
            "opportunity_id": sanitize_value(chunk.metadata.get("opportunity_id")),
        }
        
        # Add other metadata with sanitization
        for k in ["history_type", "occurred_at", "owner", "source", "file_type", "created_at", "updated_at"]:
            if k in chunk.metadata:
                payload[k] = sanitize_value(chunk.metadata[k])
        
        points.append(PointStruct(
            id=i,
            vector=all_embeddings[i].tolist(),
            payload=payload,
        ))
    
    qdrant.upsert(
        collection_name=collection_name,
        points=points,
    )
    
    # ---------------------------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Collection: {collection_name}")
    print(f"Documents: {len(docs)}")
    print(f"Chunks: {len(all_chunks)}")
    print(f"Qdrant path: {QDRANT_PATH}")
    
    # Count companies
    companies = set(c.metadata.get("company_id", "") for c in all_chunks)
    print(f"Companies: {len(companies)}")
    for company in sorted(companies):
        count = sum(1 for c in all_chunks if c.metadata.get("company_id") == company)
        print(f"  - {company}: {count} chunks")


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main():
    """Main entrypoint."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest private CRM texts into Qdrant")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection even if it exists",
    )
    parser.add_argument(
        "--collection",
        default=PRIVATE_COLLECTION_NAME,
        help=f"Collection name (default: {PRIVATE_COLLECTION_NAME})",
    )
    
    args = parser.parse_args()
    
    ingest_private_texts(
        collection_name=args.collection,
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()
