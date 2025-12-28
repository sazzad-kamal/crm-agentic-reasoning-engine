"""
Private text ingestion into Qdrant (MVP2).

Ingests private CRM text (history, opportunity notes, attachments) into a
separate Qdrant collection for account-scoped RAG.
"""

import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path

# Python 3.12+ has itertools.batched, add polyfill for 3.11
if sys.version_info >= (3, 12):
    from itertools import batched
else:
    from itertools import islice

    def batched(iterable, n):
        """Batch data into lists of length n. The last batch may be shorter."""
        it = iter(iterable)
        while batch := list(islice(it, n)):
            yield batch
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)
from sentence_transformers import SentenceTransformer

from backend.rag.utils import find_csv_dir
from backend.rag.models import DocumentChunk
from backend.rag.retrieval.constants import PRIVATE_COLLECTION, QDRANT_PATH
from backend.rag.retrieval.constants import EMBEDDING_MODEL, EMBEDDING_DIM
from backend.rag.ingest.chunking import estimate_tokens, chunk_text, MIN_CHUNK_SIZE


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# JSONL Builder Functions
# =============================================================================

def _synthesize_opportunity_descriptions(csv_dir: Path) -> pd.DataFrame:
    """
    Synthesize opportunity_descriptions.csv from opportunities.csv.
    
    Creates realistic notes based on opportunity data.
    """
    opp_path = csv_dir / "opportunities.csv"
    if not opp_path.exists():
        return pd.DataFrame()
    
    opps = pd.read_csv(opp_path)
    
    # Use deterministic seed
    random.seed(42)
    
    # Templates for generating notes
    focus_templates = [
        "streamlining contact management and deduplication",
        "automating follow-ups and reminders",
        "consolidating history/notes into a single customer timeline",
        "segmenting contacts for targeted campaigns",
        "cleaning up pipeline stages and forecasting accuracy",
    ]
    
    risk_templates = [
        "duplicate contacts causing confusion",
        "inconsistent activity logging across the team",
        "low engagement from key contacts",
    ]
    
    next_step_templates = [
        "schedule a 30‑minute admin setup call",
        "import an updated contact list and resolve duplicates",
        "build a renewal dashboard and set weekly review cadence",
        "create a saved group for renewals and start outreach",
    ]
    
    records = []
    for _, row in opps.iterrows():
        opp_id = row.get("opportunity_id", "")
        company_id = row.get("company_id", "")
        contact_id = row.get("primary_contact_id", "")
        company_name = row.get("company_name", company_id)
        stage = row.get("stage", "Unknown")
        value = row.get("value", 0)
        currency = row.get("currency", "USD")
        expected_close = row.get("expected_close_date", "TBD")
        
        # Deterministic selection based on opp_id hash
        hash_val = hash(opp_id)
        focus = focus_templates[hash_val % len(focus_templates)]
        risk = risk_templates[hash_val % len(risk_templates)]
        next_step = next_step_templates[hash_val % len(next_step_templates)]
        
        text = f"""{company_name} opportunity context:
- Stage: {stage}
- Value: {value} {currency}
- Expected close: {expected_close}

Summary: Customer is exploring changes focused on {focus}. Primary risk noted is {risk}. Recommended next step: {next_step}."""
        
        records.append({
            "opportunity_id": opp_id,
            "company_id": company_id,
            "primary_contact_id": contact_id,
            "title": f"Opportunity Notes – {opp_id}",
            "text": text,
            "created_at": row.get("created_at", ""),
            "updated_at": row.get("updated_at", ""),
        })
    
    return pd.DataFrame(records)


def build_private_texts_jsonl(csv_dir: Path, out_path: Path) -> None:
    """
    Build private_texts.jsonl from CRM CSV files.
    
    Args:
        csv_dir: Directory containing CSV files
        out_path: Output path for JSONL file
        
    This file is used for private account-scoped RAG ingestion.
    """
    logger.info(f"Building private texts from {csv_dir}")
    
    all_docs = []
    
    # 1. Process history.csv
    history_path = csv_dir / "history.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        
        for _, row in history_df.iterrows():
            history_id = row.get("history_id", "")
            doc = {
                "id": f"history::{history_id}",
                "company_id": str(row.get("company_id", "")),
                "contact_id": str(row.get("contact_id", "")),
                "opportunity_id": row.get("opportunity_id") if pd.notna(row.get("opportunity_id")) else None,
                "type": "history",
                "title": str(row.get("subject", "")),
                "text": str(row.get("description", "")),
                "metadata": {
                    "history_type": str(row.get("type", "")),
                    "occurred_at": str(row.get("occurred_at", "")),
                    "owner": str(row.get("owner", "")),
                    "source": str(row.get("source", "")),
                },
            }
            all_docs.append(doc)
        
        logger.info(f"  Processed {len(history_df)} history records")
    
    # 2. Process opportunity_descriptions.csv (or synthesize)
    opp_desc_path = csv_dir / "opportunity_descriptions.csv"
    if opp_desc_path.exists():
        opp_desc_df = pd.read_csv(opp_desc_path)
    else:
        opp_desc_df = _synthesize_opportunity_descriptions(csv_dir)
    
    if not opp_desc_df.empty:
        for _, row in opp_desc_df.iterrows():
            opp_id = row.get("opportunity_id", "")
            doc = {
                "id": f"opp_note::{opp_id}",
                "company_id": str(row.get("company_id", "")),
                "contact_id": str(row.get("primary_contact_id", "")),
                "opportunity_id": str(opp_id),
                "type": "opportunity_note",
                "title": str(row.get("title", f"Opportunity Notes – {opp_id}")),
                "text": str(row.get("text", "")),
                "metadata": {
                    "updated_at": str(row.get("updated_at", "")),
                },
            }
            all_docs.append(doc)
        
        logger.info(f"  Processed {len(opp_desc_df)} opportunity descriptions")
    
    # 3. Process attachments.csv
    attachments_path = csv_dir / "attachments.csv"
    if attachments_path.exists():
        attachments_df = pd.read_csv(attachments_path)
        
        for _, row in attachments_df.iterrows():
            att_id = row.get("attachment_id", "")
            doc = {
                "id": f"attachment::{att_id}",
                "company_id": str(row.get("company_id", "")),
                "contact_id": str(row.get("contact_id", "")),
                "opportunity_id": row.get("opportunity_id") if pd.notna(row.get("opportunity_id")) else None,
                "type": "attachment",
                "title": str(row.get("title", "")),
                "text": str(row.get("summary", "")),
                "metadata": {
                    "file_type": str(row.get("file_type", "")),
                    "created_at": str(row.get("created_at", "")),
                },
            }
            all_docs.append(doc)
        
        logger.info(f"  Processed {len(attachments_df)} attachments")
    
    # 4. Write JSONL output (stable, deterministic order)
    all_docs.sort(key=lambda x: x["id"])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    logger.info(f"Wrote {len(all_docs)} documents to {out_path}")


# =============================================================================
# Utility Functions
# =============================================================================

def sanitize_value(value: Any) -> Any:
    """Sanitize a value for JSON serialization (handles NaN, Inf, etc)."""
    match value:
        case None:
            return None
        case float() if math.isnan(value) or math.isinf(value):
            return None
        case str() | int() | bool():
            return value
        case dict():
            return {k: sanitize_value(v) for k, v in value.items()}
        case list() | tuple():
            return [sanitize_value(v) for v in value]
        case _:
            return str(value) if value else None


# =============================================================================
# Private Text Loading
# =============================================================================

def load_private_texts(jsonl_path: Path) -> list[dict]:
    """Load private texts from JSONL file."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
    
    # Base metadata for all chunks
    base_metadata = {
        "company_id": doc.get("company_id", ""),
        "type": doc.get("type", ""),
        "source_id": doc.get("id", ""),
        "contact_id": doc.get("contact_id"),
        "opportunity_id": doc.get("opportunity_id"),
    } | doc.get("metadata", {})
    
    return [
        DocumentChunk(
            chunk_id=f"{doc['id']}::chunk_{chunk_idx_start + i}",
            doc_id=doc.get("id", ""),
            title=doc.get("title", ""),
            text=txt_chunk,
            metadata=base_metadata,
        )
        for i, txt_chunk in enumerate(text_chunks)
    ]


# =============================================================================
# Ingestion
# =============================================================================

def ingest_private_texts(
    collection_name: str = PRIVATE_COLLECTION,
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
    type_counts = Counter(c.metadata.get("type", "unknown") for c in all_chunks)
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
    
    all_embeddings = []
    for batch in batched(texts, 32):
        embeddings = embed_model.encode(
            list(batch),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.extend(embeddings)
        print(f"  Embedded {len(all_embeddings)}/{len(texts)} chunks")
    
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
    console = Console()
    
    # Count companies
    companies = Counter(c.metadata.get("company_id", "") for c in all_chunks)
    
    # Summary table
    table = Table(title="Ingestion Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    
    table.add_row("Collection", collection_name)
    table.add_row("Documents", str(len(docs)))
    table.add_row("Chunks", str(len(all_chunks)))
    table.add_row("Companies", str(len(companies)))
    table.add_row("Qdrant path", str(QDRANT_PATH))
    
    console.print(table)
    
    # Per-company table
    company_table = Table(title="Chunks by Company", show_header=True)
    company_table.add_column("Company", style="cyan")
    company_table.add_column("Chunks", justify="right")
    
    for company in sorted(companies):
        company_table.add_row(company, str(companies[company]))
    
    console.print(company_table)
