"""
Base evaluation utilities for agent evaluation.

Re-exports shared utilities from backend.common.eval_base
for use by agent eval modules.
"""


def ensure_qdrant_collections() -> None:
    """
    Ensure Qdrant collections exist, ingesting data if needed.
    Shared by e2e_eval and flow_eval.
    """
    from backend.agent.rag.config import DOCS_COLLECTION, PRIVATE_COLLECTION, QDRANT_PATH
    from backend.agent.rag.client import get_qdrant_client
    from backend.agent.rag.ingest import ingest_docs, ingest_private_texts

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    qdrant = get_qdrant_client()

    docs_exists = (
        qdrant.collection_exists(DOCS_COLLECTION)
        and qdrant.get_collection(DOCS_COLLECTION).points_count > 0
    )
    private_exists = (
        qdrant.collection_exists(PRIVATE_COLLECTION)
        and qdrant.get_collection(PRIVATE_COLLECTION).points_count > 0
    )

    if docs_exists and private_exists:
        print("Qdrant collections ready.")
        return

    if not docs_exists:
        print("Ingesting docs into Qdrant...")
        chunk_count = ingest_docs()
        print(f"  Docs collection created with {chunk_count} chunks")

    if not private_exists:
        print("Ingesting private texts into Qdrant...")
        ingest_private_texts()
        print("  Private collection created")


# Re-export utilities from focused modules
from backend.eval.formatting import (
    console,
    create_summary_table,
    format_check_mark,
    format_percentage,
    print_eval_header,
)
from backend.eval.baseline import (
    compare_to_baseline,
    save_baseline,
    print_baseline_comparison,
    REGRESSION_THRESHOLD,
)

__all__ = [
    "ensure_qdrant_collections",
    "console",
    "create_summary_table",
    "format_check_mark",
    "format_percentage",
    "print_eval_header",
    "compare_to_baseline",
    "save_baseline",
    "print_baseline_comparison",
    "REGRESSION_THRESHOLD",
]
