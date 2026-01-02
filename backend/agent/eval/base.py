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
    from backend.agent.rag_tools import (
        DOCS_COLLECTION,
        PRIVATE_COLLECTION,
        QDRANT_PATH,
        get_qdrant_client,
        ingest_docs,
        ingest_private_texts,
    )

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    qdrant = get_qdrant_client()

    docs_exists = (
        qdrant.collection_exists(DOCS_COLLECTION) and
        qdrant.get_collection(DOCS_COLLECTION).points_count > 0
    )
    private_exists = (
        qdrant.collection_exists(PRIVATE_COLLECTION) and
        qdrant.get_collection(PRIVATE_COLLECTION).points_count > 0
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


# Re-export all shared utilities
from backend.agent.eval.shared import (
    console,
    create_summary_table,
    create_detail_table,
    create_comparison_table,
    format_check_mark,
    format_percentage,
    format_latency,
    format_delta,
    print_eval_header,
    print_issues_panel,
    print_success_panel,
    add_separator_row,
    add_metric_row,
    save_results_json,
    load_results_json,
    compute_p95,
    compute_pass_rate,
    compare_to_baseline,
    save_baseline,
    print_baseline_comparison,
    REGRESSION_THRESHOLD,
)

__all__ = [
    "ensure_qdrant_collections",
    "console",
    "create_summary_table",
    "create_detail_table",
    "create_comparison_table",
    "format_check_mark",
    "format_percentage",
    "format_latency",
    "format_delta",
    "print_eval_header",
    "print_issues_panel",
    "print_success_panel",
    "add_separator_row",
    "add_metric_row",
    "save_results_json",
    "load_results_json",
    "compute_p95",
    "compute_pass_rate",
    "compare_to_baseline",
    "save_baseline",
    "print_baseline_comparison",
    "REGRESSION_THRESHOLD",
]
