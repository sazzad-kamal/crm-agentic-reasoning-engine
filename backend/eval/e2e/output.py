"""Output and display functions for E2E evaluation."""

from __future__ import annotations

from rich.table import Table

from backend.eval.base import console, format_percentage, format_check_mark
from backend.eval.slo import print_slo_result
from backend.eval.formatting import build_eval_table
from backend.eval.models import (
    E2EEvalResult,
    E2EEvalSummary,
    SLO_ROUTER_ACCURACY,
    SLO_ANSWER_RELEVANCE,
    SLO_GROUNDEDNESS,
)


def print_e2e_eval_results(
    results: list[E2EEvalResult],
    summary: E2EEvalSummary,
) -> bool:
    """
    Print end-to-end evaluation results with comprehensive SLO status.

    Returns:
        True if all SLOs passed
    """
    # Compute SLO pass/fail
    intent_slo_pass = summary.intent_accuracy >= SLO_ROUTER_ACCURACY
    relevance_slo_pass = summary.answer_relevance_rate >= SLO_ANSWER_RELEVANCE
    groundedness_slo_pass = summary.groundedness_rate >= SLO_GROUNDEDNESS

    # Build table sections: (section_name, [(label, value, slo_target, slo_passed)])
    sections = [
        ("", [("Total Tests", str(summary.total_tests), None, None)]),
        (
            "Routing",
            [
                ("  Company Extraction", format_percentage(summary.company_extraction_accuracy), None, None),
                (
                    "  Intent Classification",
                    format_percentage(summary.intent_accuracy),
                    f">={format_percentage(SLO_ROUTER_ACCURACY)}",
                    intent_slo_pass,
                ),
            ],
        ),
        (
            "Answer Quality (RAGAS)",
            [
                (
                    "  Answer Relevance",
                    format_percentage(summary.answer_relevance_rate),
                    f">={format_percentage(SLO_ANSWER_RELEVANCE)}",
                    relevance_slo_pass,
                ),
                (
                    "  Groundedness",
                    format_percentage(summary.groundedness_rate),
                    f">={format_percentage(SLO_GROUNDEDNESS)}",
                    groundedness_slo_pass,
                ),
                ("  Context Relevance", format_percentage(summary.context_relevance_rate), None, None),
                ("  Faithfulness", format_percentage(summary.faithfulness_rate), None, None),
            ],
        ),
        (
            "Latency",
            [
                ("  Avg Latency", f"{summary.avg_latency_ms:.0f}ms", None, None),
                ("  P95 Latency", f"{summary.p95_latency_ms:.0f}ms", None, None),
            ],
        ),
    ]

    table = build_eval_table("E2E Evaluation Summary", sections)
    console.print(table)

    # SLO Summary Panel
    slo_checks = [
        (
            "Intent Classification",
            intent_slo_pass,
            format_percentage(summary.intent_accuracy),
            f">={format_percentage(SLO_ROUTER_ACCURACY)}",
        ),
        (
            "Answer Relevance",
            relevance_slo_pass,
            format_percentage(summary.answer_relevance_rate),
            f">={format_percentage(SLO_ANSWER_RELEVANCE)}",
        ),
        (
            "Groundedness",
            groundedness_slo_pass,
            format_percentage(summary.groundedness_rate),
            f">={format_percentage(SLO_GROUNDEDNESS)}",
        ),
    ]

    all_slos_passed = print_slo_result(slo_checks)

    # By-category breakdown
    console.print()
    cat_table = Table(title="Results by Category", show_header=True)
    cat_table.add_column("Category")
    cat_table.add_column("Tests", justify="right")
    cat_table.add_column("Relev", justify="right")
    cat_table.add_column("Ground", justify="right")
    cat_table.add_column("CtxRel", justify="right")
    cat_table.add_column("Faith", justify="right")

    for cat, stats in sorted(summary.by_category.items()):
        cat_table.add_row(
            cat,
            str(stats["count"]),
            format_percentage(stats["relevance_rate"]),
            format_percentage(stats["groundedness_rate"]),
            format_percentage(stats["context_relevance_rate"]),
            format_percentage(stats["faithfulness_rate"]),
        )

    console.print(cat_table)

    # Issue Details
    _print_issues(results)

    return all_slos_passed


def _print_issues(results: list[E2EEvalResult]) -> None:
    """Print details of issues found during evaluation."""
    company_issues = [r for r in results if not r.company_correct]
    intent_issues = [r for r in results if not r.intent_correct]
    quality_issues = [
        r
        for r in results
        if r.answer_relevance == 0
        or r.answer_grounded == 0
        or r.faithfulness == 0
        or r.has_forbidden_content
    ]

    if company_issues:
        console.print("\n[yellow bold]Company Extraction Errors:[/yellow bold]")
        for r in company_issues[:5]:
            console.print(
                f"  [{r.test_case_id}] expected={r.expected_company_id}, got={r.actual_company_id}"
            )

    if intent_issues:
        console.print("\n[yellow bold]Intent Classification Errors:[/yellow bold]")
        for r in intent_issues[:5]:
            console.print(
                f"  [{r.test_case_id}] expected={r.expected_intent}, got={r.actual_intent}"
            )

    if quality_issues:
        console.print("\n[yellow bold]Quality Issues:[/yellow bold]")
        for r in quality_issues[:5]:
            console.print(f"\n  [{r.test_case_id}] {r.question[:50]}...")
            console.print(
                f"    Relev: {r.answer_relevance}, Ground: {r.answer_grounded}, "
                f"CtxRel: {r.context_relevance}, Faith: {r.faithfulness}"
            )
            if r.judge_explanation:
                console.print(f"    Judge: {r.judge_explanation[:100]}...")
            if r.error:
                console.print(f"    [red]Error: {r.error}[/red]")
