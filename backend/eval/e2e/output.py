"""Output and display functions for E2E evaluation."""

from __future__ import annotations

from rich.table import Table

from backend.eval.base import console, format_percentage
from backend.eval.formatting import build_eval_table
from backend.eval.models import (
    SECURITY_CATEGORIES,
    SLO_ANSWER_CORRECTNESS,
    SLO_ANSWER_RELEVANCE,
    SLO_CONTEXT_PRECISION,
    SLO_FAITHFULNESS,
    SLO_LATENCY_P95_MS,
    SLO_ROUTER_ACCURACY,
    SLO_SECURITY_PASS_RATE,
    E2EEvalResult,
    E2EEvalSummary,
)


def print_e2e_eval_results(
    results: list[E2EEvalResult],
    summary: E2EEvalSummary,
) -> None:
    """Print end-to-end evaluation results."""
    # Compute SLO pass/fail
    intent_slo_pass = summary.intent_accuracy >= SLO_ROUTER_ACCURACY
    relevance_slo_pass = summary.answer_relevance_rate >= SLO_ANSWER_RELEVANCE
    faithfulness_slo_pass = summary.faithfulness_rate >= SLO_FAITHFULNESS
    ctx_precision_slo_pass = summary.context_precision_rate >= SLO_CONTEXT_PRECISION
    answer_correctness_slo_pass = summary.answer_correctness_rate >= SLO_ANSWER_CORRECTNESS
    security_slo_pass = summary.security_pass_rate >= SLO_SECURITY_PASS_RATE

    # Build table sections: (section_name, [(label, value, slo_target, slo_passed)])
    sections: list[tuple[str, list[tuple[str, str, str | None, bool | None]]]] = [
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
            f"RAG Quality ({summary.rag_tests_total} tests)",
            [
                (
                    "  Relevance",
                    format_percentage(summary.answer_relevance_rate),
                    f">={format_percentage(SLO_ANSWER_RELEVANCE)}",
                    relevance_slo_pass,
                ),
                (
                    "  Faithfulness",
                    format_percentage(summary.faithfulness_rate),
                    f">={format_percentage(SLO_FAITHFULNESS)}",
                    faithfulness_slo_pass,
                ),
                (
                    "  Context Precision",
                    format_percentage(summary.context_precision_rate),
                    f">={format_percentage(SLO_CONTEXT_PRECISION)}",
                    ctx_precision_slo_pass,
                ),
                (
                    "  Answer Correctness",
                    format_percentage(summary.answer_correctness_rate),
                    f">={format_percentage(SLO_ANSWER_CORRECTNESS)}",
                    answer_correctness_slo_pass,
                ),
            ],
        ),
        (
            f"Security Tests ({summary.security_tests_total} tests)",
            [
                (
                    "  Pass Rate",
                    f"{summary.security_tests_passed}/{summary.security_tests_total}",
                    "100%",
                    security_slo_pass,
                ),
            ],
        ),
    ]

    table = build_eval_table("E2E Evaluation Summary", sections)
    console.print(table)

    # By-category breakdown
    console.print()
    cat_table = Table(title="Results by Category", show_header=True)
    cat_table.add_column("Category")
    cat_table.add_column("Tests", justify="right")
    cat_table.add_column("Passed", justify="right")
    cat_table.add_column("Relev", justify="right")
    cat_table.add_column("Faith", justify="right")
    cat_table.add_column("CtxP", justify="right")
    cat_table.add_column("AnsC", justify="right")

    for cat, stats in sorted(summary.by_category.items()):
        if cat in SECURITY_CATEGORIES:
            # Security tests: show pass/fail count, not RAGAS metrics
            cat_table.add_row(
                cat,
                str(stats["count"]),
                f"{stats['passed']}/{stats['count']}",
                "-",
                "-",
                "-",
                "-",
            )
        else:
            # RAG tests: show RAGAS metrics
            cat_table.add_row(
                cat,
                str(stats["count"]),
                f"{stats['passed']}/{stats['count']}",
                format_percentage(stats["relevance_rate"]),
                format_percentage(stats["faithfulness_rate"]),
                format_percentage(stats.get("context_precision_rate", 0)),
                format_percentage(stats.get("answer_correctness_rate", 0)),
            )

    console.print(cat_table)

    # Issue Details
    _print_issues(results)


def _print_issues(results: list[E2EEvalResult]) -> None:
    """Print details of issues found during evaluation."""
    company_issues = [r for r in results if not r.company_correct]
    intent_issues = [r for r in results if not r.intent_correct]
    # Flag issues where RAGAS scores are below threshold (0.5)
    quality_issues = [
        r
        for r in results
        if r.answer_relevance < 0.5
        or r.faithfulness < 0.5
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
                f"    Relev: {r.answer_relevance:.2f}, Faith: {r.faithfulness:.2f}, "
                f"CtxP: {r.context_precision:.2f}, AnsC: {r.answer_correctness:.2f}"
            )
            if r.judge_explanation:
                console.print(f"    Judge: {r.judge_explanation[:100]}...")
            if r.error:
                console.print(f"    [red]Error: {r.error}[/red]")
