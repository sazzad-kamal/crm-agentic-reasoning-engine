"""Route node evaluation - tests query planner in isolation."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from backend.agent.route.query_planner import get_slot_plan
from backend.agent.route.slot_query import SlotPlan

_DIR = Path(__file__).parent
console = Console()


@dataclass
class FieldResult:
    """Result for a single field comparison."""

    field: str
    expected: str
    actual: str
    match: bool


@dataclass
class CaseResult:
    """Result for a single test case."""

    question: str
    passed: bool
    field_results: list[FieldResult] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvalResults:
    """Aggregated evaluation results."""

    total: int = 0
    passed: int = 0
    cases: list[CaseResult] = field(default_factory=list)

    # Per-field accuracy
    query_count_correct: int = 0
    table_correct: int = 0
    filters_correct: int = 0
    order_by_correct: int = 0
    needs_rag_correct: int = 0


def _load_eval_cases() -> list[dict]:
    """Load test cases from eval_cases.json."""
    with open(_DIR / "eval_cases.json") as f:
        data: list[dict] = json.load(f)
        return data


def _compare_filters(expected: list[dict], actual: list[dict]) -> bool:
    """Compare filter lists (order-insensitive)."""
    if len(expected) != len(actual):
        return False

    def normalize(f: dict) -> tuple:
        value = f.get("value")
        if isinstance(value, list):
            value = tuple(sorted(value))
        return (f.get("field"), f.get("op"), value)

    expected_set = {normalize(f) for f in expected}
    actual_set = {normalize(f) for f in actual}
    return expected_set == actual_set


def _compare_queries(expected: list[dict], actual: list[dict]) -> tuple[bool, bool, bool]:
    """
    Compare query lists.

    Returns: (table_match, filters_match, order_by_match)
    """
    if len(expected) != len(actual):
        return False, False, False

    table_match = True
    filters_match = True
    order_by_match = True

    for exp, act in zip(expected, actual, strict=True):
        if exp.get("table") != act.get("table"):
            table_match = False

        exp_filters = exp.get("filters", [])
        act_filters = act.get("filters", [])
        if not _compare_filters(exp_filters, act_filters):
            filters_match = False

        if exp.get("order_by") != act.get("order_by"):
            order_by_match = False

    return table_match, filters_match, order_by_match


def _slot_plan_to_dict(plan: SlotPlan) -> dict:
    """Convert SlotPlan to dict for comparison."""
    return {
        "queries": [
            {
                "table": q.table,
                "filters": [{"field": f.field, "op": f.op, "value": f.value} for f in q.filters],
                "order_by": q.order_by,
            }
            for q in plan.queries
        ],
        "needs_rag": plan.needs_rag,
    }


def run_route_eval(limit: int | None = None, verbose: bool = False) -> EvalResults:
    """
    Run route node evaluation.

    Args:
        limit: Maximum number of test cases to run
        verbose: Print detailed output for each case

    Returns:
        EvalResults with per-case and aggregate metrics
    """
    cases = _load_eval_cases()
    if limit:
        cases = cases[:limit]

    results = EvalResults(total=len(cases))

    for i, case in enumerate(cases):
        question = case["question"]
        expected = case["output"]

        if verbose:
            console.print(f"\n[bold]Case {i + 1}/{len(cases)}:[/bold] {question}")

        try:
            actual_plan = get_slot_plan(question)
            actual = _slot_plan_to_dict(actual_plan)

            # Compare query count
            query_count_match = len(expected["queries"]) == len(actual["queries"])

            # Compare queries
            table_match, filters_match, order_by_match = _compare_queries(
                expected["queries"], actual["queries"]
            )

            # Compare needs_rag (tracked but not included in pass/fail)
            needs_rag_match = expected["needs_rag"] == actual["needs_rag"]

            # Overall pass (excludes needs_rag for now)
            passed = all(
                [query_count_match, table_match, filters_match, order_by_match]
            )

            # Update counters
            if query_count_match:
                results.query_count_correct += 1
            if table_match:
                results.table_correct += 1
            if filters_match:
                results.filters_correct += 1
            if order_by_match:
                results.order_by_correct += 1
            if needs_rag_match:
                results.needs_rag_correct += 1
            if passed:
                results.passed += 1

            # Build field results
            field_results = [
                FieldResult("query_count", str(len(expected["queries"])), str(len(actual["queries"])), query_count_match),
                FieldResult("table", _tables_str(expected), _tables_str(actual), table_match),
                FieldResult("filters", _filters_str(expected), _filters_str(actual), filters_match),
                FieldResult("order_by", _order_by_str(expected), _order_by_str(actual), order_by_match),
                FieldResult("needs_rag", str(expected["needs_rag"]), str(actual["needs_rag"]), needs_rag_match),
            ]

            case_result = CaseResult(question=question, passed=passed, field_results=field_results)

            if verbose:
                status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
                console.print(f"  {status}")
                if not passed:
                    for fr in field_results:
                        if not fr.match:
                            console.print(f"    [yellow]{fr.field}[/yellow]: expected={fr.expected}, actual={fr.actual}")

        except Exception as e:
            case_result = CaseResult(question=question, passed=False, error=str(e))
            if verbose:
                console.print(f"  [red]ERROR[/red]: {e}")

        results.cases.append(case_result)

    return results


def _tables_str(output: dict) -> str:
    """Format tables for display."""
    return ", ".join(q.get("table", "") for q in output.get("queries", []))


def _filters_str(output: dict) -> str:
    """Format filters for display."""
    parts = []
    for q in output.get("queries", []):
        filters = q.get("filters", [])
        if filters:
            parts.append(str(len(filters)))
        else:
            parts.append("0")
    return ", ".join(parts)


def _order_by_str(output: dict) -> str:
    """Format order_by for display."""
    parts = []
    for q in output.get("queries", []):
        ob = q.get("order_by")
        parts.append(ob if ob else "null")
    return ", ".join(parts)


def print_summary(results: EvalResults) -> None:
    """Print evaluation summary."""
    console.print("\n[bold]Route Evaluation Summary[/bold]")
    console.print(f"Total: {results.total}, Passed: {results.passed}, Failed: {results.total - results.passed}")
    console.print(f"Pass Rate: {results.passed / results.total * 100:.1f}%")

    # Per-field accuracy table
    table = Table(title="Per-Field Accuracy")
    table.add_column("Field", style="cyan")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right")

    fields = [
        ("Query Count", results.query_count_correct),
        ("Table", results.table_correct),
        ("Filters", results.filters_correct),
        ("Order By", results.order_by_correct),
        ("Needs RAG", results.needs_rag_correct),
    ]

    for name, correct in fields:
        pct = correct / results.total * 100 if results.total > 0 else 0
        table.add_row(name, str(correct), f"{pct:.1f}%")

    console.print(table)

    # Failed cases
    failed = [c for c in results.cases if not c.passed]
    if failed:
        console.print(f"\n[bold red]Failed Cases ({len(failed)}):[/bold red]")
        for c in failed[:10]:  # Show first 10
            console.print(f"  - {c.question}")
            if c.error:
                console.print(f"    [red]Error: {c.error}[/red]")
