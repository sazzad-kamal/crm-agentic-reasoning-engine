"""Multi-database Act! demo evaluation runner.

Runs the 5 Gold Standard demo questions across all 6 Act! databases,
evaluating answer quality (RAGAS) and action quality (LLM judge).
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from backend.act_fetch import (  # noqa: E402
    AVAILABLE_DATABASES,
    DEMO_STARTERS,
    clear_api_cache,
    get_database,
    set_database,
)
from backend.eval.act.runner import (  # noqa: E402
    SLO_AVG_FAITHFULNESS,
    SLO_AVG_LATENCY_MS,
    SLO_AVG_RELEVANCY,
    SLO_FAITHFULNESS,
    SLO_RELEVANCY,
    QuestionConfig,
    QuestionResult,
    evaluate_question,
)

logger = logging.getLogger(__name__)


@dataclass
class DatabaseResult:
    """Results for one database across all questions."""

    database: str
    questions: list[QuestionResult] = field(default_factory=list)

    # Aggregated metrics
    pass_count: int = 0
    avg_faithfulness: float = 0.0
    avg_relevancy: float = 0.0
    avg_latency_ms: float = 0.0
    action_pass_rate: float = 0.0
    total_time_ms: int = 0

    # Error tracking
    connection_failed: bool = False
    connection_error: str | None = None

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from question results."""
        if not self.questions:
            return

        self.pass_count = sum(1 for q in self.questions if q.passed)

        faithfulness_scores = [q.faithfulness for q in self.questions if q.faithfulness > 0]
        relevancy_scores = [q.relevancy for q in self.questions if q.relevancy > 0]
        latencies = [q.total_latency_ms for q in self.questions]

        self.avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
        self.avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
        self.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
        self.action_pass_rate = sum(1 for q in self.questions if q.action_passed) / len(self.questions)


@dataclass
class MultiDbEvalSummary:
    """Complete evaluation results across all databases."""

    databases: list[DatabaseResult] = field(default_factory=list)

    # Overall metrics
    total_evaluations: int = 0
    total_passed: int = 0
    overall_pass_rate: float = 0.0
    overall_avg_faithfulness: float = 0.0
    overall_avg_relevancy: float = 0.0
    overall_avg_latency_ms: float = 0.0
    overall_action_pass_rate: float = 0.0

    # Per-question analysis
    question_pass_rates: dict[str, float] = field(default_factory=dict)

    # Timing
    total_time_ms: int = 0

    # Failures
    databases_failed: list[str] = field(default_factory=list)

    def compute_aggregates(self, questions: list[str]) -> None:
        """Compute overall metrics from database results."""
        all_questions: list[QuestionResult] = []
        for db in self.databases:
            if not db.connection_failed:
                all_questions.extend(db.questions)
            else:
                self.databases_failed.append(db.database)

        self.total_evaluations = len(all_questions)
        self.total_passed = sum(1 for q in all_questions if q.passed)
        self.overall_pass_rate = self.total_passed / self.total_evaluations if self.total_evaluations else 0

        faithfulness_scores = [q.faithfulness for q in all_questions if q.faithfulness > 0]
        relevancy_scores = [q.relevancy for q in all_questions if q.relevancy > 0]
        latencies = [q.total_latency_ms for q in all_questions]

        self.overall_avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
        self.overall_avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
        self.overall_avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
        self.overall_action_pass_rate = sum(1 for q in all_questions if q.action_passed) / len(all_questions) if all_questions else 0

        # Per-question pass rates across all databases
        for question in questions:
            question_results = [q for q in all_questions if q.question == question]
            if question_results:
                self.question_pass_rates[question] = sum(1 for q in question_results if q.passed) / len(question_results)


def run_multi_db_eval(
    databases: list[str] | None = None,
    questions: list[str] | None = None,
    verbose: bool = False,
    skip_ragas: bool = False,
) -> MultiDbEvalSummary:
    """Run evaluation across multiple databases.

    Args:
        databases: List of database names to evaluate (default: all 6)
        questions: List of question texts to evaluate (default: all 5 demo starters)
        verbose: Show full answers
        skip_ragas: Skip RAGAS evaluation for faster runs

    Returns:
        MultiDbEvalSummary with all results
    """
    databases = databases or AVAILABLE_DATABASES
    questions = questions or DEMO_STARTERS

    summary = MultiDbEvalSummary()
    total_start = time.time()

    # Header
    print("=" * 80)
    print("Act! Demo Multi-Database Evaluation")
    print("=" * 80)
    print(f"Databases: {', '.join(databases)}")
    print(f"Questions: {', '.join(questions)}")
    print(f"Total evaluations: {len(databases)} x {len(questions)} = {len(databases) * len(questions)}")
    print("=" * 80)

    original_db = get_database()

    for db_idx, db_name in enumerate(databases, 1):
        db_result = DatabaseResult(database=db_name)
        db_start = time.time()

        print(f"\n[{db_idx}/{len(databases)}] Database: {db_name}")
        print("-" * 70)

        # Switch database
        try:
            set_database(db_name)
            clear_api_cache()
        except Exception as e:
            db_result.connection_failed = True
            db_result.connection_error = str(e)
            print(f"  [X] Connection failed: {e}")
            summary.databases.append(db_result)
            continue

        # Evaluate each question
        for q_idx, question_text in enumerate(questions, 1):
            config = QuestionConfig(
                text=question_text,
                expected_action=True,
                expected_min_rows=1,
            )

            result = evaluate_question(config, verbose=verbose)
            db_result.questions.append(result)

            # Print result
            status = "PASS" if result.passed else "FAIL"
            if result.warnings and result.passed:
                status = "WARN"

            print(f"  [{q_idx}/{len(questions)}] {status} \"{question_text}\" ({result.total_latency_ms / 1000:.1f}s)")
            print(f"        Fetch: {result.fetch_rows} rows {'[OK]' if result.fetch_passed else '[X]'}  "
                  f"Faith: {result.faithfulness:.2f} {'[OK]' if result.faithfulness >= SLO_FAITHFULNESS else '[X]'}  "
                  f"Rel: {result.relevancy:.2f} {'[OK]' if result.relevancy >= SLO_RELEVANCY else '[X]'}  "
                  f"Action: {'PASS' if result.action_passed else 'FAIL'}")

            for error in result.errors:
                print(f"        [X] {error}")

        db_result.total_time_ms = int((time.time() - db_start) * 1000)
        db_result.compute_aggregates()

        # Database summary
        print(f"  Summary: {db_result.pass_count}/{len(questions)} passed | "
              f"Avg Faith: {db_result.avg_faithfulness:.2f} | "
              f"Avg Rel: {db_result.avg_relevancy:.2f} | "
              f"Time: {db_result.total_time_ms / 1000:.1f}s")

        summary.databases.append(db_result)

    # Restore original database
    try:
        set_database(original_db)
        clear_api_cache()
    except Exception:
        pass

    summary.total_time_ms = int((time.time() - total_start) * 1000)
    summary.compute_aggregates(questions)

    # Print overall summary
    _print_summary(summary, databases, questions)

    return summary


def _print_summary(summary: MultiDbEvalSummary, databases: list[str], questions: list[str]) -> None:
    """Print the overall evaluation summary."""
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    dbs_evaluated = len(databases) - len(summary.databases_failed)
    print(f"Databases Evaluated: {dbs_evaluated}/{len(databases)}", end="")
    if summary.databases_failed:
        print(f" (failed: {', '.join(summary.databases_failed)})")
    else:
        print()

    print(f"Total Evaluations: {summary.total_evaluations} ({len(questions)} questions x {dbs_evaluated} databases)")
    print(f"Pass Rate: {summary.total_passed}/{summary.total_evaluations} ({summary.overall_pass_rate * 100:.1f}%)")

    # Per-question pass rates
    print("\nPer-Question Pass Rate (across all DBs):")
    for question in questions:
        rate = summary.question_pass_rates.get(question, 0)
        dbs_tested = dbs_evaluated
        passed = int(rate * dbs_tested)
        print(f"  {question:25s} {passed}/{dbs_tested} ({rate * 100:.1f}%)")

    # Per-database pass rates
    print("\nPer-Database Pass Rate:")
    for db_result in summary.databases:
        if db_result.connection_failed:
            print(f"  {db_result.database:25s} FAILED ({db_result.connection_error})")
        else:
            rate = db_result.pass_count / len(questions) if questions else 0
            print(f"  {db_result.database:25s} {db_result.pass_count}/{len(questions)} ({rate * 100:.1f}%)")

    # Metrics vs SLOs
    print("\nMetrics:")
    faith_ok = summary.overall_avg_faithfulness >= SLO_AVG_FAITHFULNESS
    rel_ok = summary.overall_avg_relevancy >= SLO_AVG_RELEVANCY
    lat_ok = summary.overall_avg_latency_ms < SLO_AVG_LATENCY_MS
    action_ok = summary.overall_action_pass_rate >= 0.8

    print(f"  Avg Faithfulness: {summary.overall_avg_faithfulness:.2f} (>= {SLO_AVG_FAITHFULNESS} SLO) {'[OK]' if faith_ok else '[X]'}")
    print(f"  Avg Relevancy: {summary.overall_avg_relevancy:.2f} (>= {SLO_AVG_RELEVANCY} SLO) {'[OK]' if rel_ok else '[X]'}")
    print(f"  Avg Latency: {summary.overall_avg_latency_ms / 1000:.1f}s (< {SLO_AVG_LATENCY_MS / 1000}s SLO) {'[OK]' if lat_ok else '[X]'}")
    print(f"  Action Pass Rate: {summary.overall_action_pass_rate * 100:.1f}% (>= 80% SLO) {'[OK]' if action_ok else '[X]'}")

    print(f"\nTotal Time: {summary.total_time_ms / 1000 / 60:.1f}m")

    # Exit code determination
    slo_passed = faith_ok and rel_ok and lat_ok and action_ok and summary.overall_pass_rate >= 0.8
    print(f"\nEXIT CODE: {0 if slo_passed else 1} ({'all SLOs passed' if slo_passed else 'SLO failures'})")
    print("=" * 80)


def save_json_results(summary: MultiDbEvalSummary, path: Path) -> None:
    """Save results to JSON file."""
    databases_list: list[dict[str, object]] = []
    data = {
        "summary": {
            "total_evaluations": summary.total_evaluations,
            "total_passed": summary.total_passed,
            "overall_pass_rate": summary.overall_pass_rate,
            "overall_avg_faithfulness": summary.overall_avg_faithfulness,
            "overall_avg_relevancy": summary.overall_avg_relevancy,
            "overall_avg_latency_ms": summary.overall_avg_latency_ms,
            "overall_action_pass_rate": summary.overall_action_pass_rate,
            "total_time_ms": summary.total_time_ms,
            "databases_failed": summary.databases_failed,
        },
        "question_pass_rates": summary.question_pass_rates,
        "databases": databases_list,
    }

    for db_result in summary.databases:
        questions_list: list[dict[str, object]] = []
        db_data: dict[str, object] = {
            "database": db_result.database,
            "pass_count": db_result.pass_count,
            "avg_faithfulness": db_result.avg_faithfulness,
            "avg_relevancy": db_result.avg_relevancy,
            "avg_latency_ms": db_result.avg_latency_ms,
            "action_pass_rate": db_result.action_pass_rate,
            "total_time_ms": db_result.total_time_ms,
            "connection_failed": db_result.connection_failed,
            "connection_error": db_result.connection_error,
            "questions": questions_list,
        }

        for q in db_result.questions:
            questions_list.append({
                "question": q.question,
                "passed": q.passed,
                "fetch_rows": q.fetch_rows,
                "fetch_latency_ms": q.fetch_latency_ms,
                "faithfulness": q.faithfulness,
                "relevancy": q.relevancy,
                "answer_latency_ms": q.answer_latency_ms,
                "action_passed": q.action_passed,
                "action_relevance": q.action_relevance,
                "action_actionability": q.action_actionability,
                "action_appropriateness": q.action_appropriateness,
                "total_latency_ms": q.total_latency_ms,
                "errors": q.errors,
                "warnings": q.warnings,
            })

        databases_list.append(db_data)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {path}")


def save_csv_results(summary: MultiDbEvalSummary, path: Path) -> None:
    """Save results to CSV file."""
    fieldnames = [
        "database", "question", "passed", "fetch_rows", "fetch_latency_ms",
        "faithfulness", "relevancy", "action_passed", "action_relevance",
        "action_actionability", "action_appropriateness", "total_latency_ms", "errors",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for db_result in summary.databases:
            if db_result.connection_failed:
                writer.writerow({
                    "database": db_result.database,
                    "question": "",
                    "passed": False,
                    "errors": db_result.connection_error,
                })
            else:
                for q in db_result.questions:
                    writer.writerow({
                        "database": db_result.database,
                        "question": q.question,
                        "passed": q.passed,
                        "fetch_rows": q.fetch_rows,
                        "fetch_latency_ms": q.fetch_latency_ms,
                        "faithfulness": q.faithfulness,
                        "relevancy": q.relevancy,
                        "action_passed": q.action_passed,
                        "action_relevance": q.action_relevance,
                        "action_actionability": q.action_actionability,
                        "action_appropriateness": q.action_appropriateness,
                        "total_latency_ms": q.total_latency_ms,
                        "errors": "; ".join(q.errors),
                    })

    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Act! demo evaluation across multiple databases")
    parser.add_argument("--databases", "-d", type=str, help="Comma-separated list of databases (default: all 6)")
    parser.add_argument("--questions", "-q", type=str, help="Comma-separated list of questions (default: all 5)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full answers")
    parser.add_argument("--output", "-o", type=Path, help="Save results to JSON file")
    parser.add_argument("--csv", type=Path, help="Save results to CSV file")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS evaluation")
    args = parser.parse_args()

    databases = args.databases.split(",") if args.databases else None
    questions = args.questions.split(",") if args.questions else None

    summary = run_multi_db_eval(
        databases=databases,
        questions=questions,
        verbose=args.verbose,
        skip_ragas=args.skip_ragas,
    )

    if args.output:
        save_json_results(summary, args.output)

    if args.csv:
        save_csv_results(summary, args.csv)

    # Exit with appropriate code
    slo_passed = (
        summary.overall_avg_faithfulness >= SLO_AVG_FAITHFULNESS
        and summary.overall_avg_relevancy >= SLO_AVG_RELEVANCY
        and summary.overall_action_pass_rate >= 0.8
        and summary.overall_avg_latency_ms < SLO_AVG_LATENCY_MS
        and summary.overall_pass_rate >= 0.8
    )
    sys.exit(0 if slo_passed else 1)
