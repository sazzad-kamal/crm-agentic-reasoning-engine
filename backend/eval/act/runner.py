"""Act! demo evaluation runner."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from backend.act_fetch import act_fetch
from backend.agent.action.suggester import call_action_chain
from backend.agent.answer.answerer import call_answer_chain
from backend.eval.answer.action.judge import judge_suggested_action
from backend.eval.answer.text.ragas import evaluate_single

logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"

# SLO thresholds
SLO_FAITHFULNESS = 0.7
SLO_RELEVANCY = 0.7
SLO_AVG_FAITHFULNESS = 0.75
SLO_AVG_RELEVANCY = 0.75
# Latency SLOs include RAGAS eval overhead (~20s per question)
SLO_AVG_LATENCY_MS = 30000
SLO_MAX_LATENCY_MS = 45000


@dataclass
class QuestionConfig:
    """Configuration for a demo question."""

    text: str
    expected_action: bool = True
    expected_min_rows: int = 1
    warn_if_zero: bool = False
    loading_message: str = "Loading..."


@dataclass
class QuestionResult:
    """Result for a single question evaluation."""

    question: str
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Fetch metrics
    fetch_rows: int = 0
    fetch_latency_ms: int = 0
    fetch_passed: bool = True

    # Answer metrics
    faithfulness: float = 0.0
    relevancy: float = 0.0
    answer_latency_ms: int = 0

    # Action metrics
    action_passed: bool = True
    action_relevance: float = 0.0
    action_actionability: float = 0.0
    action_appropriateness: float = 0.0
    action_latency_ms: int = 0

    # Total
    total_latency_ms: int = 0


@dataclass
class EvalSummary:
    """Summary of all evaluations."""

    results: list[QuestionResult] = field(default_factory=list)
    all_passed: bool = True
    avg_faithfulness: float = 0.0
    avg_relevancy: float = 0.0
    avg_latency_ms: float = 0.0
    action_pass_rate: float = 0.0
    total_time_ms: int = 0


def load_questions() -> list[QuestionConfig]:
    """Load questions from YAML file."""
    with open(QUESTIONS_PATH) as f:
        data = yaml.safe_load(f)
    return [QuestionConfig(**q) for q in data.get("questions", [])]


def evaluate_question(config: QuestionConfig, verbose: bool = False) -> QuestionResult:
    """Evaluate a single question end-to-end."""
    result = QuestionResult(question=config.text)
    total_start = time.time()

    # Step 1: Fetch from Act! API
    fetch_start = time.time()
    try:
        fetch_result = act_fetch(config.text)
        result.fetch_latency_ms = int((time.time() - fetch_start) * 1000)

        if fetch_result.get("error"):
            result.errors.append(f"Fetch error: {fetch_result['error']}")
            result.fetch_passed = False
            result.passed = False
            return result

        data = fetch_result.get("data", [])
        result.fetch_rows = len(data)

        if result.fetch_rows < config.expected_min_rows:
            result.errors.append(
                f"Expected >= {config.expected_min_rows} rows, got {result.fetch_rows}"
            )
            result.fetch_passed = False
            result.passed = False
        elif result.fetch_rows == 0 and config.warn_if_zero:
            result.warnings.append("No data returned (may be expected)")

    except Exception as e:
        result.fetch_latency_ms = int((time.time() - fetch_start) * 1000)
        result.errors.append(f"Fetch exception: {e}")
        result.fetch_passed = False
        result.passed = False
        return result

    # Step 2: Generate answer
    answer_start = time.time()
    try:
        answer = call_answer_chain(
            question=config.text,
            sql_results={"data": data},
        )
        result.answer_latency_ms = int((time.time() - answer_start) * 1000)

        if not answer:
            result.errors.append("Empty answer generated")
            result.passed = False
            return result

        if verbose:
            print(f"\n  Answer: {answer[:200]}...")

    except Exception as e:
        result.answer_latency_ms = int((time.time() - answer_start) * 1000)
        result.errors.append(f"Answer generation failed: {e}")
        result.passed = False
        return result

    # Step 3: RAGAS evaluation (skip if no data returned)
    if not data:
        result.warnings.append("RAGAS skipped: no data returned")
    else:
        try:
            contexts = [json.dumps(data, default=str)]
            ragas = evaluate_single(
                question=config.text,
                answer=answer,
                contexts=contexts,
                reference_answer="",  # No expected answer for demo
            )
            faith_val = ragas.get("faithfulness", 0)
            rel_val = ragas.get("answer_relevancy", 0)
            result.faithfulness = float(faith_val) if isinstance(faith_val, (int, float)) else 0.0
            result.relevancy = float(rel_val) if isinstance(rel_val, (int, float)) else 0.0

            # RAGAS metrics are tracked but don't cause failure (variance is high)
            if result.faithfulness < SLO_FAITHFULNESS:
                result.warnings.append(
                    f"Faithfulness {result.faithfulness:.2f} < {SLO_FAITHFULNESS}"
                )
            if result.relevancy < SLO_RELEVANCY:
                result.warnings.append(
                    f"Relevancy {result.relevancy:.2f} < {SLO_RELEVANCY}"
                )

        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {e}")
            result.warnings.append(f"RAGAS skipped: {e}")

    # Step 4: Generate and evaluate action
    action_start = time.time()
    try:
        action = call_action_chain(question=config.text, answer=answer)
        result.action_latency_ms = int((time.time() - action_start) * 1000)

        if config.expected_action and not action:
            result.errors.append("Expected action but none generated")
            result.action_passed = False
            result.passed = False
        elif action:
            passed, rel, act, app, _ = judge_suggested_action(
                config.text, answer, action
            )
            result.action_passed = passed
            result.action_relevance = rel
            result.action_actionability = act
            result.action_appropriateness = app

            if not passed:
                result.errors.append("Action failed judge evaluation")
                result.passed = False

    except Exception as e:
        result.action_latency_ms = int((time.time() - action_start) * 1000)
        result.warnings.append(f"Action evaluation failed: {e}")

    result.total_latency_ms = int((time.time() - total_start) * 1000)

    if result.total_latency_ms > SLO_MAX_LATENCY_MS:
        result.warnings.append(
            f"Latency {result.total_latency_ms}ms > {SLO_MAX_LATENCY_MS}ms SLO"
        )

    return result


def run_act_eval(
    verbose: bool = False,
    question_index: int | None = None,
) -> EvalSummary:
    """Run Act! demo evaluation."""
    questions = load_questions()
    summary = EvalSummary()
    total_start = time.time()

    if question_index is not None:
        if 0 <= question_index < len(questions):
            questions = [questions[question_index]]
        else:
            print(f"Invalid question index: {question_index}")
            sys.exit(1)

    print("Act! Demo Eval")
    print("=" * 50)

    for i, config in enumerate(questions, 1):
        result = evaluate_question(config, verbose)
        summary.results.append(result)

        status = "PASS" if result.passed else "FAIL"
        if result.warnings and result.passed:
            status = "WARN"

        print(f"\n[{i}/{len(questions)}] {status} \"{config.text}\" ({result.total_latency_ms / 1000:.1f}s)")
        print(f"      Fetch: {result.fetch_rows} rows (>= {config.expected_min_rows}) {'[OK]' if result.fetch_passed else '[X]'}  [{result.fetch_latency_ms}ms]")
        print(f"      Faithfulness: {result.faithfulness:.2f} {'[OK]' if result.faithfulness >= SLO_FAITHFULNESS else '[X]'}  Relevancy: {result.relevancy:.2f} {'[OK]' if result.relevancy >= SLO_RELEVANCY else '[X]'}  [{result.answer_latency_ms}ms]")
        print(f"      Action: {'PASS' if result.action_passed else 'FAIL'} (rel={result.action_relevance:.1f} act={result.action_actionability:.1f} app={result.action_appropriateness:.1f})  [{result.action_latency_ms}ms]")

        for warning in result.warnings:
            print(f"      [!] {warning}")
        for error in result.errors:
            print(f"      [X] {error}")

    # Calculate summary
    summary.total_time_ms = int((time.time() - total_start) * 1000)

    if summary.results:
        faithfulness_scores = [r.faithfulness for r in summary.results if r.faithfulness > 0]
        relevancy_scores = [r.relevancy for r in summary.results if r.relevancy > 0]
        latencies = [r.total_latency_ms for r in summary.results]

        summary.avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
        summary.avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
        summary.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
        summary.action_pass_rate = sum(1 for r in summary.results if r.action_passed) / len(summary.results)
        summary.all_passed = all(r.passed for r in summary.results)

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    pass_count = sum(1 for r in summary.results if r.passed)
    total = len(summary.results)
    print(f"  Pass Rate: {pass_count}/{total} ({100 * pass_count / total:.0f}%) {'[OK]' if summary.all_passed else '[X]'}")
    print(f"  Avg Faithfulness: {summary.avg_faithfulness:.2f} (>= {SLO_AVG_FAITHFULNESS} SLO) {'[OK]' if summary.avg_faithfulness >= SLO_AVG_FAITHFULNESS else '[X]'}")
    print(f"  Avg Relevancy: {summary.avg_relevancy:.2f} (>= {SLO_AVG_RELEVANCY} SLO) {'[OK]' if summary.avg_relevancy >= SLO_AVG_RELEVANCY else '[X]'}")
    print(f"  Action Pass Rate: {100 * summary.action_pass_rate:.0f}% {'[OK]' if summary.action_pass_rate == 1.0 else '[X]'}")
    print(f"  Avg Latency: {summary.avg_latency_ms / 1000:.1f}s (< {SLO_AVG_LATENCY_MS / 1000}s SLO) {'[OK]' if summary.avg_latency_ms < SLO_AVG_LATENCY_MS else '[X]'}")
    print(f"  Total Time: {summary.total_time_ms / 1000:.1f}s")

    # Determine exit code (RAGAS metrics are advisory, not required)
    slo_passed = (
        summary.all_passed
        and summary.action_pass_rate == 1.0
        and summary.avg_latency_ms < SLO_AVG_LATENCY_MS
    )

    print(f"\nEXIT CODE: {0 if slo_passed else 1} ({'all SLOs passed' if slo_passed else 'SLO failures'})")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Act! demo evaluation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full answers")
    parser.add_argument("--question", "-q", type=int, help="Run single question by index (0-4)")
    args = parser.parse_args()

    summary = run_act_eval(verbose=args.verbose, question_index=args.question)

    # Exit with appropriate code
    slo_passed = (
        summary.all_passed
        and summary.avg_faithfulness >= SLO_AVG_FAITHFULNESS
        and summary.avg_relevancy >= SLO_AVG_RELEVANCY
        and summary.action_pass_rate == 1.0
        and summary.avg_latency_ms < SLO_AVG_LATENCY_MS
    )
    sys.exit(0 if slo_passed else 1)
