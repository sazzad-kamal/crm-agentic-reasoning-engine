"""Evaluation runner - tests conversation paths."""

from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Any, cast

from dotenv import load_dotenv

load_dotenv()

import typer

from backend.eval.answer.action.judge import judge_suggested_action
from backend.eval.answer.text.ragas import evaluate_single
from backend.eval.integration.models import (
    SLO_CONVO_STEP_PASS_RATE,
    ConvoEvalResults,
    ConvoStepResult,
)
from backend.eval.integration.tree import get_all_paths, get_expected_action, get_expected_answer, get_tree_stats

logger = logging.getLogger(__name__)


def _invoke_agent(question: str, session_id: str | None = None) -> dict[str, Any]:
    """Invoke the agent graph and return result."""
    from backend.agent.graph import agent_graph, build_thread_config
    from backend.agent.state import AgentState

    state: AgentState = {"question": question}
    config = build_thread_config(session_id)
    return cast(dict[str, Any], agent_graph.invoke(state, config=config))


def _evaluate_ragas(
    question: str, answer: str, sql_results: dict,
) -> dict[str, Any]:
    """Run RAGAS evaluation. Returns ConvoStepResult fields."""
    contexts = [json.dumps(sql_results, default=str)]
    expected = get_expected_answer(question)
    try:
        ragas = evaluate_single(question, answer, contexts, expected or "")
        nan_metrics = cast(list[str], ragas.get("nan_metrics", []))
        return {
            "relevance_score": cast(float, ragas["answer_relevancy"]),
            "answer_correctness_score": cast(float, ragas["answer_correctness"]),
            "ragas_metrics_total": 2,
            "ragas_metrics_failed": len(nan_metrics),
        }
    except Exception as e:
        logger.warning(f"RAGAS failed: {e}")
        return {}


def _evaluate_action(
    question: str, answer: str, suggested_action: str | None,
) -> dict[str, Any]:
    """Evaluate action quality. Returns ConvoStepResult fields."""
    expected_action = get_expected_action(question)
    action_rel = action_act = action_app = 0.0
    action_passed = True

    if expected_action is True and suggested_action is None:
        action_passed = False
    elif expected_action is False and suggested_action is not None:
        action_passed = False
    elif suggested_action:
        try:
            action_passed, action_rel, action_act, action_app, _ = judge_suggested_action(
                question, answer, suggested_action,
            )
        except Exception as e:
            logger.warning(f"Action judge failed: {e}")
            action_passed = False

    return {
        "expected_action": expected_action,
        "suggested_action": suggested_action,
        "action_relevance": action_rel,
        "action_actionability": action_act,
        "action_appropriateness": action_app,
        "action_passed": action_passed,
    }


def test_single_question(question: str, session_id: str) -> ConvoStepResult:
    """Test a single question and return answer with metrics."""
    try:
        result = _invoke_agent(question=question, session_id=session_id)
        answer = result.get("answer", "")

        kwargs: dict[str, Any] = {"question": question, "answer": answer}

        sql_results = result.get("sql_results", {})
        if answer and sql_results:
            kwargs.update(_evaluate_ragas(question, answer, sql_results))

        suggested_action = result.get("suggested_action")
        kwargs.update(_evaluate_action(question, answer, suggested_action))

        return ConvoStepResult(**kwargs)  # type: ignore[arg-type]

    except Exception as e:
        logger.error(f"Error testing question '{question}': {e}")
        return ConvoStepResult(question=question, answer="", errors=[str(e)])


def run_convo_eval(max_paths: int | None = None) -> ConvoEvalResults:
    """Run conversation evaluation on all paths."""
    all_paths = get_all_paths()
    paths_to_test = all_paths[:max_paths] if max_paths is not None else all_paths

    total_questions = sum(len(p) for p in paths_to_test)
    results = ConvoEvalResults(total=total_questions)
    question_num = 0

    for path_idx, path in enumerate(paths_to_test):
        session_id = f"convo_eval_{path_idx}_{int(time.time())}"
        for question in path:
            question_num += 1
            step = test_single_question(question, session_id)
            results.cases.append(step)
            status = "PASS" if step.passed else "FAIL"
            print(f"  [{question_num}/{total_questions}] {status} {question[:50]}...")

    results.compute_aggregates()
    return results


def print_summary(results: ConvoEvalResults) -> None:
    """Print evaluation summary."""
    passed = results.pass_rate >= SLO_CONVO_STEP_PASS_RATE
    status = "PASS" if passed else "FAIL"

    print("\nConversation Evaluation Summary")
    print(f"Pass Rate: {results.pass_rate:.1%} (>={SLO_CONVO_STEP_PASS_RATE:.1%} SLO) {status}")
    print(f"Questions: {results.passed}/{results.total}")
    print(f"Avg Relevance: {results.avg_relevance:.2f}, Correctness: {results.avg_answer_correctness:.2f}")

    ragas_ok = results.ragas_metrics_total - results.ragas_metrics_failed
    print(f"RAGAS: {ragas_ok}/{results.ragas_metrics_total} ({results.ragas_success_rate:.1%})")

    if results.actions_judged > 0 or results.actions_missing > 0 or results.actions_spurious > 0:
        print(f"Actions: {results.actions_passed}/{results.actions_judged} judged passed, {results.actions_missing} missing, {results.actions_spurious} spurious")
        if results.actions_judged > 0:
            print(f"  Relevance: {results.avg_action_relevance:.2f}, Actionability: {results.avg_action_actionability:.2f}, Appropriateness: {results.avg_action_appropriateness:.2f}")

    # Failed questions
    failed = [c for c in results.cases if not c.passed]
    if failed:
        print(f"\nFailed Questions ({len(failed)})")
        for c in failed[:5]:
            print(f"  {c.question[:60]}...")
            if c.errors:
                print(f"    Error: {c.errors[0]}")


def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of paths to test"),
) -> None:
    """Run conversation evaluation."""
    logging.basicConfig(level=logging.WARNING)

    stats = get_tree_stats()
    print("\nQuestion Tree Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    try:
        results = run_convo_eval(max_paths=limit)
    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
        traceback.print_exc()
        return

    print_summary(results)


if __name__ == "__main__":
    typer.run(main)
