"""Score captured answers using GPT-5.2-pro.

Evaluates each answer on:
- Usefulness: Would this help a sales manager?
- Accuracy: Does the answer match the data?
- Freshness: Is it about recent/relevant records?
- Actionability: Can you do something with this?

Also performs root cause analysis for poor scores.

Run with: python -m backend.eval.act.answer_scorer <capture_file.json>
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for Unicode output
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from backend.core.llm import create_openai_chain


class AnswerScore(BaseModel):
    """Scores for a single answer."""

    usefulness: int = Field(ge=0, le=5, description="Would this help a sales manager? 0=useless, 5=very helpful")
    accuracy: int = Field(ge=0, le=5, description="Does the answer match the fetched data? 0=wrong/hallucinated, 5=perfectly accurate")
    freshness: int = Field(ge=0, le=5, description="Is it about recent/relevant records? 0=stale/outdated, 5=current and timely")
    actionability: int = Field(ge=0, le=5, description="Can you do something concrete with this? 0=no action possible, 5=clear next steps")

    usefulness_reason: str = Field(description="Brief reason for usefulness score")
    accuracy_reason: str = Field(description="Brief reason for accuracy score")
    freshness_reason: str = Field(description="Brief reason for freshness score")
    actionability_reason: str = Field(description="Brief reason for actionability score")

    root_cause: str = Field(description="If total score < 12, what's the root cause? Options: 'wrong_question_for_data', 'bad_stale_data', 'fetch_logic_issue', 'answer_logic_issue', 'acceptable'")
    root_cause_explanation: str = Field(description="Detailed explanation of the root cause")
    recommended_fix: str = Field(description="What would fix this issue?")


_SYSTEM_PROMPT = """You are evaluating CRM assistant answers for a sales manager named Ken.

Score each dimension 0-5:
- **Usefulness**: Would Ken find this helpful for his daily work? 0=useless, 5=very valuable
- **Accuracy**: Does the answer correctly reflect the fetched data? Check for hallucinations or mismatches. 0=wrong, 5=accurate
- **Freshness**: Are the dates/records recent enough to be relevant? Dates from years ago score low. 0=stale, 5=current
- **Actionability**: Can Ken take concrete action based on this? 0=vague, 5=clear next steps

For root_cause, choose ONE of:
- **wrong_question_for_data**: The question asks for something the data doesn't support
- **bad_stale_data**: Data exists but is too old/empty to be useful
- **fetch_logic_issue**: Question is good, data could work, but we're fetching wrong things
- **answer_logic_issue**: Data is fine, but the LLM misinterpreted or hallucinated
- **acceptable**: Total score >= 12 and no major issues

Be critical. A sales manager needs actionable, current insights - not stale data dumps."""

_HUMAN_PROMPT = """## Question
"{question}"

## Fetched Data (from Act! CRM API)
```json
{fetched_data}
```

## Generated Answer
"{answer}"

## Generated Action
"{action}"

## Answer Guidance Used
"{answer_guidance}"

## Action Guidance Used
"{action_guidance}"

Evaluate this answer. Today's date is {today} - use this to judge freshness of dates in the data."""


@dataclass
class ScoredQuestion:
    """A question with its scores."""

    question: str
    answer: str
    action: str
    scores: dict = field(default_factory=dict)
    total_score: int = 0


@dataclass
class ScoringReport:
    """Full scoring report."""

    source_file: str
    database: str
    timestamp: str
    questions: list[ScoredQuestion] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def score_answer(
    question: str,
    fetched_data: dict,
    answer: str,
    action: str,
    answer_guidance: str,
    action_guidance: str,
) -> AnswerScore:
    """Score a single answer using GPT-5.2-pro."""

    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=1024,
        structured_output=AnswerScore,
        streaming=False,
        model="gpt-5.2-pro",
    )

    # Truncate data if too large
    data_str = json.dumps(fetched_data, default=str)
    if len(data_str) > 15000:
        data_str = data_str[:15000] + "\n... [truncated]"

    result: AnswerScore = chain.invoke({
        "question": question,
        "fetched_data": data_str,
        "answer": answer,
        "action": action,
        "answer_guidance": answer_guidance,
        "action_guidance": action_guidance,
        "today": datetime.utcnow().strftime("%Y-%m-%d"),
    })

    return result


def run_scoring(capture_path: Path) -> ScoringReport:
    """Score all questions from a capture file."""

    with open(capture_path, encoding="utf-8") as f:
        capture_data = json.load(f)

    report = ScoringReport(
        source_file=str(capture_path),
        database=capture_data["database"],
        timestamp=datetime.utcnow().isoformat(),
    )

    print("=" * 70)
    print(f"Answer Scoring - Database: {capture_data['database']}")
    print("Using: GPT-5.2-pro")
    print("=" * 70)

    total_scores = {"usefulness": 0, "accuracy": 0, "freshness": 0, "actionability": 0}
    root_causes: dict[str, int] = {}

    for i, q in enumerate(capture_data["questions"], 1):
        print(f"\n[{i}/5] {q['question']}")
        print("-" * 50)

        if q.get("fetch_error"):
            print(f"  SKIPPED: Fetch error - {q['fetch_error']}")
            continue

        scores = score_answer(
            question=q["question"],
            fetched_data=q["fetched_data"],
            answer=q["answer"],
            action=q["action"],
            answer_guidance=q.get("answer_guidance", ""),
            action_guidance=q.get("action_guidance", ""),
        )

        total = scores.usefulness + scores.accuracy + scores.freshness + scores.actionability

        scored = ScoredQuestion(
            question=q["question"],
            answer=q["answer"],
            action=q["action"],
            scores=scores.model_dump(),
            total_score=total,
        )
        report.questions.append(scored)

        # Accumulate totals
        total_scores["usefulness"] += scores.usefulness
        total_scores["accuracy"] += scores.accuracy
        total_scores["freshness"] += scores.freshness
        total_scores["actionability"] += scores.actionability
        root_causes[scores.root_cause] = root_causes.get(scores.root_cause, 0) + 1

        # Print scores
        status = "PASS" if total >= 12 else "FAIL"
        print(f"  [{status}] Total: {total}/20")
        print(f"    Usefulness:    {scores.usefulness}/5 - {scores.usefulness_reason}")
        print(f"    Accuracy:      {scores.accuracy}/5 - {scores.accuracy_reason}")
        print(f"    Freshness:     {scores.freshness}/5 - {scores.freshness_reason}")
        print(f"    Actionability: {scores.actionability}/5 - {scores.actionability_reason}")
        print(f"    Root Cause: {scores.root_cause}")
        print(f"    Fix: {scores.recommended_fix}")

    # Summary
    n = len(report.questions) or 1
    report.summary = {
        "avg_usefulness": total_scores["usefulness"] / n,
        "avg_accuracy": total_scores["accuracy"] / n,
        "avg_freshness": total_scores["freshness"] / n,
        "avg_actionability": total_scores["actionability"] / n,
        "avg_total": sum(total_scores.values()) / n,
        "root_causes": root_causes,
        "pass_count": sum(1 for q in report.questions if q.total_score >= 12),
        "fail_count": sum(1 for q in report.questions if q.total_score < 12),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Avg Usefulness:    {report.summary['avg_usefulness']:.1f}/5")
    print(f"  Avg Accuracy:      {report.summary['avg_accuracy']:.1f}/5")
    print(f"  Avg Freshness:     {report.summary['avg_freshness']:.1f}/5")
    print(f"  Avg Actionability: {report.summary['avg_actionability']:.1f}/5")
    print(f"  Avg Total:         {report.summary['avg_total']:.1f}/20")
    print(f"  Pass/Fail:         {report.summary['pass_count']}/{report.summary['fail_count']}")
    print("\nRoot Causes:")
    for cause, count in root_causes.items():
        print(f"    {cause}: {count}")

    return report


def save_report(report: ScoringReport, output_path: Path) -> None:
    """Save scoring report to JSON."""
    data = {
        "source_file": report.source_file,
        "database": report.database,
        "timestamp": report.timestamp,
        "summary": report.summary,
        "questions": [asdict(q) for q in report.questions],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nReport saved to: {output_path}")


def main() -> None:
    """Run scoring on a capture file."""
    import argparse

    parser = argparse.ArgumentParser(description="Score captured answers using GPT-5.2-pro")
    parser.add_argument("capture_file", type=Path, help="Path to capture JSON file")
    parser.add_argument("--output", "-o", type=Path, help="Output path (default: auto-generated)")
    args = parser.parse_args()

    if not args.capture_file.exists():
        print(f"Error: File not found: {args.capture_file}")
        return

    report = run_scoring(args.capture_file)

    # Save report
    if args.output:
        output_path = args.output
    else:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"scores_{report.database}_{timestamp}.json"

    save_report(report, output_path)


if __name__ == "__main__":
    main()
