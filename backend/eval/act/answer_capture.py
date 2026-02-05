"""Capture answers from the 5 Gold Standard questions without changes.

Runs each question against the current database and saves:
- Raw fetched data
- Generated answer
- Generated action

Run with: python -m backend.eval.act.answer_capture
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from backend.act_fetch import DEMO_PROMPTS, DEMO_STARTERS, act_fetch, get_database
from backend.agent.action.suggester import call_action_chain
from backend.agent.answer.answerer import call_answer_chain


@dataclass
class QuestionCapture:
    """Captured output for a single question."""

    question: str
    database: str
    timestamp: str

    # Raw data
    fetched_data: dict = field(default_factory=dict)
    fetch_error: str | None = None
    fetch_latency_ms: int = 0

    # Answer
    answer: str = ""
    answer_latency_ms: int = 0

    # Action
    action: str = ""
    action_latency_ms: int = 0

    # Prompts used
    answer_guidance: str = ""
    action_guidance: str = ""


@dataclass
class CaptureReport:
    """Full capture report for all questions."""

    database: str
    timestamp: str
    questions: list[QuestionCapture] = field(default_factory=list)
    total_time_ms: int = 0


def capture_question(question: str) -> QuestionCapture:
    """Capture all outputs for a single question."""
    capture = QuestionCapture(
        question=question,
        database=get_database(),
        timestamp=datetime.utcnow().isoformat(),
    )

    # Get prompts
    prompts = DEMO_PROMPTS.get(question, {})
    capture.answer_guidance = prompts.get("answer", "")
    capture.action_guidance = prompts.get("action", "")

    # Step 1: Fetch data
    fetch_start = time.time()
    try:
        result = act_fetch(question)
        capture.fetch_latency_ms = int((time.time() - fetch_start) * 1000)

        if result.get("error"):
            capture.fetch_error = result["error"]
            return capture

        capture.fetched_data = result.get("data", {})

    except Exception as e:
        capture.fetch_latency_ms = int((time.time() - fetch_start) * 1000)
        capture.fetch_error = str(e)
        return capture

    # Step 2: Generate answer
    answer_start = time.time()
    try:
        capture.answer = call_answer_chain(
            question=question,
            sql_results={"data": capture.fetched_data},
            guidance=capture.answer_guidance,
        )
        capture.answer_latency_ms = int((time.time() - answer_start) * 1000)
    except Exception as e:
        capture.answer_latency_ms = int((time.time() - answer_start) * 1000)
        capture.answer = f"[ERROR: {e}]"

    # Step 3: Generate action
    action_start = time.time()
    try:
        capture.action = call_action_chain(
            question=question,
            answer=capture.answer,
            guidance=capture.action_guidance,
        ) or ""
        capture.action_latency_ms = int((time.time() - action_start) * 1000)
    except Exception as e:
        capture.action_latency_ms = int((time.time() - action_start) * 1000)
        capture.action = f"[ERROR: {e}]"

    return capture


def run_capture() -> CaptureReport:
    """Run capture for all 5 questions."""
    report = CaptureReport(
        database=get_database(),
        timestamp=datetime.utcnow().isoformat(),
    )

    total_start = time.time()

    print("=" * 70)
    print(f"Answer Capture - Database: {get_database()}")
    print("=" * 70)

    for i, question in enumerate(DEMO_STARTERS, 1):
        print(f"\n[{i}/5] {question}")
        print("-" * 50)

        capture = capture_question(question)
        report.questions.append(capture)

        if capture.fetch_error:
            print(f"  FETCH ERROR: {capture.fetch_error}")
        else:
            data_keys = list(capture.fetched_data.keys())
            print(f"  Data: {data_keys} ({capture.fetch_latency_ms}ms)")
            print(f"  Answer: {capture.answer[:100]}..." if len(capture.answer) > 100 else f"  Answer: {capture.answer}")
            print(f"  Action: {capture.action[:100]}..." if len(capture.action) > 100 else f"  Action: {capture.action}")

    report.total_time_ms = int((time.time() - total_start) * 1000)

    print("\n" + "=" * 70)
    print(f"Total time: {report.total_time_ms / 1000:.1f}s")

    return report


def save_report(report: CaptureReport, output_path: Path) -> None:
    """Save report to JSON file."""
    # Convert to dict, handling nested dataclasses
    data = {
        "database": report.database,
        "timestamp": report.timestamp,
        "total_time_ms": report.total_time_ms,
        "questions": [asdict(q) for q in report.questions],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nReport saved to: {output_path}")


def main() -> None:
    """Run capture and save report."""
    report = run_capture()

    # Save to eval output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"capture_{report.database}_{timestamp}.json"

    save_report(report, output_path)


if __name__ == "__main__":
    main()
