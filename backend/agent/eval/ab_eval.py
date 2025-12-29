"""
A/B Evaluation for Agent Pipeline Configuration.

Runs the same evaluation questions with different agent configurations
to quantify the quality vs latency tradeoffs of:
- Follow-up Suggestions enabled/disabled
- Docs Integration enabled/disabled

Usage:
    python -m backend.agent.eval.ab_eval
    python -m backend.agent.eval.ab_eval --limit 5
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_project_root / ".env")

import typer
from rich.table import Table
from rich.progress import track

from backend.agent.config import reset_config
from backend.agent.eval.e2e_eval import E2E_TEST_CASES, judge_e2e_response
from backend.agent.eval.models import SLO_LATENCY_P95_MS
from backend.agent.eval.base import (
    console,
    format_check_mark,
    format_percentage,
    format_delta,
    print_eval_header,
)
app = typer.Typer(help="A/B Evaluation for agent configurations")


# =============================================================================
# Configuration Matrix
# =============================================================================

AGENT_AB_CONFIGS: dict[str, dict[str, Any]] = {
    # Full quality (all features on)
    "full_quality": {
        "AGENT_ENABLE_FOLLOW_UP_SUGGESTIONS": "true",
        "AGENT_ENABLE_DOCS_INTEGRATION": "true",
    },

    # No follow-up suggestions
    "no_followups": {
        "AGENT_ENABLE_FOLLOW_UP_SUGGESTIONS": "false",
        "AGENT_ENABLE_DOCS_INTEGRATION": "true",
    },

    # No docs integration (data-only mode)
    "no_docs": {
        "AGENT_ENABLE_FOLLOW_UP_SUGGESTIONS": "true",
        "AGENT_ENABLE_DOCS_INTEGRATION": "false",
    },

    # Minimal (no extras)
    "minimal": {
        "AGENT_ENABLE_FOLLOW_UP_SUGGESTIONS": "false",
        "AGENT_ENABLE_DOCS_INTEGRATION": "false",
    },
}


# =============================================================================
# A/B Evaluation
# =============================================================================

def apply_config(config: dict[str, str]) -> None:
    """Apply configuration by setting environment variables."""
    for key, value in config.items():
        os.environ[key] = value
    # Reset config to pick up new env vars
    reset_config()


def restore_config(original: dict[str, str | None]) -> None:
    """Restore original environment variables."""
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    reset_config()


def run_config_eval(
    config_name: str,
    config: dict[str, str],
    questions: list[dict],
) -> dict[str, Any]:
    """
    Run evaluation with a specific configuration.

    Returns:
        Dict with metrics for this configuration
    """
    # Save original env vars
    original_env = {
        key: os.environ.get(key)
        for key in config.keys()
    }

    try:
        apply_config(config)

        # Import after config is applied
        from backend.agent.orchestrator import answer_question

        results = []

        for q in questions:
            question = q["question"]
            expected_mode = q.get("expected_mode", "auto")

            start_time = time.time()

            try:
                result = answer_question(question)
                latency = (time.time() - start_time) * 1000

                answer = result.get("answer", "")
                sources = [s.get("id", "") for s in result.get("sources", [])]
                meta = result.get("meta", {})
                actual_mode = meta.get("mode_used", "unknown")

                # Judge the response
                judge_result = judge_e2e_response(question, answer, sources)

                results.append({
                    "latency_ms": latency,
                    "answer_relevance": judge_result["answer_relevance"],
                    "answer_grounded": judge_result["answer_grounded"],
                    "mode_correct": actual_mode == expected_mode or expected_mode == "auto",
                    "has_sources": len(sources) > 0,
                })

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                results.append({
                    "latency_ms": (time.time() - start_time) * 1000,
                    "answer_relevance": 0,
                    "answer_grounded": 0,
                    "mode_correct": False,
                    "has_sources": False,
                })

        # Aggregate metrics
        n = len(results)
        latencies = sorted([r["latency_ms"] for r in results])
        p95_idx = int(n * 0.95) if n > 0 else 0

        return {
            "config_name": config_name,
            "follow_ups": config.get("AGENT_ENABLE_FOLLOW_UP_SUGGESTIONS") == "true",
            "docs_integration": config.get("AGENT_ENABLE_DOCS_INTEGRATION") == "true",
            "num_questions": n,
            "avg_latency_ms": sum(r["latency_ms"] for r in results) / n if n else 0,
            "p95_latency_ms": latencies[min(p95_idx, n - 1)] if n else 0,
            "answer_relevance": sum(r["answer_relevance"] for r in results) / n if n else 0,
            "answer_grounded": sum(r["answer_grounded"] for r in results) / n if n else 0,
            "mode_accuracy": sum(1 for r in results if r["mode_correct"]) / n if n else 0,
        }

    finally:
        restore_config(original_env)


def run_agent_ab_evaluation(
    limit: int | None = None,
    configs: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Run A/B evaluation across agent configurations.

    Args:
        limit: Limit number of questions per config
        configs: Specific configs to run (default: all)

    Returns:
        List of results per configuration
    """
    print_eval_header(
        "[bold blue]Agent A/B Evaluation[/bold blue]",
        "Running A/B comparison across agent configurations",
    )

    # Use E2E test cases
    questions = E2E_TEST_CASES[:limit] if limit else E2E_TEST_CASES
    console.print(f"Questions per config: {len(questions)}")

    # Select configs
    configs_to_run = configs or list(AGENT_AB_CONFIGS.keys())

    all_results = []

    for config_name in track(configs_to_run, description="Running configs..."):
        config = AGENT_AB_CONFIGS[config_name]
        console.print(f"\n[cyan]Config: {config_name}[/cyan]")

        result = run_config_eval(config_name, config, questions)
        all_results.append(result)

        console.print(f"  Relevance: {result['answer_relevance']:.1%}, Latency: {result['avg_latency_ms']:.0f}ms")

    return all_results


def print_agent_ab_results(results: list[dict[str, Any]]) -> None:
    """Print agent A/B comparison results."""
    # Sort by relevance
    results_sorted = sorted(results, key=lambda x: x["answer_relevance"], reverse=True)

    # Main comparison table
    table = Table(title="Agent A/B Configuration Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Config", style="bold")
    table.add_column("Follow-ups", justify="center")
    table.add_column("Docs", justify="center")
    table.add_column("Latency", justify="right")
    table.add_column("Relevance", justify="right")
    table.add_column("Grounded", justify="right")

    for r in results_sorted:
        table.add_row(
            r["config_name"],
            format_check_mark(r["follow_ups"]),
            format_check_mark(r["docs_integration"]),
            f"{r['avg_latency_ms']:.0f}ms",
            format_percentage(r["answer_relevance"], (0.8, 0.6)),
            format_percentage(r["answer_grounded"], (0.8, 0.6)),
        )

    console.print(table)

    # Feature impact analysis
    console.print("\n[bold]Feature Impact Analysis[/bold]")

    baseline = next((r for r in results if r["config_name"] == "full_quality"), results[0])

    impact_table = Table(show_header=True, header_style="bold")
    impact_table.add_column("Feature Disabled")
    impact_table.add_column("Quality Impact", justify="right")
    impact_table.add_column("Latency Saved", justify="right")

    feature_map = {
        "heuristic_router": "LLM Router",
        "no_followups": "Follow-ups",
        "no_docs": "Docs Integration",
    }

    for config_name, feature in feature_map.items():
        r = next((r for r in results if r["config_name"] == config_name), None)
        if r:
            quality_delta = r["answer_relevance"] - baseline["answer_relevance"]
            latency_delta = baseline["avg_latency_ms"] - r["avg_latency_ms"]

            impact_table.add_row(
                feature,
                format_delta(quality_delta, is_positive_good=True),
                f"[green]{latency_delta:+.0f}ms[/green]" if latency_delta > 0 else f"{latency_delta:.0f}ms",
            )

    console.print(impact_table)


def save_agent_ab_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save A/B results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


# =============================================================================
# CLI
# =============================================================================

@app.command()
def run(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit questions per config"),
    output: Path = typer.Option(
        Path("backend/data/processed/agent_ab_eval_results.json"),
        "--output", "-o",
        help="Output file for results",
    ),
    configs: str | None = typer.Option(
        None,
        "--configs", "-c",
        help="Comma-separated configs to run (default: all)",
    ),
) -> None:
    """Run A/B evaluation across agent configurations."""
    config_list = configs.split(",") if configs else None

    results = run_agent_ab_evaluation(limit=limit, configs=config_list)
    print_agent_ab_results(results)
    save_agent_ab_results(results, output)


@app.command()
def quick() -> None:
    """Quick A/B test with 5 questions and 3 main configs."""
    results = run_agent_ab_evaluation(
        limit=5,
        configs=["full_quality", "heuristic_router", "fast"],
    )
    print_agent_ab_results(results)


if __name__ == "__main__":
    app()
