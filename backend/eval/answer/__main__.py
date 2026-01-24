"""CLI entry point for answer eval."""

import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[3] / ".env")

from backend.eval.answer.runner import print_summary, run_answer_eval


def main(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of questions"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run answer node evaluation."""
    logging.basicConfig(level=logging.WARNING)
    print_summary(run_answer_eval(limit=limit, verbose=verbose))


if __name__ == "__main__":
    typer.run(main)
