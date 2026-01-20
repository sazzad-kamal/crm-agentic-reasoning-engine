"""CLI entry point for followup suggestion evaluation."""

import argparse

from backend.eval.followup.runner import print_summary, run_followup_eval


def main() -> None:
    """Run followup suggestion evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate follow-up suggestion quality"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed output for each case",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate",
    )
    parser.add_argument(
        "--no-tree",
        action="store_true",
        help="Disable hardcoded tree, use LLM only",
    )

    args = parser.parse_args()

    results = run_followup_eval(
        verbose=args.verbose,
        limit=args.limit,
        use_hardcoded_tree=not args.no_tree,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
