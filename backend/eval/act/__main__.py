"""Allow running as: python -m backend.eval.act"""

import argparse
import sys
from pathlib import Path

from backend.eval.act.runner import (
    SLO_AVG_FAITHFULNESS,
    SLO_AVG_LATENCY_MS,
    SLO_AVG_RELEVANCY,
    run_act_eval,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Act! demo evaluation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full answers")
    parser.add_argument("--question", "-q", type=int, help="Run single question by index (0-4)")

    # Multi-database evaluation options
    parser.add_argument("--multi-db", action="store_true", help="Run evaluation across all databases")
    parser.add_argument("--databases", "-d", type=str, help="Comma-separated list of databases (with --multi-db)")
    parser.add_argument("--questions", type=str, help="Comma-separated list of questions (with --multi-db)")
    parser.add_argument("--output", "-o", type=Path, help="Save results to JSON file")
    parser.add_argument("--csv", type=Path, help="Save results to CSV file")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS evaluation")

    args = parser.parse_args()

    if args.multi_db:
        # Run multi-database evaluation
        from backend.eval.act.multi_db_runner import (
            run_multi_db_eval,
            save_csv_results,
            save_json_results,
        )

        databases = args.databases.split(",") if args.databases else None
        questions = args.questions.split(",") if args.questions else None

        multi_summary = run_multi_db_eval(
            databases=databases,
            questions=questions,
            verbose=args.verbose,
            skip_ragas=args.skip_ragas,
        )

        if args.output:
            save_json_results(multi_summary, args.output)

        if args.csv:
            save_csv_results(multi_summary, args.csv)

        # Exit with appropriate code
        slo_passed = (
            multi_summary.overall_avg_faithfulness >= SLO_AVG_FAITHFULNESS
            and multi_summary.overall_avg_relevancy >= SLO_AVG_RELEVANCY
            and multi_summary.overall_action_pass_rate >= 0.8
            and multi_summary.overall_avg_latency_ms < SLO_AVG_LATENCY_MS
            and multi_summary.overall_pass_rate >= 0.8
        )
        sys.exit(0 if slo_passed else 1)
    else:
        # Run single-database evaluation
        single_summary = run_act_eval(verbose=args.verbose, question_index=args.question)

        # Exit with appropriate code
        slo_passed = (
            single_summary.all_passed
            and single_summary.avg_faithfulness >= SLO_AVG_FAITHFULNESS
            and single_summary.avg_relevancy >= SLO_AVG_RELEVANCY
            and single_summary.action_pass_rate == 1.0
            and single_summary.avg_latency_ms < SLO_AVG_LATENCY_MS
        )
        sys.exit(0 if slo_passed else 1)
