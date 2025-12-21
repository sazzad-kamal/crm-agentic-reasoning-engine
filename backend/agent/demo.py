#!/usr/bin/env python3
"""
CLI demo for the CRM Agent.

Usage:
    python -m backend.agent.demo "What's going on with Acme Manufacturing in the last 90 days?"
    
    # With mock LLM (no API key needed):
    MOCK_LLM=1 python -m backend.agent.demo "What's going on with Acme Manufacturing?"
"""

import sys
import json
import argparse

from backend.agent.orchestrator import answer_question


def format_raw_data_summary(raw_data: dict) -> str:
    """Format raw data counts for display."""
    lines = []
    
    companies = raw_data.get("companies", [])
    activities = raw_data.get("activities", [])
    opportunities = raw_data.get("opportunities", [])
    history = raw_data.get("history", [])
    renewals = raw_data.get("renewals", [])
    
    lines.append(f"  Companies: {len(companies)}")
    lines.append(f"  Activities: {len(activities)}")
    lines.append(f"  Opportunities: {len(opportunities)}")
    lines.append(f"  History: {len(history)}")
    lines.append(f"  Renewals: {len(renewals)}")
    
    pipeline = raw_data.get("pipeline_summary")
    if pipeline:
        lines.append(f"  Pipeline: {pipeline.get('total_count', 0)} deals, ${pipeline.get('total_value', 0):,.0f}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="CRM Agent CLI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backend.agent.demo "What's going on with Acme Manufacturing in the last 90 days?"
  python -m backend.agent.demo "Which accounts have upcoming renewals?"
  python -m backend.agent.demo "Show the pipeline for Beta Tech"
  
Use MOCK_LLM=1 environment variable to test without an API key.
"""
    )
    parser.add_argument("question", help="The question to ask the CRM agent")
    parser.add_argument("--mode", choices=["auto", "docs", "data", "data+docs"], 
                       default="auto", help="Mode for answering (default: auto)")
    parser.add_argument("--company", help="Pre-specify a company ID")
    parser.add_argument("--json", action="store_true", help="Output raw JSON response")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("CRM Agent Demo")
    print(f"{'='*60}")
    print(f"\nQuestion: {args.question}")
    
    if args.mode != "auto":
        print(f"Mode: {args.mode}")
    if args.company:
        print(f"Company: {args.company}")
    
    print(f"\n{'-'*60}")
    print("Processing...")
    print(f"{'-'*60}\n")
    
    # Call the agent
    result = answer_question(
        question=args.question,
        mode=args.mode,
        company_id=args.company,
    )
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return
    
    # Formatted output
    print(f"Mode Used: {result['meta']['mode_used']}")
    print(f"Latency: {result['meta']['latency_ms']}ms")
    
    if result['meta'].get('company_id'):
        print(f"Company ID: {result['meta']['company_id']}")
    
    print(f"\n{'='*60}")
    print("STEPS")
    print(f"{'='*60}")
    for step in result['steps']:
        status_icon = "✓" if step['status'] == "done" else "✗" if step['status'] == "error" else "○"
        print(f"  {status_icon} {step['label']}")
    
    print(f"\n{'='*60}")
    print("SOURCES")
    print(f"{'='*60}")
    if result['sources']:
        for src in result['sources']:
            print(f"  [{src['type']}] {src['label']}")
    else:
        print("  (No sources)")
    
    print(f"\n{'='*60}")
    print("RAW DATA SUMMARY")
    print(f"{'='*60}")
    print(format_raw_data_summary(result['raw_data']))
    
    if args.verbose and result['raw_data'].get('companies'):
        print(f"\n  Company Details:")
        for c in result['raw_data']['companies'][:1]:
            print(f"    Name: {c.get('name')}")
            print(f"    Status: {c.get('status')}")
            print(f"    Plan: {c.get('plan')}")
            print(f"    Renewal: {c.get('renewal_date')}")
    
    if args.verbose and result['raw_data'].get('opportunities'):
        print(f"\n  Open Opportunities:")
        for opp in result['raw_data']['opportunities'][:3]:
            print(f"    - {opp.get('name')}: {opp.get('stage')} (${opp.get('value', 0):,})")
    
    print(f"\n{'='*60}")
    print("ANSWER")
    print(f"{'='*60}")
    print(result['answer'])
    print()


if __name__ == "__main__":
    main()
