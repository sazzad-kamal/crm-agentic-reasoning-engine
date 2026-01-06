#!/usr/bin/env python3
"""
E2E latency profiler for agent queries.

Runs sample queries and breaks down latency by step.

Usage:
    python -m backend.agent.eval.e2e.profile
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Load .env from project root
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(project_root / ".env")

from backend.agent.nodes.graph import run_agent
from backend.agent.rag.tools import tool_docs_rag

# Configure logging to show timing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Warmup models (simulates server startup)
print("Warming up models...")
warmup_start = time.time()
tool_docs_rag("warmup", top_k=1)  # Trigger embedding model load
warmup_ms = int((time.time() - warmup_start) * 1000)
print(f"Models loaded in {warmup_ms}ms")
print()

# Check LangSmith status
langsmith_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
langsmith_key = os.environ.get("LANGCHAIN_API_KEY", "")
langsmith_project = os.environ.get("LANGCHAIN_PROJECT", "default")

print("=" * 70)
print("E2E LATENCY PROFILER")
print("=" * 70)
print()

if langsmith_enabled and langsmith_key:
    print(f"[OK] LangSmith ENABLED - Project: {langsmith_project}")
    print("     View traces at: https://smith.langchain.com/")
else:
    print("[!] LangSmith NOT enabled")
    print("    To enable, set in .env:")
    print("      LANGCHAIN_TRACING_V2=true")
    print("      LANGCHAIN_API_KEY=lsv2_pt_...")
    print("      LANGCHAIN_PROJECT=acme-crm-companion")

print()
print("-" * 70)

# Test queries representing different modes
TEST_QUERIES = [
    # Data-only query (company lookup)
    ("What's happening with Acme Manufacturing?", "data"),
    # Docs-only query (product question)
    ("How do I create a new opportunity?", "docs"),
    # Data+docs query (requires both)
    ("What renewals are coming up and how does the renewal process work?", "data+docs"),
]


def run_profiled_query(question: str, expected_mode: str):
    """Run a query and print latency breakdown."""
    print(f"\n>> Query: {question[:60]}...")
    print(f"   Expected mode: {expected_mode}")
    print()

    start = time.time()
    result = run_agent(question, session_id=f"profile-{int(time.time())}")
    total_ms = int((time.time() - start) * 1000)

    meta = result.get("meta", {})

    print(f"   Mode used: {meta.get('mode_used', 'unknown')}")
    print(f"   Company: {meta.get('company_id', 'None')}")
    print(f"   Sources: {len(result.get('sources', []))}")
    print()
    print(f"   TOTAL LATENCY: {total_ms}ms ({meta.get('latency_ms', 0)}ms reported)")
    print()

    # Show answer preview
    answer = result.get("answer", "")
    print(f"   Answer preview: {answer[:100]}...")

    return {
        "question": question,
        "mode": meta.get("mode_used"),
        "latency_ms": total_ms,
        "sources": len(result.get("sources", [])),
    }


def main():
    results = []

    for question, expected_mode in TEST_QUERIES:
        print()
        print("=" * 70)
        try:
            result = run_profiled_query(question, expected_mode)
            results.append(result)
        except Exception as e:
            print(f"   ERROR: {e}")
            results.append({"question": question, "error": str(e)})

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Query':<45} {'Mode':<12} {'Latency':<10} {'Sources'}")
    print("-" * 80)

    for r in results:
        if "error" in r:
            print(f"{r['question'][:43]:<45} ERROR")
        else:
            print(
                f"{r['question'][:43]:<45} {r['mode']:<12} {r['latency_ms']:>6}ms   {r['sources']}"
            )

    # Average latency
    valid = [r for r in results if "latency_ms" in r]
    if valid:
        avg = sum(r["latency_ms"] for r in valid) / len(valid)
        print()
        print(f"Average latency: {avg:.0f}ms")

    if langsmith_enabled:
        print()
        print("View detailed traces in LangSmith:")
        print(f"   https://smith.langchain.com/o/default/projects/p/{langsmith_project}")


if __name__ == "__main__":
    main()
