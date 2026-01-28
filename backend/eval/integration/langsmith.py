"""LangSmith latency breakdown for eval runs."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

AGENT_NODES = {"fetch", "answer", "followup"}


def get_latency_percentages(
    minutes_ago: int = 10,
    project_name: str | None = None,
) -> dict[str, float]:
    """Get latency percentages by agent node for display in eval summary.

    Fetches recent runs from LangSmith, computes average latency per agent node,
    and returns each as a proportion of total agent latency.

    Returns:
        Dict with keys: fetch, answer, followup. Values are proportions (0.0 to 1.0).
    """
    try:
        from langsmith import Client
    except ImportError:
        logger.warning("langsmith not installed. Run: pip install langsmith")
        return {}

    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        logger.warning("LANGCHAIN_API_KEY not set - skipping latency breakdown")
        return {}

    project = project_name or os.getenv("LANGCHAIN_PROJECT", "default")
    client = Client()

    start_time = datetime.now(UTC) - timedelta(minutes=minutes_ago)

    try:
        runs = list(client.list_runs(
            project_name=project,
            start_time=start_time,
            is_root=False,
            limit=1000,
        ))
    except Exception as e:
        logger.warning(f"Could not fetch LangSmith runs: {e}")
        return {}

    if not runs:
        logger.info(f"No runs found in last {minutes_ago} minutes")
        return {}

    # Aggregate average latencies for agent nodes only
    node_latencies: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        if run.end_time and run.start_time and run.name in AGENT_NODES:
            latency_ms = (run.end_time - run.start_time).total_seconds() * 1000
            node_latencies[run.name].append(latency_ms)

    if not node_latencies:
        return {}

    node_avgs = {name: sum(lats) / len(lats) for name, lats in node_latencies.items()}
    total_avg = sum(node_avgs.values())

    if total_avg == 0:
        return {}

    return {node: node_avgs.get(node, 0) / total_avg for node in AGENT_NODES}
