"""
Parallel evaluation utilities.
"""


def calculate_p95_latency(latencies: list[float | int]) -> float:
    """
    Calculate P95 latency from a list of latencies.

    Args:
        latencies: List of latency values in milliseconds

    Returns:
        P95 latency value, or 0.0 if list is empty
    """
    if not latencies:
        return 0.0

    sorted_latencies = sorted(latencies)
    p95_index = int(len(sorted_latencies) * 0.95)
    return float(sorted_latencies[min(p95_index, len(sorted_latencies) - 1)])


__all__ = [
    "calculate_p95_latency",
]
