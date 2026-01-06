"""Tests for graph-level features."""

import pytest


# MetaInfo schema was removed as part of YAGNI cleanup.
# Per-node latencies are tracked in AgentState (router_latency_ms, answer_latency_ms, etc.)
# and returned in the streaming response, not via MetaInfo.
