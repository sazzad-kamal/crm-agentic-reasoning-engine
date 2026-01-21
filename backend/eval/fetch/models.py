"""Models for fetch node evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Question(BaseModel):
    """A question from the evaluation set."""

    text: str
    difficulty: int


class CaseResult(BaseModel):
    """Result for a single test case."""

    question: str
    difficulty: int
    sql: str
    passed: bool
    row_count: int = 0
    errors: list[str] = Field(default_factory=list)

    # Latency tracking (milliseconds)
    sql_gen_latency_ms: float = 0.0
    sql_exec_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class EvalResults(BaseModel):
    """Aggregated evaluation results."""

    total: int = 0
    passed: int = 0
    sql_executed: int = 0
    sql_failed: int = 0
    cases: list[CaseResult] = Field(default_factory=list)

    # Latency stats (averages in milliseconds)
    avg_sql_gen_latency_ms: float = 0.0
    avg_sql_exec_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def sql_correctness(self) -> float:
        """Percentage of SQL queries that executed successfully and passed judge."""
        if not self.cases:
            return 0.0
        passed_sql = sum(1 for c in self.cases if c.passed and not c.errors)
        return passed_sql / len(self.cases)

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from individual case results."""
        if not self.cases:
            return

        # Latency averages
        total_sql_gen = sum(c.sql_gen_latency_ms for c in self.cases)
        total_sql_exec = sum(c.sql_exec_latency_ms for c in self.cases)
        total_latency = sum(c.total_latency_ms for c in self.cases)

        self.avg_sql_gen_latency_ms = total_sql_gen / len(self.cases)
        self.avg_sql_exec_latency_ms = total_sql_exec / len(self.cases)
        self.avg_total_latency_ms = total_latency / len(self.cases)


__all__ = ["Question", "CaseResult", "EvalResults"]
