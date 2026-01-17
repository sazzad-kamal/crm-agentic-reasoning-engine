"""Models for fetch node evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Question:
    """A question from the evaluation set."""

    text: str
    difficulty: int
    rag_only: bool = False


@dataclass
class CaseResult:
    """Result for a single test case."""

    question: str
    difficulty: int
    rag_only: bool
    sql: str
    passed: bool
    row_count: int = 0
    errors: list[str] = field(default_factory=list)

    # Latency tracking (milliseconds)
    sql_gen_latency_ms: float = 0.0
    sql_exec_latency_ms: float = 0.0
    rag_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # RAG metrics (optional, only when RAG is invoked)
    rag_precision: float | None = None
    rag_recall: float | None = None


@dataclass
class EvalResults:
    """Aggregated evaluation results."""

    total: int = 0
    passed: int = 0
    sql_executed: int = 0
    sql_failed: int = 0
    rag_invoked: int = 0
    cases: list[CaseResult] = field(default_factory=list)

    # Latency stats (averages in milliseconds)
    avg_sql_gen_latency_ms: float = 0.0
    avg_sql_exec_latency_ms: float = 0.0
    avg_rag_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0

    # RAG aggregate metrics
    avg_rag_precision: float = 0.0
    avg_rag_recall: float = 0.0

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def sql_correctness(self) -> float:
        """Percentage of SQL queries that executed successfully and passed judge."""
        sql_questions = [c for c in self.cases if not c.rag_only]
        if not sql_questions:
            return 0.0
        passed_sql = sum(1 for c in sql_questions if c.passed and not c.errors)
        return passed_sql / len(sql_questions)

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from individual case results."""
        if not self.cases:
            return

        # Latency averages
        total_sql_gen = sum(c.sql_gen_latency_ms for c in self.cases)
        total_sql_exec = sum(c.sql_exec_latency_ms for c in self.cases)
        total_rag = sum(c.rag_latency_ms for c in self.cases if c.rag_latency_ms > 0)
        total_latency = sum(c.total_latency_ms for c in self.cases)

        self.avg_sql_gen_latency_ms = total_sql_gen / len(self.cases)
        self.avg_sql_exec_latency_ms = total_sql_exec / len(self.cases)
        self.avg_total_latency_ms = total_latency / len(self.cases)

        # RAG latency average (only for cases where RAG was invoked)
        rag_cases = [c for c in self.cases if c.rag_latency_ms > 0]
        if rag_cases:
            self.avg_rag_latency_ms = total_rag / len(rag_cases)
            self.rag_invoked = len(rag_cases)

        # RAG precision/recall averages (only for cases with metrics)
        rag_precision_cases = [c for c in self.cases if c.rag_precision is not None]
        rag_recall_cases = [c for c in self.cases if c.rag_recall is not None]

        if rag_precision_cases:
            precisions = [c.rag_precision for c in rag_precision_cases if c.rag_precision is not None]
            self.avg_rag_precision = sum(precisions) / len(precisions) if precisions else 0.0
        if rag_recall_cases:
            recalls = [c.rag_recall for c in rag_recall_cases if c.rag_recall is not None]
            self.avg_rag_recall = sum(recalls) / len(recalls) if recalls else 0.0


__all__ = ["Question", "CaseResult", "EvalResults"]
