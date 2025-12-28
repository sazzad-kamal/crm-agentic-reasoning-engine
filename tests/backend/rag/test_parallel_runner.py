"""Tests for the parallel evaluation runner module."""

import threading
import time

import pytest

from backend.rag.eval.parallel_runner import run_parallel_evaluation


class TestRunParallelEvaluation:
    """Tests for run_parallel_evaluation function."""

    def test_returns_results_in_order(self):
        """Results should be returned in the same order as input items."""
        items = [
            {"id": "a", "value": 1},
            {"id": "b", "value": 2},
            {"id": "c", "value": 3},
        ]

        def evaluate_fn(item, lock):
            return item["value"] * 2

        results = run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=2,
            description="Test",
        )

        assert results == [2, 4, 6]

    def test_handles_empty_items(self):
        """Should handle empty input list gracefully."""
        def evaluate_fn(item, lock):
            return item["value"]

        results = run_parallel_evaluation(
            items=[],
            evaluate_fn=evaluate_fn,
            max_workers=2,
            description="Test",
        )

        assert results == []

    def test_single_item(self):
        """Should handle single item correctly."""
        items = [{"id": "only", "value": 42}]

        def evaluate_fn(item, lock):
            return item["value"]

        results = run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=2,
            description="Test",
        )

        assert results == [42]

    def test_custom_id_field(self):
        """Should support custom ID field name."""
        items = [
            {"question_id": "q1", "data": "a"},
            {"question_id": "q2", "data": "b"},
        ]

        def evaluate_fn(item, lock):
            return item["data"].upper()

        results = run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=2,
            description="Test",
            id_field="question_id",
        )

        assert results == ["A", "B"]

    def test_lock_is_passed_to_evaluate_fn(self):
        """Lock should be passed to evaluate function when use_lock=True."""
        items = [{"id": "test", "value": 1}]
        received_lock = [None]

        def evaluate_fn(item, lock):
            received_lock[0] = lock
            return item["value"]

        run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=1,
            description="Test",
            use_lock=True,
        )

        assert received_lock[0] is not None
        assert isinstance(received_lock[0], type(threading.Lock()))

    def test_no_lock_when_disabled(self):
        """Lock should be None when use_lock=False."""
        items = [{"id": "test", "value": 1}]
        received_lock = [None]

        def evaluate_fn(item, lock):
            received_lock[0] = lock
            return item["value"]

        run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=1,
            description="Test",
            use_lock=False,
        )

        assert received_lock[0] is None

    def test_parallel_execution(self):
        """Should execute items in parallel."""
        items = [
            {"id": "1", "delay": 0.05},
            {"id": "2", "delay": 0.05},
            {"id": "3", "delay": 0.05},
            {"id": "4", "delay": 0.05},
        ]

        def evaluate_fn(item, lock):
            time.sleep(item["delay"])
            return item["id"]

        start = time.time()
        results = run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=4,
            description="Test",
        )
        elapsed = time.time() - start

        # With 4 workers, 4 items of 0.05s each should complete in ~0.05s
        # Sequential would take ~0.2s
        assert elapsed < 0.15  # Allow some overhead
        assert len(results) == 4

    def test_handles_exceptions_gracefully(self):
        """Should handle exceptions and continue processing other items."""
        items = [
            {"id": "ok1", "fail": False},
            {"id": "fail", "fail": True},
            {"id": "ok2", "fail": False},
        ]

        def evaluate_fn(item, lock):
            if item["fail"]:
                raise ValueError("Test error")
            return item["id"]

        results = run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=2,
            description="Test",
        )

        # Failed item should be excluded from results
        assert len(results) == 2
        assert "ok1" in results
        assert "ok2" in results

    def test_preserves_complex_results(self):
        """Should preserve complex result types."""
        items = [{"id": "1"}, {"id": "2"}]

        def evaluate_fn(item, lock):
            return {"processed": item["id"], "nested": {"value": int(item["id"])}}

        results = run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=2,
            description="Test",
        )

        assert results[0] == {"processed": "1", "nested": {"value": 1}}
        assert results[1] == {"processed": "2", "nested": {"value": 2}}


class TestParallelRunnerEdgeCases:
    """Edge case tests for parallel runner."""

    def test_max_workers_one(self):
        """Should work with single worker (sequential execution)."""
        items = [{"id": str(i)} for i in range(5)]

        def evaluate_fn(item, lock):
            return int(item["id"])

        results = run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=1,
            description="Test",
        )

        assert results == [0, 1, 2, 3, 4]

    def test_more_workers_than_items(self):
        """Should work when workers exceed items."""
        items = [{"id": "1"}, {"id": "2"}]

        def evaluate_fn(item, lock):
            return item["id"]

        results = run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=10,
            description="Test",
        )

        assert results == ["1", "2"]

    def test_returns_none_values(self):
        """Should handle None as a valid result."""
        items = [{"id": "1"}, {"id": "2"}]

        def evaluate_fn(item, lock):
            return None

        results = run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=2,
            description="Test",
        )

        assert results == [None, None]

    def test_thread_safety_with_lock(self):
        """Lock should ensure thread-safe access to shared resource."""
        items = [{"id": str(i)} for i in range(10)]
        counter = {"value": 0}

        def evaluate_fn(item, lock):
            if lock:
                with lock:
                    current = counter["value"]
                    time.sleep(0.001)  # Simulate work
                    counter["value"] = current + 1
            return counter["value"]

        run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=4,
            description="Test",
            use_lock=True,
        )

        # With proper locking, counter should be exactly 10
        assert counter["value"] == 10
