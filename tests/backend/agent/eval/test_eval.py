"""
Tests for backend.agent.eval module.

Tests the agent evaluation models, tracking, and history functions.
"""

import os

import pytest

# Set mock mode
os.environ["MOCK_LLM"] = "1"

from backend.agent.eval.models import (
    ToolEvalResult,
    RouterEvalResult,
    E2EEvalResult,
    ToolEvalSummary,
    RouterEvalSummary,
    E2EEvalSummary,
    AgentEvalSummary,
    SLO_LATENCY_P95_MS,
    SLO_TOOL_ACCURACY,
    SLO_ROUTER_ACCURACY,
    SLO_ANSWER_RELEVANCE,
    SLO_GROUNDEDNESS,
)


# =============================================================================
# Model Tests
# =============================================================================

class TestToolEvalResult:
    """Tests for ToolEvalResult model."""

    def test_tool_eval_result_creation(self):
        """Test creating a ToolEvalResult."""
        result = ToolEvalResult(
            tool_name="tool_company_lookup",
            test_case_id="tc1",
            input_params={"company_id": "ACME-MFG"},
            expected_found=True,
            actual_found=True,
            expected_company_id="ACME-MFG",
            actual_company_id="ACME-MFG",
            data_correct=True,
            sources_present=True,
            latency_ms=50.0,
        )

        assert result.tool_name == "tool_company_lookup"
        assert result.expected_found is True
        assert result.actual_found is True
        assert result.data_correct is True

    def test_tool_eval_result_with_error(self):
        """Test ToolEvalResult with error."""
        result = ToolEvalResult(
            tool_name="tool_company_lookup",
            test_case_id="tc1",
            input_params={"company_id": "INVALID"},
            expected_found=True,
            actual_found=False,
            data_correct=False,
            sources_present=False,
            error="Company not found",
        )

        assert result.actual_found is False
        assert result.error == "Company not found"


class TestRouterEvalResult:
    """Tests for RouterEvalResult model."""

    def test_router_eval_result_creation(self):
        """Test creating a RouterEvalResult."""
        result = RouterEvalResult(
            test_case_id="r1",
            question="What is going on with Acme?",
            expected_mode="data",
            actual_mode="data",
            expected_company_id="ACME-MFG",
            actual_company_id="ACME-MFG",
            mode_correct=True,
            company_correct=True,
            intent_expected="company_status",
            intent_actual="company_status",
            intent_correct=True,
        )

        assert result.mode_correct is True
        assert result.company_correct is True
        assert result.intent_correct is True

    def test_router_eval_result_mode_mismatch(self):
        """Test RouterEvalResult with mode mismatch."""
        result = RouterEvalResult(
            test_case_id="r1",
            question="How do I import contacts?",
            expected_mode="docs",
            actual_mode="data",
            mode_correct=False,
            company_correct=True,
        )

        assert result.mode_correct is False


class TestE2EEvalResult:
    """Tests for E2EEvalResult model."""

    def test_e2e_eval_result_creation(self):
        """Test creating an E2EEvalResult."""
        result = E2EEvalResult(
            test_case_id="e2e1",
            question="What's the status of Acme Manufacturing?",
            category="company_status",
            expected_company_id="ACME-MFG",
            actual_company_id="ACME-MFG",
            company_correct=True,
            expected_intent="company_status",
            actual_intent="company_status",
            intent_correct=True,
            answer="Acme Manufacturing is doing well.",
            answer_relevance=1,
            answer_grounded=1,
            has_sources=True,
            latency_ms=200.0,
            total_tokens=500,
        )

        assert result.answer_relevance == 1
        assert result.answer_grounded == 1
        assert result.intent_correct is True

    def test_e2e_eval_result_with_judge_explanation(self):
        """Test E2EEvalResult with judge explanation."""
        result = E2EEvalResult(
            test_case_id="e2e1",
            question="Test question",
            category="test",
            expected_intent="general",
            actual_intent="general",
            intent_correct=True,
            answer="Test answer",
            answer_relevance=0,
            answer_grounded=0,
            has_sources=False,
            latency_ms=100.0,
            total_tokens=100,
            judge_explanation="Answer was not relevant to the question.",
        )

        assert result.judge_explanation == "Answer was not relevant to the question."


class TestToolEvalSummary:
    """Tests for ToolEvalSummary model."""

    def test_tool_eval_summary_creation(self):
        """Test creating a ToolEvalSummary."""
        summary = ToolEvalSummary(
            total_tests=20,
            passed=18,
            failed=2,
            accuracy=0.90,
            by_tool={
                "tool_company_lookup": {"passed": 10, "failed": 0, "accuracy": 1.0},
                "tool_activities": {"passed": 8, "failed": 2, "accuracy": 0.8},
            },
        )

        assert summary.total_tests == 20
        assert summary.accuracy == 0.90
        assert summary.by_tool["tool_company_lookup"]["accuracy"] == 1.0


class TestRouterEvalSummary:
    """Tests for RouterEvalSummary model."""

    def test_router_eval_summary_creation(self):
        """Test creating a RouterEvalSummary."""
        summary = RouterEvalSummary(
            total_tests=30,
            mode_accuracy=0.93,
            company_extraction_accuracy=0.97,
            intent_accuracy=0.90,
            by_mode={
                "data": {"expected": 15, "correct": 14, "accuracy": 0.93},
                "docs": {"expected": 15, "correct": 14, "accuracy": 0.93},
            },
        )

        assert summary.mode_accuracy == 0.93
        assert summary.intent_accuracy == 0.90


class TestE2EEvalSummary:
    """Tests for E2EEvalSummary model."""

    def test_e2e_eval_summary_creation(self):
        """Test creating an E2EEvalSummary."""
        summary = E2EEvalSummary(
            total_tests=25,
            company_extraction_accuracy=0.95,
            intent_accuracy=0.92,
            answer_relevance_rate=0.88,
            groundedness_rate=0.84,
            avg_latency_ms=350.0,
            p95_latency_ms=800.0,
            latency_slo_pass=True,
            by_category={
                "company_status": {"count": 10, "relevance": 0.9, "grounded": 0.8},
                "docs_query": {"count": 15, "relevance": 0.87, "grounded": 0.87},
            },
        )

        assert summary.answer_relevance_rate == 0.88
        assert summary.latency_slo_pass is True


class TestAgentEvalSummary:
    """Tests for AgentEvalSummary model."""

    def test_agent_eval_summary_creation(self):
        """Test creating an AgentEvalSummary."""
        e2e = E2EEvalSummary(
            total_tests=25,
            company_extraction_accuracy=0.95,
            intent_accuracy=0.92,
            answer_relevance_rate=0.88,
            groundedness_rate=0.84,
            avg_latency_ms=350.0,
            p95_latency_ms=800.0,
            by_category={},
        )

        summary = AgentEvalSummary(
            e2e_eval=e2e,
            overall_score=0.88,
            all_slos_passed=True,
        )

        assert summary.e2e_eval is not None
        assert summary.overall_score == 0.88
        assert summary.all_slos_passed is True


# =============================================================================
# SLO Constants Tests
# =============================================================================

class TestAgentSLOConstants:
    """Tests for agent SLO constant values."""

    def test_slo_latency(self):
        """Test SLO latency threshold."""
        assert SLO_LATENCY_P95_MS == 5000

    def test_slo_tool_accuracy(self):
        """Test SLO tool accuracy threshold."""
        assert SLO_TOOL_ACCURACY == 0.90

    def test_slo_router_accuracy(self):
        """Test SLO router accuracy threshold."""
        assert SLO_ROUTER_ACCURACY == 0.90

    def test_slo_answer_relevance(self):
        """Test SLO answer relevance threshold."""
        assert SLO_ANSWER_RELEVANCE == 0.80

    def test_slo_groundedness(self):
        """Test SLO groundedness threshold."""
        assert SLO_GROUNDEDNESS == 0.80


# =============================================================================
# Agent Tracking Module Tests
# =============================================================================

class TestAgentTrackingModule:
    """Tests for the agent tracking module functions."""

    def test_compare_e2e_with_previous_no_previous(self):
        """Test comparison when no previous run exists."""
        from backend.agent.eval.tracking import compare_e2e_with_previous

        current = E2EEvalSummary(
            total_tests=20,
            company_extraction_accuracy=0.95,
            intent_accuracy=0.90,
            answer_relevance_rate=0.85,
            groundedness_rate=0.80,
            avg_latency_ms=400.0,
            p95_latency_ms=1000.0,
            by_category={},
        )

        comparison = compare_e2e_with_previous(current, None)

        assert comparison["has_previous"] is False
        assert comparison["regressions"] == []
        assert comparison["improvements"] == []

    def test_compare_e2e_with_previous_detects_regression(self):
        """Test that comparison detects regressions."""
        from backend.agent.eval.tracking import compare_e2e_with_previous

        previous = E2EEvalSummary(
            total_tests=20,
            company_extraction_accuracy=0.95,
            intent_accuracy=0.95,
            answer_relevance_rate=0.90,
            groundedness_rate=0.90,
            avg_latency_ms=300.0,
            p95_latency_ms=800.0,
            by_category={},
        )

        current = E2EEvalSummary(
            total_tests=20,
            company_extraction_accuracy=0.95,
            intent_accuracy=0.95,
            answer_relevance_rate=0.70,  # Regression
            groundedness_rate=0.90,
            avg_latency_ms=300.0,
            p95_latency_ms=800.0,
            by_category={},
        )

        comparison = compare_e2e_with_previous(current, previous)

        assert comparison["has_previous"] is True
        assert len(comparison["regressions"]) >= 1

    def test_analyze_e2e_budget_violations_no_violations(self):
        """Test budget analysis with no violations."""
        from backend.agent.eval.tracking import analyze_e2e_budget_violations

        results = [
            E2EEvalResult(
                test_case_id="t1",
                question="Test",
                category="test",
                expected_intent="general",
                actual_intent="general",
                intent_correct=True,
                answer="Test",
                answer_relevance=1,
                answer_grounded=1,
                has_sources=True,
                latency_ms=1000.0,  # Well under budget
                total_tokens=100,
            )
        ]

        analysis = analyze_e2e_budget_violations(results)

        assert len(analysis["total_violations"]) == 0

    def test_analyze_e2e_budget_violations_with_violations(self):
        """Test budget analysis with violations."""
        from backend.agent.eval.tracking import analyze_e2e_budget_violations

        results = [
            E2EEvalResult(
                test_case_id="t1",
                question="Test",
                category="test",
                expected_intent="general",
                actual_intent="general",
                intent_correct=True,
                answer="Test",
                answer_relevance=1,
                answer_grounded=1,
                has_sources=True,
                latency_ms=6000.0,  # Over budget (5000ms)
                total_tokens=100,
            )
        ]

        analysis = analyze_e2e_budget_violations(results)

        assert len(analysis["total_violations"]) == 1
        assert analysis["total_violations"][0]["test_case_id"] == "t1"


# =============================================================================
# Agent History Module Tests
# =============================================================================

class TestAgentHistoryModule:
    """Tests for the agent history module functions."""

    def test_compute_agent_trends_insufficient_data(self):
        """Test trend computation with insufficient data."""
        from backend.agent.eval.history import compute_agent_trends

        history = [{"metrics": {"answer_relevance": 0.8}}]
        result = compute_agent_trends(history, "answer_relevance")

        assert result["has_trend"] is False

    def test_compute_agent_trends_with_data(self):
        """Test trend computation with sufficient data."""
        from backend.agent.eval.history import compute_agent_trends

        history = [
            {"metrics": {"answer_relevance": 0.70}},
            {"metrics": {"answer_relevance": 0.75}},
            {"metrics": {"answer_relevance": 0.85}},
        ]

        result = compute_agent_trends(history, "answer_relevance")

        assert result["has_trend"] is True
        assert result["min"] == 0.70
        assert result["max"] == 0.85
        assert result["current"] == 0.85

    def test_compute_agent_trends_direction(self):
        """Test trend direction computation."""
        from backend.agent.eval.history import compute_agent_trends

        # Upward trend
        history_up = [
            {"metrics": {"score": 0.60}},
            {"metrics": {"score": 0.85}},
        ]
        result = compute_agent_trends(history_up, "score")
        assert result["trend_direction"] == "up"

        # Downward trend
        history_down = [
            {"metrics": {"score": 0.85}},
            {"metrics": {"score": 0.60}},
        ]
        result = compute_agent_trends(history_down, "score")
        assert result["trend_direction"] == "down"

    def test_compute_agent_trends_stable(self):
        """Test trend direction stable when values are close."""
        from backend.agent.eval.history import compute_agent_trends

        history = [
            {"metrics": {"score": 0.80}},
            {"metrics": {"score": 0.805}},
        ]
        result = compute_agent_trends(history, "score")
        assert result["trend_direction"] == "stable"

    def test_compute_agent_trends_six_or_more_runs(self):
        """Test trend computation with 6+ runs uses 3-run average."""
        from backend.agent.eval.history import compute_agent_trends

        history = [
            {"metrics": {"score": 0.60}},
            {"metrics": {"score": 0.62}},
            {"metrics": {"score": 0.64}},
            {"metrics": {"score": 0.80}},
            {"metrics": {"score": 0.82}},
            {"metrics": {"score": 0.84}},
        ]
        result = compute_agent_trends(history, "score")
        assert result["has_trend"] is True
        assert result["num_runs"] == 6
        # Recent avg (0.80+0.82+0.84)/3 = 0.82
        # Prev avg (0.60+0.62+0.64)/3 = 0.62
        # Trend = 0.82 - 0.62 = 0.20
        assert result["trend"] == pytest.approx(0.20, abs=0.01)


# =============================================================================
# Shared Module Tests
# =============================================================================

class TestSharedFormatters:
    """Tests for shared formatting functions."""

    def test_format_check_mark_true(self):
        """Test format_check_mark with True."""
        from backend.agent.eval.shared import format_check_mark

        result = format_check_mark(True)
        assert "[green]Y[/green]" in result

    def test_format_check_mark_false(self):
        """Test format_check_mark with False."""
        from backend.agent.eval.shared import format_check_mark

        result = format_check_mark(False)
        assert "[red]X[/red]" in result

    def test_format_percentage_high(self):
        """Test format_percentage with high value (green)."""
        from backend.agent.eval.shared import format_percentage

        result = format_percentage(0.95)
        assert "[green]" in result
        assert "95.0%" in result

    def test_format_percentage_medium(self):
        """Test format_percentage with medium value (yellow)."""
        from backend.agent.eval.shared import format_percentage

        result = format_percentage(0.75)
        assert "[yellow]" in result
        assert "75.0%" in result

    def test_format_percentage_low(self):
        """Test format_percentage with low value (red)."""
        from backend.agent.eval.shared import format_percentage

        result = format_percentage(0.50)
        assert "[red]" in result
        assert "50.0%" in result

    def test_format_percentage_custom_thresholds(self):
        """Test format_percentage with custom thresholds."""
        from backend.agent.eval.shared import format_percentage

        # With custom thresholds (0.8, 0.6), 0.75 should be yellow
        result = format_percentage(0.75, thresholds=(0.8, 0.6))
        assert "[yellow]" in result


class TestSharedTables:
    """Tests for shared table creation functions."""

    def test_create_summary_table(self):
        """Test create_summary_table creates valid table."""
        from backend.agent.eval.shared import create_summary_table

        table = create_summary_table("Test Summary")
        assert table.title == "Test Summary"
        assert len(table.columns) == 2

    def test_create_slo_table(self):
        """Test create_slo_table creates valid table."""
        from backend.agent.eval.shared import create_slo_table

        slo_checks = [
            ("P95 Latency", True, "4.5s", "5.0s"),
            ("Answer Relevance", True, "85%", "80%"),
            ("Groundedness", False, "75%", "80%"),
        ]

        table = create_slo_table(slo_checks, "Test SLOs")
        assert table.title == "Test SLOs"
        assert len(table.columns) == 4


class TestSharedSLOFunctions:
    """Tests for SLO-related shared functions."""

    def test_get_failed_slos_none_failed(self):
        """Test get_failed_slos with all passing."""
        from backend.agent.eval.shared import get_failed_slos

        slo_checks = [
            ("P95 Latency", True, "4.5s", "5.0s"),
            ("Answer Relevance", True, "85%", "80%"),
        ]

        failed = get_failed_slos(slo_checks)
        assert failed == []

    def test_get_failed_slos_some_failed(self):
        """Test get_failed_slos with failures."""
        from backend.agent.eval.shared import get_failed_slos

        slo_checks = [
            ("P95 Latency", True, "4.5s", "5.0s"),
            ("Answer Relevance", False, "70%", "80%"),
            ("Groundedness", False, "60%", "80%"),
        ]

        failed = get_failed_slos(slo_checks)
        assert len(failed) == 2
        assert "Answer Relevance" in failed
        assert "Groundedness" in failed

    def test_determine_exit_code_all_pass(self):
        """Test determine_exit_code with all passing."""
        from backend.agent.eval.shared import determine_exit_code

        assert determine_exit_code(all_slos_passed=True, is_regression=False) == 0

    def test_determine_exit_code_slo_fail(self):
        """Test determine_exit_code with SLO failure."""
        from backend.agent.eval.shared import determine_exit_code

        assert determine_exit_code(all_slos_passed=False, is_regression=False) == 1

    def test_determine_exit_code_regression(self):
        """Test determine_exit_code with regression."""
        from backend.agent.eval.shared import determine_exit_code

        assert determine_exit_code(all_slos_passed=True, is_regression=True) == 1


class TestSharedLatency:
    """Tests for latency calculation functions."""

    def test_calculate_p95_latency_empty_list(self):
        """Test calculate_p95_latency with empty list."""
        from backend.agent.eval.shared import calculate_p95_latency

        assert calculate_p95_latency([]) == 0.0

    def test_calculate_p95_latency_single_value(self):
        """Test calculate_p95_latency with single value."""
        from backend.agent.eval.shared import calculate_p95_latency

        assert calculate_p95_latency([1000]) == 1000.0

    def test_calculate_p95_latency_multiple_values(self):
        """Test calculate_p95_latency with multiple values."""
        from backend.agent.eval.shared import calculate_p95_latency

        latencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        p95 = calculate_p95_latency(latencies)
        # P95 index = 10 * 0.95 = 9.5 -> 9, value at index 9 is 1000
        assert p95 == 1000.0

    def test_calculate_p95_latency_with_outliers(self):
        """Test calculate_p95_latency with outliers."""
        from backend.agent.eval.shared import calculate_p95_latency

        # 95 values at 100ms, 5 values at 10000ms
        latencies = [100] * 95 + [10000] * 5
        p95 = calculate_p95_latency(latencies)
        # P95 should pick up the outliers
        assert p95 >= 100


class TestSharedJSONParsing:
    """Tests for JSON parsing functions."""

    def test_parse_json_response_plain_json(self):
        """Test parse_json_response with plain JSON."""
        from backend.agent.eval.shared import parse_json_response

        result = parse_json_response('{"score": 1, "reason": "Good answer"}')
        assert result["score"] == 1
        assert result["reason"] == "Good answer"

    def test_parse_json_response_with_markdown_block(self):
        """Test parse_json_response with markdown code block."""
        from backend.agent.eval.shared import parse_json_response

        text = """Here's the analysis:
```json
{"score": 0, "reason": "Bad answer"}
```
"""
        result = parse_json_response(text)
        assert result["score"] == 0

    def test_parse_json_response_with_plain_code_block(self):
        """Test parse_json_response with plain code block."""
        from backend.agent.eval.shared import parse_json_response

        text = """
```
{"score": 1}
```
"""
        result = parse_json_response(text)
        assert result["score"] == 1

    def test_parse_json_response_invalid_json(self):
        """Test parse_json_response with invalid JSON."""
        from backend.agent.eval.shared import parse_json_response
        import json

        with pytest.raises(json.JSONDecodeError):
            parse_json_response("not valid json")


class TestSharedBaseline:
    """Tests for baseline comparison functions."""

    def test_compare_to_baseline_no_file(self, tmp_path):
        """Test compare_to_baseline when file doesn't exist."""
        from backend.agent.eval.shared import compare_to_baseline

        baseline_path = tmp_path / "nonexistent.json"
        is_regression, baseline_score = compare_to_baseline(0.85, baseline_path)

        assert is_regression is False
        assert baseline_score is None

    def test_compare_to_baseline_with_file(self, tmp_path):
        """Test compare_to_baseline with existing baseline."""
        from backend.agent.eval.shared import compare_to_baseline
        import json

        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps({"overall_score": 0.80}))

        is_regression, baseline_score = compare_to_baseline(0.85, baseline_path)

        assert is_regression is False
        assert baseline_score == 0.80

    def test_compare_to_baseline_regression_detected(self, tmp_path):
        """Test compare_to_baseline detects regression."""
        from backend.agent.eval.shared import compare_to_baseline, REGRESSION_THRESHOLD
        import json

        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps({"overall_score": 0.90}))

        # Score below baseline - REGRESSION_THRESHOLD is regression
        current_score = 0.90 - REGRESSION_THRESHOLD - 0.01
        is_regression, baseline_score = compare_to_baseline(current_score, baseline_path)

        assert is_regression is True

    def test_compare_to_baseline_with_summary_structure(self, tmp_path):
        """Test compare_to_baseline with nested summary structure."""
        from backend.agent.eval.shared import compare_to_baseline
        import json

        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps({"summary": {"overall_score": 0.85}}))

        is_regression, baseline_score = compare_to_baseline(0.90, baseline_path)

        assert baseline_score == 0.85

    def test_compare_to_baseline_invalid_json(self, tmp_path):
        """Test compare_to_baseline with invalid JSON."""
        from backend.agent.eval.shared import compare_to_baseline

        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text("not valid json")

        is_regression, baseline_score = compare_to_baseline(0.85, baseline_path)

        assert is_regression is False
        assert baseline_score is None

    def test_save_baseline(self, tmp_path):
        """Test save_baseline writes correct file."""
        from backend.agent.eval.shared import save_baseline
        import json

        baseline_path = tmp_path / "subdir" / "baseline.json"
        summary = {"overall_score": 0.88, "answer_relevance": 0.90}

        save_baseline(summary, baseline_path)

        assert baseline_path.exists()
        saved_data = json.loads(baseline_path.read_text())
        assert saved_data["summary"]["overall_score"] == 0.88


class TestSharedPrintFunctions:
    """Tests for print/output functions (verify they don't crash)."""

    def test_print_eval_header(self, capsys):
        """Test print_eval_header runs without error."""
        from backend.agent.eval.shared import print_eval_header

        # Should not raise
        print_eval_header("Test Header", "Test Subtitle")

    def test_print_baseline_comparison_no_baseline(self, capsys):
        """Test print_baseline_comparison with no baseline."""
        from backend.agent.eval.shared import print_baseline_comparison

        # Should not raise
        print_baseline_comparison(0.85, None, False)

    def test_print_baseline_comparison_with_baseline(self, capsys):
        """Test print_baseline_comparison with baseline."""
        from backend.agent.eval.shared import print_baseline_comparison

        # Should not raise
        print_baseline_comparison(0.85, 0.80, False)

    def test_print_baseline_comparison_regression(self, capsys):
        """Test print_baseline_comparison with regression."""
        from backend.agent.eval.shared import print_baseline_comparison

        # Should not raise
        print_baseline_comparison(0.70, 0.80, True)

    def test_print_slo_result_all_pass(self, capsys):
        """Test print_slo_result with all passing."""
        from backend.agent.eval.shared import print_slo_result

        slo_checks = [
            ("P95 Latency", True, "4.5s", "5.0s"),
            ("Answer Relevance", True, "85%", "80%"),
        ]

        result = print_slo_result(slo_checks)
        assert result is True

    def test_print_slo_result_some_fail(self, capsys):
        """Test print_slo_result with failures."""
        from backend.agent.eval.shared import print_slo_result

        slo_checks = [
            ("P95 Latency", True, "4.5s", "5.0s"),
            ("Answer Relevance", False, "70%", "80%"),
        ]

        result = print_slo_result(slo_checks)
        assert result is False

    def test_print_overall_result_panel_pass(self, capsys):
        """Test print_overall_result_panel with pass."""
        from backend.agent.eval.shared import print_overall_result_panel

        # Should not raise
        print_overall_result_panel(True, [], "All tests passed!")

    def test_print_overall_result_panel_fail(self, capsys):
        """Test print_overall_result_panel with failure."""
        from backend.agent.eval.shared import print_overall_result_panel

        # Should not raise
        print_overall_result_panel(False, ["SLO failed", "Regression detected"], "")

    def test_print_debug_failures_empty(self, capsys):
        """Test print_debug_failures with empty list."""
        from backend.agent.eval.shared import print_debug_failures

        # Should not raise, should print nothing
        print_debug_failures([], "No Failures")

    def test_print_debug_failures_with_items(self, capsys):
        """Test print_debug_failures with items."""
        from backend.agent.eval.shared import print_debug_failures

        failures = [
            {"id": "t1", "error": "Test error 1"},
            {"id": "t2", "error": "Test error 2"},
        ]

        # Should not raise
        print_debug_failures(failures, "Test Failures")

    def test_print_debug_failures_with_custom_formatter(self, capsys):
        """Test print_debug_failures with custom formatter."""
        from backend.agent.eval.shared import print_debug_failures, console

        failures = [{"id": "t1", "score": 0.5}]

        def custom_formatter(i, item):
            console.print(f"Custom: {item['id']}")

        # Should not raise
        print_debug_failures(failures, "Custom", format_item=custom_formatter)


class TestSharedParallelRunner:
    """Tests for parallel evaluation runner."""

    def test_run_parallel_evaluation_basic(self):
        """Test run_parallel_evaluation with simple function."""
        from backend.agent.eval.shared import run_parallel_evaluation

        items = [
            {"id": "1", "value": 10},
            {"id": "2", "value": 20},
            {"id": "3", "value": 30},
        ]

        def evaluate_fn(item, lock):
            return item["value"] * 2

        results = run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=2,
            description="Test",
        )

        assert len(results) == 3
        assert sorted(results) == [20, 40, 60]

    def test_run_parallel_evaluation_no_lock(self):
        """Test run_parallel_evaluation without lock."""
        from backend.agent.eval.shared import run_parallel_evaluation

        items = [{"id": "1", "value": 5}]

        def evaluate_fn(item, lock):
            assert lock is None
            return item["value"]

        results = run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=1,
            description="Test",
            use_lock=False,
        )

        assert results == [5]

    def test_run_parallel_evaluation_custom_id_field(self):
        """Test run_parallel_evaluation with custom ID field."""
        from backend.agent.eval.shared import run_parallel_evaluation

        items = [
            {"test_id": "a", "value": 1},
            {"test_id": "b", "value": 2},
        ]

        def evaluate_fn(item, lock):
            return item["value"]

        results = run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=2,
            description="Test",
            id_field="test_id",
        )

        assert len(results) == 2

    def test_run_parallel_evaluation_handles_errors(self, capsys):
        """Test run_parallel_evaluation handles errors gracefully."""
        from backend.agent.eval.shared import run_parallel_evaluation

        items = [
            {"id": "good", "value": 10},
            {"id": "bad", "value": None},
        ]

        def evaluate_fn(item, lock):
            if item["value"] is None:
                raise ValueError("Bad value")
            return item["value"]

        results = run_parallel_evaluation(
            items=items,
            evaluate_fn=evaluate_fn,
            max_workers=2,
            description="Test",
        )

        # Only the good item should have a result
        assert len(results) == 1
        assert results[0] == 10


# =============================================================================
# History Module Extended Tests
# =============================================================================

class TestHistoryFileOperations:
    """Tests for history file operations."""

    def test_load_agent_history_no_file(self, tmp_path, monkeypatch):
        """Test load_agent_history when file doesn't exist."""
        from backend.agent.eval import history

        # Point to non-existent file
        monkeypatch.setattr(history, "HISTORY_FILE", tmp_path / "nonexistent.json")

        result = history.load_agent_history()
        assert result == []

    def test_load_agent_history_with_file(self, tmp_path, monkeypatch):
        """Test load_agent_history with existing file."""
        from backend.agent.eval import history
        import json

        history_file = tmp_path / "history.json"
        history_file.write_text(json.dumps([{"run_id": "test1"}]))
        monkeypatch.setattr(history, "HISTORY_FILE", history_file)

        result = history.load_agent_history()
        assert len(result) == 1
        assert result[0]["run_id"] == "test1"

    def test_load_agent_history_invalid_json(self, tmp_path, monkeypatch):
        """Test load_agent_history with invalid JSON."""
        from backend.agent.eval import history

        history_file = tmp_path / "history.json"
        history_file.write_text("not valid json")
        monkeypatch.setattr(history, "HISTORY_FILE", history_file)

        result = history.load_agent_history()
        assert result == []

    def test_save_agent_history(self, tmp_path, monkeypatch):
        """Test save_agent_history writes file."""
        from backend.agent.eval import history
        import json

        data_dir = tmp_path / "data"
        history_file = data_dir / "history.json"
        monkeypatch.setattr(history, "DATA_DIR", data_dir)
        monkeypatch.setattr(history, "HISTORY_FILE", history_file)

        test_history = [{"run_id": "r1"}, {"run_id": "r2"}]
        history.save_agent_history(test_history)

        assert history_file.exists()
        saved = json.loads(history_file.read_text())
        assert len(saved) == 2

    def test_save_agent_history_limits_entries(self, tmp_path, monkeypatch):
        """Test save_agent_history limits to MAX_HISTORY_ENTRIES."""
        from backend.agent.eval import history
        import json

        data_dir = tmp_path / "data"
        history_file = data_dir / "history.json"
        monkeypatch.setattr(history, "DATA_DIR", data_dir)
        monkeypatch.setattr(history, "HISTORY_FILE", history_file)
        monkeypatch.setattr(history, "MAX_HISTORY_ENTRIES", 5)

        # Create more entries than the limit
        test_history = [{"run_id": f"r{i}"} for i in range(10)]
        history.save_agent_history(test_history)

        saved = json.loads(history_file.read_text())
        assert len(saved) == 5
        # Should keep the last 5
        assert saved[0]["run_id"] == "r5"

    def test_add_to_agent_history(self, tmp_path, monkeypatch):
        """Test add_to_agent_history adds entry."""
        from backend.agent.eval import history
        from backend.agent.eval.models import E2EEvalSummary
        import json

        data_dir = tmp_path / "data"
        history_file = data_dir / "history.json"
        monkeypatch.setattr(history, "DATA_DIR", data_dir)
        monkeypatch.setattr(history, "HISTORY_FILE", history_file)

        summary = E2EEvalSummary(
            total_tests=10,
            answer_relevance_rate=0.85,
            groundedness_rate=0.80,
            company_extraction_accuracy=0.95,
            intent_accuracy=0.90,
            avg_latency_ms=2000,
            p95_latency_ms=4000,
            by_category={},
        )

        history.add_to_agent_history(summary, run_id="test-run", tags=["test"])

        saved = json.loads(history_file.read_text())
        assert len(saved) == 1
        assert saved[0]["run_id"] == "test-run"
        assert saved[0]["tags"] == ["test"]
        assert saved[0]["metrics"]["answer_relevance"] == 0.85

    def test_add_to_agent_history_tracks_slo_failures(self, tmp_path, monkeypatch):
        """Test add_to_agent_history tracks SLO failures."""
        from backend.agent.eval import history
        from backend.agent.eval.models import E2EEvalSummary
        import json

        data_dir = tmp_path / "data"
        history_file = data_dir / "history.json"
        monkeypatch.setattr(history, "DATA_DIR", data_dir)
        monkeypatch.setattr(history, "HISTORY_FILE", history_file)

        # Summary with failing SLOs
        summary = E2EEvalSummary(
            total_tests=10,
            answer_relevance_rate=0.50,  # Below SLO (0.80)
            groundedness_rate=0.50,  # Below SLO (0.80)
            company_extraction_accuracy=0.95,
            intent_accuracy=0.90,
            avg_latency_ms=2000,
            p95_latency_ms=4000,
            by_category={},
        )

        history.add_to_agent_history(summary)

        saved = json.loads(history_file.read_text())
        assert saved[0]["all_slos_passed"] is False
        assert len(saved[0]["failed_slos"]) >= 2

    def test_print_agent_trend_report_no_history(self, tmp_path, monkeypatch, capsys):
        """Test print_agent_trend_report with no history."""
        from backend.agent.eval import history

        monkeypatch.setattr(history, "HISTORY_FILE", tmp_path / "nonexistent.json")

        # Should not raise
        history.print_agent_trend_report()

    def test_print_agent_trend_report_with_history(self, tmp_path, monkeypatch, capsys):
        """Test print_agent_trend_report with history."""
        from backend.agent.eval import history
        import json
        from datetime import datetime

        history_file = tmp_path / "history.json"
        test_history = [
            {
                "run_id": "r1",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "answer_relevance": 0.80,
                    "groundedness": 0.75,
                    "company_extraction": 0.90,
                    "intent_accuracy": 0.85,
                    "p95_latency_ms": 4000,
                    "avg_latency_ms": 2000,
                },
                "all_slos_passed": True,
                "failed_slos": [],
            },
            {
                "run_id": "r2",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "answer_relevance": 0.85,
                    "groundedness": 0.80,
                    "company_extraction": 0.92,
                    "intent_accuracy": 0.88,
                    "p95_latency_ms": 3500,
                    "avg_latency_ms": 1800,
                },
                "all_slos_passed": True,
                "failed_slos": [],
            },
        ]
        history_file.write_text(json.dumps(test_history))
        monkeypatch.setattr(history, "HISTORY_FILE", history_file)

        # Should not raise
        history.print_agent_trend_report(num_runs=2)


# =============================================================================
# Tracking Module Extended Tests
# =============================================================================

class TestTrackingFileOperations:
    """Tests for tracking file operations."""

    def test_load_previous_e2e_summary_no_file(self, tmp_path, monkeypatch):
        """Test load_previous_e2e_summary when file doesn't exist."""
        from backend.agent.eval import tracking

        monkeypatch.setattr(
            tracking, "PREVIOUS_RESULTS_PATH", tmp_path / "nonexistent.json"
        )

        result = tracking.load_previous_e2e_summary()
        assert result is None

    def test_load_previous_e2e_summary_with_file(self, tmp_path, monkeypatch):
        """Test load_previous_e2e_summary with existing file."""
        from backend.agent.eval import tracking
        import json

        results_file = tmp_path / "previous.json"
        summary_data = {
            "summary": {
                "total_tests": 10,
                "answer_relevance_rate": 0.85,
                "groundedness_rate": 0.80,
                "company_extraction_accuracy": 0.95,
                "intent_accuracy": 0.90,
                "avg_latency_ms": 2000,
                "p95_latency_ms": 4000,
                "by_category": {},
            }
        }
        results_file.write_text(json.dumps(summary_data))
        monkeypatch.setattr(tracking, "PREVIOUS_RESULTS_PATH", results_file)

        result = tracking.load_previous_e2e_summary()
        assert result is not None
        assert result.answer_relevance_rate == 0.85

    def test_load_previous_e2e_summary_invalid_json(self, tmp_path, monkeypatch):
        """Test load_previous_e2e_summary with invalid JSON."""
        from backend.agent.eval import tracking

        results_file = tmp_path / "previous.json"
        results_file.write_text("not valid json")
        monkeypatch.setattr(tracking, "PREVIOUS_RESULTS_PATH", results_file)

        result = tracking.load_previous_e2e_summary()
        assert result is None

    def test_save_e2e_as_previous(self, tmp_path, monkeypatch):
        """Test save_e2e_as_previous writes file."""
        from backend.agent.eval import tracking
        from backend.agent.eval.models import E2EEvalResult, E2EEvalSummary
        import json

        data_dir = tmp_path / "data"
        results_file = data_dir / "previous.json"
        monkeypatch.setattr(tracking, "DATA_DIR", data_dir)
        monkeypatch.setattr(tracking, "PREVIOUS_RESULTS_PATH", results_file)

        results = [
            E2EEvalResult(
                test_case_id="t1",
                question="Test?",
                category="test",
                answer="Answer",
                answer_relevance=1,
                answer_grounded=1,
                has_sources=True,
                latency_ms=1000,
                total_tokens=100,
            )
        ]

        summary = E2EEvalSummary(
            total_tests=1,
            answer_relevance_rate=1.0,
            groundedness_rate=1.0,
            avg_latency_ms=1000,
            p95_latency_ms=1000,
            by_category={},
        )

        tracking.save_e2e_as_previous(results, summary)

        assert results_file.exists()
        saved = json.loads(results_file.read_text())
        assert "results" in saved
        assert "summary" in saved


class TestTrackingComparison:
    """Tests for tracking comparison functions."""

    def test_compare_e2e_with_previous_improvement(self):
        """Test compare_e2e_with_previous detects improvement."""
        from backend.agent.eval.tracking import compare_e2e_with_previous
        from backend.agent.eval.models import E2EEvalSummary

        previous = E2EEvalSummary(
            total_tests=10,
            answer_relevance_rate=0.70,
            groundedness_rate=0.70,
            avg_latency_ms=3000,
            p95_latency_ms=5000,
            by_category={},
        )

        current = E2EEvalSummary(
            total_tests=10,
            answer_relevance_rate=0.90,  # Improvement
            groundedness_rate=0.90,  # Improvement
            avg_latency_ms=2000,
            p95_latency_ms=3500,  # Improvement
            by_category={},
        )

        comparison = compare_e2e_with_previous(current, previous)

        assert comparison["has_previous"] is True
        assert len(comparison["improvements"]) >= 2

    def test_compare_e2e_with_previous_latency_regression(self):
        """Test compare_e2e_with_previous detects latency regression."""
        from backend.agent.eval.tracking import compare_e2e_with_previous
        from backend.agent.eval.models import E2EEvalSummary

        previous = E2EEvalSummary(
            total_tests=10,
            answer_relevance_rate=0.85,
            groundedness_rate=0.85,
            avg_latency_ms=2000,
            p95_latency_ms=3000,
            by_category={},
        )

        current = E2EEvalSummary(
            total_tests=10,
            answer_relevance_rate=0.85,
            groundedness_rate=0.85,
            avg_latency_ms=3000,
            p95_latency_ms=4000,  # Latency regression (>500ms increase)
            by_category={},
        )

        comparison = compare_e2e_with_previous(current, previous)

        # Check for latency regression
        latency_regressions = [
            r for r in comparison["regressions"] if r["metric"] == "P95 Latency"
        ]
        assert len(latency_regressions) == 1

    def test_print_e2e_regression_report_no_previous(self, capsys):
        """Test print_e2e_regression_report with no previous."""
        from backend.agent.eval.tracking import print_e2e_regression_report

        comparison = {"has_previous": False, "regressions": [], "improvements": []}

        # Should not raise
        print_e2e_regression_report(comparison)

    def test_print_e2e_regression_report_with_comparison(self, capsys):
        """Test print_e2e_regression_report with comparison."""
        from backend.agent.eval.tracking import print_e2e_regression_report
        from backend.agent.eval.models import E2EEvalSummary

        previous = E2EEvalSummary(
            total_tests=10,
            answer_relevance_rate=0.80,
            groundedness_rate=0.80,
            avg_latency_ms=2000,
            p95_latency_ms=4000,
            by_category={},
        )

        current = E2EEvalSummary(
            total_tests=10,
            answer_relevance_rate=0.85,
            groundedness_rate=0.85,
            avg_latency_ms=1800,
            p95_latency_ms=3500,
            by_category={},
        )

        comparison = {
            "has_previous": True,
            "regressions": [],
            "improvements": [{"metric": "Answer Relevance", "delta": 0.05}],
            "previous_summary": previous,
            "current_summary": current,
        }

        # Should not raise
        print_e2e_regression_report(comparison)


class TestTrackingBudget:
    """Tests for budget analysis functions."""

    def test_analyze_e2e_budget_violations_category_stats(self):
        """Test analyze_e2e_budget_violations computes category stats."""
        from backend.agent.eval.tracking import analyze_e2e_budget_violations
        from backend.agent.eval.models import E2EEvalResult

        results = [
            E2EEvalResult(
                test_case_id="t1",
                question="Q1",
                category="cat_a",
                answer="A",
                answer_relevance=1,
                answer_grounded=1,
                has_sources=True,
                latency_ms=1000,
                total_tokens=100,
            ),
            E2EEvalResult(
                test_case_id="t2",
                question="Q2",
                category="cat_a",
                answer="A",
                answer_relevance=1,
                answer_grounded=1,
                has_sources=True,
                latency_ms=2000,
                total_tokens=100,
            ),
            E2EEvalResult(
                test_case_id="t3",
                question="Q3",
                category="cat_b",
                answer="A",
                answer_relevance=1,
                answer_grounded=1,
                has_sources=True,
                latency_ms=1500,
                total_tokens=100,
            ),
        ]

        analysis = analyze_e2e_budget_violations(results)

        assert "cat_a" in analysis["category_stats"]
        assert "cat_b" in analysis["category_stats"]
        assert analysis["category_stats"]["cat_a"]["count"] == 2
        assert analysis["category_stats"]["cat_b"]["count"] == 1

    def test_print_e2e_budget_report_no_violations(self, capsys):
        """Test print_e2e_budget_report with no violations."""
        from backend.agent.eval.tracking import print_e2e_budget_report
        from backend.agent.eval.models import E2EEvalResult

        results = [
            E2EEvalResult(
                test_case_id="t1",
                question="Q1",
                category="test",
                answer="A",
                answer_relevance=1,
                answer_grounded=1,
                has_sources=True,
                latency_ms=1000,
                total_tokens=100,
            ),
        ]

        # Should not raise
        print_e2e_budget_report(results)

    def test_print_e2e_budget_report_with_violations(self, capsys):
        """Test print_e2e_budget_report with violations."""
        from backend.agent.eval.tracking import print_e2e_budget_report
        from backend.agent.eval.models import E2EEvalResult

        results = [
            E2EEvalResult(
                test_case_id="t1",
                question="Q1",
                category="test",
                answer="A",
                answer_relevance=1,
                answer_grounded=1,
                has_sources=True,
                latency_ms=10000,  # Over budget
                total_tokens=100,
            ),
        ]

        # Should not raise
        print_e2e_budget_report(results)


class TestTrackingConstants:
    """Tests for tracking constants."""

    def test_agent_latency_budgets_defined(self):
        """Test AGENT_LATENCY_BUDGETS has expected keys."""
        from backend.agent.eval.tracking import AGENT_LATENCY_BUDGETS

        assert "router" in AGENT_LATENCY_BUDGETS
        assert "synthesis" in AGENT_LATENCY_BUDGETS
        assert "rag_docs" in AGENT_LATENCY_BUDGETS

    def test_agent_total_latency_budget(self):
        """Test AGENT_TOTAL_LATENCY_BUDGET_MS is set."""
        from backend.agent.eval.tracking import AGENT_TOTAL_LATENCY_BUDGET_MS

        assert AGENT_TOTAL_LATENCY_BUDGET_MS > 0


# =============================================================================
# Base Module Tests
# =============================================================================

class TestBaseModule:
    """Tests for base module."""

    def test_base_re_exports_shared_utilities(self):
        """Test base.py re-exports shared utilities."""
        from backend.agent.eval.base import (
            console,
            create_summary_table,
            format_check_mark,
            format_percentage,
            print_eval_header,
            compare_to_baseline,
            save_baseline,
            print_baseline_comparison,
            REGRESSION_THRESHOLD,
        )

        # All should be importable
        assert console is not None
        assert callable(create_summary_table)
        assert callable(format_check_mark)
        assert callable(format_percentage)
        assert callable(print_eval_header)
        assert callable(compare_to_baseline)
        assert callable(save_baseline)
        assert callable(print_baseline_comparison)
        assert REGRESSION_THRESHOLD > 0

    def test_ensure_qdrant_collections_importable(self):
        """Test ensure_qdrant_collections is importable."""
        from backend.agent.eval.base import ensure_qdrant_collections

        assert callable(ensure_qdrant_collections)


# =============================================================================
# Additional Coverage Tests
# =============================================================================

class TestSharedLLMJudge:
    """Tests for run_llm_judge function with mocked LLM."""

    def test_run_llm_judge_success(self, monkeypatch):
        """Test run_llm_judge with successful response."""
        # Mock call_llm at the source module
        def mock_call_llm(*args, **kwargs):
            return '{"score": 1, "reason": "Good"}'

        monkeypatch.setattr(
            "backend.agent.eval.llm_client.call_llm",
            mock_call_llm,
        )

        from backend.agent.eval.shared import run_llm_judge

        result = run_llm_judge("Test prompt", "System prompt")
        assert result["score"] == 1

    def test_run_llm_judge_empty_response(self, monkeypatch):
        """Test run_llm_judge with empty response."""
        def mock_call_llm(*args, **kwargs):
            return ""

        monkeypatch.setattr(
            "backend.agent.eval.llm_client.call_llm",
            mock_call_llm,
        )

        from backend.agent.eval.shared import run_llm_judge

        result = run_llm_judge("Test prompt", "System prompt")
        assert "error" in result

    def test_run_llm_judge_exception(self, monkeypatch):
        """Test run_llm_judge with exception."""
        def mock_call_llm(*args, **kwargs):
            raise ValueError("LLM error")

        monkeypatch.setattr(
            "backend.agent.eval.llm_client.call_llm",
            mock_call_llm,
        )

        from backend.agent.eval.shared import run_llm_judge

        result = run_llm_judge("Test prompt", "System prompt")
        assert "error" in result
        assert "LLM error" in result["error"]


class TestBaseEnsureQdrant:
    """Tests for ensure_qdrant_collections with mocks."""

    def test_ensure_qdrant_collections_exists(self, monkeypatch, capsys):
        """Test ensure_qdrant_collections when collections exist."""
        # Mock rag_tools module
        mock_client = type('MockClient', (), {
            'collection_exists': lambda self, name: True,
            'get_collection': lambda self, name: type('C', (), {'points_count': 10})(),
        })()

        def mock_get_client():
            return mock_client

        # Patch the imports inside ensure_qdrant_collections
        import backend.agent.rag.client
        monkeypatch.setattr(backend.agent.rag.client, "get_qdrant_client", mock_get_client)

        from backend.agent.eval.base import ensure_qdrant_collections

        # This should complete without calling ingest
        ensure_qdrant_collections()

    def test_ensure_qdrant_collections_missing(self, monkeypatch, tmp_path, capsys):
        """Test ensure_qdrant_collections when collections are missing."""
        # Mock rag_tools module
        mock_client = type('MockClient', (), {
            'collection_exists': lambda self, name: False,
            'get_collection': lambda self, name: type('C', (), {'points_count': 0})(),
        })()

        def mock_get_client():
            return mock_client

        def mock_ingest_docs():
            return 10

        def mock_ingest_private():
            pass

        import backend.agent.rag.client
        import backend.agent.rag.ingest
        import backend.agent.rag.config
        monkeypatch.setattr(backend.agent.rag.client, "get_qdrant_client", mock_get_client)
        monkeypatch.setattr(backend.agent.rag.ingest, "ingest_docs", mock_ingest_docs)
        monkeypatch.setattr(backend.agent.rag.ingest, "ingest_private_texts", mock_ingest_private)
        monkeypatch.setattr(backend.agent.rag.config, "QDRANT_PATH", tmp_path / "qdrant")

        from backend.agent.eval.base import ensure_qdrant_collections

        # This should run ingest functions
        ensure_qdrant_collections()


class TestTrackingPrintRegressionWithRegression:
    """Tests for tracking print functions with regressions."""

    def test_print_e2e_regression_report_with_regressions(self, capsys):
        """Test print_e2e_regression_report with actual regressions."""
        from backend.agent.eval.tracking import print_e2e_regression_report
        from backend.agent.eval.models import E2EEvalSummary

        previous = E2EEvalSummary(
            total_tests=10,
            answer_relevance_rate=0.90,
            groundedness_rate=0.90,
            avg_latency_ms=2000,
            p95_latency_ms=3000,
            by_category={},
        )

        current = E2EEvalSummary(
            total_tests=10,
            answer_relevance_rate=0.70,  # Regression
            groundedness_rate=0.90,
            avg_latency_ms=2000,
            p95_latency_ms=4000,  # Latency regression
            by_category={},
        )

        comparison = {
            "has_previous": True,
            "regressions": [
                {"metric": "Answer Relevance", "delta": -0.20},
                {"metric": "P95 Latency", "delta": 1000, "unit": "ms"},
            ],
            "improvements": [],
            "previous_summary": previous,
            "current_summary": current,
        }

        # Should not raise
        print_e2e_regression_report(comparison)


class TestHistoryTrendReportEdgeCases:
    """Tests for history trend report edge cases."""

    def test_print_agent_trend_report_invalid_timestamp(self, tmp_path, monkeypatch, capsys):
        """Test print_agent_trend_report with invalid timestamp."""
        from backend.agent.eval import history
        import json

        history_file = tmp_path / "history.json"
        test_history = [
            {
                "run_id": "r1",
                "timestamp": "invalid-timestamp",
                "metrics": {
                    "answer_relevance": 0.80,
                    "groundedness": 0.75,
                    "company_extraction": 0.90,
                    "intent_accuracy": 0.85,
                    "p95_latency_ms": 4000,
                    "avg_latency_ms": 2000,
                },
                "all_slos_passed": True,
                "failed_slos": [],
            },
        ]
        history_file.write_text(json.dumps(test_history))
        monkeypatch.setattr(history, "HISTORY_FILE", history_file)

        # Should not raise despite invalid timestamp
        history.print_agent_trend_report()

    def test_print_agent_trend_report_stable_trends(self, tmp_path, monkeypatch, capsys):
        """Test print_agent_trend_report with stable (unchanged) metrics."""
        from backend.agent.eval import history
        import json
        from datetime import datetime

        history_file = tmp_path / "history.json"
        # Two runs with identical metrics = stable trend
        test_history = [
            {
                "run_id": "r1",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "answer_relevance": 0.85,
                    "groundedness": 0.80,
                    "company_extraction": 0.90,
                    "intent_accuracy": 0.85,
                    "p95_latency_ms": 4000,
                    "avg_latency_ms": 2000,
                },
                "all_slos_passed": True,
                "failed_slos": [],
            },
            {
                "run_id": "r2",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "answer_relevance": 0.85,  # Same as r1 - stable
                    "groundedness": 0.80,  # Same as r1 - stable
                    "company_extraction": 0.90,
                    "intent_accuracy": 0.85,
                    "p95_latency_ms": 4000,
                    "avg_latency_ms": 2000,
                },
                "all_slos_passed": True,
                "failed_slos": [],
            },
        ]
        history_file.write_text(json.dumps(test_history))
        monkeypatch.setattr(history, "HISTORY_FILE", history_file)

        # Should print with stable trend indicators
        history.print_agent_trend_report()

    def test_print_agent_trend_report_latency_only(self, tmp_path, monkeypatch, capsys):
        """Test print_agent_trend_report with latency-only metrics (lower is better)."""
        from backend.agent.eval import history
        import json
        from datetime import datetime

        history_file = tmp_path / "history.json"
        # Two runs with decreasing latency = improvement (down is good for latency)
        test_history = [
            {
                "run_id": "r1",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "answer_relevance": 0.85,
                    "groundedness": 0.80,
                    "company_extraction": 0.90,
                    "intent_accuracy": 0.85,
                    "p95_latency_ms": 5000,  # Higher
                    "avg_latency_ms": 3000,  # Higher
                },
                "all_slos_passed": True,
                "failed_slos": [],
            },
            {
                "run_id": "r2",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "answer_relevance": 0.85,
                    "groundedness": 0.80,
                    "company_extraction": 0.90,
                    "intent_accuracy": 0.85,
                    "p95_latency_ms": 3000,  # Lower - improvement
                    "avg_latency_ms": 2000,  # Lower - improvement
                },
                "all_slos_passed": True,
                "failed_slos": [],
            },
        ]
        history_file.write_text(json.dumps(test_history))
        monkeypatch.setattr(history, "HISTORY_FILE", history_file)

        # Should print with improvement indicators for latency (down is good)
        history.print_agent_trend_report()


class TestHistoryWithLatencySLOViolation:
    """Tests for history with latency SLO violations."""

    def test_add_to_agent_history_latency_slo_failure(self, tmp_path, monkeypatch):
        """Test add_to_agent_history tracks latency SLO failures."""
        from backend.agent.eval import history
        from backend.agent.eval.models import E2EEvalSummary
        import json

        data_dir = tmp_path / "data"
        history_file = data_dir / "history.json"
        monkeypatch.setattr(history, "DATA_DIR", data_dir)
        monkeypatch.setattr(history, "HISTORY_FILE", history_file)

        # Summary with failing latency SLOs
        summary = E2EEvalSummary(
            total_tests=10,
            answer_relevance_rate=0.90,
            groundedness_rate=0.90,
            company_extraction_accuracy=0.95,
            intent_accuracy=0.90,
            avg_latency_ms=20000,  # Way over SLO
            p95_latency_ms=50000,  # Way over SLO
            by_category={},
        )

        history.add_to_agent_history(summary)

        saved = json.loads(history_file.read_text())
        assert saved[0]["all_slos_passed"] is False
        assert saved[0]["p95_slo_pass"] is False
        assert saved[0]["avg_slo_pass"] is False


class TestTrackingFullReport:
    """Tests for full tracking report (integration)."""

    def test_print_e2e_tracking_report(self, tmp_path, monkeypatch, capsys):
        """Test print_e2e_tracking_report integration."""
        from backend.agent.eval import tracking, history
        from backend.agent.eval.models import E2EEvalResult, E2EEvalSummary

        # Set up temp paths
        data_dir = tmp_path / "data"
        monkeypatch.setattr(tracking, "DATA_DIR", data_dir)
        monkeypatch.setattr(tracking, "PREVIOUS_RESULTS_PATH", data_dir / "prev.json")
        monkeypatch.setattr(history, "DATA_DIR", data_dir)
        monkeypatch.setattr(history, "HISTORY_FILE", data_dir / "history.json")

        results = [
            E2EEvalResult(
                test_case_id="t1",
                question="Test?",
                category="test",
                answer="Answer",
                answer_relevance=1,
                answer_grounded=1,
                has_sources=True,
                latency_ms=1000,
                total_tokens=100,
            )
        ]

        summary = E2EEvalSummary(
            total_tests=1,
            answer_relevance_rate=1.0,
            groundedness_rate=1.0,
            avg_latency_ms=1000,
            p95_latency_ms=1000,
            by_category={},
        )

        # Should not raise
        tracking.print_e2e_tracking_report(results, summary)
