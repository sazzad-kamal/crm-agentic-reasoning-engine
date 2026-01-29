"""Tests for backend.agent.followup.entity_context module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.agent.followup.entity_context import (
    _extract_entity_ids,
    get_entity_context,
)


class TestExtractEntityIds:
    """Tests for _extract_entity_ids helper."""

    def test_extracts_company_ids(self):
        """Extracts company_id from rows."""
        sql_results = {"data": [
            {"company_id": "C001", "name": "Acme"},
            {"company_id": "C002", "name": "Beta"},
        ]}
        entities = _extract_entity_ids(sql_results)
        assert entities == {"company_id": {"C001", "C002"}}

    def test_extracts_multiple_entity_types(self):
        """Extracts company_id and contact_id from rows."""
        sql_results = {"data": [
            {"company_id": "C001", "contact_id": "CT001"},
        ]}
        entities = _extract_entity_ids(sql_results)
        assert "company_id" in entities
        assert "contact_id" in entities
        assert entities["company_id"] == {"C001"}
        assert entities["contact_id"] == {"CT001"}

    def test_deduplicates_ids(self):
        """Deduplicates entity IDs across rows."""
        sql_results = {"data": [
            {"company_id": "C001"},
            {"company_id": "C001"},
            {"company_id": "C002"},
        ]}
        entities = _extract_entity_ids(sql_results)
        assert entities["company_id"] == {"C001", "C002"}

    def test_empty_sql_results(self):
        """Empty sql_results returns empty dict."""
        assert _extract_entity_ids({}) == {}
        assert _extract_entity_ids({"data": []}) == {}

    def test_rows_without_entity_columns(self):
        """Rows without entity ID columns returns empty dict."""
        sql_results = {"data": [
            {"name": "Acme", "amount": 1000},
        ]}
        assert _extract_entity_ids(sql_results) == {}

    def test_none_values_excluded(self):
        """None entity ID values are excluded."""
        sql_results = {"data": [
            {"company_id": None, "contact_id": "CT001"},
        ]}
        entities = _extract_entity_ids(sql_results)
        assert "company_id" not in entities
        assert entities["contact_id"] == {"CT001"}


class TestGetEntityContext:
    """Tests for get_entity_context function."""

    @patch("backend.agent.followup.entity_context.execute_sql")
    def test_generates_context_string(self, mock_execute: MagicMock):
        """Generates context string with entity names and counts."""
        conn = MagicMock()
        sql_results = {"data": [{"company_id": "C001"}]}

        # Mock responses: name lookup, then 4 count queries
        mock_execute.side_effect = [
            # Name lookup
            ([{"name": "Acme Corp"}], None),
            # contacts count
            ([{"cnt": 3}], None),
            # opportunities count
            ([{"cnt": 2}], None),
            # activities count
            ([{"cnt": 5}], None),
            # history count
            ([{"cnt": 4}], None),
        ]

        result = get_entity_context(sql_results, conn)

        assert "Acme Corp (company)" in result
        assert "3 contacts" in result
        assert "2 opportunities" in result
        assert "5 activities" in result
        assert "4 history" in result

    @patch("backend.agent.followup.entity_context.execute_sql")
    def test_empty_sql_results_returns_empty(self, mock_execute: MagicMock):
        """Empty sql_results returns empty string."""
        conn = MagicMock()
        result = get_entity_context({}, conn)
        assert result == ""
        mock_execute.assert_not_called()

    @patch("backend.agent.followup.entity_context.execute_sql")
    def test_no_entities_returns_empty(self, mock_execute: MagicMock):
        """Rows without entity columns returns empty string."""
        conn = MagicMock()
        sql_results = {"data": [{"name": "Acme", "amount": 1000}]}
        result = get_entity_context(sql_results, conn)
        assert result == ""

    @patch("backend.agent.followup.entity_context.execute_sql")
    def test_zero_counts_show_no_linked_records(self, mock_execute: MagicMock):
        """Entity with zero counts shows 'no linked records'."""
        conn = MagicMock()
        sql_results = {"data": [{"company_id": "C001"}]}

        mock_execute.side_effect = [
            # Name lookup
            ([{"name": "Empty Corp"}], None),
            # All counts return 0
            ([{"cnt": 0}], None),
            ([{"cnt": 0}], None),
            ([{"cnt": 0}], None),
            ([{"cnt": 0}], None),
        ]

        result = get_entity_context(sql_results, conn)

        assert "Empty Corp (company): no linked records" in result

    @patch("backend.agent.followup.entity_context.execute_sql")
    def test_name_lookup_failure_uses_id(self, mock_execute: MagicMock):
        """Failed name lookup falls back to entity ID."""
        conn = MagicMock()
        sql_results = {"data": [{"company_id": "C001"}]}

        mock_execute.side_effect = [
            # Name lookup fails
            ([], "Not found"),
            # Count queries
            ([{"cnt": 1}], None),
            ([{"cnt": 0}], None),
            ([{"cnt": 0}], None),
            ([{"cnt": 0}], None),
        ]

        result = get_entity_context(sql_results, conn)

        assert "C001 (company)" in result

    @patch("backend.agent.followup.entity_context.execute_sql")
    def test_header_text(self, mock_execute: MagicMock):
        """Context string starts with header."""
        conn = MagicMock()
        sql_results = {"data": [{"company_id": "C001"}]}

        mock_execute.side_effect = [
            ([{"name": "Test"}], None),
            ([{"cnt": 1}], None),
            ([{"cnt": 0}], None),
            ([{"cnt": 0}], None),
            ([{"cnt": 0}], None),
        ]

        result = get_entity_context(sql_results, conn)

        assert result.startswith("Linked data for entities in the answer:")
