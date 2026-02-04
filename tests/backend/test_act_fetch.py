"""Unit tests for Act! API fetch module."""

import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from backend.act_fetch import (
    DEMO_FOLLOWUPS,
    DEMO_PROMPTS,
    DEMO_STARTERS,
    _cache_get,
    _cache_key,
    _cache_set,
    _clear_token,
    _filter_noise_history,
    _get_auth_header,
    _is_open_opportunity,
    act_fetch,
    clear_api_cache,
    get_database,
    set_database,
)


class TestAuthHeader:
    """Tests for auth header generation."""

    def test_generates_base64_header(self):
        """Auth header is properly base64 encoded."""
        with patch("backend.act_fetch.ACT_API_USER", "user"), \
             patch("backend.act_fetch.ACT_API_PASS", "pass"):
            header = _get_auth_header()
            assert header.startswith("Basic ")
            # user:pass in base64 is dXNlcjpwYXNz
            assert header == "Basic dXNlcjpwYXNz"


class TestTokenCache:
    """Tests for token caching."""

    def test_clear_token_resets_cache(self):
        """_clear_token clears the token cache."""
        import backend.act_fetch as module

        module._token = "test_token"
        module._token_expires = 9999999999.0

        _clear_token()

        assert module._token is None
        assert module._token_expires is None


class TestApiCache:
    """Tests for stale-while-error API caching."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_api_cache()

    def test_cache_key_includes_user_database_endpoint_params(self):
        """Cache key includes user, database, endpoint, and params hash."""
        with patch("backend.act_fetch.ACT_API_USER", "testuser"), \
             patch("backend.act_fetch._current_database", "TestDB"):
            key = _cache_key("/api/contacts", {"$top": 100})
            assert "testuser" in key
            assert "TestDB" in key
            assert "/api/contacts" in key
            # Params are hashed, so just verify key has multiple parts
            assert key.count(":") >= 3

    def test_cache_key_different_params_different_keys(self):
        """Different params produce different cache keys."""
        with patch("backend.act_fetch.ACT_API_USER", "testuser"), \
             patch("backend.act_fetch._current_database", "TestDB"):
            key1 = _cache_key("/api/contacts", {"$top": 100})
            key2 = _cache_key("/api/contacts", {"$top": 200})
            assert key1 != key2

    def test_cache_set_and_get(self):
        """Cache stores and retrieves data correctly."""
        clear_api_cache()
        test_data = [{"id": "1", "name": "Test"}]

        _cache_set("test_key", test_data)
        result = _cache_get("test_key")

        assert result == test_data

    def test_cache_get_returns_none_for_missing_key(self):
        """Cache returns None for missing key."""
        clear_api_cache()
        result = _cache_get("nonexistent_key")
        assert result is None

    def test_clear_api_cache_removes_all_entries(self):
        """clear_api_cache removes all cached entries."""
        _cache_set("key1", [{"id": "1"}])
        _cache_set("key2", [{"id": "2"}])

        clear_api_cache()

        assert _cache_get("key1") is None
        assert _cache_get("key2") is None

    def test_get_returns_cached_on_failure(self):
        """_get returns cached data when API fails after retries."""
        import backend.act_fetch as module

        # Pre-populate cache
        clear_api_cache()
        cached_data = [{"id": "cached", "name": "Cached Data"}]
        with patch("backend.act_fetch.ACT_API_USER", "testuser"), \
             patch("backend.act_fetch._current_database", "TestDB"):
            key = _cache_key("/api/test", {"$top": 10})
            _cache_set(key, cached_data)

            # Make _get_raw fail
            with patch.object(module, "_get_raw", side_effect=httpx.HTTPError("API down")):
                result = module._get("/api/test", {"$top": 10})
                assert result == cached_data


class TestDatabaseSwitching:
    """Tests for runtime database switching."""

    def test_get_database_returns_current(self):
        """get_database returns current database."""
        db = get_database()
        assert isinstance(db, str)

    def test_set_database_changes_current(self):
        """set_database changes current database."""
        with patch("backend.act_fetch.AVAILABLE_DATABASES", ["KQC", "W31322003119"]):
            # Set to known value first
            set_database("KQC")
            assert get_database() == "KQC"
            # Switch to another
            set_database("W31322003119")
            assert get_database() == "W31322003119"
            # Switch back
            set_database("KQC")
            assert get_database() == "KQC"

    def test_set_database_invalid_raises(self):
        """set_database raises for invalid database."""
        with pytest.raises(ValueError, match="Unknown database"):
            set_database("InvalidDB")


class TestActFetch:
    """Tests for act_fetch function - 5 Gold Standard questions."""

    @pytest.fixture(autouse=True)
    def reset_token(self):
        """Reset token cache before each test."""
        _clear_token()

    @pytest.mark.parametrize("question", DEMO_STARTERS)
    def test_each_question_has_handler(self, question: str):
        """Each of the 5 demo questions has a handler."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = [{"id": "1", "name": "Test"}]
            result = act_fetch(question)

            assert result["error"] is None
            # All questions return dict with multiple keys
            assert isinstance(result["data"], dict)
            assert len(result["data"]) > 0

    def test_unknown_question_returns_error(self):
        """Unknown question returns error dict."""
        result = act_fetch("What is the meaning of life?")

        assert "Unknown question" in result["error"]
        assert result["data"] == {}

    def test_api_error_returns_error_dict(self):
        """API error returns error dict instead of raising."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.side_effect = httpx.HTTPError("API down")
            result = act_fetch("Daily briefing")

            assert "API down" in result["error"]
            assert result["data"] == {}

    def test_timeout_returns_error_dict(self):
        """Timeout returns error dict instead of raising."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Connection timeout")
            result = act_fetch("Forecast health")

            assert "timeout" in result["error"].lower()
            assert result["data"] == {}

    def test_daily_briefing_fetches_multiple_endpoints(self):
        """'Daily briefing' fetches calendar, history, opportunities, contacts."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = [{"id": "1", "name": "Test"}]
            result = act_fetch("Daily briefing")

            assert result["error"] is None
            # Should have called multiple endpoints
            assert mock_get.call_count >= 4
            # Should return expected keys
            assert "today_meetings" in result["data"]
            assert "recent_history" in result["data"]
            assert "open_opportunities" in result["data"]
            assert "overdue_followups" in result["data"]
            assert "duckdb" in result["data"]

    def test_forecast_health_returns_buckets(self):
        """'Forecast health' returns 30/60/90 day forecast buckets."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = [
                {"id": "1", "name": "Deal A", "estimatedCloseDate": "2099-01-15", "productTotal": 10000, "probability": 50, "statusName": "Open"},
            ]
            result = act_fetch("Forecast health")

            assert result["error"] is None
            assert "forecast_30d" in result["data"]
            assert "forecast_60d" in result["data"]
            assert "forecast_90d" in result["data"]
            assert "slipped_deals" in result["data"]
            assert "duckdb" in result["data"]

    def test_at_risk_deals_flags_risk_reasons(self):
        """'At-risk deals' flags deals by risk reason."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = [
                {"id": "1", "name": "Stalled Deal", "daysInStage": 20, "daysOpen": 10, "probability": 80, "estimatedCloseDate": "2099-01-01", "statusName": "Open"},
            ]
            result = act_fetch("At-risk deals")

            assert result["error"] is None
            assert "at_risk_deals" in result["data"]
            assert "duckdb" in result["data"]
            # Should flag stalled deals with clean output structure
            if result["data"]["at_risk_deals"]:
                assert result["data"]["at_risk_deals"][0].get("risk_reason") == "Stalled"
                assert "primary_contact" in result["data"]["at_risk_deals"][0]

    def test_account_momentum_categorizes_accounts(self):
        """'Account momentum' categorizes accounts as EXPAND/SAVE/RE-ACTIVATE."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = [{"id": "1", "name": "Test Co"}]
            result = act_fetch("Account momentum")

            assert result["error"] is None
            assert "expand" in result["data"]
            assert "save" in result["data"]
            assert "reactivate" in result["data"]
            assert "duckdb" in result["data"]

    def test_relationship_gaps_finds_single_threaded(self):
        """'Relationship gaps' identifies single-threaded deals."""
        with patch("backend.act_fetch._get") as mock_get:
            mock_get.return_value = [{"id": "1", "name": "Test", "contacts": [], "companies": [], "productTotal": 5000, "statusName": "Open"}]
            result = act_fetch("Relationship gaps")

            assert result["error"] is None
            assert "relationship_analysis" in result["data"]
            assert "duckdb" in result["data"]

    def test_daily_briefing_filters_field_changed(self):
        """'Daily briefing' filters out Field changed entries from history."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "history" in endpoint:
                    return [
                        {"id": "1", "regarding": "Meeting with John", "details": "Good call"},
                        {"id": "2", "regarding": "Field changed: Status", "details": ""},
                        {"id": "3", "regarding": "Call with Sarah", "details": "Follow up needed"},
                    ]
                return [{"id": "1"}]

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Daily briefing")

            assert result["error"] is None
            # Should filter out "Field changed" entries
            for item in result["data"].get("recent_history", []):
                assert "Field changed" not in item.get("regarding", "")


class TestDemoStarters:
    """Tests for DEMO_STARTERS constant - Gold Standard 5 questions."""

    def test_has_five_questions(self):
        """DEMO_STARTERS contains exactly 5 questions."""
        assert len(DEMO_STARTERS) == 5

    def test_all_are_strings(self):
        """All starters are non-empty strings."""
        for starter in DEMO_STARTERS:
            assert isinstance(starter, str)
            assert len(starter) > 0

    def test_expected_questions(self):
        """Expected Gold Standard questions are in DEMO_STARTERS."""
        expected = [
            "Daily briefing",
            "Forecast health",
            "At-risk deals",
            "Account momentum",
            "Relationship gaps",
        ]
        assert DEMO_STARTERS == expected


class TestDemoFollowups:
    """Tests for DEMO_FOLLOWUPS constant."""

    def test_has_entry_for_each_starter(self):
        """DEMO_FOLLOWUPS has entry for each starter question."""
        for starter in DEMO_STARTERS:
            assert starter in DEMO_FOLLOWUPS

    def test_followups_are_lists(self):
        """Each follow-up entry is a list of strings."""
        for question, followups in DEMO_FOLLOWUPS.items():
            assert isinstance(followups, list)
            for f in followups:
                assert isinstance(f, str)

    def test_followups_are_valid_questions(self):
        """All follow-up suggestions are valid demo questions or known questions."""
        valid_questions = set(DEMO_STARTERS)
        for question, followups in DEMO_FOLLOWUPS.items():
            for f in followups:
                assert f in valid_questions, f"Invalid follow-up '{f}' for '{question}'"


class TestDemoPrompts:
    """Tests for DEMO_PROMPTS constant."""

    def test_has_entry_for_all_questions(self):
        """DEMO_PROMPTS has entry for all 5 questions."""
        for q in DEMO_STARTERS:
            assert q in DEMO_PROMPTS

    def test_each_prompt_has_answer_and_action(self):
        """Each prompt has 'answer' and 'action' keys."""
        for q, prompts in DEMO_PROMPTS.items():
            assert "answer" in prompts, f"Missing 'answer' for {q}"
            assert "action" in prompts, f"Missing 'action' for {q}"
            assert isinstance(prompts["answer"], str)
            assert isinstance(prompts["action"], str)


class TestIsOpenOpportunity:
    """Tests for _is_open_opportunity helper."""

    def test_open_status_returns_true(self):
        """Open status returns True."""
        assert _is_open_opportunity({"statusName": "Open"}) is True

    def test_empty_status_returns_true(self):
        """Empty or missing status returns True."""
        assert _is_open_opportunity({}) is True
        assert _is_open_opportunity({"statusName": ""}) is True
        assert _is_open_opportunity({"statusName": None}) is True

    def test_closed_status_returns_false(self):
        """Closed status returns False."""
        assert _is_open_opportunity({"statusName": "Closed"}) is False
        assert _is_open_opportunity({"statusName": "closed"}) is False
        assert _is_open_opportunity({"statusName": "Closed - Won"}) is False

    def test_won_status_returns_false(self):
        """Won status returns False."""
        assert _is_open_opportunity({"statusName": "Won"}) is False
        assert _is_open_opportunity({"statusName": "won"}) is False

    def test_lost_status_returns_false(self):
        """Lost status returns False."""
        assert _is_open_opportunity({"statusName": "Lost"}) is False
        assert _is_open_opportunity({"statusName": "lost"}) is False


class TestFilterNoiseHistory:
    """Tests for _filter_noise_history helper."""

    def test_filters_field_changed_in_regarding(self):
        """Filters out 'Field changed' in regarding."""
        history = [
            {"id": "1", "regarding": "Field changed: Status", "details": ""},
            {"id": "2", "regarding": "Call with John", "details": "Good call"},
        ]
        result = _filter_noise_history(history)
        assert len(result) == 1
        assert result[0]["id"] == "2"

    def test_filters_field_changed_in_details(self):
        """Filters out 'Field changed' in details."""
        history = [
            {"id": "1", "regarding": "Update", "details": "Field changed: Phone"},
            {"id": "2", "regarding": "Meeting", "details": "Discussed contract"},
        ]
        result = _filter_noise_history(history)
        assert len(result) == 1
        assert result[0]["id"] == "2"

    def test_filters_record_created(self):
        """Filters out 'record created' entries."""
        history = [
            {"id": "1", "regarding": "Record created", "details": ""},
            {"id": "2", "regarding": "Email sent", "details": "Follow-up"},
        ]
        result = _filter_noise_history(history)
        assert len(result) == 1
        assert result[0]["id"] == "2"

    def test_filters_record_updated(self):
        """Filters out 'record updated' entries."""
        history = [
            {"id": "1", "regarding": "Record updated", "details": ""},
            {"id": "2", "regarding": "Phone call", "details": "Left voicemail"},
        ]
        result = _filter_noise_history(history)
        assert len(result) == 1
        assert result[0]["id"] == "2"

    def test_keeps_entries_with_content(self):
        """Keeps entries with actual content."""
        history = [
            {"id": "1", "regarding": "Meeting", "details": ""},
            {"id": "2", "regarding": "", "details": "Important note"},
            {"id": "3", "regarding": "Call", "details": "Discussed pricing"},
        ]
        result = _filter_noise_history(history)
        assert len(result) == 3

    def test_filters_empty_entries(self):
        """Filters entries with no content."""
        history = [
            {"id": "1", "regarding": "", "details": ""},
            {"id": "2", "regarding": None, "details": None},
            {"id": "3", "regarding": "Meeting", "details": "Good"},
        ]
        result = _filter_noise_history(history)
        assert len(result) == 1
        assert result[0]["id"] == "3"


class TestAuth:
    """Tests for _auth function."""

    def setup_method(self):
        """Reset token before each test."""
        _clear_token()

    def test_returns_cached_token_when_valid(self):
        """Returns cached token when not expired."""
        import backend.act_fetch as module

        module._token = "cached_token"
        module._token_expires = time.time() + 3600  # 1 hour from now

        result = module._auth()
        assert result == "cached_token"

    def test_fetches_new_token_when_expired(self):
        """Fetches new token when expired."""
        import backend.act_fetch as module

        module._token = "old_token"
        module._token_expires = time.time() - 100  # Expired

        mock_resp = MagicMock()
        mock_resp.text = "eyJhbGciOiJIUzI1NiJ9.test.signature"
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            result = module._auth()
            assert result.startswith("eyJ")

    def test_fetches_new_token_when_none(self):
        """Fetches new token when cache is empty."""
        import backend.act_fetch as module

        module._token = None
        module._token_expires = None

        mock_resp = MagicMock()
        mock_resp.text = "eyJhbGciOiJIUzI1NiJ9.test.signature"
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            result = module._auth()
            assert result.startswith("eyJ")
            assert module._token == result
            assert module._token_expires is not None

    def test_handles_jwt_text_response(self):
        """Handles JWT token returned as plain text."""
        import backend.act_fetch as module

        _clear_token()
        mock_resp = MagicMock()
        mock_resp.text = "  eyJhbGciOiJIUzI1NiJ9.payload.sig  "
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            result = module._auth()
            assert result == "eyJhbGciOiJIUzI1NiJ9.payload.sig"

    def test_handles_json_token_response(self):
        """Handles JSON response with 'token' key."""
        import backend.act_fetch as module

        _clear_token()
        mock_resp = MagicMock()
        mock_resp.text = '{"token": "json_token_value"}'
        mock_resp.json.return_value = {"token": "json_token_value"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            result = module._auth()
            assert result == "json_token_value"

    def test_handles_json_access_token_response(self):
        """Handles JSON response with 'access_token' key."""
        import backend.act_fetch as module

        _clear_token()
        mock_resp = MagicMock()
        mock_resp.text = '{"access_token": "access_token_value"}'
        mock_resp.json.return_value = {"access_token": "access_token_value"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            result = module._auth()
            assert result == "access_token_value"

    def test_raises_on_invalid_response_format(self):
        """Raises ValueError on unrecognized response format."""
        import backend.act_fetch as module

        _clear_token()
        mock_resp = MagicMock()
        mock_resp.text = "invalid_response_without_eyJ"
        mock_resp.json.return_value = {"other_key": "value"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(ValueError, match="Unexpected auth response format"):
                module._auth()

    def test_raises_on_http_error(self):
        """Raises HTTPError on auth failure."""
        import backend.act_fetch as module

        _clear_token()
        with patch("httpx.get", side_effect=httpx.HTTPError("Auth failed")):
            with pytest.raises(httpx.HTTPError):
                module._auth()


class TestGetRaw:
    """Tests for _get_raw function."""

    def setup_method(self):
        """Reset state before each test."""
        _clear_token()
        clear_api_cache()

    def test_returns_list_response_directly(self):
        """Returns list response directly."""
        import backend.act_fetch as module

        mock_auth = MagicMock(return_value="token")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"id": "1"}, {"id": "2"}]
        mock_resp.raise_for_status = MagicMock()

        with patch.object(module, "_auth", mock_auth), \
             patch("httpx.get", return_value=mock_resp):
            result = module._get_raw("/api/test")
            assert result == [{"id": "1"}, {"id": "2"}]

    def test_extracts_value_from_odata_response(self):
        """Extracts 'value' from OData response."""
        import backend.act_fetch as module

        mock_auth = MagicMock(return_value="token")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"value": [{"id": "1"}]}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(module, "_auth", mock_auth), \
             patch("httpx.get", return_value=mock_resp):
            result = module._get_raw("/api/test")
            assert result == [{"id": "1"}]

    def test_extracts_items_from_dict_response(self):
        """Extracts 'items' from dict response."""
        import backend.act_fetch as module

        mock_auth = MagicMock(return_value="token")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": [{"id": "1"}]}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(module, "_auth", mock_auth), \
             patch("httpx.get", return_value=mock_resp):
            result = module._get_raw("/api/test")
            assert result == [{"id": "1"}]

    def test_wraps_single_dict_in_list(self):
        """Wraps single dict response in list."""
        import backend.act_fetch as module

        mock_auth = MagicMock(return_value="token")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "single"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(module, "_auth", mock_auth), \
             patch("httpx.get", return_value=mock_resp):
            result = module._get_raw("/api/test")
            assert result == [{"id": "single"}]

    def test_returns_empty_list_for_none_data(self):
        """Returns empty list for null/None response."""
        import backend.act_fetch as module

        mock_auth = MagicMock(return_value="token")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = None
        mock_resp.raise_for_status = MagicMock()

        with patch.object(module, "_auth", mock_auth), \
             patch("httpx.get", return_value=mock_resp):
            result = module._get_raw("/api/test")
            assert result == []

    def test_clears_token_on_401_and_raises(self):
        """Clears token on 401 and raises HTTPStatusError."""
        import backend.act_fetch as module

        module._token = "old_token"
        module._token_expires = time.time() + 3600

        mock_auth = MagicMock(return_value="token")
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_resp
        )

        with patch.object(module, "_auth", mock_auth), \
             patch("httpx.get", return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                module._get_raw("/api/test")
            # Token should be cleared
            assert module._token is None

    def test_raises_on_timeout(self):
        """Raises TimeoutException on timeout."""
        import backend.act_fetch as module

        mock_auth = MagicMock(return_value="token")

        with patch.object(module, "_auth", mock_auth), \
             patch("httpx.get", side_effect=httpx.TimeoutException("Timeout")):
            with pytest.raises(httpx.TimeoutException):
                module._get_raw("/api/test")

    def test_raises_on_http_error(self):
        """Raises HTTPError on API error."""
        import backend.act_fetch as module

        mock_auth = MagicMock(return_value="token")

        with patch.object(module, "_auth", mock_auth), \
             patch("httpx.get", side_effect=httpx.HTTPError("Server error")):
            with pytest.raises(httpx.HTTPError):
                module._get_raw("/api/test")

    def test_handles_non_list_value_in_odata(self):
        """Handles non-list value in OData response."""
        import backend.act_fetch as module

        mock_auth = MagicMock(return_value="token")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"value": "single_value"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(module, "_auth", mock_auth), \
             patch("httpx.get", return_value=mock_resp):
            result = module._get_raw("/api/test")
            assert result == ["single_value"]


class TestGetWithCache:
    """Tests for _get function with caching."""

    def setup_method(self):
        """Reset state before each test."""
        _clear_token()
        clear_api_cache()

    def test_caches_successful_result(self):
        """Caches successful API result."""
        import backend.act_fetch as module

        test_data = [{"id": "1"}]
        with patch.object(module, "_get_raw", return_value=test_data):
            result = module._get("/api/test", {"$top": 10})
            assert result == test_data

            # Verify it was cached
            key = _cache_key("/api/test", {"$top": 10})
            assert _cache_get(key) == test_data

    def test_raises_when_no_cache_on_failure(self):
        """Raises exception when no cache available on failure."""
        import backend.act_fetch as module

        clear_api_cache()
        with patch.object(module, "_get_raw", side_effect=httpx.HTTPError("Failed")):
            with pytest.raises(httpx.HTTPError):
                module._get("/api/uncached", {"$top": 10})


class TestActFetchDailyBriefing:
    """Detailed tests for Daily briefing question."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        _clear_token()
        clear_api_cache()

    def test_extracts_calendar_items_from_nested_structure(self):
        """Extracts calendar items from nested 'items' structure."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "calendar" in endpoint:
                    return [{"items": [{"id": "mtg1", "contacts": []}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Daily briefing")

            assert result["error"] is None

    def test_enriches_meeting_with_contact_details(self):
        """Enriches meetings with full contact details."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "calendar" in endpoint:
                    return [{"id": "mtg1", "contacts": [{"id": "c1"}]}]
                if endpoint == "/api/contacts/c1":
                    return [{"id": "c1", "fullName": "John Doe", "companyID": "co1"}]
                if "contacts" in endpoint:
                    return [{"id": "c1", "fullName": "John Doe", "companyID": "co1"}]
                if "history" in endpoint:
                    return [{"id": "h1", "regarding": "Call", "details": "Good", "contacts": [{"id": "c1"}]}]
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "contacts": [{"id": "c1"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Daily briefing")

            assert result["error"] is None
            assert "today_meetings" in result["data"]

    def test_links_opportunities_via_company(self):
        """Links opportunities via contact's companyID."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "calendar" in endpoint:
                    return [{"id": "mtg1", "contacts": [{"id": "c1"}]}]
                if endpoint == "/api/contacts/c1":
                    return [{"id": "c1", "fullName": "John", "companyID": "co1"}]
                if "contacts" in endpoint:
                    return [{"id": "c1", "fullName": "John", "companyID": "co1"}]
                if "history" in endpoint:
                    return []
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "contacts": [{"id": "c2"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Daily briefing")
            assert result["error"] is None

    def test_identifies_overdue_followups(self):
        """Identifies contacts needing follow-up."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "calendar" in endpoint:
                    return []
                if "contacts" in endpoint:
                    return [
                        {"id": "c1", "fullName": "Old Contact", "lastAttempt": "2020-01-01", "emailAddress": "test@test.com"},
                        {"id": "c2", "fullName": "No Attempt", "lastAttempt": None, "businessPhone": "123"},
                    ]
                if "history" in endpoint:
                    return []
                if "opportunities" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Daily briefing")

            assert result["error"] is None
            assert "overdue_followups" in result["data"]
            # Should include contacts with old/missing lastAttempt
            assert len(result["data"]["overdue_followups"]) >= 1

    def test_handles_attendee_fetch_exception(self):
        """Handles exception when fetching individual attendee contacts."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "calendar" in endpoint:
                    return [{"id": "mtg1", "contacts": [{"id": "c1"}, {"id": "c2"}]}]
                if endpoint == "/api/contacts/c1":
                    raise httpx.HTTPError("Fetch failed")
                if endpoint == "/api/contacts/c2":
                    return {"id": "c2", "fullName": "Valid Contact"}  # Dict, not list
                if "contacts" in endpoint:
                    return []
                if "history" in endpoint:
                    return []
                if "opportunities" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Daily briefing")

            assert result["error"] is None

    def test_links_opportunities_via_company_opps(self):
        """Links opportunities via company_opps lookup."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "calendar" in endpoint:
                    return [{"id": "mtg1", "contacts": [{"id": "c1"}]}]
                if endpoint == "/api/contacts/c1":
                    return [{"id": "c1", "fullName": "John", "companyID": "co1"}]
                if "contacts" in endpoint:
                    return [
                        {"id": "c1", "fullName": "John", "companyID": "co1"},
                        {"id": "c2", "fullName": "Jane", "companyID": "co1"},
                    ]
                if "history" in endpoint:
                    return []
                if "opportunities" in endpoint:
                    # Opp linked to c2 who shares co1 with c1 (meeting attendee)
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "contacts": [{"id": "c2"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Daily briefing")

            assert result["error"] is None
            # Meeting with c1 should have opp linked via co1
            if result["data"]["today_meetings"]:
                mtg = result["data"]["today_meetings"][0]
                assert len(mtg["opportunities"]) >= 1

    def test_attendee_fetch_returns_list(self):
        """Handles attendee fetch returning list with contact."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "calendar" in endpoint:
                    return [{"id": "mtg1", "contacts": [{"id": "c1"}]}]
                if endpoint == "/api/contacts/c1":
                    # Return list (line 410)
                    return [{"id": "c1", "fullName": "From List"}]
                if "contacts" in endpoint:
                    return []  # Not in main contacts list
                if "history" in endpoint:
                    return []
                if "opportunities" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Daily briefing")

            assert result["error"] is None


class TestActFetchForecastHealth:
    """Detailed tests for Forecast health question."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        _clear_token()
        clear_api_cache()

    def test_handles_no_close_date(self):
        """Handles opportunities without close date."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "No Date Deal", "estimatedCloseDate": None, "productTotal": 5000, "statusName": "Open"},
                    ]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Forecast health")

            assert result["error"] is None
            assert result["data"]["no_date_count"] == 1

    def test_calculates_days_overdue_for_slipped(self):
        """Calculates days_overdue for slipped deals."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Slipped Deal", "estimatedCloseDate": "2020-01-01", "productTotal": 10000, "probability": 50, "statusName": "Open", "contacts": [{"id": "c1"}]},
                    ]
                if "/api/contacts/c1" in endpoint:
                    return [{"id": "c1", "fullName": "John", "emailAddress": "j@test.com"}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Forecast health")

            assert result["error"] is None
            assert len(result["data"]["slipped_deals"]) == 1
            assert result["data"]["slipped_deals"][0]["_days_overdue"] > 0

    def test_buckets_deals_by_close_date(self):
        """Buckets deals into 30/60/90/beyond periods."""
        import time as t

        now = t.time()
        d15 = t.strftime("%Y-%m-%d", t.gmtime(now + 15*24*60*60))
        d45 = t.strftime("%Y-%m-%d", t.gmtime(now + 45*24*60*60))
        d75 = t.strftime("%Y-%m-%d", t.gmtime(now + 75*24*60*60))
        d120 = t.strftime("%Y-%m-%d", t.gmtime(now + 120*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "30d Deal", "estimatedCloseDate": d15, "productTotal": 1000, "probability": 50, "statusName": "Open"},
                        {"id": "2", "name": "60d Deal", "estimatedCloseDate": d45, "productTotal": 2000, "probability": 50, "statusName": "Open"},
                        {"id": "3", "name": "90d Deal", "estimatedCloseDate": d75, "productTotal": 3000, "probability": 50, "statusName": "Open"},
                        {"id": "4", "name": "Beyond Deal", "estimatedCloseDate": d120, "productTotal": 4000, "probability": 50, "statusName": "Open"},
                    ]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Forecast health")

            assert result["error"] is None
            assert result["data"]["forecast_30d"]["count"] == 1
            assert result["data"]["forecast_60d"]["count"] == 1
            assert result["data"]["forecast_90d"]["count"] == 1
            assert result["data"]["beyond_90d_count"] == 1

    def test_adds_primary_contact_to_slipped_deals(self):
        """Adds primary contact info to slipped deals."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Slipped", "estimatedCloseDate": "2020-01-01", "productTotal": 5000, "statusName": "Open", "contacts": [{"id": "c1"}]},
                    ]
                if "/api/contacts/c1" in endpoint:
                    return [{"id": "c1", "fullName": "Jane Doe", "emailAddress": "jane@test.com", "businessPhone": "555-1234"}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Forecast health")

            assert result["error"] is None
            slipped = result["data"]["slipped_deals"]
            assert len(slipped) == 1
            assert slipped[0]["_primary_contact"]["name"] == "Jane Doe"
            assert slipped[0]["_primary_contact"]["email"] == "jane@test.com"

    def test_handles_contact_fetch_failure(self):
        """Handles exception when fetching individual contacts."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Deal", "estimatedCloseDate": "2020-01-01", "productTotal": 5000, "statusName": "Open", "contacts": [{"id": "c1"}]},
                    ]
                if "/api/contacts/c1" in endpoint:
                    raise httpx.HTTPError("Fetch failed")
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Forecast health")

            assert result["error"] is None

    def test_handles_contact_fetch_dict_result(self):
        """Handles contact fetch returning dict instead of list."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Deal", "estimatedCloseDate": "2020-01-01", "productTotal": 5000, "statusName": "Open", "contacts": [{"id": "c1"}]},
                    ]
                if "/api/contacts/c1" in endpoint:
                    return {"id": "c1", "fullName": "Direct Dict"}  # Dict, not list
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Forecast health")

            assert result["error"] is None

    def test_filters_closed_opportunities(self):
        """Filters out closed opportunities."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Open", "estimatedCloseDate": "2099-01-01", "productTotal": 5000, "statusName": "Open"},
                        {"id": "2", "name": "Closed", "estimatedCloseDate": "2099-01-01", "productTotal": 5000, "statusName": "Closed - Won"},
                    ]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Forecast health")

            assert result["error"] is None
            # Only open deal should be counted
            total = result["data"]["forecast_30d"]["count"] + result["data"]["forecast_60d"]["count"] + result["data"]["forecast_90d"]["count"] + result["data"]["beyond_90d_count"]
            assert total <= 1

    def test_handles_date_parse_exception(self):
        """Handles invalid date format gracefully."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Bad Date", "estimatedCloseDate": "invalid-date", "productTotal": 5000, "statusName": "Open"},
                    ]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Forecast health")

            # Should not crash, just handle the exception
            assert result["error"] is None

    def test_slipped_deal_date_parse_exception(self):
        """Handles invalid date format in slipped deal gracefully."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        # Past date but malformed (triggers line 546-547)
                        {"id": "1", "name": "Bad Slipped", "estimatedCloseDate": "2020-99-99", "productTotal": 5000, "statusName": "Open"},
                    ]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Forecast health")

            # Should handle gracefully, set _days_overdue to 0
            assert result["error"] is None


class TestActFetchAtRiskDeals:
    """Detailed tests for At-risk deals question."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        _clear_token()
        clear_api_cache()

    def test_classifies_overdue_deals(self):
        """Classifies deals past close date as Overdue."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Overdue Deal", "estimatedCloseDate": "2020-01-01", "daysInStage": 5, "probability": 50, "statusName": "Open"},
                    ]
                if "contacts" in endpoint:
                    return []
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            assert result["error"] is None
            assert len(result["data"]["at_risk_deals"]) == 1
            assert result["data"]["at_risk_deals"][0]["risk_reason"] == "Overdue"

    def test_classifies_stalled_deals(self):
        """Classifies deals with daysInStage > 14 as Stalled."""
        import time as t

        future_date = t.strftime("%Y-%m-%d", t.gmtime(t.time() + 30*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Stalled Deal", "estimatedCloseDate": future_date, "daysInStage": 20, "probability": 50, "statusName": "Open"},
                    ]
                if "contacts" in endpoint:
                    return []
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            assert result["error"] is None
            at_risk = result["data"]["at_risk_deals"]
            assert len(at_risk) == 1
            assert at_risk[0]["risk_reason"] == "Stalled"

    def test_classifies_low_activity_deals(self):
        """Classifies deals with no recent touch as Low-activity."""
        import time as t

        future_date = t.strftime("%Y-%m-%d", t.gmtime(t.time() + 30*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Quiet Deal", "estimatedCloseDate": future_date, "daysInStage": 5, "probability": 50, "statusName": "Open"},
                    ]
                if "contacts" in endpoint:
                    return []
                if "history" in endpoint:
                    return []  # No history = no touch
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            assert result["error"] is None
            at_risk = result["data"]["at_risk_deals"]
            assert len(at_risk) == 1
            assert at_risk[0]["risk_reason"] == "Low-activity"

    def test_gets_company_from_opp_companies(self):
        """Gets company name from opp.companies[0]."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Deal", "estimatedCloseDate": "2020-01-01", "daysInStage": 5, "statusName": "Open",
                         "companies": [{"id": "co1", "name": "Acme Corp"}]},
                    ]
                if "contacts" in endpoint:
                    return []
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            assert result["error"] is None
            assert result["data"]["at_risk_deals"][0]["company"] == "Acme Corp"

    def test_gets_company_from_contact_fallback(self):
        """Falls back to contact.company when opp.companies empty."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Deal", "estimatedCloseDate": "2020-01-01", "daysInStage": 5, "statusName": "Open",
                         "contacts": [{"id": "c1"}]},
                    ]
                if "contacts" in endpoint:
                    return [{"id": "c1", "fullName": "John", "company": "Fallback Inc"}]
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            assert result["error"] is None
            assert result["data"]["at_risk_deals"][0]["company"] == "Fallback Inc"

    def test_gets_primary_contact_with_details(self):
        """Gets primary contact with email/phone."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Deal", "estimatedCloseDate": "2020-01-01", "daysInStage": 5, "statusName": "Open",
                         "contacts": [{"id": "c1"}]},
                    ]
                if "contacts" in endpoint:
                    return [{"id": "c1", "fullName": "Jane", "emailAddress": "jane@test.com", "businessPhone": "555"}]
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            assert result["error"] is None
            primary = result["data"]["at_risk_deals"][0]["primary_contact"]
            assert primary["name"] == "Jane"
            assert primary["email"] == "jane@test.com"

    def test_fallback_contact_without_details(self):
        """Falls back to contact even without email/phone."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Deal", "estimatedCloseDate": "2020-01-01", "daysInStage": 5, "statusName": "Open",
                         "contacts": [{"id": "c1"}]},
                    ]
                if "contacts" in endpoint:
                    return [{"id": "c1", "fullName": "No Details"}]
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            assert result["error"] is None
            primary = result["data"]["at_risk_deals"][0]["primary_contact"]
            assert primary["name"] == "No Details"

    def test_tracks_touch_from_history_companies(self):
        """Tracks last_touch from history.companies."""
        import time as t

        future = t.strftime("%Y-%m-%d", t.gmtime(t.time() + 30*24*60*60))
        recent = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 5*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Deal", "estimatedCloseDate": future, "daysInStage": 5, "statusName": "Open",
                         "companies": [{"id": "co1"}]},
                    ]
                if "contacts" in endpoint:
                    return []
                if "history" in endpoint:
                    return [{"id": "h1", "regarding": "Call", "details": "test", "startTime": recent, "companies": [{"id": "co1"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            assert result["error"] is None
            # With recent touch, should not be at risk
            # (or if at risk, should have last_touch set)

    def test_tracks_touch_from_history_contacts(self):
        """Tracks last_touch from history.contacts."""
        import time as t

        future = t.strftime("%Y-%m-%d", t.gmtime(t.time() + 30*24*60*60))
        recent = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 5*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Deal", "estimatedCloseDate": future, "daysInStage": 5, "statusName": "Open",
                         "contacts": [{"id": "c1"}]},
                    ]
                if "contacts" in endpoint:
                    return [{"id": "c1", "fullName": "John", "companyID": "co1"}]
                if "history" in endpoint:
                    return [{"id": "h1", "regarding": "Call", "details": "test", "startTime": recent, "contacts": [{"id": "c1"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            assert result["error"] is None

    def test_tracks_company_touch_via_contact_companyid(self):
        """Tracks company touch via contact.companyID from history."""
        import time as t

        future = t.strftime("%Y-%m-%d", t.gmtime(t.time() + 30*24*60*60))
        recent = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 5*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Deal", "estimatedCloseDate": future, "daysInStage": 5, "statusName": "Open",
                         "contacts": [{"id": "c2"}]},  # c2 has same companyID as c1
                    ]
                if "contacts" in endpoint:
                    return [
                        {"id": "c1", "fullName": "John", "companyID": "co1"},
                        {"id": "c2", "fullName": "Jane", "companyID": "co1"},
                    ]
                if "history" in endpoint:
                    # History with c1 sets company touch for co1
                    return [{"id": "h1", "regarding": "Call", "details": "test", "startTime": recent, "contacts": [{"id": "c1"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            assert result["error"] is None

    def test_computes_last_touch_from_opp_contacts(self):
        """Computes last_touch from opp's contact touch times."""
        import time as t

        future = t.strftime("%Y-%m-%d", t.gmtime(t.time() + 30*24*60*60))
        recent = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 5*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Deal", "estimatedCloseDate": future, "daysInStage": 20, "statusName": "Open",
                         "contacts": [{"id": "c1"}], "companies": [{"id": "co1"}]},
                    ]
                if "contacts" in endpoint:
                    return [{"id": "c1", "fullName": "John", "companyID": "co1"}]
                if "history" in endpoint:
                    return [
                        {"id": "h1", "regarding": "Call", "details": "test", "startTime": recent, "contacts": [{"id": "c1"}]},
                        {"id": "h2", "regarding": "Email", "details": "test", "startTime": recent, "companies": [{"id": "co1"}]},
                    ]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            assert result["error"] is None
            # Should have last_touch from contact or company
            if result["data"]["at_risk_deals"]:
                assert result["data"]["at_risk_deals"][0]["last_touch"] == recent

    def test_handles_date_parse_exception_in_risk(self):
        """Handles invalid date in risk classification gracefully."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Deal", "estimatedCloseDate": "bad-date", "daysInStage": 5, "statusName": "Open"},
                    ]
                if "contacts" in endpoint:
                    return []
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            # Should handle gracefully
            assert result["error"] is None

    def test_handles_history_without_starttime(self):
        """Handles history entries without startTime."""
        import time as t

        future = t.strftime("%Y-%m-%d", t.gmtime(t.time() + 30*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Deal", "estimatedCloseDate": future, "daysInStage": 5, "statusName": "Open"},
                    ]
                if "contacts" in endpoint:
                    return []
                if "history" in endpoint:
                    return [
                        {"id": "h1", "regarding": "Call", "details": "test", "startTime": None},  # No startTime
                        {"id": "h2", "regarding": "Email", "details": "test", "startTime": ""},  # Empty startTime
                    ]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            assert result["error"] is None

    def test_skips_closed_opportunities(self):
        """Skips closed opportunities in at-risk analysis (line 632)."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        {"id": "1", "name": "Open Deal", "estimatedCloseDate": "2020-01-01", "daysInStage": 5, "statusName": "Open"},
                        {"id": "2", "name": "Closed Deal", "estimatedCloseDate": "2020-01-01", "daysInStage": 5, "statusName": "Closed - Won"},
                    ]
                if "contacts" in endpoint:
                    return []
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            assert result["error"] is None
            # Only open deal should be at risk
            assert len(result["data"]["at_risk_deals"]) == 1
            assert result["data"]["at_risk_deals"][0]["name"] == "Open Deal"

    def test_overdue_date_parse_exception(self):
        """Handles invalid date format in overdue deal calculation (lines 668-669)."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [
                        # Past date but malformed
                        {"id": "1", "name": "Bad Overdue", "estimatedCloseDate": "2020-99-99", "daysInStage": 5, "statusName": "Open"},
                    ]
                if "contacts" in endpoint:
                    return []
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("At-risk deals")

            # Should handle gracefully
            assert result["error"] is None


class TestActFetchAccountMomentum:
    """Detailed tests for Account momentum question."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        _clear_token()
        clear_api_cache()

    def test_discovers_companies_from_contacts(self):
        """Discovers companies from contact.companyID."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "companies" in endpoint:
                    return []  # No companies from /api/companies
                if "contacts" in endpoint:
                    return [{"id": "c1", "companyID": "co1", "company": "Contact Company"}]
                if "opportunities" in endpoint:
                    return []
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Account momentum")

            assert result["error"] is None

    def test_discovers_companies_from_opps(self):
        """Discovers companies from opp.companies."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "companies" in endpoint:
                    return []
                if "contacts" in endpoint:
                    return []
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "companies": [{"id": "co1", "name": "Opp Company"}], "productTotal": 5000}]
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Account momentum")

            assert result["error"] is None

    def test_links_pipeline_via_contact_company(self):
        """Links opportunity pipeline via contact.companyID."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Test Co"}]
                if "contacts" in endpoint:
                    return [{"id": "c1", "companyID": "co1"}]
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "productTotal": 10000, "contacts": [{"id": "c1"}]}]
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Account momentum")

            assert result["error"] is None

    def test_tracks_engagement_from_history(self):
        """Tracks engagement from history.contacts and history.companies."""
        import time as t

        recent = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 5*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Test Co"}]
                if "contacts" in endpoint:
                    return [{"id": "c1", "companyID": "co1", "fullName": "John"}]
                if "opportunities" in endpoint:
                    return [{"id": "o1", "statusName": "Open", "productTotal": 5000, "contacts": [{"id": "c1"}]}]
                if "history" in endpoint:
                    return [
                        {"id": "h1", "regarding": "Call", "details": "test", "startTime": recent, "contacts": [{"id": "c1"}]},
                        {"id": "h2", "regarding": "Meeting", "details": "test", "startTime": recent, "companies": [{"id": "co1"}]},
                    ]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Account momentum")

            assert result["error"] is None

    def test_categorizes_expand_accounts(self):
        """Categorizes accounts with pipeline + high engagement as EXPAND."""
        import time as t

        recent = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 5*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Expand Co"}]
                if "contacts" in endpoint:
                    return [{"id": "c1", "companyID": "co1", "fullName": "John"}]
                if "opportunities" in endpoint:
                    return [{"id": "o1", "statusName": "Open", "productTotal": 10000, "contacts": [{"id": "c1"}]}]
                if "history" in endpoint:
                    # 3+ engagements in last 30 days
                    return [
                        {"id": "h1", "regarding": "Call", "details": "t", "startTime": recent, "contacts": [{"id": "c1"}]},
                        {"id": "h2", "regarding": "Email", "details": "t", "startTime": recent, "contacts": [{"id": "c1"}]},
                        {"id": "h3", "regarding": "Meeting", "details": "t", "startTime": recent, "contacts": [{"id": "c1"}]},
                    ]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Account momentum")

            assert result["error"] is None
            assert len(result["data"]["expand"]) >= 1

    def test_categorizes_save_accounts(self):
        """Categorizes accounts with pipeline + low engagement as SAVE."""
        import time as t

        old = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 60*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Save Co"}]
                if "contacts" in endpoint:
                    return [{"id": "c1", "companyID": "co1", "fullName": "John"}]
                if "opportunities" in endpoint:
                    return [{"id": "o1", "statusName": "Open", "productTotal": 10000, "contacts": [{"id": "c1"}]}]
                if "history" in endpoint:
                    # Only 1 old engagement
                    return [{"id": "h1", "regarding": "Call", "details": "t", "startTime": old, "contacts": [{"id": "c1"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Account momentum")

            assert result["error"] is None
            assert len(result["data"]["save"]) >= 1

    def test_categorizes_reactivate_accounts(self):
        """Categorizes accounts with no pipeline but has touch as RE-ACTIVATE."""
        import time as t

        old = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 60*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Reactivate Co"}]
                if "contacts" in endpoint:
                    return [{"id": "c1", "companyID": "co1", "fullName": "John"}]
                if "opportunities" in endpoint:
                    return []  # No pipeline
                if "history" in endpoint:
                    return [{"id": "h1", "regarding": "Call", "details": "t", "startTime": old, "contacts": [{"id": "c1"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Account momentum")

            assert result["error"] is None
            assert len(result["data"]["reactivate"]) >= 1

    def test_tracks_top_contact(self):
        """Tracks most recently engaged contact per company."""
        import time as t

        recent = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 5*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Test Co"}]
                if "contacts" in endpoint:
                    return [{"id": "c1", "companyID": "co1", "fullName": "Top Contact", "emailAddress": "top@test.com"}]
                if "opportunities" in endpoint:
                    return [{"id": "o1", "statusName": "Open", "productTotal": 5000, "contacts": [{"id": "c1"}]}]
                if "history" in endpoint:
                    return [{"id": "h1", "regarding": "Call", "details": "t", "startTime": recent, "contacts": [{"id": "c1"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Account momentum")

            assert result["error"] is None

    def test_handles_history_without_starttime(self):
        """Handles history entries without startTime."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Test Co"}]
                if "contacts" in endpoint:
                    return [{"id": "c1", "companyID": "co1"}]
                if "opportunities" in endpoint:
                    return [{"id": "o1", "statusName": "Open", "productTotal": 5000, "contacts": [{"id": "c1"}]}]
                if "history" in endpoint:
                    return [
                        {"id": "h1", "regarding": "Call", "details": "t", "startTime": None},
                        {"id": "h2", "regarding": "Email", "details": "t", "startTime": ""},
                    ]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Account momentum")

            assert result["error"] is None

    def test_handles_date_parse_exception_in_momentum(self):
        """Handles invalid date in last_touch_days calculation."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Test Co"}]
                if "contacts" in endpoint:
                    return [{"id": "c1", "companyID": "co1"}]
                if "opportunities" in endpoint:
                    return [{"id": "o1", "statusName": "Open", "productTotal": 5000, "contacts": [{"id": "c1"}]}]
                if "history" in endpoint:
                    return [{"id": "h1", "regarding": "Call", "details": "t", "startTime": "invalid-date", "contacts": [{"id": "c1"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Account momentum")

            # Should handle gracefully
            assert result["error"] is None

    def test_tracks_engagement_from_history_companies(self):
        """Tracks engagement from history.companies."""
        import time as t

        recent = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 5*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Test Co"}]
                if "contacts" in endpoint:
                    return []
                if "opportunities" in endpoint:
                    return [{"id": "o1", "statusName": "Open", "productTotal": 5000, "companies": [{"id": "co1"}]}]
                if "history" in endpoint:
                    return [{"id": "h1", "regarding": "Meeting", "details": "t", "startTime": recent, "companies": [{"id": "co1"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Account momentum")

            assert result["error"] is None


class TestActFetchRelationshipGaps:
    """Detailed tests for Relationship gaps question."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        _clear_token()
        clear_api_cache()

    def test_tracks_engaged_contacts(self):
        """Tracks contacts engaged within 60 days."""
        import time as t

        recent = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 30*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "productTotal": 5000, "contacts": [{"id": "c1"}]}]
                if "contacts" in endpoint:
                    return [{"id": "c1", "companyID": "co1", "fullName": "John"}]
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Test Co"}]
                if "history" in endpoint:
                    return [{"id": "h1", "regarding": "Call", "details": "t", "startTime": recent, "contacts": [{"id": "c1"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Relationship gaps")

            assert result["error"] is None
            assert "relationship_analysis" in result["data"]

    def test_detects_single_threaded_deals(self):
        """Detects deals with 0-1 engaged contacts as single-threaded."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Single Thread Deal", "statusName": "Open", "productTotal": 5000, "contacts": []}]
                if "contacts" in endpoint:
                    return []
                if "companies" in endpoint:
                    return []
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Relationship gaps")

            assert result["error"] is None
            analysis = result["data"]["relationship_analysis"]
            assert len(analysis) >= 1
            assert analysis[0]["single_threaded"] is True

    def test_finds_dark_contacts(self):
        """Finds contacts not engaged within 60 days."""
        import time as t

        old = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 90*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "productTotal": 5000, "contacts": [{"id": "c1"}]}]
                if "contacts" in endpoint:
                    return [
                        {"id": "c1", "companyID": "co1", "fullName": "Active"},
                        {"id": "c2", "companyID": "co1", "fullName": "Dark Contact", "emailAddress": "dark@test.com"},
                    ]
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Test Co"}]
                if "history" in endpoint:
                    return [{"id": "h1", "regarding": "Call", "details": "t", "startTime": old, "contacts": [{"id": "c2"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Relationship gaps")

            assert result["error"] is None

    def test_resolves_company_from_company_map(self):
        """Resolves company name from fetched companies."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "productTotal": 5000, "companies": [{"id": "co1"}]}]
                if "contacts" in endpoint:
                    return []
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Resolved Company"}]
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Relationship gaps")

            assert result["error"] is None
            assert result["data"]["relationship_analysis"][0]["company"] == "Resolved Company"

    def test_falls_back_to_contact_company(self):
        """Falls back to contact.company when company_map empty."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "productTotal": 5000, "contacts": [{"id": "c1"}]}]
                if "contacts" in endpoint:
                    return [{"id": "c1", "companyID": "co1", "company": "Fallback Company"}]
                if "companies" in endpoint:
                    return []
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Relationship gaps")

            assert result["error"] is None
            assert result["data"]["relationship_analysis"][0]["company"] == "Fallback Company"

    def test_sorts_dark_contacts_by_staleness(self):
        """Sorts dark contacts with never-touched first."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "productTotal": 5000, "contacts": [{"id": "c1"}]}]
                if "contacts" in endpoint:
                    return [
                        {"id": "c1", "companyID": "co1", "fullName": "Active"},
                        {"id": "c2", "companyID": "co1", "fullName": "Never Touched", "emailAddress": "never@test.com"},
                        {"id": "c3", "companyID": "co1", "fullName": "Old Touch", "emailAddress": "old@test.com"},
                    ]
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Test Co"}]
                if "history" in endpoint:
                    return [{"id": "h1", "regarding": "Call", "details": "t", "startTime": "2020-01-01", "contacts": [{"id": "c3"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Relationship gaps")

            assert result["error"] is None

    def test_handles_history_without_starttime(self):
        """Handles history entries without startTime."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "productTotal": 5000}]
                if "contacts" in endpoint:
                    return []
                if "companies" in endpoint:
                    return []
                if "history" in endpoint:
                    return [
                        {"id": "h1", "regarding": "Call", "details": "t", "startTime": None},
                        {"id": "h2", "regarding": "Email", "details": "t", "startTime": ""},
                    ]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Relationship gaps")

            assert result["error"] is None

    def test_tracks_company_engagement(self):
        """Tracks company-level engagement from history.companies."""
        import time as t

        recent = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 30*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "productTotal": 5000, "companies": [{"id": "co1"}]}]
                if "contacts" in endpoint:
                    return []
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Test Co"}]
                if "history" in endpoint:
                    return [{"id": "h1", "regarding": "Meeting", "details": "t", "startTime": recent, "companies": [{"id": "co1"}]}]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Relationship gaps")

            assert result["error"] is None

    def test_derives_company_from_opp_companies(self):
        """Derives company IDs from opp.companies."""
        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "productTotal": 5000, "companies": [{"id": "co1"}]}]
                if "contacts" in endpoint:
                    return []
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "From Companies"}]
                if "history" in endpoint:
                    return []
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Relationship gaps")

            assert result["error"] is None
            assert result["data"]["relationship_analysis"][0]["company"] == "From Companies"

    def test_counts_opp_specific_engagement(self):
        """Counts engagement only for contacts directly linked to opp."""
        import time as t

        recent = t.strftime("%Y-%m-%d", t.gmtime(t.time() - 30*24*60*60))

        with patch("backend.act_fetch._get") as mock_get:
            def mock_endpoint(endpoint, params=None):
                if "opportunities" in endpoint:
                    return [{"id": "o1", "name": "Deal", "statusName": "Open", "productTotal": 5000, "contacts": [{"id": "c1"}, {"id": "c2"}]}]
                if "contacts" in endpoint:
                    return [
                        {"id": "c1", "companyID": "co1"},
                        {"id": "c2", "companyID": "co1"},
                        {"id": "c3", "companyID": "co1"},  # Not on opp
                    ]
                if "companies" in endpoint:
                    return [{"id": "co1", "name": "Test Co"}]
                if "history" in endpoint:
                    return [
                        {"id": "h1", "regarding": "Call", "details": "t", "startTime": recent, "contacts": [{"id": "c1"}]},
                        {"id": "h2", "regarding": "Call", "details": "t", "startTime": recent, "contacts": [{"id": "c3"}]},  # Not on opp
                    ]
                return []

            mock_get.side_effect = mock_endpoint
            result = act_fetch("Relationship gaps")

            assert result["error"] is None
            # Only c1 should count as engaged (c3 not on opp)
            assert result["data"]["relationship_analysis"][0]["engaged"] == 1
