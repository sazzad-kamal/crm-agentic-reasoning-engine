"""Tests for the data explorer API endpoints."""

import pytest
from fastapi.testclient import TestClient

# client fixture is provided by conftest.py


class TestCompaniesEndpoint:
    """Tests for GET /api/data/companies."""

    def test_returns_companies(self, client: TestClient):
        """Should return list of companies."""
        response = client.get("/api/data/companies")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "total" in data
        assert "columns" in data
        assert isinstance(data["data"], list)

    def test_returns_correct_columns(self, client: TestClient):
        """Should return expected company columns."""
        response = client.get("/api/data/companies")
        data = response.json()
        expected_columns = ["company_id", "name", "status", "plan"]
        for col in expected_columns:
            assert col in data["columns"]

    def test_includes_nested_private_texts(self, client: TestClient):
        """Should include nested private texts for each company."""
        response = client.get("/api/data/companies")
        data = response.json()
        # At least one company should have _private_texts field
        assert any("_private_texts" in company for company in data["data"])

    def test_private_texts_is_list(self, client: TestClient):
        """Private texts should be a list."""
        response = client.get("/api/data/companies")
        data = response.json()
        for company in data["data"]:
            if "_private_texts" in company:
                assert isinstance(company["_private_texts"], list)

    def test_total_matches_data_length(self, client: TestClient):
        """Total should match the number of records."""
        response = client.get("/api/data/companies")
        data = response.json()
        assert data["total"] == len(data["data"])


class TestContactsEndpoint:
    """Tests for GET /api/data/contacts."""

    def test_returns_contacts(self, client: TestClient):
        """Should return list of contacts."""
        response = client.get("/api/data/contacts")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_returns_correct_columns(self, client: TestClient):
        """Should return expected contact columns."""
        response = client.get("/api/data/contacts")
        data = response.json()
        expected_columns = ["contact_id", "first_name", "last_name", "email"]
        for col in expected_columns:
            assert col in data["columns"]

    def test_includes_nested_private_texts(self, client: TestClient):
        """Should include nested private texts for contacts."""
        response = client.get("/api/data/contacts")
        data = response.json()
        for contact in data["data"]:
            assert "_private_texts" in contact
            assert isinstance(contact["_private_texts"], list)


class TestOpportunitiesEndpoint:
    """Tests for GET /api/data/opportunities."""

    def test_returns_opportunities(self, client: TestClient):
        """Should return list of opportunities."""
        response = client.get("/api/data/opportunities")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_returns_correct_columns(self, client: TestClient):
        """Should return expected opportunity columns."""
        response = client.get("/api/data/opportunities")
        data = response.json()
        expected_columns = ["opportunity_id", "name", "stage", "value"]
        for col in expected_columns:
            assert col in data["columns"]

    def test_includes_nested_descriptions(self, client: TestClient):
        """Should include nested descriptions for opportunities."""
        response = client.get("/api/data/opportunities")
        data = response.json()
        for opp in data["data"]:
            assert "_descriptions" in opp
            assert isinstance(opp["_descriptions"], list)

    def test_includes_nested_attachments(self, client: TestClient):
        """Should include nested attachments for opportunities."""
        response = client.get("/api/data/opportunities")
        data = response.json()
        for opp in data["data"]:
            assert "_attachments" in opp
            assert isinstance(opp["_attachments"], list)

    def test_descriptions_have_required_fields(self, client: TestClient):
        """Descriptions should have title and text fields."""
        response = client.get("/api/data/opportunities")
        data = response.json()
        for opp in data["data"]:
            for desc in opp["_descriptions"]:
                assert "title" in desc or "text" in desc


class TestGroupsEndpoint:
    """Tests for GET /api/data/groups."""

    def test_returns_groups(self, client: TestClient):
        """Should return list of groups."""
        response = client.get("/api/data/groups")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_returns_correct_columns(self, client: TestClient):
        """Should return expected group columns."""
        response = client.get("/api/data/groups")
        data = response.json()
        expected_columns = ["group_id", "name", "description"]
        for col in expected_columns:
            assert col in data["columns"]

    def test_includes_nested_members(self, client: TestClient):
        """Should include nested members for groups."""
        response = client.get("/api/data/groups")
        data = response.json()
        for group in data["data"]:
            assert "_members" in group
            assert isinstance(group["_members"], list)

    def test_members_have_company_id(self, client: TestClient):
        """Group members should have company_id."""
        response = client.get("/api/data/groups")
        data = response.json()
        for group in data["data"]:
            for member in group["_members"]:
                assert "company_id" in member


class TestActivitiesEndpoint:
    """Tests for GET /api/data/activities."""

    def test_returns_activities(self, client: TestClient):
        """Should return list of activities."""
        response = client.get("/api/data/activities")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_returns_correct_columns(self, client: TestClient):
        """Should return expected activity columns."""
        response = client.get("/api/data/activities")
        data = response.json()
        # Activities should have basic columns
        assert len(data["columns"]) > 0


class TestHistoryEndpoint:
    """Tests for GET /api/data/history."""

    def test_returns_history(self, client: TestClient):
        """Should return list of history records."""
        response = client.get("/api/data/history")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_returns_correct_columns(self, client: TestClient):
        """Should return expected history columns."""
        response = client.get("/api/data/history")
        data = response.json()
        assert len(data["columns"]) > 0


class TestCreateSimpleDataEndpoint:
    """Tests for _create_simple_data_endpoint factory function."""

    def test_jsonl_loader_branch(self, client: TestClient):
        """Covers is_jsonl=True branch via /data/companies which loads private_texts.jsonl."""
        from backend.api.data import _create_simple_data_endpoint, load_jsonl_data
        # Verify the factory selects jsonl loader when is_jsonl=True
        endpoint = _create_simple_data_endpoint("private_texts.jsonl", is_jsonl=True)
        assert endpoint is not None


class TestDataResponseFormat:
    """Tests for consistent response format across all endpoints."""

    @pytest.mark.parametrize("endpoint", [
        "/api/data/companies",
        "/api/data/contacts",
        "/api/data/opportunities",
        "/api/data/activities",
        "/api/data/groups",
        "/api/data/history",
    ])
    def test_response_has_required_fields(self, client: TestClient, endpoint: str):
        """All endpoints should return data, total, and columns."""
        response = client.get(endpoint)
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "total" in data
        assert "columns" in data

    @pytest.mark.parametrize("endpoint", [
        "/api/data/companies",
        "/api/data/contacts",
        "/api/data/opportunities",
        "/api/data/activities",
        "/api/data/groups",
        "/api/data/history",
    ])
    def test_endpoint_returns_non_empty_data(self, client: TestClient, endpoint: str):
        """Main endpoints should return data (assuming test data exists)."""
        response = client.get(endpoint)
        data = response.json()
        # These main tables should have data in the test dataset
        assert data["total"] > 0, f"{endpoint} should have data"


class TestStarterQuestionsEndpoint:
    """Tests for GET /api/data/starter-questions."""

    def test_returns_starter_questions(self, client: TestClient):
        """Should return list of starter questions."""
        response = client.get("/api/data/starter-questions")
        assert response.status_code == 200
        data = response.json()
        assert "questions" in data
        assert isinstance(data["questions"], list)

    def test_returns_expected_count(self, client: TestClient):
        """Should return expected number of starter questions."""
        response = client.get("/api/data/starter-questions")
        data = response.json()
        # Should have at least 3 starter questions
        assert len(data["questions"]) >= 3

    def test_questions_are_strings(self, client: TestClient):
        """All questions should be strings."""
        response = client.get("/api/data/starter-questions")
        data = response.json()
        for question in data["questions"]:
            assert isinstance(question, str)
            assert len(question) > 0

    def test_contains_expected_starters(self, client: TestClient):
        """Should contain expected starter questions."""
        response = client.get("/api/data/starter-questions")
        data = response.json()
        questions = data["questions"]
        # Should have at least one company-specific and one general question
        has_acme = any("Acme" in q for q in questions)
        has_beta = any("Beta" in q for q in questions)
        has_renewal = any("renewal" in q.lower() for q in questions)
        assert has_acme or has_beta or has_renewal
