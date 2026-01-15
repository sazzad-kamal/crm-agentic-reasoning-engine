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

    def test_includes_nested_private_texts(self, client: TestClient):
        """Should include nested private texts for opportunities."""
        response = client.get("/api/data/opportunities")
        data = response.json()
        for opp in data["data"]:
            assert "_private_texts" in opp
            assert isinstance(opp["_private_texts"], list)

    def test_includes_notes_column(self, client: TestClient):
        """Should include notes column in opportunities."""
        response = client.get("/api/data/opportunities")
        data = response.json()
        assert "notes" in data["columns"]


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


class TestDataResponseFormat:
    """Tests for consistent response format across all endpoints."""

    @pytest.mark.parametrize("endpoint", [
        "/api/data/companies",
        "/api/data/contacts",
        "/api/data/opportunities",
        "/api/data/activities",
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
        "/api/data/history",
    ])
    def test_endpoint_returns_non_empty_data(self, client: TestClient, endpoint: str):
        """Main endpoints should return data (assuming test data exists)."""
        response = client.get(endpoint)
        data = response.json()
        # These main tables should have data in the test dataset
        assert data["total"] > 0, f"{endpoint} should have data"
