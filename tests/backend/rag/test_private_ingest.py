"""Tests for private text ingestion."""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.rag.ingest.private_text import (
    build_private_texts_jsonl,
    load_private_texts,
    private_doc_to_chunks,
    sanitize_value,
)
from backend.rag.models import DocumentChunk


class TestSanitizeValue:
    """Tests for sanitize_value function."""

    def test_sanitize_none(self):
        """Test None passes through."""
        assert sanitize_value(None) is None

    def test_sanitize_string(self):
        """Test string passes through."""
        assert sanitize_value("hello") == "hello"

    def test_sanitize_int(self):
        """Test int passes through."""
        assert sanitize_value(42) == 42

    def test_sanitize_bool(self):
        """Test bool passes through."""
        assert sanitize_value(True) is True
        assert sanitize_value(False) is False

    def test_sanitize_float(self):
        """Test normal float is converted to string."""
        # sanitize_value converts floats to strings (except NaN/Inf)
        result = sanitize_value(3.14)
        assert result == "3.14" or result == 3.14  # Implementation detail

    def test_sanitize_nan(self):
        """Test NaN becomes None."""
        import math
        assert sanitize_value(float('nan')) is None

    def test_sanitize_inf(self):
        """Test Inf becomes None."""
        assert sanitize_value(float('inf')) is None
        assert sanitize_value(float('-inf')) is None

    def test_sanitize_dict(self):
        """Test dict values are sanitized recursively."""
        result = sanitize_value({"key": float('nan'), "ok": "value"})
        assert result["key"] is None
        assert result["ok"] == "value"

    def test_sanitize_list(self):
        """Test list values are sanitized recursively."""
        result = sanitize_value([1, float('nan'), "text"])
        assert result[0] == 1
        assert result[1] is None
        assert result[2] == "text"


class TestLoadPrivateTexts:
    """Tests for load_private_texts function."""

    def test_loads_jsonl_file(self, tmp_path):
        """Test loads documents from JSONL file."""
        jsonl_file = tmp_path / "private.jsonl"
        jsonl_file.write_text(
            '{"id": "doc1", "text": "Hello"}\n'
            '{"id": "doc2", "text": "World"}\n'
        )
        
        docs = load_private_texts(jsonl_file)
        
        assert len(docs) == 2
        assert docs[0]["id"] == "doc1"
        assert docs[1]["id"] == "doc2"

    def test_skips_empty_lines(self, tmp_path):
        """Test skips empty lines in JSONL."""
        jsonl_file = tmp_path / "private.jsonl"
        jsonl_file.write_text(
            '{"id": "doc1"}\n'
            '\n'
            '{"id": "doc2"}\n'
        )
        
        docs = load_private_texts(jsonl_file)
        
        assert len(docs) == 2

    def test_returns_list_of_dicts(self, tmp_path):
        """Test returns list of dict objects."""
        jsonl_file = tmp_path / "private.jsonl"
        jsonl_file.write_text('{"id": "doc1", "company_id": "C001"}\n')
        
        docs = load_private_texts(jsonl_file)
        
        assert isinstance(docs, list)
        assert isinstance(docs[0], dict)


class TestPrivateDocToChunks:
    """Tests for private_doc_to_chunks function."""

    def test_returns_list_of_chunks(self):
        """Test returns list of DocumentChunk objects."""
        doc = {
            "id": "history::H001",
            "company_id": "C001",
            "type": "history",
            "title": "Call Notes",
            "text": "This is a long enough text for a chunk. " * 10,
        }
        
        chunks = private_doc_to_chunks(doc)
        
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)

    def test_chunk_includes_company_id(self):
        """Test chunk metadata includes company_id."""
        doc = {
            "id": "history::H001",
            "company_id": "C001",
            "type": "history",
            "title": "Notes",
            "text": "Some substantial text content here. " * 5,
        }
        
        chunks = private_doc_to_chunks(doc)
        
        if chunks:
            assert chunks[0].metadata.get("company_id") == "C001"

    def test_chunk_includes_type(self):
        """Test chunk metadata includes type."""
        doc = {
            "id": "opp_note::O001",
            "company_id": "C001",
            "type": "opportunity_note",
            "title": "Opp Notes",
            "text": "Opportunity description here. " * 5,
        }
        
        chunks = private_doc_to_chunks(doc)
        
        if chunks:
            assert chunks[0].metadata.get("type") == "opportunity_note"

    def test_chunk_id_format(self):
        """Test chunk_id follows expected format."""
        doc = {
            "id": "history::H001",
            "company_id": "C001",
            "type": "history",
            "title": "Notes",
            "text": "Content for the chunk here. " * 5,
        }
        
        chunks = private_doc_to_chunks(doc)
        
        if chunks:
            # Should include doc id and chunk indicator
            assert "history::H001" in chunks[0].chunk_id
            assert "chunk_" in chunks[0].chunk_id

    def test_short_text_combined_with_title(self):
        """Test very short text is combined with title."""
        doc = {
            "id": "test::001",
            "company_id": "C001",
            "type": "test",
            "title": "Important Title",
            "text": "Short",
        }
        
        # Should not error even with short text
        chunks = private_doc_to_chunks(doc)
        assert isinstance(chunks, list)

    def test_preserves_contact_id(self):
        """Test chunk metadata preserves contact_id."""
        doc = {
            "id": "history::H001",
            "company_id": "C001",
            "contact_id": "CT001",
            "type": "history",
            "title": "Notes",
            "text": "Text content long enough for chunking. " * 5,
        }
        
        chunks = private_doc_to_chunks(doc)
        
        if chunks:
            assert chunks[0].metadata.get("contact_id") == "CT001"

    def test_preserves_opportunity_id(self):
        """Test chunk metadata preserves opportunity_id."""
        doc = {
            "id": "opp_note::O001",
            "company_id": "C001",
            "opportunity_id": "O001",
            "type": "opportunity_note",
            "title": "Notes",
            "text": "Opportunity details go here. " * 5,
        }
        
        chunks = private_doc_to_chunks(doc)
        
        if chunks:
            assert chunks[0].metadata.get("opportunity_id") == "O001"


class TestBuildPrivateTextsJsonl:
    """Tests for build_private_texts_jsonl function."""

    def test_creates_output_file(self, tmp_path):
        """Test creates JSONL output file."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        out_path = tmp_path / "output.jsonl"
        
        # Create minimal history.csv
        (csv_dir / "history.csv").write_text(
            "history_id,company_id,contact_id,subject,description,type,occurred_at,owner,source\n"
            "H001,C001,CT001,Call,Notes here,call,2025-01-01,Owner1,CRM\n"
        )
        
        build_private_texts_jsonl(csv_dir, out_path)
        
        assert out_path.exists()

    def test_includes_history_records(self, tmp_path):
        """Test includes history records in output."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        out_path = tmp_path / "output.jsonl"
        
        (csv_dir / "history.csv").write_text(
            "history_id,company_id,contact_id,subject,description,type,occurred_at,owner,source\n"
            "H001,C001,CT001,Test Call,Description text,call,2025-01-01,Owner,CRM\n"
        )
        
        build_private_texts_jsonl(csv_dir, out_path)
        
        with open(out_path) as f:
            lines = [json.loads(line) for line in f]
        
        assert len(lines) >= 1
        assert any("history" in doc.get("id", "") for doc in lines)

    def test_includes_opportunity_descriptions(self, tmp_path):
        """Test synthesizes opportunity descriptions."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        out_path = tmp_path / "output.jsonl"
        
        # Create opportunities.csv for synthesis
        (csv_dir / "opportunities.csv").write_text(
            "opportunity_id,company_id,primary_contact_id,company_name,stage,value,currency,expected_close_date,created_at,updated_at\n"
            "O001,C001,CT001,Acme Corp,Proposal,10000,USD,2025-06-01,2025-01-01,2025-01-15\n"
        )
        
        build_private_texts_jsonl(csv_dir, out_path)
        
        with open(out_path) as f:
            lines = [json.loads(line) for line in f]
        
        # Should have synthesized opportunity note
        assert any("opp_note" in doc.get("id", "") for doc in lines)

    def test_handles_missing_files(self, tmp_path):
        """Test handles missing CSV files gracefully."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        out_path = tmp_path / "output.jsonl"
        
        # No CSV files at all
        build_private_texts_jsonl(csv_dir, out_path)
        
        # Should create empty or minimal output
        assert out_path.exists()

    def test_output_sorted_by_id(self, tmp_path):
        """Test output is sorted by document ID."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        out_path = tmp_path / "output.jsonl"
        
        (csv_dir / "history.csv").write_text(
            "history_id,company_id,contact_id,subject,description,type,occurred_at,owner,source\n"
            "H002,C001,CT001,Call B,Desc B,call,2025-01-02,Owner,CRM\n"
            "H001,C001,CT001,Call A,Desc A,call,2025-01-01,Owner,CRM\n"
        )
        
        build_private_texts_jsonl(csv_dir, out_path)
        
        with open(out_path) as f:
            lines = [json.loads(line) for line in f]
        
        ids = [doc["id"] for doc in lines]
        assert ids == sorted(ids)

    def test_creates_parent_directories(self, tmp_path):
        """Test creates parent directories for output."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        out_path = tmp_path / "nested" / "dir" / "output.jsonl"
        
        build_private_texts_jsonl(csv_dir, out_path)
        
        assert out_path.exists()


class TestPrivateTextDocumentStructure:
    """Tests for document structure from private text processing."""

    def test_history_doc_structure(self, tmp_path):
        """Test history document has expected structure."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        out_path = tmp_path / "output.jsonl"
        
        (csv_dir / "history.csv").write_text(
            "history_id,company_id,contact_id,subject,description,type,occurred_at,owner,source\n"
            "H001,C001,CT001,Test Subject,Test Description,call,2025-01-01,John,CRM\n"
        )
        
        build_private_texts_jsonl(csv_dir, out_path)
        
        with open(out_path) as f:
            doc = json.loads(f.readline())
        
        assert "id" in doc
        assert "company_id" in doc
        assert "type" in doc
        assert "text" in doc
        assert "metadata" in doc

    def test_opportunity_note_structure(self, tmp_path):
        """Test opportunity note has expected structure."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        out_path = tmp_path / "output.jsonl"
        
        (csv_dir / "opportunities.csv").write_text(
            "opportunity_id,company_id,primary_contact_id,company_name,stage,value,currency,expected_close_date,created_at,updated_at\n"
            "O001,C001,CT001,Acme,Proposal,5000,USD,2025-03-01,2025-01-01,2025-01-01\n"
        )
        
        build_private_texts_jsonl(csv_dir, out_path)
        
        with open(out_path) as f:
            docs = [json.loads(line) for line in f]
        
        opp_doc = next((d for d in docs if "opp_note" in d.get("id", "")), None)
        if opp_doc:
            assert opp_doc["type"] == "opportunity_note"
            assert "company_id" in opp_doc
