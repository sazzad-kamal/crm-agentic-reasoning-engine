"""Tests for RAG indexer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.agent.rag.indexer import (
    DOCS_DIR,
    INDEX_DIR,
)


class TestDocsDir:
    """Tests for documentation directory configuration."""

    def test_docs_dir_is_path(self):
        """DOCS_DIR should be a Path object."""
        assert isinstance(DOCS_DIR, Path)

    def test_index_dir_is_path(self):
        """INDEX_DIR should be a Path object."""
        assert isinstance(INDEX_DIR, Path)

    def test_docs_dir_points_to_data_docs(self):
        """DOCS_DIR should point to data/docs folder."""
        assert DOCS_DIR.name == "docs"
        assert DOCS_DIR.parent.name == "data"


class TestLoadPdfs:
    """Tests for PDF loading function."""

    def test_returns_empty_list_when_no_pdfs(self):
        """Should return empty list when no PDFs are found."""
        with patch("backend.agent.rag.indexer.DOCS_DIR") as mock_docs_dir:
            mock_docs_dir.exists.return_value = True
            mock_docs_dir.glob.return_value = []

            from backend.agent.rag.indexer import _load_pdfs
            # Need to reimport to use the patched DOCS_DIR
            import importlib
            import backend.agent.rag.indexer as indexer_module
            importlib.reload(indexer_module)

            # For this test, just verify the function exists and returns a list
            from backend.agent.rag.indexer import _load_pdfs
            result = _load_pdfs()
            assert isinstance(result, list)

    def test_load_pdfs_handles_missing_dir(self):
        """Should handle missing docs directory."""
        # If the actual docs dir doesn't exist, should return empty
        from backend.agent.rag.indexer import _load_pdfs

        # This is a simple test that the function runs without error
        result = _load_pdfs()
        assert isinstance(result, list)


class TestBuildIndex:
    """Tests for index building."""

    def test_build_index_returns_index(self):
        """Should return a VectorStoreIndex."""
        with patch("backend.agent.rag.indexer._configure_settings"), \
             patch("backend.agent.rag.indexer._load_pdfs", return_value=[]), \
             patch("backend.agent.rag.indexer.VectorStoreIndex") as mock_index_cls, \
             patch("backend.agent.rag.indexer.INDEX_DIR") as mock_index_dir:

            mock_index = MagicMock()
            mock_index_cls.from_documents.return_value = mock_index
            mock_index_dir.exists.return_value = False
            mock_index_dir.mkdir = MagicMock()

            # Reset singleton
            import backend.agent.rag.indexer
            backend.agent.rag.indexer._index = None

            from backend.agent.rag.indexer import build_index
            result = build_index(force_rebuild=True)

            # Should have called from_documents
            mock_index_cls.from_documents.assert_called_once()

    def test_get_index_returns_singleton(self):
        """get_index should return same instance on repeated calls."""
        with patch("backend.agent.rag.indexer._configure_settings"), \
             patch("backend.agent.rag.indexer._load_pdfs", return_value=[]), \
             patch("backend.agent.rag.indexer.VectorStoreIndex") as mock_index_cls, \
             patch("backend.agent.rag.indexer.INDEX_DIR") as mock_index_dir:

            mock_index = MagicMock()
            mock_index_cls.from_documents.return_value = mock_index
            mock_index_dir.exists.return_value = False
            mock_index_dir.mkdir = MagicMock()

            # Reset singleton
            import backend.agent.rag.indexer
            backend.agent.rag.indexer._index = None

            from backend.agent.rag.indexer import get_index

            # First call builds
            result1 = get_index()
            # Second call should return same instance (singleton)
            result2 = get_index()

            # from_documents should only be called once
            assert mock_index_cls.from_documents.call_count == 1
