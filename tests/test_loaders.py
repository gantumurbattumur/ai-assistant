"""Tests for src/ingest/loaders.py"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.ingest.loaders import (
    SUPPORTED_EXTENSIONS,
    load_all_books,
    load_file,
    _docs_from_epub,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOOKS_DIR = Path("data/books")
VALID_CONTENT_TYPES = {"text", "heading", "table", "image_description"}


def _books_exist() -> bool:
    return BOOKS_DIR.exists() and any(
        Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
        for f in os.listdir(BOOKS_DIR)
    )


# ---------------------------------------------------------------------------
# Unit tests — load_file() with mocked Docling
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_docs() -> list[Document]:
    """A minimal set of Documents covering all four content_types."""
    return [
        Document(page_content="Chapter 1", metadata={"source": "book.pdf", "page": 1, "content_type": "heading"}),
        Document(page_content="Some body text here.", metadata={"source": "book.pdf", "page": 1, "content_type": "text"}),
        Document(page_content="| col1 | col2 |\n|---|---|\n| a | b |", metadata={"source": "book.pdf", "page": 2, "content_type": "table"}),
        Document(page_content="[Image] A diagram of the architecture.", metadata={"source": "book.pdf", "page": 3, "content_type": "image_description"}),
    ]


class TestLoadFile:
    def test_unsupported_extension_returns_empty(self, tmp_path):
        bad_file = tmp_path / "file.xyz"
        bad_file.write_text("content")
        result = load_file(str(bad_file))
        assert result == []

    def test_epub_routes_to_epub_loader(self, tmp_path, fake_docs):
        epub_file = tmp_path / "book.epub"
        epub_file.write_bytes(b"fake epub bytes")

        with patch("src.ingest.loaders._docs_from_epub", return_value=fake_docs) as mock_epub:
            result = load_file(str(epub_file))

        mock_epub.assert_called_once_with(str(epub_file), "book.epub")
        assert result == fake_docs

    def test_pdf_routes_to_docling_loader(self, tmp_path, fake_docs):
        pdf_file = tmp_path / "book.pdf"
        pdf_file.write_bytes(b"%PDF fake")

        with patch("src.ingest.loaders._docs_from_docling", return_value=fake_docs) as mock_docling:
            result = load_file(str(pdf_file))

        mock_docling.assert_called_once_with(str(pdf_file), "book.pdf")
        assert result == fake_docs


# ---------------------------------------------------------------------------
# Unit tests — load_all_books()
# ---------------------------------------------------------------------------


class TestLoadAllBooks:
    def test_missing_directory_returns_empty(self, tmp_path):
        with patch("src.ingest.loaders.FILE_PATH", str(tmp_path / "nonexistent")):
            result = load_all_books()
        assert result == []

    def test_returns_flat_list_of_documents(self, tmp_path, fake_docs):
        """load_all_books() must return a flat list[Document], not list[list[Document]]."""
        # Create one dummy PDF in a temp books dir
        (tmp_path / "book.pdf").write_bytes(b"%PDF fake")

        with (
            patch("src.ingest.loaders.FILE_PATH", str(tmp_path) + "/"),
            patch("src.ingest.loaders.load_file", return_value=fake_docs),
        ):
            result = load_all_books()

        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_skips_unsupported_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("ignore me")
        (tmp_path / "notes.csv").write_text("ignore me too")

        with patch("src.ingest.loaders.FILE_PATH", str(tmp_path) + "/"):
            result = load_all_books()

        assert result == []

    def test_failed_file_is_skipped_gracefully(self, tmp_path, fake_docs):
        """A broken file should not crash the whole ingestion."""
        (tmp_path / "good.pdf").write_bytes(b"%PDF good")
        (tmp_path / "bad.pdf").write_bytes(b"%PDF bad")

        def _fake_load(path: str) -> list[Document]:
            if "bad" in path:
                raise RuntimeError("Simulated parse error")
            return fake_docs

        with (
            patch("src.ingest.loaders.FILE_PATH", str(tmp_path) + "/"),
            patch("src.ingest.loaders.load_file", side_effect=_fake_load),
        ):
            result = load_all_books()

        # Only the good file's docs should be returned
        assert result == fake_docs

    def test_content_types_are_valid(self, tmp_path, fake_docs):
        (tmp_path / "book.pdf").write_bytes(b"%PDF fake")

        with (
            patch("src.ingest.loaders.FILE_PATH", str(tmp_path) + "/"),
            patch("src.ingest.loaders.load_file", return_value=fake_docs),
        ):
            result = load_all_books()

        for doc in result:
            assert doc.metadata["content_type"] in VALID_CONTENT_TYPES

    def test_metadata_fields_present(self, tmp_path, fake_docs):
        (tmp_path / "book.pdf").write_bytes(b"%PDF fake")

        with (
            patch("src.ingest.loaders.FILE_PATH", str(tmp_path) + "/"),
            patch("src.ingest.loaders.load_file", return_value=fake_docs),
        ):
            result = load_all_books()

        for doc in result:
            assert "source" in doc.metadata
            assert "page" in doc.metadata
            assert "content_type" in doc.metadata


# ---------------------------------------------------------------------------
# Unit tests — EPUB fallback loader
# ---------------------------------------------------------------------------


class TestDocsFromEpub:
    def test_epub_extracts_text(self, tmp_path):
        import ebooklib

        mock_item = MagicMock()
        mock_item.get_type.return_value = ebooklib.ITEM_DOCUMENT
        mock_item.get_content.return_value = b"<html><body><p>Hello world</p></body></html>"

        mock_book = MagicMock()
        mock_book.get_items.return_value = [mock_item]

        with patch("src.ingest.loaders.epub.read_epub", return_value=mock_book):
            docs = _docs_from_epub("fake.epub", "fake.epub")

        assert len(docs) == 1
        assert "Hello world" in docs[0].page_content
        assert docs[0].metadata["source"] == "fake.epub"
        assert docs[0].metadata["content_type"] == "text"

    def test_epub_skips_empty_items(self, tmp_path):
        import ebooklib

        mock_item = MagicMock()
        mock_item.get_type.return_value = ebooklib.ITEM_DOCUMENT
        mock_item.get_content.return_value = b"<html><body>   </body></html>"

        mock_book = MagicMock()
        mock_book.get_items.return_value = [mock_item]

        with patch("src.ingest.loaders.epub.read_epub", return_value=mock_book):
            docs = _docs_from_epub("fake.epub", "fake.epub")

        assert docs == []


# ---------------------------------------------------------------------------
# Integration tests — only run if data/books/ has real files
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _books_exist(), reason="No books in data/books/ — skipping integration tests")
class TestIntegration:
    def test_load_all_books_returns_documents(self):
        docs = load_all_books()
        assert len(docs) > 0

    def test_all_docs_are_documents(self):
        docs = load_all_books()
        assert all(isinstance(d, Document) for d in docs)

    def test_content_type_distribution(self):
        docs = load_all_books()
        types = Counter(d.metadata.get("content_type") for d in docs)
        print(f"\nContent type breakdown: {dict(types)}")
        # At minimum we expect text content
        assert types.get("text", 0) > 0

    def test_no_empty_page_content(self):
        docs = load_all_books()
        empty = [d for d in docs if not d.page_content.strip()]
        assert empty == [], f"{len(empty)} documents have empty page_content"

    def test_source_metadata_set(self):
        docs = load_all_books()
        missing_source = [d for d in docs if not d.metadata.get("source")]
        assert missing_source == []
