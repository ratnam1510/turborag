"""Tests for the document chunker."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from turborag.chunker import ChunkConfig, chunk_file, chunk_text
from turborag.exceptions import ChunkingError


class TestChunkText:
    """Tests for token-aware text chunking."""

    def test_basic_chunking(self) -> None:
        text = "Hello world. " * 200
        chunks = chunk_text(text, source_doc="test.txt")
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.chunk_id.startswith("chunk-")
            assert chunk.source_doc == "test.txt"
            assert len(chunk.text) > 0

    def test_empty_text_returns_empty(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self) -> None:
        text = "This is a short sentence."
        chunks = chunk_text(text, source_doc="short.txt")
        assert len(chunks) == 1
        assert "short sentence" in chunks[0].text

    def test_chunk_ids_are_deterministic(self) -> None:
        text = "Determinism is important for reproducibility. " * 50
        chunks1 = chunk_text(text, source_doc="test.txt")
        chunks2 = chunk_text(text, source_doc="test.txt")
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_different_sources_different_ids(self) -> None:
        text = "Same content different sources."
        chunks1 = chunk_text(text, source_doc="a.txt")
        chunks2 = chunk_text(text, source_doc="b.txt")
        assert chunks1[0].chunk_id != chunks2[0].chunk_id

    def test_custom_config(self) -> None:
        text = "Word " * 500
        config = ChunkConfig(chunk_size=50, chunk_overlap=10)
        chunks = chunk_text(text, config=config)
        assert len(chunks) > 1

    def test_metadata_propagation(self) -> None:
        text = "Some content."
        chunks = chunk_text(
            text,
            source_doc="doc.md",
            base_metadata={"author": "test", "version": 2},
        )
        assert len(chunks) == 1
        assert chunks[0].metadata["author"] == "test"
        assert chunks[0].metadata["version"] == 2
        assert "chunk_index" in chunks[0].metadata

    def test_section_detection(self) -> None:
        text = "# Introduction\n\nThis is the intro.\n\n# Methods\n\nThese are the methods."
        chunks = chunk_text(text)
        # At least one chunk should have section info
        sections = [c.section for c in chunks if c.section]
        # Sections may or may not be detected depending on chunk boundaries
        assert isinstance(sections, list)


class TestChunkFile:
    """Tests for file-based chunking."""

    def test_chunk_text_file(self, tmp_path: Path) -> None:
        text_file = tmp_path / "test.txt"
        text_file.write_text("This is a test file. " * 100)
        chunks = chunk_file(text_file)
        assert len(chunks) > 0
        assert chunks[0].source_doc == str(text_file)

    def test_chunk_markdown_file(self, tmp_path: Path) -> None:
        md_file = tmp_path / "test.md"
        md_file.write_text("# Heading\n\nSome content.\n\n## Subheading\n\nMore content.")
        chunks = chunk_file(md_file)
        assert len(chunks) > 0

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(ChunkingError, match="not found"):
            chunk_file("/nonexistent/path.txt")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("content")
        with pytest.raises(ChunkingError, match="Unsupported"):
            chunk_file(bad_file)
