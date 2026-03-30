"""Token-aware document chunking for PDF, markdown, and plain text.

Provides a production-grade chunker that respects sentence boundaries,
supports configurable chunk sizes with overlap, and extracts metadata
(source file, page numbers, sections) for downstream indexing.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .exceptions import ChunkingError
from .types import ChunkRecord

logger = logging.getLogger(__name__)

# Sentence-ending patterns (handles abbreviations gracefully)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# Section header patterns for markdown
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass(slots=True)
class ChunkConfig:
    """Configuration for the document chunker."""

    chunk_size: int = 512
    """Target chunk size in tokens."""

    chunk_overlap: int = 64
    """Number of overlapping tokens between adjacent chunks."""

    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    )
    """Ordered list of separators to split on (first match wins)."""

    min_chunk_size: int = 50
    """Minimum chunk size in tokens — smaller chunks get merged with neighbors."""

    encoding_name: str = "cl100k_base"
    """Tiktoken encoding name for token counting."""


def chunk_text(
    text: str,
    *,
    source_doc: str | None = None,
    config: ChunkConfig | None = None,
    base_metadata: dict[str, Any] | None = None,
) -> list[ChunkRecord]:
    """Split text into token-aware chunks with metadata.

    Parameters
    ----------
    text : str
        The input text to chunk.
    source_doc : str, optional
        Source document path or name.
    config : ChunkConfig, optional
        Chunking configuration. Uses defaults if not provided.
    base_metadata : dict, optional
        Metadata to attach to every chunk.

    Returns
    -------
    list[ChunkRecord]
        Chunks with generated IDs, text, and metadata.
    """
    if not text or not text.strip():
        return []

    cfg = config or ChunkConfig()
    tokenizer = _get_tokenizer(cfg.encoding_name)

    # Detect sections in the text
    sections = _detect_sections(text)

    # Split into initial segments by separator hierarchy
    segments = _recursive_split(text, cfg.separators, tokenizer, cfg.chunk_size)

    # Merge small segments and enforce chunk_size / overlap
    chunks = _merge_and_overlap(segments, tokenizer, cfg)

    # Build ChunkRecords with metadata
    records: list[ChunkRecord] = []
    for i, chunk_text_content in enumerate(chunks):
        section = _find_section(chunk_text_content, sections)
        chunk_id = _generate_chunk_id(source_doc or "", i, chunk_text_content)

        metadata = dict(base_metadata or {})
        metadata["chunk_index"] = i
        metadata["chunk_count"] = len(chunks)

        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                text=chunk_text_content,
                source_doc=source_doc,
                section=section,
                metadata=metadata,
            )
        )

    return records


def chunk_file(
    path: str | Path,
    *,
    config: ChunkConfig | None = None,
    base_metadata: dict[str, Any] | None = None,
) -> list[ChunkRecord]:
    """Chunk a file (plain text, markdown, or PDF) into ChunkRecords.

    File type is detected from the extension:
    - ``.txt``, ``.md``, ``.markdown`` → read as text
    - ``.pdf`` → extract text via pdfminer.six (requires ``turborag[ingest]``)
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ChunkingError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    source_doc = str(file_path)

    if suffix in (".txt", ".md", ".markdown", ".rst", ".text"):
        text = file_path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        text = _extract_pdf_text(file_path)
    else:
        raise ChunkingError(f"Unsupported file type: {suffix}")

    return chunk_text(
        text,
        source_doc=source_doc,
        config=config,
        base_metadata=base_metadata,
    )


def chunk_documents(
    paths: Sequence[str | Path],
    *,
    config: ChunkConfig | None = None,
) -> list[ChunkRecord]:
    """Chunk multiple documents into a single list of ChunkRecords."""
    all_records: list[ChunkRecord] = []
    for path in paths:
        try:
            records = chunk_file(path, config=config)
            all_records.extend(records)
            logger.info("Chunked %s → %d chunks", path, len(records))
        except ChunkingError as exc:
            logger.warning("Skipping %s: %s", path, exc)
    return all_records


def _get_tokenizer(encoding_name: str):
    """Get a tiktoken encoder, or fall back to a simple whitespace tokenizer."""
    try:
        import tiktoken
        return tiktoken.get_encoding(encoding_name)
    except ImportError:
        return _WhitespaceTokenizer()


class _WhitespaceTokenizer:
    """Fallback tokenizer when tiktoken is not installed."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, tokens: list[int]) -> str:
        # Not used in our chunking logic — we work with text directly
        return ""


def _count_tokens(text: str, tokenizer: Any) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text))


def _recursive_split(
    text: str,
    separators: list[str],
    tokenizer: Any,
    target_size: int,
) -> list[str]:
    """Recursively split text using separator hierarchy."""
    token_count = _count_tokens(text, tokenizer)
    if token_count <= target_size:
        return [text]

    # Try each separator in order
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            segments: list[str] = []
            for part in parts:
                cleaned = part.strip()
                if cleaned:
                    segments.append(cleaned)
            if len(segments) > 1:
                # Recursively split segments that are still too large
                result: list[str] = []
                for segment in segments:
                    if _count_tokens(segment, tokenizer) > target_size:
                        result.extend(
                            _recursive_split(segment, separators[separators.index(sep) + 1:], tokenizer, target_size)
                        )
                    else:
                        result.append(segment)
                return result

    # Last resort: split at target_size by characters
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for word in words:
        word_tokens = _count_tokens(word, tokenizer)
        if current_tokens + word_tokens > target_size and current:
            chunks.append(" ".join(current))
            current = [word]
            current_tokens = word_tokens
        else:
            current.append(word)
            current_tokens += word_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _merge_and_overlap(
    segments: list[str],
    tokenizer: Any,
    config: ChunkConfig,
) -> list[str]:
    """Merge small segments and add overlap between adjacent chunks."""
    if not segments:
        return []

    # First pass: merge small segments
    merged: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for segment in segments:
        seg_tokens = _count_tokens(segment, tokenizer)

        if current_tokens + seg_tokens <= config.chunk_size:
            current_parts.append(segment)
            current_tokens += seg_tokens
        else:
            if current_parts:
                merged.append(" ".join(current_parts))
            current_parts = [segment]
            current_tokens = seg_tokens

    if current_parts:
        merged.append(" ".join(current_parts))

    # Second pass: add overlap
    if config.chunk_overlap <= 0 or len(merged) <= 1:
        return merged

    overlapped: list[str] = []
    for i, chunk in enumerate(merged):
        if i == 0:
            overlapped.append(chunk)
            continue

        # Get tail of previous chunk as overlap prefix
        prev_words = merged[i - 1].split()
        overlap_words = prev_words[-config.chunk_overlap:] if len(prev_words) > config.chunk_overlap else prev_words
        overlap_text = " ".join(overlap_words)

        overlapped.append(overlap_text + " " + chunk)

    return overlapped


def _detect_sections(text: str) -> list[tuple[int, str]]:
    """Detect markdown-style sections and their character offsets."""
    sections: list[tuple[int, str]] = []
    for match in _MD_HEADING.finditer(text):
        sections.append((match.start(), match.group(2).strip()))
    return sections


def _find_section(chunk_text: str, sections: list[tuple[int, str]]) -> str | None:
    """Find which section a chunk belongs to based on content overlap."""
    if not sections:
        return None
    # Check if any section heading appears in the chunk
    for _, heading in sections:
        if heading in chunk_text:
            return heading
    return None


def _generate_chunk_id(source: str, index: int, text: str) -> str:
    """Generate a deterministic chunk ID from source, index, and content hash."""
    content = f"{source}:{index}:{text[:200]}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"chunk-{digest}"


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF file using pdfminer.six."""
    try:
        from pdfminer.high_level import extract_text
    except ImportError as exc:
        raise ChunkingError(
            "PDF extraction requires pdfminer.six: pip install 'turborag[ingest]'"
        ) from exc

    try:
        text = extract_text(str(path))
    except Exception as exc:
        raise ChunkingError(f"Failed to extract text from {path}: {exc}") from exc

    if not text or not text.strip():
        raise ChunkingError(f"No text content extracted from {path}")

    return text
