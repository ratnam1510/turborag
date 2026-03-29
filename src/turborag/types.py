from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    """A stored text chunk and its provenance metadata."""

    chunk_id: str
    text: str
    source_doc: str | None = None
    page_num: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """A user-facing retrieval result."""

    chunk_id: str
    text: str
    score: float
    source_doc: str | None = None
    page_num: int | None = None
    graph_path: list[str] | None = None
    explanation: str | None = None
