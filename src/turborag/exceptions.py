"""TurboRAG domain-specific exceptions.

Provides a clean exception hierarchy so callers can catch specific error
categories instead of bare ValueError/RuntimeError.
"""
from __future__ import annotations


class TurboRAGError(Exception):
    """Base exception for all TurboRAG errors."""


class IndexError(TurboRAGError):
    """Error during index operations (add, search, save, load)."""


class IndexConfigError(IndexError):
    """Invalid or incompatible index configuration."""


class DuplicateIDError(IndexError):
    """Attempted to add a vector with an ID that already exists in the index."""

    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id
        super().__init__(f"duplicate id detected: {chunk_id}")


class IDNotFoundError(IndexError):
    """Attempted to access a vector ID that does not exist in the index."""

    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id
        super().__init__(f"id not found: {chunk_id}")


class IngestError(TurboRAGError):
    """Error during data ingestion (parsing, validation, import)."""


class DatasetFormatError(IngestError):
    """Invalid or unsupported dataset format."""


class ChunkingError(IngestError):
    """Error during document chunking."""


class ServiceError(TurboRAGError):
    """Error in the HTTP service or MCP server layer."""


class QueryError(TurboRAGError):
    """Error during query processing."""


class EmbeddingError(TurboRAGError):
    """Error during embedding generation."""
