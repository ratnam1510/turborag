from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from ..index import TurboIndex
from ..types import ChunkRecord, RetrievalResult


FloatVector = NDArray[np.float32]
FetchRecords = Callable[[Sequence[str]], Sequence[ChunkRecord | Mapping[str, Any]]]


class QueryEmbedder(Protocol):
    def embed_query(self, text: str): ...


class DocumentEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]): ...


@dataclass(slots=True)
class SearchHit:
    """A low-level search result used by compatibility adapters."""

    chunk_id: str
    score: float


class ExistingRAGAdapter:
    """Attach TurboRAG to an existing RAG stack without changing its database.

    The adapter stores only compressed embeddings in TurboRAG while your current metadata
    store remains the source of truth for chunk text and metadata. Results are hydrated by
    calling `fetch_records(ids)` with the same chunk IDs already used in the application.
    """

    def __init__(
        self,
        index: TurboIndex,
        query_embedder: QueryEmbedder | Callable[[str], Any],
        fetch_records: FetchRecords,
    ) -> None:
        self.index = index
        self.query_embedder = query_embedder
        self.fetch_records = fetch_records

    @classmethod
    def from_embeddings(
        cls,
        embeddings: Sequence[Sequence[float]] | FloatVector,
        ids: Sequence[str],
        query_embedder: QueryEmbedder | Callable[[str], Any],
        fetch_records: FetchRecords,
        **index_kwargs: Any,
    ) -> "ExistingRAGAdapter":
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D array-like object")

        index = TurboIndex(dim=matrix.shape[1], **index_kwargs)
        index.add(matrix, ids)
        return cls(index=index, query_embedder=query_embedder, fetch_records=fetch_records)

    def add_embeddings(self, embeddings: Sequence[Sequence[float]] | FloatVector, ids: Sequence[str]) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D array-like object")
        self.index.add(matrix, ids)

    def search_ids(self, query: str, k: int = 10) -> list[SearchHit]:
        vector = _embed_query(self.query_embedder, query)
        return self.search_ids_by_vector(vector, k=k)

    def search_ids_by_vector(self, vector: Sequence[float] | FloatVector, k: int = 10) -> list[SearchHit]:
        query = _as_vector(vector)
        return [SearchHit(chunk_id=chunk_id, score=score) for chunk_id, score in self.index.search(query, k=k)]

    def query(self, text: str, k: int = 10) -> list[RetrievalResult]:
        hits = self.search_ids(text, k=k)
        return self._hydrate_hits(hits)

    def query_by_vector(self, vector: Sequence[float] | FloatVector, k: int = 10) -> list[RetrievalResult]:
        hits = self.search_ids_by_vector(vector, k=k)
        return self._hydrate_hits(hits)

    def similarity_search(self, query: str, k: int = 4) -> list[ChunkRecord]:
        return [self._to_chunk(result) for result in self.query(query, k=k)]

    def similarity_search_by_vector(self, vector: Sequence[float] | FloatVector, k: int = 4) -> list[ChunkRecord]:
        return [self._to_chunk(result) for result in self.query_by_vector(vector, k=k)]

    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[ChunkRecord, float]]:
        results = self.query(query, k=k)
        return [(self._to_chunk(result), result.score) for result in results]

    def _hydrate_hits(self, hits: Sequence[SearchHit]) -> list[RetrievalResult]:
        if not hits:
            return []

        ordered_ids = [hit.chunk_id for hit in hits]
        fetched = self.fetch_records(ordered_ids)
        by_id = {record.chunk_id: record for record in (coerce_chunk_record(item) for item in fetched)}

        hydrated: list[RetrievalResult] = []
        for hit in hits:
            record = by_id.get(hit.chunk_id)
            if record is None:
                continue
            hydrated.append(
                RetrievalResult(
                    chunk_id=record.chunk_id,
                    text=record.text,
                    score=hit.score,
                    source_doc=record.source_doc,
                    page_num=record.page_num,
                    explanation="Retrieved from TurboRAG using the existing application record store.",
                )
            )
        return hydrated

    @staticmethod
    def _to_chunk(result: RetrievalResult) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=result.chunk_id,
            text=result.text,
            source_doc=result.source_doc,
            page_num=result.page_num,
        )


def coerce_chunk_record(item: ChunkRecord | Mapping[str, Any]) -> ChunkRecord:
    if isinstance(item, ChunkRecord):
        return item
    if not isinstance(item, Mapping):
        raise TypeError("records must be ChunkRecord instances or mapping-like objects")

    chunk_id = item.get("chunk_id") or item.get("id")
    text = item.get("text") or item.get("page_content") or item.get("content")
    if chunk_id is None or text is None:
        raise ValueError("record mappings must provide chunk_id/id and text/page_content/content")

    metadata = dict(item.get("metadata", {}))
    if "section" in item and item["section"] is not None:
        metadata.setdefault("section", item["section"])

    page_num = item["page_num"] if "page_num" in item else item.get("page")

    return ChunkRecord(
        chunk_id=str(chunk_id),
        text=str(text),
        source_doc=item.get("source_doc") or item.get("source"),
        page_num=page_num,
        section=item.get("section"),
        metadata=metadata,
    )


def embed_texts(embedder: DocumentEmbedder | Callable[[str], Any], texts: Sequence[str]) -> FloatVector:
    if not texts:
        raise ValueError("texts must not be empty")

    if hasattr(embedder, "embed_documents"):
        values = embedder.embed_documents(list(texts))
    elif hasattr(embedder, "embed_texts"):
        values = embedder.embed_texts(list(texts))
    elif hasattr(embedder, "encode"):
        values = embedder.encode(list(texts))
    elif callable(embedder):
        values = [embedder(text) for text in texts]
    else:
        raise TypeError("embedder must be callable or provide embed_documents/embed_texts/encode")

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1 and len(texts) == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("embedded texts must produce a 2D matrix")
    return matrix


def _embed_query(embedder: QueryEmbedder | Callable[[str], Any], text: str) -> FloatVector:
    if hasattr(embedder, "embed_query"):
        value = embedder.embed_query(text)
    elif hasattr(embedder, "embed"):
        value = embedder.embed(text)
    elif hasattr(embedder, "encode"):
        value = embedder.encode(text)
    elif callable(embedder):
        value = embedder(text)
    else:
        raise TypeError("query_embedder must be callable or provide embed_query/embed/encode")
    return _as_vector(value)


def _as_vector(vector: Sequence[float] | FloatVector) -> FloatVector:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError("query vectors must be one-dimensional")
    return array
