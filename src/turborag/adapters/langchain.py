from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..types import ChunkRecord
from .compat import ExistingRAGAdapter, embed_texts


class TurboVectorStore:
    """LangChain-style vector store surface backed by TurboRAG.

    This adapter keeps the API intentionally familiar: `from_texts`, `add_texts`,
    `similarity_search`, and `similarity_search_with_score`. If `langchain_core` is
    installed, results are returned as `Document` objects; otherwise `ChunkRecord`
    instances are returned.
    """

    def __init__(
        self,
        embedding: Any,
        adapter: ExistingRAGAdapter,
        resolver: Callable[[Sequence[str]], Sequence[ChunkRecord | Mapping[str, Any]]],
        managed_records: dict[str, ChunkRecord] | None = None,
    ) -> None:
        self.embedding = embedding
        self.adapter = adapter
        self._resolver = resolver
        self._managed_records = managed_records

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        embedding: Any,
        metadatas: Sequence[Mapping[str, Any] | None] | None = None,
        ids: Sequence[str] | None = None,
        **index_kwargs: Any,
    ) -> "TurboVectorStore":
        records = _build_records(texts=texts, metadatas=metadatas, ids=ids)
        embeddings = embed_texts(embedding, [record.text for record in records])
        record_store = {record.chunk_id: record for record in records}

        adapter = ExistingRAGAdapter.from_embeddings(
            embeddings=embeddings,
            ids=[record.chunk_id for record in records],
            query_embedder=embedding,
            fetch_records=lambda requested_ids: [record_store[chunk_id] for chunk_id in requested_ids if chunk_id in record_store],
            **index_kwargs,
        )
        return cls(
            embedding=embedding,
            adapter=adapter,
            resolver=lambda ids: [record_store[chunk_id] for chunk_id in ids if chunk_id in record_store],
            managed_records=record_store,
        )

    @classmethod
    def from_existing_records(
        cls,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        embedding: Any,
        resolver: Callable[[Sequence[str]], Sequence[ChunkRecord | Mapping[str, Any]]],
        **index_kwargs: Any,
    ) -> "TurboVectorStore":
        adapter = ExistingRAGAdapter.from_embeddings(
            embeddings=embeddings,
            ids=ids,
            query_embedder=embedding,
            fetch_records=resolver,
            **index_kwargs,
        )
        return cls(embedding=embedding, adapter=adapter, resolver=resolver, managed_records=None)

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any] | None] | None = None,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        if self._managed_records is None:
            raise RuntimeError(
                "add_texts is only supported when TurboVectorStore owns the records in-memory. "
                "For external databases, write the new records through your existing storage path "
                "and call adapter.add_embeddings(...) with the matching chunk IDs."
            )

        records = _build_records(texts=texts, metadatas=metadatas, ids=ids)
        embeddings = embed_texts(self.embedding, [record.text for record in records])
        self.adapter.add_embeddings(embeddings, [record.chunk_id for record in records])
        for record in records:
            self._managed_records[record.chunk_id] = record
        self._resolver = lambda requested_ids, store=self._managed_records: [store[chunk_id] for chunk_id in requested_ids if chunk_id in store]
        self.adapter.fetch_records = self._resolver
        return [record.chunk_id for record in records]

    def similarity_search(self, query: str, k: int = 4):
        return [_to_document(record) for record in self.adapter.similarity_search(query, k=k)]

    def similarity_search_by_vector(self, embedding: Sequence[float], k: int = 4):
        return [_to_document(record) for record in self.adapter.similarity_search_by_vector(embedding, k=k)]

    def similarity_search_with_score(self, query: str, k: int = 4):
        return [(_to_document(record), score) for record, score in self.adapter.similarity_search_with_score(query, k=k)]

    def as_retriever(self, search_kwargs: Mapping[str, Any] | None = None) -> "TurboRetriever":
        return TurboRetriever(store=self, search_kwargs=dict(search_kwargs or {}))


@dataclass(slots=True)
class TurboRetriever:
    """Small retriever wrapper with a LangChain-like `invoke` method."""

    store: TurboVectorStore
    search_kwargs: Mapping[str, Any]

    def invoke(self, query: str):
        k = int(self.search_kwargs.get("k", 4))
        return self.store.similarity_search(query, k=k)

    def get_relevant_documents(self, query: str):
        return self.invoke(query)


def _build_records(
    texts: Sequence[str],
    metadatas: Sequence[Mapping[str, Any] | None] | None,
    ids: Sequence[str] | None,
) -> list[ChunkRecord]:
    if not texts:
        raise ValueError("texts must not be empty")
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas must be the same length as texts")
    if ids is not None and len(ids) != len(texts):
        raise ValueError("ids must be the same length as texts")

    built: list[ChunkRecord] = []
    for index, text in enumerate(texts):
        metadata = dict((metadatas[index] if metadatas is not None else None) or {})
        chunk_id = ids[index] if ids is not None else str(uuid.uuid4())
        built.append(
            ChunkRecord(
                chunk_id=str(chunk_id),
                text=text,
                source_doc=metadata.get("source_doc") or metadata.get("source"),
                page_num=metadata.get("page_num") or metadata.get("page"),
                section=metadata.get("section"),
                metadata=metadata,
            )
        )
    return built


def _to_document(record: ChunkRecord):
    try:
        from langchain_core.documents import Document
    except ImportError:
        return record

    metadata = dict(record.metadata)
    metadata.setdefault("chunk_id", record.chunk_id)
    if record.source_doc is not None:
        metadata.setdefault("source_doc", record.source_doc)
    if record.page_num is not None:
        metadata.setdefault("page_num", record.page_num)
    if record.section is not None:
        metadata.setdefault("section", record.section)
    return Document(page_content=record.text, metadata=metadata)
