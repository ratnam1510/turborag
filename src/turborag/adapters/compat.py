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
        *,
        allow_unhydrated: bool = True,
    ) -> None:
        self.index = index
        self.query_embedder = query_embedder
        self.fetch_records = fetch_records
        self.allow_unhydrated = allow_unhydrated

    @classmethod
    def from_embeddings(
        cls,
        embeddings: Sequence[Sequence[float]] | FloatVector,
        ids: Sequence[str],
        query_embedder: QueryEmbedder | Callable[[str], Any],
        fetch_records: FetchRecords,
        *,
        allow_unhydrated: bool = True,
        **index_kwargs: Any,
    ) -> "ExistingRAGAdapter":
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D array-like object")

        index = TurboIndex(dim=matrix.shape[1], **index_kwargs)
        index.add(matrix, ids)
        return cls(
            index=index,
            query_embedder=query_embedder,
            fetch_records=fetch_records,
            allow_unhydrated=allow_unhydrated,
        )

    @classmethod
    def from_existing_backend(
        cls,
        embeddings: Sequence[Sequence[float]] | FloatVector,
        ids: Sequence[str],
        query_embedder: QueryEmbedder | Callable[[str], Any],
        records_backend: Any,
        *,
        allow_unhydrated: bool = True,
        **index_kwargs: Any,
    ) -> "ExistingRAGAdapter":
        """Create an adapter over a pre-existing database client or store.

        `records_backend` can be:
        - a callable `fetch_records(ids)`,
        - a mapping keyed by chunk ID,
        - a client object exposing `fetch(ids=...)`, `retrieve(ids=...)`, or `get(ids=...)`.

        This helper keeps TurboRAG as a compressed retrieval sidecar while your existing
        database remains the source of truth for text and metadata.
        """

        fetch_records = resolve_records_backend(records_backend)
        return cls.from_embeddings(
            embeddings=embeddings,
            ids=ids,
            query_embedder=query_embedder,
            fetch_records=fetch_records,
            allow_unhydrated=allow_unhydrated,
            **index_kwargs,
        )

    def add_embeddings(
        self, embeddings: Sequence[Sequence[float]] | FloatVector, ids: Sequence[str]
    ) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D array-like object")
        self.index.add(matrix, ids)

    def search_ids(self, query: str, k: int = 10) -> list[SearchHit]:
        vector = _embed_query(self.query_embedder, query)
        return self.search_ids_by_vector(vector, k=k)

    def search_ids_by_vector(
        self, vector: Sequence[float] | FloatVector, k: int = 10
    ) -> list[SearchHit]:
        query = _as_vector(vector)
        return [
            SearchHit(chunk_id=chunk_id, score=score)
            for chunk_id, score in self.index.search(query, k=k)
        ]

    def query(self, text: str, k: int = 10) -> list[RetrievalResult]:
        hits = self.search_ids(text, k=k)
        return self._hydrate_hits(hits)

    def query_by_vector(
        self, vector: Sequence[float] | FloatVector, k: int = 10
    ) -> list[RetrievalResult]:
        hits = self.search_ids_by_vector(vector, k=k)
        return self._hydrate_hits(hits)

    def similarity_search(self, query: str, k: int = 4) -> list[ChunkRecord]:
        return [self._to_chunk(result) for result in self.query(query, k=k)]

    def similarity_search_by_vector(
        self, vector: Sequence[float] | FloatVector, k: int = 4
    ) -> list[ChunkRecord]:
        return [self._to_chunk(result) for result in self.query_by_vector(vector, k=k)]

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[ChunkRecord, float]]:
        results = self.query(query, k=k)
        return [(self._to_chunk(result), result.score) for result in results]

    def _hydrate_hits(self, hits: Sequence[SearchHit]) -> list[RetrievalResult]:
        if not hits:
            return []

        ordered_ids = [hit.chunk_id for hit in hits]
        fetched = self.fetch_records(ordered_ids)
        by_id = {
            record.chunk_id: record
            for record in (coerce_chunk_record(item) for item in fetched)
        }

        hydrated: list[RetrievalResult] = []
        for hit in hits:
            record = by_id.get(hit.chunk_id)
            if record is None:
                if not self.allow_unhydrated:
                    continue
                hydrated.append(
                    RetrievalResult(
                        chunk_id=hit.chunk_id,
                        text="",
                        score=hit.score,
                        source_doc=None,
                        page_num=None,
                        explanation=(
                            "Retrieved from TurboRAG sidecar by chunk ID. "
                            "Hydrate text/metadata from the existing database using chunk_id."
                        ),
                    )
                )
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

    metadata_source = _first_mapping(
        item.get("metadata"), item.get("payload"), item.get("fields")
    )
    metadata = dict(metadata_source or {})

    chunk_id = _first_non_none(
        item.get("chunk_id"),
        item.get("id"),
        item.get("_id"),
        item.get("point_id"),
        item.get("vector_id"),
        metadata.get("chunk_id"),
        metadata.get("id"),
        metadata.get("_id"),
    )

    text = _first_non_none(
        item.get("text"),
        item.get("page_content"),
        item.get("content"),
        item.get("document"),
        metadata.get("text"),
        metadata.get("page_content"),
        metadata.get("content"),
        metadata.get("document"),
    )

    if chunk_id is None or text is None:
        raise ValueError(
            "record mappings must provide chunk_id/id and text/page_content/content"
        )

    if "metadata" in item and isinstance(item.get("metadata"), Mapping):
        metadata = dict(item.get("metadata", {}))

    for key in (
        "chunk_id",
        "id",
        "_id",
        "text",
        "page_content",
        "content",
        "document",
        "source_doc",
        "source",
        "page_num",
        "page",
        "page_number",
        "section",
    ):
        metadata.pop(key, None)

    if "section" in item and item["section"] is not None:
        metadata.setdefault("section", item["section"])
    elif metadata_source is not None and metadata_source.get("section") is not None:
        metadata.setdefault("section", metadata_source.get("section"))

    page_num = _coerce_optional_int(
        _first_non_none(
            item["page_num"] if "page_num" in item else None,
            item.get("page"),
            item.get("page_number"),
            metadata_source.get("page_num") if metadata_source else None,
            metadata_source.get("page") if metadata_source else None,
            metadata_source.get("page_number") if metadata_source else None,
        )
    )

    source_doc = _first_non_none(
        item.get("source_doc"),
        item.get("source"),
        item.get("document_id"),
        metadata_source.get("source_doc") if metadata_source else None,
        metadata_source.get("source") if metadata_source else None,
        metadata_source.get("document_id") if metadata_source else None,
    )

    section = _first_non_none(
        item.get("section"),
        metadata_source.get("section") if metadata_source else None,
    )

    return ChunkRecord(
        chunk_id=str(chunk_id),
        text=str(text),
        source_doc=None if source_doc is None else str(source_doc),
        page_num=page_num,
        section=None if section is None else str(section),
        metadata=metadata,
    )


def resolve_records_backend(records_backend: Any) -> FetchRecords:
    """Build a `fetch_records(ids)` callback from common backend shapes.

    This allows sidecar adoption over existing stores with minimal glue code.
    Supported inputs:

    - callable: `fetch(ids) -> Sequence[ChunkRecord|Mapping]`
    - mapping: keyed by chunk ID
    - backend object exposing one of: `fetch(ids=...)`, `retrieve(ids=...)`, `get(ids=...)`
    """

    if callable(records_backend):
        return records_backend

    if isinstance(records_backend, Mapping):
        return lambda ids, store=records_backend: [
            store[chunk_id] for chunk_id in ids if chunk_id in store
        ]

    for method_name in ("fetch", "retrieve", "get"):
        if hasattr(records_backend, method_name):
            return lambda ids, backend=records_backend, name=method_name: (
                _extract_backend_items(
                    _call_backend_method(backend, name, ids),
                    requested_ids=ids,
                )
            )

    raise TypeError(
        "records_backend must be a callable, mapping, or client with fetch/retrieve/get methods"
    )


def _call_backend_method(backend: Any, method_name: str, ids: Sequence[str]) -> Any:
    method = getattr(backend, method_name)
    id_list = list(ids)

    try:
        return method(ids=id_list)
    except TypeError:
        pass

    try:
        return method(id_list)
    except TypeError:
        pass

    return method()


def _extract_backend_items(response: Any, *, requested_ids: Sequence[str]) -> list[Any]:
    if response is None:
        return []

    if isinstance(response, Mapping):
        if "vectors" in response and isinstance(response["vectors"], Mapping):
            return _extract_vector_map_response(response["vectors"], requested_ids)

        if "result" in response:
            return _extract_backend_items(
                response["result"], requested_ids=requested_ids
            )

        if "points" in response and isinstance(response["points"], Sequence):
            return [_coerce_backend_item(item) for item in response["points"]]

        if "records" in response and isinstance(response["records"], Sequence):
            return [_coerce_backend_item(item) for item in response["records"]]

        if "matches" in response and isinstance(response["matches"], Sequence):
            return [_coerce_backend_item(item) for item in response["matches"]]

        if "ids" in response and ("documents" in response or "metadatas" in response):
            return _extract_chroma_get_response(response)

        if "id" in response and (
            "payload" in response or "metadata" in response or "text" in response
        ):
            return [_coerce_backend_item(response)]

    if hasattr(response, "vectors"):
        return _extract_backend_items(
            {"vectors": getattr(response, "vectors")}, requested_ids=requested_ids
        )

    if hasattr(response, "points"):
        return _extract_backend_items(
            getattr(response, "points"), requested_ids=requested_ids
        )

    if hasattr(response, "result"):
        return _extract_backend_items(
            getattr(response, "result"), requested_ids=requested_ids
        )

    if isinstance(response, Sequence) and not isinstance(response, (str, bytes)):
        return [_coerce_backend_item(item) for item in response]

    return [_coerce_backend_item(response)]


def _coerce_backend_item(item: Any) -> Any:
    if isinstance(item, Mapping):
        return item

    if hasattr(item, "model_dump") and callable(item.model_dump):
        dumped = item.model_dump()
        if isinstance(dumped, Mapping):
            return dumped

    if hasattr(item, "dict") and callable(item.dict):
        dumped = item.dict()
        if isinstance(dumped, Mapping):
            return dumped

    mapped: dict[str, Any] = {}
    for field in (
        "chunk_id",
        "id",
        "point_id",
        "vector_id",
        "text",
        "content",
        "page_content",
        "document",
        "payload",
        "fields",
        "metadata",
        "source_doc",
        "source",
        "page_num",
        "page",
        "page_number",
        "section",
    ):
        if hasattr(item, field):
            mapped[field] = getattr(item, field)

    return mapped if mapped else item


def _extract_vector_map_response(
    vector_map: Mapping[str, Any], requested_ids: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk_id in requested_ids:
        if chunk_id not in vector_map:
            continue
        raw = vector_map[chunk_id]
        if isinstance(raw, Mapping):
            row: dict[str, Any] = {"chunk_id": chunk_id}
            if isinstance(raw.get("metadata"), Mapping):
                row["metadata"] = dict(raw["metadata"])
            for key in (
                "text",
                "page_content",
                "content",
                "document",
                "source_doc",
                "source",
                "page_num",
                "page",
                "section",
                "payload",
                "fields",
            ):
                if key in raw:
                    row[key] = raw[key]
            rows.append(row)
        else:
            rows.append({"chunk_id": chunk_id, "text": str(raw)})
    return rows


def _extract_chroma_get_response(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    ids = [str(value) for value in _flatten_values(payload.get("ids", []))]
    documents = _flatten_values(payload.get("documents", []))
    metadatas = _flatten_values(payload.get("metadatas", []))

    rows: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        record: dict[str, Any] = {"chunk_id": chunk_id}

        if index < len(documents) and documents[index] is not None:
            record["text"] = str(documents[index])

        if index < len(metadatas) and isinstance(metadatas[index], Mapping):
            metadata = dict(metadatas[index])
            record["metadata"] = metadata
            if "source_doc" in metadata:
                record.setdefault("source_doc", metadata["source_doc"])
            if "source" in metadata:
                record.setdefault("source", metadata["source"])
            if "page_num" in metadata:
                record.setdefault("page_num", metadata["page_num"])
            if "page" in metadata:
                record.setdefault("page", metadata["page"])
            if "section" in metadata:
                record.setdefault("section", metadata["section"])

        rows.append(record)
    return rows


def _flatten_values(values: Any) -> list[Any]:
    if isinstance(values, (str, bytes)):
        return [values]
    if not isinstance(values, Sequence):
        return [values]

    if not values:
        return []

    first = values[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes, Mapping)):
        flattened: list[Any] = []
        for nested in values:
            if isinstance(nested, Sequence) and not isinstance(
                nested, (str, bytes, Mapping)
            ):
                flattened.extend(list(nested))
            else:
                flattened.append(nested)
        return flattened

    return list(values)


def _first_mapping(*values: Any) -> Mapping[str, Any] | None:
    for value in values:
        if isinstance(value, Mapping):
            return value
    return None


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def embed_texts(
    embedder: DocumentEmbedder | Callable[[str], Any], texts: Sequence[str]
) -> FloatVector:
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
        raise TypeError(
            "embedder must be callable or provide embed_documents/embed_texts/encode"
        )

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1 and len(texts) == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("embedded texts must produce a 2D matrix")
    return matrix


def _embed_query(
    embedder: QueryEmbedder | Callable[[str], Any], text: str
) -> FloatVector:
    if hasattr(embedder, "embed_query"):
        value = embedder.embed_query(text)
    elif hasattr(embedder, "embed"):
        value = embedder.embed(text)
    elif hasattr(embedder, "encode"):
        value = embedder.encode(text)
    elif callable(embedder):
        value = embedder(text)
    else:
        raise TypeError(
            "query_embedder must be callable or provide embed_query/embed/encode"
        )
    return _as_vector(value)


def _as_vector(vector: Sequence[float] | FloatVector) -> FloatVector:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError("query vectors must be one-dimensional")
    return array
