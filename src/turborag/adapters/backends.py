from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, Sequence

from ..types import ChunkRecord

FetchRecords = Callable[[Sequence[str]], Sequence[ChunkRecord | Mapping[str, Any]]]


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

__all__ = [
    "build_postgres_fetch_records",
    "build_neon_fetch_records",
    "build_supabase_fetch_records",
    "build_pinecone_fetch_records",
    "build_qdrant_fetch_records",
    "build_chroma_fetch_records",
    "as_chunk_records",
]


def build_postgres_fetch_records(
    *,
    connection: Any | None = None,
    dsn: str | None = None,
    table: str = "chunks",
    id_column: str = "chunk_id",
    text_column: str = "text",
    source_doc_column: str = "source_doc",
    page_num_column: str = "page_num",
    section_column: str = "section",
    metadata_column: str = "metadata",
) -> FetchRecords:
    """Build a `fetch_records(ids)` callback backed by PostgreSQL.

    This works for any Postgres provider, including Neon and Supabase Postgres.

    Pass either:
    - a live `connection`, or
    - a `dsn` string (opens a short-lived connection per fetch call).
    """

    if (connection is None) == (dsn is None):
        raise ValueError("Provide exactly one of connection or dsn")

    table_sql = _quote_table_name(table)
    id_sql = _quote_identifier(id_column)
    text_sql = _quote_identifier(text_column)
    source_doc_sql = _quote_identifier(source_doc_column)
    page_num_sql = _quote_identifier(page_num_column)
    section_sql = _quote_identifier(section_column)
    metadata_sql = _quote_identifier(metadata_column)

    query = (
        f"SELECT {id_sql} AS chunk_id, "
        f"{text_sql} AS text, "
        f"{source_doc_sql} AS source_doc, "
        f"{page_num_sql} AS page_num, "
        f"{section_sql} AS section, "
        f"{metadata_sql} AS metadata "
        f"FROM {table_sql} "
        f"WHERE {id_sql} = ANY(%s)"
    )

    def fetch_records(ids: Sequence[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        id_list = [str(chunk_id) for chunk_id in ids]

        if connection is not None:
            rows, columns = _execute_sql(connection, query, [id_list])
        else:
            connect = _load_psycopg_connect()
            conn = connect(dsn)
            try:
                rows, columns = _execute_sql(conn, query, [id_list])
            finally:
                close = getattr(conn, "close", None)
                if callable(close):
                    close()

        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            mapped = _row_to_mapping(row, columns)
            chunk_id = mapped.get("chunk_id")
            if chunk_id is None:
                continue
            by_id[str(chunk_id)] = {
                "chunk_id": str(chunk_id),
                "text": _coerce_text(mapped.get("text")),
                "source_doc": _coerce_optional_str(mapped.get("source_doc")),
                "page_num": _coerce_optional_int(mapped.get("page_num")),
                "section": _coerce_optional_str(mapped.get("section")),
                "metadata": _coerce_metadata(mapped.get("metadata")),
            }

        return [by_id[chunk_id] for chunk_id in id_list if chunk_id in by_id]

    return fetch_records


def build_neon_fetch_records(**kwargs: Any):
    """Alias for `build_postgres_fetch_records` (Neon is PostgreSQL)."""

    return build_postgres_fetch_records(**kwargs)


def build_supabase_fetch_records(
    client: Any,
    *,
    table: str = "chunks",
    id_column: str = "chunk_id",
    text_column: str = "text",
    source_doc_column: str = "source_doc",
    page_num_column: str = "page_num",
    section_column: str = "section",
    metadata_column: str = "metadata",
) -> FetchRecords:
    """Build a `fetch_records(ids)` callback from a supabase-py client."""

    selected_columns = _dedupe(
        [
            id_column,
            text_column,
            source_doc_column,
            page_num_column,
            section_column,
            metadata_column,
        ]
    )
    select_clause = ",".join(selected_columns)

    def fetch_records(ids: Sequence[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        id_list = [str(chunk_id) for chunk_id in ids]
        response = (
            client.table(table).select(select_clause).in_(id_column, id_list).execute()
        )
        data = _response_data(response)

        by_id: dict[str, dict[str, Any]] = {}
        for item in data:
            if not isinstance(item, Mapping):
                continue
            chunk_id = item.get(id_column)
            if chunk_id is None:
                continue
            chunk_id_str = str(chunk_id)
            metadata = _coerce_metadata(item.get(metadata_column))
            by_id[chunk_id_str] = {
                "chunk_id": chunk_id_str,
                "text": _coerce_text(item.get(text_column)),
                "source_doc": _coerce_optional_str(
                    _first_non_none(
                        item.get(source_doc_column),
                        metadata.get(source_doc_column),
                        metadata.get("source_doc"),
                        metadata.get("source"),
                    )
                ),
                "page_num": _coerce_optional_int(
                    _first_non_none(
                        item.get(page_num_column),
                        metadata.get(page_num_column),
                        metadata.get("page_num"),
                        metadata.get("page"),
                    )
                ),
                "section": _coerce_optional_str(
                    _first_non_none(
                        item.get(section_column),
                        metadata.get(section_column),
                        metadata.get("section"),
                    )
                ),
                "metadata": metadata,
            }

        return [by_id[chunk_id] for chunk_id in id_list if chunk_id in by_id]

    return fetch_records


def build_pinecone_fetch_records(
    index: Any,
    *,
    namespace: str | None = None,
    text_key: str = "text",
    source_doc_key: str = "source_doc",
    page_num_key: str = "page_num",
    section_key: str = "section",
) -> FetchRecords:
    """Build a `fetch_records(ids)` callback from a Pinecone index client."""

    def fetch_records(ids: Sequence[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        id_list = [str(chunk_id) for chunk_id in ids]
        response = _pinecone_fetch(index, id_list, namespace)
        vectors = _get_attr_or_key(response, "vectors", default={})
        if not isinstance(vectors, Mapping):
            return []

        by_id: dict[str, dict[str, Any]] = {}
        for chunk_id in id_list:
            vector = vectors.get(chunk_id)
            if vector is None:
                continue

            metadata_raw = _get_attr_or_key(vector, "metadata", default={})
            metadata = _coerce_metadata(metadata_raw)
            by_id[chunk_id] = {
                "chunk_id": chunk_id,
                "text": _coerce_text(
                    _first_non_none(
                        metadata.get(text_key),
                        metadata.get("text"),
                        metadata.get("page_content"),
                        metadata.get("content"),
                    )
                ),
                "source_doc": _coerce_optional_str(
                    _first_non_none(
                        metadata.get(source_doc_key),
                        metadata.get("source_doc"),
                        metadata.get("source"),
                    )
                ),
                "page_num": _coerce_optional_int(
                    _first_non_none(
                        metadata.get(page_num_key),
                        metadata.get("page_num"),
                        metadata.get("page"),
                    )
                ),
                "section": _coerce_optional_str(
                    _first_non_none(metadata.get(section_key), metadata.get("section"))
                ),
                "metadata": metadata,
            }

        return [by_id[chunk_id] for chunk_id in id_list if chunk_id in by_id]

    return fetch_records


def build_qdrant_fetch_records(
    client: Any,
    *,
    collection_name: str,
    text_key: str = "text",
    source_doc_key: str = "source_doc",
    page_num_key: str = "page_num",
    section_key: str = "section",
) -> FetchRecords:
    """Build a `fetch_records(ids)` callback from a Qdrant client."""

    def fetch_records(ids: Sequence[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        id_list = [str(chunk_id) for chunk_id in ids]
        points = _qdrant_retrieve(client, collection_name, id_list)

        by_id: dict[str, dict[str, Any]] = {}
        for point in points:
            point_id = _get_attr_or_key(point, "id")
            payload_raw = _get_attr_or_key(point, "payload", default={})
            payload = _coerce_metadata(payload_raw)
            chunk_id = _first_non_none(
                payload.get("chunk_id"), payload.get("id"), point_id
            )
            if chunk_id is None:
                continue

            chunk_id_str = str(chunk_id)
            by_id[chunk_id_str] = {
                "chunk_id": chunk_id_str,
                "text": _coerce_text(
                    _first_non_none(
                        payload.get(text_key),
                        payload.get("text"),
                        payload.get("page_content"),
                        payload.get("content"),
                    )
                ),
                "source_doc": _coerce_optional_str(
                    _first_non_none(
                        payload.get(source_doc_key),
                        payload.get("source_doc"),
                        payload.get("source"),
                    )
                ),
                "page_num": _coerce_optional_int(
                    _first_non_none(
                        payload.get(page_num_key),
                        payload.get("page_num"),
                        payload.get("page"),
                    )
                ),
                "section": _coerce_optional_str(
                    _first_non_none(payload.get(section_key), payload.get("section"))
                ),
                "metadata": payload,
            }

        return [by_id[chunk_id] for chunk_id in id_list if chunk_id in by_id]

    return fetch_records


def build_chroma_fetch_records(collection: Any) -> FetchRecords:
    """Build a `fetch_records(ids)` callback from a Chroma collection."""

    def fetch_records(ids: Sequence[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        id_list = [str(chunk_id) for chunk_id in ids]
        response = collection.get(ids=id_list, include=["documents", "metadatas"])
        if not isinstance(response, Mapping):
            return []

        response_ids = [str(value) for value in _flatten(response.get("ids", []))]
        documents = _flatten(response.get("documents", []))
        metadatas = _flatten(response.get("metadatas", []))

        by_id: dict[str, dict[str, Any]] = {}
        for i, chunk_id in enumerate(response_ids):
            metadata = _coerce_metadata(metadatas[i] if i < len(metadatas) else {})
            text = _coerce_text(
                documents[i] if i < len(documents) else metadata.get("text")
            )
            by_id[chunk_id] = {
                "chunk_id": chunk_id,
                "text": text,
                "source_doc": _coerce_optional_str(
                    _first_non_none(metadata.get("source_doc"), metadata.get("source"))
                ),
                "page_num": _coerce_optional_int(
                    _first_non_none(metadata.get("page_num"), metadata.get("page"))
                ),
                "section": _coerce_optional_str(metadata.get("section")),
                "metadata": metadata,
            }

        return [by_id[chunk_id] for chunk_id in id_list if chunk_id in by_id]

    return fetch_records


def _load_psycopg_connect():
    try:
        import psycopg  # type: ignore

        return psycopg.connect
    except ImportError:
        pass

    try:
        import psycopg2  # type: ignore

        return psycopg2.connect
    except ImportError as exc:
        raise ImportError(
            "Postgres backends require psycopg (v3) or psycopg2. Install one of them."
        ) from exc


def _execute_sql(
    connection: Any, query: str, params: list[Any]
) -> tuple[list[Any], list[str]]:
    cursor = connection.cursor()
    try:
        cursor.execute(query, params)
        columns = [desc[0] for desc in (cursor.description or [])]
        rows = list(cursor.fetchall())
        return rows, columns
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            close()


def _row_to_mapping(row: Any, columns: Sequence[str]) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        return {columns[i]: row[i] for i in range(min(len(columns), len(row)))}
    return {}


def _quote_table_name(table: str) -> str:
    parts = [part for part in table.split(".") if part]
    if not parts:
        raise ValueError("table must not be empty")
    return ".".join(_quote_identifier(part) for part in parts)


def _quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _response_data(response: Any) -> list[Any]:
    if isinstance(response, Mapping):
        data = response.get("data", [])
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return list(data)
        return []

    data = getattr(response, "data", None)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return list(data)
    return []


def _pinecone_fetch(index: Any, ids: list[str], namespace: str | None) -> Any:
    if namespace is None:
        try:
            return index.fetch(ids=ids)
        except TypeError:
            return index.fetch(ids)

    try:
        return index.fetch(ids=ids, namespace=namespace)
    except TypeError:
        return index.fetch(ids, namespace=namespace)


def _qdrant_retrieve(client: Any, collection_name: str, ids: list[str]) -> list[Any]:
    try:
        response = client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
    except TypeError:
        try:
            response = client.retrieve(
                collection_name, ids, with_payload=True, with_vectors=False
            )
        except TypeError:
            response = client.retrieve(collection_name=collection_name, ids=ids)

    if isinstance(response, Sequence) and not isinstance(response, (str, bytes)):
        return list(response)
    if isinstance(response, Mapping):
        points = response.get("result") or response.get("points") or []
        if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
            return list(points)
    result = getattr(response, "result", None)
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        return list(result)
    points = getattr(response, "points", None)
    if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
        return list(points)
    return []


def _get_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    if hasattr(value, key):
        return getattr(value, key)
    return default


def _coerce_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(decoded, Mapping):
            return dict(decoded)
    return {}


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text != "" else None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if not isinstance(value, Sequence):
        return [value]
    if not value:
        return []

    if any(
        isinstance(item, Sequence) and not isinstance(item, (str, bytes, Mapping))
        for item in value
    ):
        flattened: list[Any] = []
        for item in value:
            if isinstance(item, Sequence) and not isinstance(
                item, (str, bytes, Mapping)
            ):
                flattened.extend(list(item))
            else:
                flattened.append(item)
        return flattened
    return list(value)


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def as_chunk_records(
    fetch_records: FetchRecords, ids: Sequence[str]
) -> list[ChunkRecord]:
    """Helper for quickly normalizing backend callback output to `ChunkRecord` objects."""

    from .compat import coerce_chunk_record

    return [coerce_chunk_record(item) for item in fetch_records(ids)]
