from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .backends import (
    FetchRecords,
    build_chroma_fetch_records,
    build_neon_fetch_records,
    build_pinecone_fetch_records,
    build_postgres_fetch_records,
    build_qdrant_fetch_records,
    build_supabase_fetch_records,
)

ADAPTER_CONFIG_FILE_NAME = "adapter.json"
ADAPTER_CONFIG_SCHEMA_VERSION = 1

__all__ = [
    "ADAPTER_CONFIG_FILE_NAME",
    "ADAPTER_CONFIG_SCHEMA_VERSION",
    "default_adapter_config_path",
    "load_adapter_config",
    "save_adapter_config",
    "maybe_load_adapter_config",
    "build_fetch_records_from_config",
    "normalize_adapter_backend",
    "validate_adapter_config",
]


def default_adapter_config_path(index_path: str | Path) -> Path:
    return Path(index_path) / ADAPTER_CONFIG_FILE_NAME


def load_adapter_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("adapter config must be a JSON object")
    return payload


def save_adapter_config(config: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(config), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return target


def maybe_load_adapter_config(
    index_path: str | Path,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, Path | None]:
    if config_path is not None:
        path = Path(config_path)
        return load_adapter_config(path), path

    default_path = default_adapter_config_path(index_path)
    if not default_path.exists():
        return None, None
    return load_adapter_config(default_path), default_path


def normalize_adapter_backend(value: str) -> str:
    backend = value.strip().lower()
    aliases = {
        "postgresql": "postgres",
        "supabase-postgres": "supabase_postgres",
    }
    return aliases.get(backend, backend)


def build_fetch_records_from_config(config: Mapping[str, Any]) -> FetchRecords:
    validated = validate_adapter_config(config, resolve_env=True)
    backend = validated["backend"]
    options = validated["options"]

    if backend in {"postgres", "postgresql", "supabase_postgres"}:
        return build_postgres_fetch_records(**options)

    if backend == "neon":
        return build_neon_fetch_records(**options)

    if backend == "supabase":
        client = _build_supabase_client(options)
        table = str(options.get("table", "chunks"))
        id_column = str(options.get("id_column", "chunk_id"))
        text_column = str(options.get("text_column", "text"))
        source_doc_column = str(options.get("source_doc_column", "source_doc"))
        page_num_column = str(options.get("page_num_column", "page_num"))
        section_column = str(options.get("section_column", "section"))
        metadata_column = str(options.get("metadata_column", "metadata"))
        return build_supabase_fetch_records(
            client,
            table=table,
            id_column=id_column,
            text_column=text_column,
            source_doc_column=source_doc_column,
            page_num_column=page_num_column,
            section_column=section_column,
            metadata_column=metadata_column,
        )

    if backend == "pinecone":
        index = _build_pinecone_index(options)
        namespace = options.get("namespace")
        return build_pinecone_fetch_records(
            index,
            namespace=None if namespace is None else str(namespace),
            text_key=str(options.get("text_key", "text")),
            source_doc_key=str(options.get("source_doc_key", "source_doc")),
            page_num_key=str(options.get("page_num_key", "page_num")),
            section_key=str(options.get("section_key", "section")),
        )

    if backend == "qdrant":
        client = _build_qdrant_client(options)
        collection_name = _require_str(options, "collection_name")
        return build_qdrant_fetch_records(
            client,
            collection_name=collection_name,
            text_key=str(options.get("text_key", "text")),
            source_doc_key=str(options.get("source_doc_key", "source_doc")),
            page_num_key=str(options.get("page_num_key", "page_num")),
            section_key=str(options.get("section_key", "section")),
        )

    if backend == "chroma":
        collection = _build_chroma_collection(options)
        return build_chroma_fetch_records(collection)

    raise ValueError(f"unsupported adapter backend: {backend}")


def validate_adapter_config(
    config: Mapping[str, Any], *, resolve_env: bool = False
) -> dict[str, Any]:
    backend = normalize_adapter_backend(str(config.get("backend", "")))
    if not backend:
        raise ValueError("adapter config requires 'backend'")

    options_raw = config.get("options", {})
    if not isinstance(options_raw, Mapping):
        raise ValueError("adapter config 'options' must be an object")

    options = {
        key: (_resolve_config_value(value) if resolve_env else value)
        for key, value in options_raw.items()
    }

    if backend in {"postgres", "neon", "supabase_postgres"}:
        _require_option(options, "dsn", resolve_env=resolve_env)
    elif backend == "supabase":
        _require_option(options, "url", resolve_env=resolve_env)
        _require_option(options, "key", resolve_env=resolve_env)
    elif backend == "pinecone":
        _require_option(options, "api_key", resolve_env=resolve_env)
        _require_option(options, "index_name", resolve_env=resolve_env)
    elif backend == "qdrant":
        _require_option(options, "collection_name", resolve_env=resolve_env)
        has_url = _has_option(options, "url", resolve_env=resolve_env)
        has_path = _has_option(options, "path", resolve_env=resolve_env)
        if not has_url and not has_path:
            raise ValueError("qdrant adapter config requires either 'url' or 'path'")
    elif backend == "chroma":
        _require_option(options, "path", resolve_env=resolve_env)
        _require_option(options, "collection_name", resolve_env=resolve_env)
    else:
        raise ValueError(f"unsupported adapter backend: {backend}")

    return {
        "schema_version": int(
            config.get("schema_version", ADAPTER_CONFIG_SCHEMA_VERSION)
        ),
        "backend": backend,
        "options": dict(options),
    }


def _build_supabase_client(options: Mapping[str, Any]) -> Any:
    url = _require_str(options, "url")
    key = _require_str(options, "key")

    try:
        from supabase import create_client  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Supabase adapter requires supabase-py. Install with turborag[adapters]."
        ) from exc

    return create_client(url, key)


def _build_pinecone_index(options: Mapping[str, Any]) -> Any:
    api_key = _require_str(options, "api_key")
    index_name = _require_str(options, "index_name")

    try:
        from pinecone import Pinecone  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Pinecone adapter requires pinecone client. Install with turborag[adapters]."
        ) from exc

    client = Pinecone(api_key=api_key)
    return client.Index(index_name)


def _build_qdrant_client(options: Mapping[str, Any]) -> Any:
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Qdrant adapter requires qdrant-client. Install with turborag[adapters]."
        ) from exc

    url = options.get("url")
    path = options.get("path")
    api_key = options.get("api_key")

    if url is None and path is None:
        raise ValueError("qdrant config requires either 'url' or 'path'")

    kwargs: dict[str, Any] = {}
    if url is not None:
        kwargs["url"] = str(url)
    if path is not None:
        kwargs["path"] = str(path)
    if api_key is not None:
        kwargs["api_key"] = str(api_key)
    return QdrantClient(**kwargs)


def _build_chroma_collection(options: Mapping[str, Any]) -> Any:
    path = _require_str(options, "path")
    collection_name = _require_str(options, "collection_name")

    try:
        import chromadb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Chroma adapter requires chromadb. Install with turborag[adapters]."
        ) from exc

    client = chromadb.PersistentClient(path=path)
    return client.get_collection(collection_name)


def _resolve_config_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("${") and stripped.endswith("}"):
            env_name = stripped[2:-1].strip()
            return os.getenv(env_name)
        if stripped.startswith("$") and len(stripped) > 1:
            return os.getenv(stripped[1:])
    return value


def _has_option(options: Mapping[str, Any], key: str, *, resolve_env: bool) -> bool:
    if key not in options:
        return False
    value = options.get(key)
    if resolve_env:
        value = _resolve_config_value(value)
    if value is None:
        return False
    return str(value).strip() != ""


def _require_option(options: Mapping[str, Any], key: str, *, resolve_env: bool) -> str:
    value = options.get(key)
    if value is None:
        raise ValueError(f"adapter option '{key}' is required")
    if resolve_env:
        value = _resolve_config_value(value)
    text = str(value).strip()
    if not text:
        raise ValueError(f"adapter option '{key}' must not be empty")
    return text


def _require_str(options: Mapping[str, Any], key: str) -> str:
    value = options.get(key)
    if value is None:
        raise ValueError(f"adapter option '{key}' is required")
    text = str(value).strip()
    if not text:
        raise ValueError(f"adapter option '{key}' must not be empty")
    return text
