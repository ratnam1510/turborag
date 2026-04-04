"""Compatibility adapters for adopting TurboRAG incrementally."""

from .backends import (
    FetchRecords,
    as_chunk_records,
    build_chroma_fetch_records,
    build_neon_fetch_records,
    build_pinecone_fetch_records,
    build_postgres_fetch_records,
    build_qdrant_fetch_records,
    build_supabase_fetch_records,
)
from .config import (
    ADAPTER_CONFIG_FILE_NAME,
    ADAPTER_CONFIG_SCHEMA_VERSION,
    build_fetch_records_from_config,
    default_adapter_config_path,
    load_adapter_config,
    maybe_load_adapter_config,
    normalize_adapter_backend,
    save_adapter_config,
    validate_adapter_config,
)
from .compat import ExistingRAGAdapter, resolve_records_backend
from .langchain import TurboRetriever, TurboVectorStore

__all__ = [
    "ExistingRAGAdapter",
    "FetchRecords",
    "build_postgres_fetch_records",
    "build_neon_fetch_records",
    "build_supabase_fetch_records",
    "build_pinecone_fetch_records",
    "build_qdrant_fetch_records",
    "build_chroma_fetch_records",
    "as_chunk_records",
    "ADAPTER_CONFIG_FILE_NAME",
    "ADAPTER_CONFIG_SCHEMA_VERSION",
    "default_adapter_config_path",
    "load_adapter_config",
    "save_adapter_config",
    "maybe_load_adapter_config",
    "build_fetch_records_from_config",
    "normalize_adapter_backend",
    "validate_adapter_config",
    "resolve_records_backend",
    "TurboRetriever",
    "TurboVectorStore",
]
