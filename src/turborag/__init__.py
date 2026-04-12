"""TurboRAG public package interface."""

from .adapters import (
    ADAPTER_CONFIG_FILE_NAME,
    ADAPTER_CONFIG_SCHEMA_VERSION,
    ExistingRAGAdapter,
    FetchRecords,
    TurboRetriever,
    TurboVectorStore,
    as_chunk_records,
    build_fetch_records_from_config,
    build_chroma_fetch_records,
    build_neon_fetch_records,
    build_pinecone_fetch_records,
    build_postgres_fetch_records,
    build_qdrant_fetch_records,
    build_supabase_fetch_records,
    default_adapter_config_path,
    load_adapter_config,
    maybe_load_adapter_config,
    normalize_adapter_backend,
    resolve_records_backend,
    save_adapter_config,
    validate_adapter_config,
)
from .benchmark import (
    BenchmarkCase,
    BenchmarkComparison,
    BenchmarkReport,
    BenchmarkSuite,
    ExactSearchBackend,
    FaissSearchBackend,
    TurboIndexBackend,
    available_baselines,
    build_baselines,
    build_faiss_backend,
    load_query_cases,
    write_benchmark_artifact,
)
from .chunker import ChunkConfig, chunk_documents, chunk_file, chunk_text
from .compress import (
    compressed_dot,
    compressed_dot_naive,
    dequantize_qjl,
    generate_rotation,
    quantize_qjl,
)
from .embeddings import Embedder, SentenceTransformerEmbedder
from .exceptions import (
    ChunkingError,
    DatasetFormatError,
    DuplicateIDError,
    EmbeddingError,
    IDNotFoundError,
    IndexConfigError,
    IngestError,
    QueryError,
    ServiceError,
    TurboRAGError,
)
from .fast_kernels import build_query_lut, score_shard_lut, topk_shard_lut
from .graph import ENTITY_PROMPT, GraphBuilder
from .hybrid import HybridRetriever
from .ingest import (
    ImportedDataset,
    build_sidecar_index,
    load_dataset,
    open_sidecar_adapter,
)
from .index import TurboIndex
from .types import ChunkRecord, RetrievalResult

__all__ = [
    # Core
    "TurboIndex",
    "ChunkRecord",
    "RetrievalResult",
    # Compression
    "compressed_dot",
    "compressed_dot_naive",
    "dequantize_qjl",
    "generate_rotation",
    "quantize_qjl",
    # Fast kernels
    "build_query_lut",
    "score_shard_lut",
    "topk_shard_lut",
    # Chunking
    "ChunkConfig",
    "chunk_text",
    "chunk_file",
    "chunk_documents",
    # Graph
    "ENTITY_PROMPT",
    "GraphBuilder",
    "HybridRetriever",
    # Adapters
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
    # Embeddings
    "Embedder",
    "SentenceTransformerEmbedder",
    # Ingest
    "ImportedDataset",
    "build_sidecar_index",
    "load_dataset",
    "open_sidecar_adapter",
    # Benchmark
    "BenchmarkCase",
    "BenchmarkComparison",
    "BenchmarkReport",
    "BenchmarkSuite",
    "ExactSearchBackend",
    "FaissSearchBackend",
    "TurboIndexBackend",
    "available_baselines",
    "build_baselines",
    "build_faiss_backend",
    "load_query_cases",
    "write_benchmark_artifact",
    # Exceptions
    "TurboRAGError",
    "IndexConfigError",
    "DuplicateIDError",
    "IDNotFoundError",
    "IngestError",
    "DatasetFormatError",
    "ChunkingError",
    "ServiceError",
    "QueryError",
    "EmbeddingError",
]
