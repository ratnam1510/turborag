"""TurboRAG public package interface."""

from .adapters import ExistingRAGAdapter, TurboRetriever, TurboVectorStore
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
from .compress import compressed_dot, compressed_dot_naive, dequantize_qjl, generate_rotation, quantize_qjl
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
from .fast_kernels import build_query_lut, score_shard_lut
from .graph import ENTITY_PROMPT, GraphBuilder
from .hybrid import HybridRetriever
from .ingest import ImportedDataset, build_sidecar_index, load_dataset, open_sidecar_adapter
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
