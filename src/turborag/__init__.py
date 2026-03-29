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
from .compress import compressed_dot, dequantize_qjl, generate_rotation, quantize_qjl
from .embeddings import Embedder, SentenceTransformerEmbedder
from .graph import ENTITY_PROMPT, GraphBuilder
from .hybrid import HybridRetriever
from .ingest import ImportedDataset, build_sidecar_index, load_dataset, open_sidecar_adapter
from .index import TurboIndex
from .types import ChunkRecord, RetrievalResult

__all__ = [
    "BenchmarkCase",
    "BenchmarkComparison",
    "BenchmarkReport",
    "BenchmarkSuite",
    "ChunkRecord",
    "ENTITY_PROMPT",
    "Embedder",
    "ExactSearchBackend",
    "FaissSearchBackend",
    "ExistingRAGAdapter",
    "GraphBuilder",
    "HybridRetriever",
    "ImportedDataset",
    "RetrievalResult",
    "SentenceTransformerEmbedder",
    "TurboRetriever",
    "TurboIndexBackend",
    "TurboIndex",
    "TurboVectorStore",
    "available_baselines",
    "build_baselines",
    "build_faiss_backend",
    "build_sidecar_index",
    "compressed_dot",
    "dequantize_qjl",
    "generate_rotation",
    "load_dataset",
    "load_query_cases",
    "open_sidecar_adapter",
    "quantize_qjl",
    "write_benchmark_artifact",
]
