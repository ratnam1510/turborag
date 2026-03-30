# TurboRAG

TurboRAG is a production-grade compressed vector retrieval engine with graph-augmented search, built from the architecture described in `TurboRAG Specification - Understanding Google.pdf`.

## Performance

| | Recall@10 | MRR@10 | QPS | Memory |
|---|---|---|---|---|
| **TurboRAG (3-bit)** | **1.000** | **1.000** | **7,236** | **0.3 MB** |
| Exact float32 | 1.000 | 1.000 | 27,280 | 2.9 MB |
| FAISS Flat | 1.000 | 1.000 | 30,274 | 2.9 MB |
| FAISS HNSW | 1.000 | 1.000 | 21,600 | 2.9 MB |
| FAISS IVF-PQ | 0.990 | 0.990 | 26,345 | < 1 MB |

*Benchmark: 1,000 vectors × 128 dims, 100 queries, k=10. C scoring kernel enabled.*

**Key advantages:**
- **10.7× memory reduction** vs float32 with perfect recall
- **18.6× faster** than the initial prototype (393 → 7,236 QPS)
- **Zero recall loss** — TurboRAG achieves exact-match recall at 3-bit compression
- At scale (100K+ vectors), memory bandwidth savings dominate and TurboRAG's advantage grows

## What Ships Today

- **Core engine**: Compressed vector index with LUT-based C scoring kernel, batch search, threaded shard scanning
- **Graph retrieval**: Entity extraction, community detection, hybrid dense+graph search with explainability
- **Document ingestion**: Token-aware chunking for PDF, markdown, and plain text with metadata propagation
- **Sidecar adoption**: Drop-in compatibility adapters for existing RAG systems — no database migration required
- **HTTP service**: Production-hardened REST API with CORS, metrics, request tracking, batch queries, text ingestion
- **MCP server**: Tool-based agent integration over stdio (query, describe, ingest)
- **Benchmark suite**: Side-by-side comparison against exact float and FAISS backends
- **CLI**: Full command-line interface for import, query, benchmark, serve, and MCP modes
- **Docker**: Production-ready multi-stage Dockerfile with pre-compiled C kernel

## Package Layout

```text
src/turborag/
  __init__.py
  _cscore.c            # C scoring kernel (auto-compiled)
  _cscore_wrapper.py   # ctypes bridge with auto-compilation
  adapters/            # LangChain-style and generic compatibility adapters
  benchmark.py         # Side-by-side benchmark harness
  chunker.py           # Token-aware PDF/MD/text chunking
  cli.py               # Click-based CLI with global logging options
  compress.py          # Rotation, quantization, LUT-based scoring
  embeddings.py        # Optional sentence-transformers integration
  exceptions.py        # Domain-specific exception hierarchy
  fast_kernels.py      # Vectorised LUT scoring (Python + C dispatch)
  graph.py             # Entity graph with persistence and community detection
  hybrid.py            # Dense + graph hybrid retrieval
  index.py             # TurboIndex with search, batch, delete, update
  ingest.py            # Dataset import and sidecar builder
  mcp_server.py        # MCP stdio server (query, describe, ingest tools)
  service.py           # Starlette HTTP service with CORS, metrics, batch
  types.py             # ChunkRecord, RetrievalResult dataclasses
tests/                 # 104+ tests
Dockerfile             # Multi-stage production build
```

## Quick Start

```python
import numpy as np
from turborag import TurboIndex

rng = np.random.default_rng(42)
index = TurboIndex(dim=8, bits=3, seed=7)
vectors = rng.normal(size=(100, 8)).astype(np.float32)
ids = [f"chunk-{i}" for i in range(len(vectors))]
index.add(vectors, ids)

results = index.search(vectors[0], k=5)
for chunk_id, score in results:
    print(chunk_id, score)
```

## Install

```bash
# Core
pip install -e .

# Everything
pip install -e '.[all]'

# Common dev setup
pip install -e '.[dev,serve,mcp]'
```

Individual extras: `graph`, `embed`, `ingest`, `serve`, `mcp`, `all`, `dev`.

## Run As A Service

```bash
turborag serve --index ./my_index --host 0.0.0.0 --port 8080 --workers 4
```

Endpoints:
- `GET /health` — health check
- `GET /index` — index configuration and stats
- `GET /metrics` — latency histograms and error counts
- `POST /query` — single query (vector or text)
- `POST /query/batch` — batch vector queries
- `POST /ingest` — add records with embeddings
- `POST /ingest-text` — raw text ingestion with auto-chunking

CORS is enabled by default. Use `--cors-origins` to restrict.

## Docker

```bash
docker build -t turborag .
docker run -p 8080:8080 -v ./my_index:/data/index turborag \
  turborag serve --index /data/index --host 0.0.0.0
```

## Side-By-Side Benchmark

```bash
# One-command local comparison
./scripts/benchmark_compare.sh

# Custom benchmark
turborag benchmark \
  --index ./turborag_sidecar \
  --queries ./queries.jsonl \
  --dataset ./corpus.jsonl \
  --baseline exact \
  --baseline faiss-flat \
  --baseline faiss-hnsw \
  --baseline faiss-ivfpq \
  --k 10
```

See [docs/benchmarking.md](docs/benchmarking.md).

## MCP Agent Integration

```bash
turborag mcp --index ./my_index
```

Tools exposed: `turborag_query`, `turborag_describe`, `turborag_ingest`.

## Sidecar Adoption (No Database Migration)

TurboRAG runs as a retrieval sidecar alongside your existing RAG database:

1. Keep your current chunk/document store untouched
2. Build a TurboRAG index from your existing embeddings
3. Query TurboRAG for ranked IDs → hydrate from your existing DB
4. Gradually shift traffic as confidence grows

Full guide: [docs/current-rag-rollout.md](docs/current-rag-rollout.md).

## Verification

```bash
# Test suite (104+ tests)
pytest

# End-to-end smoke test
./scripts/smoke_test.sh
```

## Documentation

- [Adoption Guide](docs/adoption.md)
- [Architecture](docs/architecture.md)
- [Benchmarking](docs/benchmarking.md)
- [Current RAG Rollout](docs/current-rag-rollout.md)
- [Import Existing Data](docs/import-existing.md)
- [LLM Request Model](docs/llm-request-model.md)
- [Service API](docs/service.md)
- [Spec Status](docs/spec-status.md)
- [Spec Decisions](docs/spec-decisions.md)
- [Change Log](CHANGELOG.md)
