# TurboRAG

TurboRAG is a production-grade compressed vector retrieval engine with graph-augmented search, implementing the quantization techniques from Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), [Quantized Johnson-Lindenstrauss (QJL)](https://arxiv.org/abs/2406.03482) (AAAI 2025), and [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) papers by Amir Zandieh and Vahab Mirrokni.

## Performance

### Small Scale (1K vectors, 128-dim, 100 queries, k=10, 4-bit)

| Backend | Recall@10 | MRR | QPS | Memory |
|---|---|---|---|---|
| **TurboRAG 4-bit** | **1.000** | **1.000** | **6,209** | **0.08 MB** |
| Exact float32 | 1.000 | 1.000 | 26,774 | 0.49 MB |
| FAISS Flat | 1.000 | 1.000 | 32,384 | 0.49 MB |
| FAISS HNSW | 1.000 | 1.000 | 23,640 | 0.55 MB |
| FAISS IVF-PQ | 0.990 | 0.990 | 27,438 | < 0.49 MB |

### Large Scale (100K vectors, 384-dim, 200 queries, k=10, 3-bit)

| Backend | Recall@10 | MRR | QPS | Memory | Notes |
|---|---|---|---|---|---|
| **TurboRAG exact** | **1.000** | **1.000** | **67** | **18.3 MB** | Guaranteed perfect recall |
| **TurboRAG fast** | **0.975** | **0.975** | **274** | **18.3 MB** | Binary sketch head + LUT refine |
| Exact float32 | 1.000 | 1.000 | 240 | 146.5 MB | Brute force |
| FAISS Flat | 1.000 | 1.000 | 232 | 146.5 MB | Brute force |
| FAISS HNSW | 0.645 | 0.645 | 1,928 | 152.6 MB | Vectors + graph |

TurboRAG `auto` mode selects the best strategy per query: `exact` for small indexes, `fast` for large ones.

**Key advantages:**
- **8× memory compression** — 18.3 MB vs 146.5 MB float32 at 100K scale, with perfect recall in exact mode
- **Adaptive two-stage search** — binary sketch pre-filter (SimHash + POPCNT) with full LUT refine gives 4× speedup over exact with 97.5% recall
- **Guaranteed exact mode** — `mode="exact"` always gives perfect recall when accuracy is non-negotiable
- **64% higher recall than HNSW** — TurboRAG fast (0.975) vs FAISS HNSW (0.645), while using 8× less memory
- Production-hardened with atomic persistence, concurrency-safe HTTP service, and input validation

## What Ships Today

- **Core engine**: Compressed vector index with adaptive two-stage search (binary sketch head + fused LUT refine), C scoring kernel with POPCNT, batch search, threaded shard scanning
- **Graph retrieval**: Entity extraction, community detection, hybrid dense+graph search with explainability
- **Document ingestion**: Token-aware chunking for PDF, markdown, and plain text with metadata propagation
- **Sidecar adoption**: Drop-in compatibility adapters for existing RAG systems — no database migration required
- **HTTP service**: Production-hardened REST API with CORS, metrics, request tracking, batch queries, text ingestion, concurrency-safe mutations
- **Atomic persistence**: `save()` clears stale shard files before writing, preventing deleted vectors from reappearing
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

### From PyPI

```bash
# Core
pip install turborag

# Everything
pip install turborag[all]
```

### From Source

```bash
git clone https://github.com/ratnam1510/turborag.git
cd turborag
pip install -e '.[all,dev]'
```

Individual extras: `graph`, `embed`, `ingest`, `serve`, `mcp`, `all`, `dev`. See [docs/installation.md](docs/installation.md) for details.

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

- [Installation](docs/installation.md)
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
