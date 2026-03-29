# TurboRAG

TurboRAG is a Python package scaffold for the architecture described in `TurboRAG Specification - Understanding Google.pdf`. This repository now contains a working implementation of the compressed indexing layer, graph and hybrid retrieval primitives, sidecar adoption tooling, benchmark utilities, an MCP query server, and an HTTP service surface for deployment.

The codebase is being built from that PDF, but it does not yet implement every section literally. The current implementation status is tracked in [docs/spec-status.md](docs/spec-status.md).

## What Exists Today

- Deterministic rotation generation for embedding compression.
- Bit-packed scalar quantization and decompression utilities.
- `TurboIndex`, a compressed vector index with in-memory storage and shard-based on-disk persistence.
- `GraphBuilder`, a pluggable graph construction layer that can use an LLM client and optionally Leiden community detection.
- `HybridRetriever`, which merges dense retrieval with graph-guided expansion and emits result explanations.
- Compatibility adapters that let an existing RAG system keep its current database and swap retrieval underneath it.
- Import/sync tooling and a CLI for building a TurboRAG sidecar from existing embeddings.
- A benchmark harness and CLI for recall and MRR evaluation.
- An MCP query server for stdio-based agent integration.
- An HTTP sidecar service with health, index-inspection, query, and incremental-ingest endpoints.
- A documentation set that captures architecture, implementation choices, roadmap, and the exact status of the project.

## What Does Not Exist Yet

The original PDF still describes a larger product surface than what ships today. The main remaining gaps are raw-document ingest for PDF/markdown/plain text, persisted graph APIs and graph visualisation, deeper framework-native adapters, Docker packaging, and reproducible benchmark artifacts with headline numbers.

## Package Layout

```text
src/turborag/
  __init__.py
  adapters/
  benchmark.py
  cli.py
  compress.py
  embeddings.py
  graph.py
  hybrid.py
  ingest.py
  index.py
  mcp_server.py
  service.py
  types.py
docs/
  adoption.md
  architecture.md
  import-existing.md
  implementation-plan.md
  llm-request-model.md
  service.md
  spec-decisions.md
  spec-status.md
tests/
  test_adapters.py
  test_benchmark.py
  test_cli.py
  test_compress.py
  test_ingest.py
  test_index.py
  test_service.py
```

## Core Concepts

### Compression

The prototype uses three steps:

1. Optional L2 normalisation to keep vector magnitude bounded and search numerically stable.
2. A deterministic orthogonal rotation generated from a seed.
3. Uniform bit-packing quantization into 2, 3, or 4 bits per dimension.

This is intentionally simple and production-safe for a first cut. The PDF blends several ideas together, including Johnson-Lindenstrauss style projection, sign quantization, and scalar quantization. The current implementation chooses a scalar-quantized rotated representation because it supports incremental indexing cleanly and keeps the code auditable.

### Indexing

`TurboIndex` stores compressed vectors in either:

- memory-only shards for local or test usage, or
- on-disk shard files backed by `numpy.memmap` for larger corpora.

Search rotates and quantizes the query vector with the same parameters, computes approximate scores against each shard, and returns the highest-scoring IDs.

### Graph Retrieval

`GraphBuilder` accepts an LLM client with a `complete(prompt: str) -> str` method. It caches responses in SQLite when a cache directory is configured. When graph extras are available, it can run Leiden community detection; otherwise it falls back to connected components.

`HybridRetriever` combines dense matches from `TurboIndex` with graph-derived candidates gathered by breadth-first search over named entities that appear in the query.

### Adoption Without Database Changes

TurboRAG can act as a sidecar retrieval engine instead of a replacement database.

- Keep your current chunk table, document store, or metadata database.
- Build a TurboRAG index using the same chunk IDs you already store.
- Let TurboRAG return ranked IDs.
- Hydrate those IDs from your existing database exactly the way your app does today.

The concrete integration path is documented in [docs/adoption.md](docs/adoption.md) and the full rollout sequence is in [docs/current-rag-rollout.md](docs/current-rag-rollout.md).

### Import Existing Embeddings

TurboRAG now supports importing existing embeddings from JSONL or NPZ into a sidecar index and querying that sidecar through a CLI or Python helpers. The import format and command flow are documented in [docs/import-existing.md](docs/import-existing.md).

### Benchmarking

TurboRAG ships with a benchmark harness for recall and mean reciprocal rank over precomputed query vectors and relevant IDs. It supports TurboRAG-only evaluation as well as side-by-side comparison against exact float search and optional FAISS baselines when the `faiss` module is available.

### Service And Agent Surfaces

TurboRAG can also be exposed without embedding it into application code:

- `turborag serve` starts an HTTP sidecar service over an existing index.
- `turborag mcp` starts an MCP stdio server for tool-based agent integration.

The current service and MCP layers are intentionally thin wrappers over the sidecar index, not full replacements for the richer ingest and graph APIs described in the PDF.

### LLM Requests And Cost

If you switch from a normal vector store to TurboRAG dense retrieval, the number of LLM requests in a standard RAG pipeline usually stays about the same. TurboRAG primarily improves the retrieval layer: compression, memory footprint, and retrieval efficiency.

- Dense TurboRAG alone does not inherently remove your final answer-generation call.
- The graph layer can add build-time LLM calls for extraction and summarisation.
- Better retrieval can still reduce total spend indirectly by improving relevance and reducing wasted context.

The full explanation is in [docs/llm-request-model.md](docs/llm-request-model.md).

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

Core install:

```bash
pip install -e .
```

With graph tooling:

```bash
pip install -e '.[graph]'
```

With developer tooling:

```bash
pip install -e '.[dev]'
```

With local text-query CLI support:

```bash
pip install -e '.[embed]'
```

With HTTP service support:

```bash
pip install -e '.[serve]'
```

With MCP support:

```bash
pip install -e '.[mcp]'
```

For the most common existing-RAG evaluation setup:

```bash
pip install -e '.[dev,serve,mcp]'
```

## Side-By-Side Benchmark

```bash
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

For a deterministic local fixture and artifact generation, see [docs/benchmarking.md](docs/benchmarking.md).

## Run As A Service

```bash
turborag serve --index ./turborag_sidecar --host 0.0.0.0 --port 8080
```

Available endpoints:

- `GET /health`
- `GET /index`
- `POST /query`
- `POST /ingest`

## Verification

Run the test suite with:

```bash
pytest
```

## Documentation

- [Adoption Guide](docs/adoption.md)
- [Architecture](docs/architecture.md)
- [Benchmarking](docs/benchmarking.md)
- [Current RAG Rollout](docs/current-rag-rollout.md)
- [Import Existing Data](docs/import-existing.md)
- [Implementation Plan](docs/implementation-plan.md)
- [LLM Request Model](docs/llm-request-model.md)
- [Service API](docs/service.md)
- [Spec Status](docs/spec-status.md)
- [Spec Decisions](docs/spec-decisions.md)
- [Change Log](CHANGELOG.md)
