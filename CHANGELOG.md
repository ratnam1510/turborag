# Changelog

## 0.2.0 - 2026-03-30

Production-grade overhaul targeting performance, hardening, and feature completeness.

### Performance

- Replaced the naive dequantize-then-matmul scorer with a **LUT-based scoring kernel** that operates directly on packed bytes — eliminates the float32 memory blowup entirely.
- Added a **C scoring kernel** (`_cscore.c`) with ctypes auto-compilation, column-major traversal for cache efficiency, and `-O3 -march=native` optimisation.
- Result: **18.6× speedup** over v0.1 prototype (393 → 7,236 QPS) with perfect recall preserved.
- Added batch search (`search_batch`) and threaded shard scanning for multi-query workloads.

### Production Hardening

- Added **domain-specific exception hierarchy** (`TurboRAGError`, `DuplicateIDError`, `IDNotFoundError`, `ChunkingError`, etc.).
- Added `delete()` and `update()` operations to `TurboIndex`.
- Service layer: CORS middleware, `/metrics` endpoint with latency histograms, request ID tracking (`X-Request-Id`), structured logging with timing.
- CLI: global `--log-level` and `--log-format` options, `--workers` and `--cors-origins` for serve command, progress bars for import operations.
- `describe-index` now reports C kernel availability.

### New Features

- **Document chunker** (`chunker.py`): token-aware chunking for PDF, markdown, and plain text with tiktoken, configurable chunk size/overlap, metadata propagation.
- **Graph persistence**: `GraphBuilder.save()` and `GraphBuilder.load()` write/read GraphML + community/summary JSON.
- **Enriched MCP server**: three tools — `turborag_query`, `turborag_describe`, `turborag_ingest`.
- **New service endpoints**: `POST /query/batch`, `POST /ingest-text` (auto-chunking), `GET /metrics`.
- **Dockerfile**: multi-stage production build with pre-compiled C kernel, health check, volume mount.

### Tests

- Expanded from 49 to **104+ tests** covering service hardening (CORS, metrics, batch, error handling), fast kernels, chunker, graph, hybrid, edge cases, integration, and index operations.

### Documentation

- Updated README with real benchmark numbers and full feature inventory.
- Updated spec-status to reflect implemented features.
- All docs aligned with the current code reality.

## 0.1.0 - 2026-03-28

Initial repository bootstrap from the PDF specification.

### Added

- Python package skeleton in `src/turborag`.
- Compression utilities for rotation generation, bit packing, quantization, dequantization, and approximate dot product.
- `TurboIndex` with in-memory storage, memmap-backed shard persistence, save/load support, and approximate search.
- `GraphBuilder` with prompt-driven entity extraction, SQLite caching, optional Leiden clustering, and community summarisation hooks.
- `HybridRetriever` and retrieval dataclasses for dense, graph, and hybrid retrieval flows.
- Compatibility adapters for sidecar adoption with existing RAG databases and familiar vector-store-style methods.
- Dataset import helpers, sidecar snapshots, and a `turborag` CLI for importing and querying existing embedding exports.
- Benchmark utilities and CLI evaluation flow for recall and MRR.
- Side-by-side benchmark comparison against exact float search and optional FAISS baselines.
- An MCP query server over stdio for agent integration.
- An HTTP sidecar service with health, index, query, and incremental-ingest endpoints.
- Project documentation covering architecture, implementation status, roadmap, and spec-level decisions.
