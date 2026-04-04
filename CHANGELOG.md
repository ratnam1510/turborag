# Changelog

## Unreleased

### New Features

- Added `ExistingRAGAdapter.from_existing_backend(...)` plus `resolve_records_backend(...)` so TurboRAG can attach directly on top of existing database/client integrations without schema migration.
- Expanded compatibility hydration to accept common backend payload shapes (`fetch/retrieve/get`, mapping stores, record/match/point responses, and nested metadata payloads).
- Added optional ID-only retrieval mode in the HTTP service via `hydrate=false` for `POST /query` and `POST /query/batch`.
- Added low-memory serving flags to CLI:
  - `--no-load-snapshot` skips loading `records.jsonl` into memory at startup.
  - `--require-hydrated` drops hits that cannot be hydrated.
- Added low-memory CLI query mode with `turborag query --ids-only`.
- Added known backend helper builders in `turborag.adapters.backends` for Postgres/Neon/Supabase, Pinecone, Qdrant, and Chroma.
- Added plug-and-play adapter config + CLI workflow:
  - `turborag adapt` auto mode (env-based backend detection)
  - `turborag adapt set ...`
  - `turborag adapt supabase|neon|pinecone|qdrant|chroma ...`
  - `turborag adapt show`
  - `turborag adapt remove`
  - `turborag adapt demo <backend>`
  - `turborag serve --adapter-config ...`
- Added TypeScript client support for ID-only query flows:
  - `query({ ..., hydrate })`
  - `queryBatch({ ..., hydrate })`
  - `queryIds(...)`
  - `queryBatchIds(...)`

### Tests

- Added adapter, ingest, service, and CLI tests covering:
  - external backend hydration,
  - unhydrated ID-only results,
  - service low-memory startup flags,
  - TypeScript client hydration toggles and ID-only helpers.

### Documentation

- Updated README and adoption/service/import/architecture/rollout docs with explicit "run on top of existing DB" guidance and low-memory/ID-only retrieval paths.

## 0.4.0 - 2026-04-01

### New Features

- **Adaptive two-stage search** with three modes: `exact`, `fast`, and `auto` (system-selected).
- **Binary sketch head** — SimHash sign-bit pre-filter with C-level POPCNT Hamming scanner. At index time, each vector's rotated sign bits are stored as a 48-byte sketch. At query time, XOR + POPCNT scans all sketches in ~0.5ms, then the top candidates are refined with the full LUT scorer.
- Result: **4× speedup** over exhaustive on 100K vectors (67 → 274 QPS) with 97.5% recall.
- `mode="auto"` picks `exact` for indexes <10K vectors and `fast` for larger ones.
- Sketches are persisted alongside shard files (`.sketch.bin`), survive save/load/delete round-trips.
- `TurboIndexBackend` benchmark backend now supports `mode` parameter for side-by-side exact vs fast comparison.

### Documentation

- Updated README with memory columns in all benchmark tables.
- Changed source reference from local PDF to the actual TurboQuant/QJL/PolarQuant papers (Google Research, ICLR/AAAI/AISTATS 2025-2026).
- Updated spec-status and spec-decisions to reference the published papers.

## 0.3.0 - 2026-03-30

### Performance

- Rewrote the 3-bit C scoring kernel with a **fused byte-triplet approach**: processes 3 bytes (8 dims) at a time with precomputed per-byte LUTs instead of per-dimension extraction.
- Result: **4.4× speedup** on large-scale benchmark (15 → 65 QPS at 100K×384 3-bit).
- Avoided redundant `np.ascontiguousarray` copies in ctypes wrapper when arrays are already contiguous.

### Production Hardening

- **Atomic save/load**: `save()` now clears stale shard files before writing, preventing deleted vectors from reappearing.
- **C scorer**: replaced VLAs with heap allocation to prevent stack overflow on large dims; added input shape validation in the ctypes wrapper.
- Removed `allow_pickle=True` from NPZ loading (security fix).
- Renamed `IndexError` to `IndexOperationError` to stop shadowing the Python builtin.
- **HTTP service**: added `threading.Lock` around ingest mutations for concurrency safety; all handlers now catch `TurboRAGError`.
- `TurboIndex.open()` skips expensive rotation generation since `load()` overwrites it immediately.
- `add()` now processes vectors in shard-sized chunks to avoid memory spikes on large batches.

### Bug Fixes

- Fixed C kernel stack overflow risk on high-dimensional vectors by switching from VLAs to heap allocation.
- Fixed potential stale shard data after deletions by making `save()` atomic.
- Fixed `IndexError` name collision with Python builtin.

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
