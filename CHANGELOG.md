# Changelog

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
- Side-by-side benchmark comparison against exact float search and optional FAISS baselines, plus local benchmark fixture and artifact scripts.
- An MCP query server over stdio for agent integration.
- An HTTP sidecar service with health, index, query, and incremental-ingest endpoints.
- Project documentation covering architecture, implementation status, roadmap, and spec-level decisions.
- Adoption documentation for zero-database-migration rollout.
- Explicit spec-status and LLM-request-model documentation.
- Service documentation and tests for compression, indexing, adapters, ingest, service, benchmark, and CLI behavior.
- Explicit current-RAG rollout documentation covering install choices, sidecar deployment modes, and cutover flow.

### Notes

- The current build is a disciplined first slice of the larger TurboRAG vision, not the full end-state product described in the source specification.
- Raw-document ingest, persisted graph APIs, deeper framework adapters, Docker packaging, and published benchmark artifacts are still pending.
