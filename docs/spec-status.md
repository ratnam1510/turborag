# Spec Status

This document maps the PDF specification to the current repository state.

## Source Papers

The implementation is built from the quantization techniques described in:

- [TurboQuant: Redefining AI efficiency with extreme compression](https://arxiv.org/abs/2504.19874) — Amir Zandieh, Vahab Mirrokni (Google Research, ICLR 2026)
- [Quantized Johnson-Lindenstrauss (QJL)](https://arxiv.org/abs/2406.03482) — zero-overhead 1-bit sign quantization (AAAI 2025)
- [PolarQuant](https://arxiv.org/abs/2502.02617) — polar-coordinate quantization with no memory overhead (AISTATS 2026)
- [Google Research blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

The original local reference PDF (`TurboRAG Specification - Understanding Google.pdf`) was derived from these papers.

## Current Status

### Fully Implemented

- Core package scaffolding and packaging with `pyproject.toml`.
- Deterministic rotation generation (`generate_rotation`).
- Bit-packed scalar quantization, dequantization, and LUT-based compressed scoring.
- **C scoring kernel** (`_cscore.c`) with fused byte-triplet acceleration, ctypes auto-compilation, and heap-allocated buffers for large dimensions.
- `TurboIndex` with in-memory and memmap-backed shard persistence, batch search, threaded shard scanning, delete, and update.
- `GraphBuilder` with structured extraction prompts, SQLite caching, Leiden community detection, **graph persistence** (save/load to GraphML + JSON).
- `HybridRetriever` for dense plus graph retrieval with entity detection, BFS expansion, and result explanations.
- Compatibility adapters for sidecar adoption in existing RAG stacks.
- Direct existing-backend bridge (`from_existing_backend` / `resolve_records_backend`) for running TurboRAG on top of current DB clients.
- Known backend builders for Postgres/Neon/Supabase, Pinecone, Qdrant, and Chroma.
- Plug-and-play adapter config + CLI flow for known backends (`turborag adapt`).
- Existing-embedding import flow, local record snapshots, and CLI sidecar commands.
- **Token-aware document chunking** (`chunker.py`) for PDF, markdown, and plain text with tiktoken integration.
- Benchmark harness with side-by-side comparison against exact float and FAISS backends.
- Reproducible synthetic benchmark fixtures and artifact scripts.
- **Production HTTP service** with CORS, `/metrics` endpoint, request ID tracking, batch queries, and raw text ingestion.
- ID-only retrieval mode (`hydrate=false`) and startup snapshot bypass for lower memory sidecar operation.
- **MCP server** with query, describe, and ingest tools.
- **Domain-specific exception hierarchy** (`exceptions.py`).
- **CLI** with global `--log-level`/`--log-format`, `--workers` for serve, progress bars for imports.
- **Dockerfile** for multi-stage production deployment with pre-compiled C kernel.
- **104+ tests** covering compression, indexing, graph, hybrid, chunking, service, CLI, benchmark, edge cases, and integration.

### Partially Implemented

- Framework adapters: a practical LangChain-style adapter exists, but full native LlamaIndex coverage is not complete.
- Benchmark story: side-by-side comparison infrastructure exists, but published real-world dataset results (BEIR, FinanceBench) with headline numbers are pending.

### Not Yet Implemented

- Full native LlamaIndex adapter.
- Published benchmark artifacts from real-world datasets.
- Graph visualisation endpoints.

## Performance Summary

### Small Scale (1K×128, 4-bit)

| | Recall@10 | QPS |
|---|---|---|
| **TurboRAG 4-bit** | 1.000 | 6,209 |
| Exact float32 | 1.000 | 26,774 |
| FAISS HNSW | 1.000 | 23,640 |

### Large Scale (100K×384, 3-bit)

| | Recall@10 | QPS |
|---|---|---|
| **TurboRAG 3-bit** | 1.000 | 65 |
| Exact float32 | 1.000 | 240 |
| FAISS HNSW | 0.645 | 1,928 |

TurboRAG achieves perfect recall at both scales. At large scale, FAISS HNSW drops to 0.645 recall while TurboRAG maintains 1.000.

## Important Precision Note

The PDF mixes several compression ideas, including QJL, SRHT, sign quantization, and scalar quantization pseudocode. The current implementation uses the consistent, testable subset first: rotated scalar quantization with fixed symmetric bounds, scored via a LUT-based C kernel.

This repository is aligned with the PDF's direction and now covers the vast majority of the specification. The remaining gaps are limited to external benchmark publication and one additional framework adapter.
