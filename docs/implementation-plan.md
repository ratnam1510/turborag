# Implementation Plan

## Status Snapshot

The exact mapping from the PDF to current implementation status is documented in [docs/spec-status.md](file:///Users/ratnamshah/turborag/docs/spec-status.md).

### Implemented

- Core package and packaging metadata.
- Rotation generation.
- Bit-packing quantization utilities.
- LUT-based C scoring kernel with fused byte-triplet acceleration.
- In-memory and on-disk compressed vector indexing.
- Binary sketch head and adaptive two-stage search (auto/exact/fast modes).
- Batch search and threaded shard scanning.
- Token-aware chunking for PDF, markdown, and plain text.
- Document ingestion with metadata propagation.
- Graph construction hooks with caching and persistence.
- Hybrid retrieval scaffolding.
- Compatibility adapters for sidecar adoption with existing RAG stacks.
- Existing backend bridge for adapters and service hydration (`from_existing_backend`, `resolve_records_backend`).
- Known backend helper builders for Postgres/Neon/Supabase, Pinecone, Qdrant, and Chroma.
- Adapter config persistence and CLI (`turborag adapt`) for plug-and-play backend binding.
- Existing-embedding import/sync flow and sidecar CLI commands.
- Benchmark harness, side-by-side baseline comparison, and local artifact generation flow.
- HTTP sidecar service with CORS, metrics, request tracking, batch queries, and text ingestion.
- Low-memory sidecar mode with optional ID-only query payloads and snapshot-load bypass.
- MCP query, describe, and ingest tools over stdio.
- Domain-specific exception hierarchy.
- Docker packaging with multi-stage production build.
- Core documentation and test coverage (104+ tests).

### Not Yet Implemented

- Richer embedding model wrappers and provider integrations.
- Cross-encoder reranking.
- Deeper LangChain integration and a full LlamaIndex adapter.
- Persisted graph/community storage and graph API surfaces.
- Reproducible published benchmark artifacts on real external datasets.

## Recommended Build Sequence

### Phase 1 (Done)

Stabilise the core numeric path.

- Batch search support.
- LUT-based C scoring kernel replacing the prototype dequantize-and-matmul scorer.
- Fused byte-triplet acceleration for 2-bit and 4-bit paths.
- Top-k optimisation and threaded shard scanning.
- Precision and recall benchmarks against brute-force float32 search.

### Phase 2 (Done)

Make ingestion real.

- Token-aware chunking.
- PDF extraction and markdown/plain-text ingestion.
- Metadata store for chunk text, source path, page number, and section.
- Ingest path wired into `GraphBuilder` and `TurboIndex`.

### Phase 3 (Done)

Production hardening and developer ergonomics.

- CORS middleware and configurable origins.
- `/metrics` endpoint with latency histograms per endpoint.
- Request ID tracking and structured logging.
- Multi-worker support for the HTTP service.
- Domain-specific exception hierarchy (`TurboRAGError`).
- Docker packaging with pre-compiled C kernel.
- MCP ingest tool alongside query and describe.

### Phase 4 (Mostly Done)

Deepen graph retrieval and distribution.

- Extend the service layer with raw-text ingest and graph inspection endpoints.
- Harden the MCP surface with ingest and graph-aware tools.
- Add LangChain and LlamaIndex adapters.
- Persisted graph and community summaries to disk.

Remaining: deeper LangChain/LlamaIndex adapters, persisted graph API surfaces.

### Phase 5 (Done)

Adaptive search and sketch head.

- Binary SimHash sketch generation and persistence (`.sketch.bin` per shard).
- Two-stage search: POPCNT Hamming pre-filter followed by full LUT refine.
- Three search modes: `auto`, `exact`, `fast`.
- `auto` mode selects strategy based on index size.
- Atomic persistence with stale shard cleanup on `save()`.

### Phase 6

Proof and positioning.

- Publish real-world benchmark scripts and results on external datasets, building on the local fixture and comparison tooling.
- Add example notebooks and end-to-end demos.
- Harden the public API for a TestPyPI release.

## Documentation Expectations For Each Future Phase

Every phase should update all of the following, not just code:

- `README.md`
- `CHANGELOG.md`
- `docs/architecture.md`
- `docs/spec-decisions.md`
- tests and verification notes

That keeps the implementation, project narrative, and technical claims aligned.
