# Implementation Plan

## Status Snapshot

The exact mapping from the PDF to current implementation status is documented in [docs/spec-status.md](file:///Users/ratnamshah/turborag/docs/spec-status.md).

### Implemented

- Core package and packaging metadata.
- Rotation generation.
- Bit-packing quantization utilities.
- Approximate compressed scoring.
- In-memory and on-disk compressed vector indexing.
- Graph construction hooks with caching.
- Hybrid retrieval scaffolding.
- Compatibility adapters for sidecar adoption with existing RAG stacks.
- Existing-embedding import/sync flow and sidecar CLI commands.
- Benchmark harness, side-by-side baseline comparison, and local artifact generation flow.
- HTTP sidecar service and MCP query server.
- Core documentation and test coverage.

### Not Yet Implemented

- Document ingestion for PDF, markdown, and plain text.
- Richer embedding model wrappers and provider integrations.
- Cross-encoder reranking.
- Deeper LangChain integration and a full LlamaIndex adapter.
- Persisted graph/community storage and graph API surfaces.
- Docker packaging for the service deployment path.
- Reproducible published benchmark artifacts on real external datasets.

## Recommended Build Sequence

### Phase 1

Stabilise the core numeric path.

- Add batch search support.
- Replace the prototype dequantize-and-matmul scorer with a faster kernel.
- Profile shard search and add top-k optimisation for large shard counts.
- Add precision and recall benchmarks against brute-force float32 search.

### Phase 2

Make ingestion real.

- Add token-aware chunking.
- Add PDF extraction and markdown/plain-text ingestion.
- Introduce a metadata store for chunk text, source path, page number, and section.
- Wire the ingest path into `GraphBuilder` and `TurboIndex`.

### Phase 3

Deepen graph retrieval.

- Replace exact string entity matching with a query entity extractor.
- Add proper relationship weighting into BFS scoring.
- Persist graph and community summaries to disk.
- Introduce community summary embeddings for retrieval-time fusion.

### Phase 4

Developer ergonomics and distribution.

- Extend the service layer with raw-document ingest and graph inspection endpoints.
- Harden the MCP surface with ingest and graph-aware tools.
- Add LangChain and LlamaIndex adapters.
- Package the service into a Docker deployment flow.

### Phase 5

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
