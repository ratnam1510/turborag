# Spec Status

This document maps the PDF specification to the current repository state.

## Source Of Truth

The implementation is being built from [TurboRAG Specification - Understanding Google.pdf](file:///Users/ratnamshah/turborag/TurboRAG%20Specification%20-%20Understanding%20Google.pdf).

## Current Status

### Implemented Now

- Core package scaffolding and packaging.
- Deterministic rotation generation.
- Bit-packed scalar quantization and dequantization.
- Approximate compressed similarity scoring.
- `TurboIndex` with in-memory and memmap-backed shard persistence.
- `GraphBuilder` scaffolding with structured extraction prompts, caching, and community assignment hooks.
- `HybridRetriever` scaffolding for dense plus graph retrieval.
- Compatibility adapters for sidecar adoption in existing RAG stacks.
- Existing-embedding import flow, local record snapshots, and CLI sidecar commands.
- Benchmark harness plus `turborag benchmark` CLI evaluation.
- Side-by-side benchmark comparison against exact float search and optional FAISS baselines.
- Reproducible synthetic local benchmark fixture and artifact scripts.
- HTTP sidecar service with health, index, query, and incremental-ingest endpoints.
- MCP stdio server for TurboRAG query access.
- Expanded test coverage for compression, indexing, adapters, service, and CLI behavior.

### Partially Implemented

- Graph layer: the architecture exists, but production graph persistence, richer routing, and summary embedding fusion are still pending.
- Framework adapters: a practical LangChain-style adapter exists, but full native framework coverage is not complete.
- Service layer: the HTTP API currently targets sidecar query and embedding-level ingest, not raw-document ingest or graph visualisation.
- MCP layer: query access is implemented, but ingest tooling and richer graph-aware tools from the PDF are still pending.
- Benchmark story: side-by-side comparison and local reproducible fixtures exist, but published real-world benchmark artifacts and headline numbers are still pending.

### Not Yet Implemented

- Full ingest pipeline for PDF, markdown, and text.
- Persisted graph/community storage and graph inspection endpoints.
- Full LangChain and LlamaIndex production adapters.
- Docker packaging for the self-hosted service flow.
- End-to-end benchmark-backed positioning claims from the PDF.

## Important Precision Note

The PDF mixes several compression ideas, including QJL, SRHT, sign quantization, and scalar quantization pseudocode. The current implementation uses the consistent, testable subset first: rotated scalar quantization with fixed symmetric bounds.

That means this repository is aligned with the PDF’s direction, but it is not yet a full literal implementation of every line in the specification.

## What “Perfect” Means Right Now

At this stage, correctness means:

- the code should be explicit about what is implemented,
- the docs should not overclaim,
- the migration path should be smooth,
- the service and agent surfaces should reuse the same sidecar logic rather than fork it,
- the unfinished parts should be documented clearly,
- and each future phase should land with tests and updated docs.
