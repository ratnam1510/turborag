# TurboRAG Architecture

## Goal

Build a retrieval library that combines:

- a compressed dense retrieval layer,
- an entity graph for multi-hop expansion, and
- a clean path toward adapters, ingestion, benchmarking, and services.

## Current Module Map

### `turborag.compress`

Implements the compression primitives used by the rest of the package.

- `generate_rotation(dim, seed)` builds the deterministic orthogonal transform.
- `quantize_qjl(...)` and `dequantize_qjl(...)` encode and decode packed vectors.
- `compressed_dot(...)` provides prototype approximate scoring.

### `turborag.index`

Implements `TurboIndex`, the current storage and retrieval engine.

- Validates vector shapes and IDs.
- Normalises vectors before compression by default.
- Persists shard files plus a `config.json` and `rotation.npy`.
- Loads saved indexes with `numpy.memmap` so large indexes do not need full RAM materialisation.

### `turborag.graph`

Implements `GraphBuilder`.

- Uses a structured extraction prompt.
- Caches LLM output in SQLite.
- Builds a `networkx.Graph` when graph dependencies are installed.
- Assigns communities with Leiden when available, otherwise uses connected components.

### `turborag.hybrid`

Implements `HybridRetriever`.

- Runs dense retrieval through `TurboIndex`.
- Detects query entities by graph node name match.
- Expands graph paths with breadth-first search.
- Merges dense and graph candidates and emits explanations.

### `turborag.adapters`

Implements the adoption layer for existing RAG systems.

- `ExistingRAGAdapter` lets TurboRAG reuse an application's current chunk IDs and metadata store.
- `TurboVectorStore` exposes familiar vector-store-like methods such as `from_texts`, `from_existing_records`, and `similarity_search`.
- This is the main path for adopting TurboRAG without changing the current database schema.

### `turborag.ingest`

Implements the sidecar build and snapshot-loading workflow for existing RAG systems.

- Loads JSONL and NPZ datasets containing embeddings and chunk records.
- Builds a TurboRAG sidecar index from those embeddings.
- Writes `records.jsonl` snapshots for self-contained local hydration.
- Reopens sidecar indexes as `ExistingRAGAdapter` instances.

### `turborag.cli`

Implements the current command-line DX layer.

- `import-existing-index` builds a sidecar from an embedding export.
- `query` supports precomputed vectors and optional local text embedding.
- `describe-index` shows index metadata for inspection and debugging.
- `benchmark` evaluates recall and MRR from JSONL query cases.
- `serve` exposes the sidecar over HTTP.
- `mcp` exposes the sidecar over MCP stdio.

### `turborag.benchmark`

Implements offline retrieval evaluation.

- Loads query cases from JSONL.
- Computes recall and mean reciprocal rank over a `TurboIndex`.
- Produces human-readable and JSON reports for CLI or CI use.

### `turborag.service`

Implements the HTTP deployment surface over an existing sidecar index.

- Exposes health and index metadata endpoints.
- Accepts vector queries and optional text queries when a query embedder is configured.
- Supports incremental embedding-level ingest that appends to the existing sidecar and refreshes the record snapshot.

### `turborag.mcp_server`

Implements the MCP stdio server.

- Loads an existing sidecar index.
- Exposes TurboRAG retrieval as an MCP tool for agent clients.
- Reuses the same index and records snapshot layout as the CLI and HTTP service.

### `turborag.types`

Holds the user-facing dataclasses.

- `ChunkRecord`
- `RetrievalResult`

## Storage Layout

Saved indexes currently follow this layout:

```text
index/
  config.json
  records.jsonl
  rotation.npy
  sketch.bin
  shards/
    shard_000.bin
    shard_000.ids.json
    shard_000.sketch.bin
    shard_001.bin
    shard_001.ids.json
    shard_001.sketch.bin
```

`config.json` stores the compression and index settings. Each `.bin` file is a raw packed byte matrix with one row per vector and one file per shard. `.sketch.bin` files store binary SimHash sketches used by the adaptive search pre-filter. `records.jsonl` is optional, but when present it lets the CLI, HTTP service, and MCP server hydrate results without reaching back into an external database.

## Query Flow

TurboRAG supports three search modes selected via `mode`: `auto` (default), `exact`, and `fast`.

1. Accept a float32 query vector.
2. Optionally L2-normalise it.
3. Rotate it with the stored orthogonal matrix.
4. Quantize and bit-pack it using the configured bit width.
5. **Mode selection**:
   - `exact`: Score every vector with the LUT-based C scoring kernel with fused byte-triplet acceleration.
   - `fast`: Compute a binary SimHash sketch of the query, use POPCNT Hamming distance against per-shard `.sketch.bin` files to pre-filter candidates, then refine only the top candidates with the full LUT scorer.
   - `auto`: Select `exact` for small indexes and `fast` for large ones based on index size.
6. Merge top candidates across shards.
7. If hybrid mode is enabled, merge in graph-derived candidates.
8. If running through an adapter, hydrate the ranked chunk IDs from the existing application record store.
9. Return structured results and a human-readable explanation.

## Service Flow

1. Open an existing sidecar index from disk.
2. Load `records.jsonl` when available for local hydration.
3. Accept HTTP queries as either precomputed vectors or text.
4. Route query execution through the same `TurboIndex` and adapter logic used elsewhere in the package.
5. Optionally append new embedding-level records and update the sidecar metadata in place.

All ingest mutations are protected by a `threading.Lock` for concurrency safety when running with multiple workers. All handlers catch `TurboRAGError` and return structured error responses.

## Existing-RAG Import Flow

1. Export existing chunk IDs, text, metadata, and embeddings.
2. Load them with `turborag.ingest.load_dataset(...)`.
3. Build a sidecar with `turborag.ingest.build_sidecar_index(...)`.
4. Reopen that sidecar with `turborag.ingest.open_sidecar_adapter(...)` or the CLI.
5. Keep the original database in place unless you explicitly want a self-contained snapshot.

## Why The First Version Is Intentionally Conservative

The PDF spec mixes several compression ideas and leaves some operational details underspecified. This repository therefore starts with a conservative architecture that is easy to test and safe to evolve:

- scalar quantization instead of an incomplete SRHT pipeline,
- exact deterministic persistence instead of speculative compact rotation reconstruction,
- memmap shards instead of a custom binary container format,
- graceful graph fallbacks when optional dependencies are missing.

That keeps the codebase honest while preserving the path to a more advanced implementation.
