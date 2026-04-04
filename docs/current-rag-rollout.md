# Current RAG Rollout

This document explains exactly how TurboRAG fits into an existing RAG system that already has:

- a metadata database or document store,
- an embedding model,
- chunk IDs already in production,
- and an existing query pipeline.

The key point is simple: TurboRAG does not need to replace the current database. It can sit beside it.

## What A Team Installs

### From PyPI

```bash
# Core
pip install turborag

# With text-query embedding
pip install turborag[embed]

# With HTTP sidecar service
pip install turborag[serve]

# With MCP agent access
pip install turborag[mcp]

# Everything
pip install turborag[all]
```

### From Source

```bash
git clone https://github.com/ratnam1510/turborag.git
cd turborag
pip install -e '.[all,dev]'
```

## What Stays The Same

In a current RAG stack, these pieces usually do not change:

- the chunk table or document store,
- the source-of-truth metadata database,
- the chunk IDs,
- the answer-generation LLM call,
- the app's document hydration path,
- existing auth, tenancy, and business logic.

## What TurboRAG Replaces

TurboRAG replaces only the retrieval engine that ranks chunk IDs from embeddings.

That means the app still fetches the final chunk text and metadata from the same database it already trusts.

## Flow 1: Embedded In The Existing App

This is the lowest-latency integration.

### One-time setup

1. Export existing embeddings and chunk IDs from the current store.
2. Build a TurboRAG sidecar index from that export.
3. Keep the current database untouched.

### Runtime query flow

1. The user asks a question.
2. The app embeds the query with the same embedder it already uses.
3. TurboRAG searches the compressed sidecar index and returns ranked `chunk_id` values.
4. The app fetches those chunk IDs from the existing database.
5. The app sends the hydrated chunk text into the normal answer-generation step.

### Python integration sketch

```python
from turborag.adapters.compat import ExistingRAGAdapter

adapter = ExistingRAGAdapter.from_embeddings(
    embeddings=embeddings_matrix,
    ids=chunk_ids,
    query_embedder=embedder,
    fetch_records=fetch_records,
    bits=3,
    shard_size=100_000,
)

results = adapter.query("What changed in capex guidance?", k=5)
```

`fetch_records(ids)` should call the current database and return the same records the app already knows how to use.

## Flow 2: HTTP Sidecar Beside The Existing App

This is the easiest operational model when the team does not want TurboRAG inside the main app process.

### One-time setup

1. Export current embeddings and chunk IDs.
2. Build a TurboRAG sidecar index.
3. Configure adapter binding from environment (optional, but recommended for external DB hydration):

```bash
turborag adapt --index ./turborag_sidecar
```

3. Start the TurboRAG service with that index.

### Runtime query flow

1. The application embeds the query using its current embedding model.
2. The application sends the vector to TurboRAG over HTTP.
3. TurboRAG returns ranked `chunk_id` values and, when a snapshot exists, local chunk text.
4. The application can either:
   - trust the sidecar snapshot for fast local hydration, or
   - fetch the chunk IDs from its current database.
5. The application continues the normal answer-generation flow.

### Service startup

```bash
turborag serve --index ./turborag_sidecar --host 0.0.0.0 --port 8080
```

If you already configured adapter auto mode, this uses `adapter.json` automatically.

Memory-minimal startup (no snapshot hydration load):

```bash
turborag serve --index ./turborag_sidecar --host 0.0.0.0 --port 8080 --no-load-snapshot
```

### Query example

```bash
curl -X POST http://localhost:8080/query \
  -H 'content-type: application/json' \
  -d '{"query_vector":[0.1,0.2,0.3],"top_k":5}'
```

ID-only response path (hydrate in your existing DB):

```bash
curl -X POST http://localhost:8080/query \
  -H 'content-type: application/json' \
  -d '{"query_vector":[0.1,0.2,0.3],"top_k":5,"hydrate":false}'
```

## What The Team Actually Does In Order

### Phase A: Evaluate offline

1. Export a slice of current chunks, embeddings, and IDs.
2. Build a TurboRAG sidecar.
3. Run `turborag benchmark` with `--baseline exact` and optional FAISS baselines to compare TurboRAG against side-by-side references.

### Phase B: Shadow mode

1. Keep the current retriever live.
2. Run TurboRAG for the same queries in parallel.
3. Compare latency, overlap, and answer quality.

### Phase C: Partial cutover

1. Route a small percentage of retrieval traffic to TurboRAG.
2. Keep hydration in the current database.
3. Watch quality and operational stability.

### Phase D: Full retrieval cutover

1. Make TurboRAG the retrieval layer.
2. Keep the current metadata database as the source of truth.
3. Continue writing new chunks through the existing ingest path, then append matching embeddings into TurboRAG.

## What They Need To Export

TurboRAG needs:

- `chunk_id`
- `text`
- `embedding`

Optional but strongly recommended:

- `source_doc`
- `page_num`
- `section`
- `metadata`

## Write Path For New Data

For new chunks after rollout:

1. Write the new record into the current database exactly as before.
2. Compute the embedding exactly as before.
3. Add the new embedding to TurboRAG using the same `chunk_id`.

This preserves the no-database-migration model.

## Honest Current Limits

What works now:

- embedding-level sidecar import,
- embedded Python adapter flow,
- HTTP sidecar query flow,
- incremental embedding-level ingest,
- benchmark and MCP surfaces.

What does not exist yet:

- persisted graph APIs and graph visualisation endpoints,
- deeper production-native LangChain and LlamaIndex adapters,
- published benchmark artifacts on real external datasets.
