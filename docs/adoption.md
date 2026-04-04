# Adoption Guide

## Goal

Let an existing RAG user adopt TurboRAG without rewriting application code or migrating their metadata database.

For the full install and rollout sequence, see [docs/current-rag-rollout.md](file:///Users/ratnamshah/turborag/docs/current-rag-rollout.md).

## What To Install

### From PyPI

```bash
# Core
pip install turborag

# With local text-query embedding
pip install turborag[embed]

# With HTTP sidecar service
pip install turborag[serve]

# With MCP agent server
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

For the full installation reference, see [docs/installation.md](installation.md).

## The Easiest Migration Pattern

Keep your current system exactly as-is for:

- chunk storage,
- metadata,
- source documents,
- primary database tables,
- document hydration at query time.

Add TurboRAG only as a compressed sidecar index over the same chunk IDs you already use.

For most existing RAG teams, this should be a retrieval-layer swap, not an application rewrite.

## Sidecar Architecture

1. Export or intercept the embeddings you already compute today.
2. Build a `TurboIndex` keyed by the same `chunk_id` values already stored in your application.
3. Keep your current database untouched.
4. At query time, ask TurboRAG only for ranked IDs.
5. Hydrate those IDs from your existing database or document store.

This means TurboRAG becomes a retrieval engine, not a database migration.

## Generic Integration Example

```python
from turborag.adapters.compat import ExistingRAGAdapter

# Existing application pieces.
# - embeddings_matrix: shape (N, D)
# - ids: your existing chunk IDs
# - embedder: whatever you already use for queries
# - fetch_records(ids): existing DB lookup by chunk ID

adapter = ExistingRAGAdapter.from_embeddings(
    embeddings=embeddings_matrix,
    ids=ids,
    query_embedder=embedder,
    fetch_records=fetch_records,
    bits=3,
    shard_size=100_000,
)

results = adapter.query("What did the CFO say about capex guidance?", k=5)
```

If you want TurboRAG to attach directly to an existing DB client shape, use:

```python
adapter = ExistingRAGAdapter.from_existing_backend(
    embeddings=embeddings_matrix,
    ids=ids,
    query_embedder=embedder,
    records_backend=existing_db_client,
    bits=3,
)
```

`records_backend` can be:

- a callable `fetch_records(ids)`,
- a mapping keyed by `chunk_id`,
- or a client with `fetch(ids=...)`, `retrieve(ids=...)`, or `get(ids=...)`.

For known databases and services, use ready-made backend builders in
`turborag.adapters.backends`:

- `build_postgres_fetch_records(...)` (works for Postgres/Neon/Supabase Postgres)
- `build_neon_fetch_records(...)`
- `build_supabase_fetch_records(...)` (supabase-py client)
- `build_pinecone_fetch_records(...)`
- `build_qdrant_fetch_records(...)`
- `build_chroma_fetch_records(...)`

## Fast Trial Path With The CLI

If your team already has an embedding export, the fastest proof-of-value path is:

```bash
turborag import-existing-index --input ./dataset.jsonl --index ./turborag_sidecar --bits 3
turborag query --index ./turborag_sidecar --query-vector '[0.1, 0.2, 0.3]' --top-k 5
```

To return compact ID-only results (lowest memory), add `--ids-only`:

```bash
turborag query --index ./turborag_sidecar --query-vector '[0.1, 0.2, 0.3]' --top-k 5 --ids-only
```

That path is documented in [docs/import-existing.md](file:///Users/ratnamshah/turborag/docs/import-existing.md).

## Plug-And-Play Adapter CLI

Use `turborag adapt` to configure known backends once, then just run `turborag serve`.

Fastest path (auto-detect from your environment):

```bash
turborag adapt --index ./turborag_sidecar
```

If you're already inside the index directory, even shorter:

```bash
cd ./turborag_sidecar
turborag adapt
```

If multiple backend envs are present, force one explicitly:

```bash
turborag adapt --index ./turborag_sidecar --backend supabase
```

You can use dedicated commands directly:

- `turborag adapt supabase --index ./turborag_sidecar`
- `turborag adapt neon --index ./turborag_sidecar`
- `turborag adapt pinecone --index ./turborag_sidecar`
- `turborag adapt qdrant --index ./turborag_sidecar`
- `turborag adapt chroma --index ./turborag_sidecar`

These commands auto-detect common env vars and store references like `${SUPABASE_URL}`
in `adapter.json`, so secrets stay in your environment.

### Configure Neon

```bash
export DATABASE_URL='postgresql://user:pass@host/db'

turborag adapt set neon \
  --index ./turborag_sidecar \
  --option dsn=${DATABASE_URL} \
  --option table=public.chunks
```

### Configure Supabase (Python client path)

```bash
export SUPABASE_URL='https://xyz.supabase.co'
export SUPABASE_KEY='...'

turborag adapt set supabase \
  --index ./turborag_sidecar \
  --option url=${SUPABASE_URL} \
  --option key=${SUPABASE_KEY} \
  --option table=chunks
```

Shortcut form (preferred):

```bash
export SUPABASE_URL='https://xyz.supabase.co'
export SUPABASE_KEY='...'

turborag adapt supabase --index ./turborag_sidecar
```

### Configure Pinecone

```bash
export PINECONE_API_KEY='...'

turborag adapt set pinecone \
  --index ./turborag_sidecar \
  --option api_key=${PINECONE_API_KEY} \
  --option index_name=my-index \
  --option namespace=prod
```

Shortcut form:

```bash
export PINECONE_API_KEY='...'
export PINECONE_INDEX_NAME='my-index'

turborag adapt pinecone --index ./turborag_sidecar
```

### Configure Qdrant

```bash
turborag adapt set qdrant \
  --index ./turborag_sidecar \
  --option url=http://localhost:6333 \
  --option collection_name=chunks
```

Shortcut form:

```bash
export QDRANT_URL='http://localhost:6333'
turborag adapt qdrant --index ./turborag_sidecar
```

### Configure Chroma

```bash
turborag adapt set chroma \
  --index ./turborag_sidecar \
  --option path=./chroma \
  --option collection_name=chunks
```

### Show / Remove config

```bash
turborag adapt show --index ./turborag_sidecar
turborag adapt remove --index ./turborag_sidecar
```

Need a template command quickly?

```bash
turborag adapt demo supabase
```

## Run The Sidecar Over HTTP

If you want to keep TurboRAG out of your application process, expose the sidecar over HTTP:

```bash
turborag serve --index ./turborag_sidecar --host 0.0.0.0 --port 8080
```

For external-DB hydration and lower memory startup, disable local snapshot loading:

```bash
turborag serve --index ./turborag_sidecar --host 0.0.0.0 --port 8080 --no-load-snapshot
```

Then query it with the same sidecar snapshot:

```bash
curl -X POST http://localhost:8080/query \
  -H 'content-type: application/json' \
  -d '{"query_vector":[0.1,0.2,0.3],"top_k":5}'
```

For ID-only response payloads:

```bash
curl -X POST http://localhost:8080/query \
  -H 'content-type: application/json' \
  -d '{"query_vector":[0.1,0.2,0.3],"top_k":5,"hydrate":false}'
```

If you start the service with `--model`, it can also accept `query_text` payloads.

## What `fetch_records(ids)` Should Do

It should reuse your current storage path and return the records you already have, for example:

```python
[
    {
        "chunk_id": "c_1029",
        "text": "Capital expenditure guidance was raised...",
        "source_doc": "q3_call.pdf",
        "page_num": 18,
        "metadata": {"ticker": "ACME"},
    }
]
```

## Why This Avoids Database Changes

TurboRAG stores only:

- the compressed vectors,
- shard metadata,
- optional graph state.

Your existing SQL, Postgres, Pinecone metadata table, Mongo collection, or document store can stay exactly where it is.

## Does This Reduce LLM Requests?

Usually not directly, if you are only switching the vector retrieval layer.

- Your embedding flow is usually still the same shape.
- Your final answer-generation LLM call is usually still the same shape.
- What changes first is retrieval efficiency, memory usage, and potentially retrieval quality.

If you later enable the graph layer from the PDF, indexing can require additional LLM calls for extraction and summarisation. That is covered in [docs/llm-request-model.md](file:///Users/ratnamshah/turborag/docs/llm-request-model.md).

## Familiar Vector Store Surface

For LangChain-style code, use [src/turborag/adapters/langchain.py](file:///Users/ratnamshah/turborag/src/turborag/adapters/langchain.py). It exposes familiar methods:

- `from_texts(...)`
- `from_existing_records(...)`
- `add_texts(...)`
- `similarity_search(...)`
- `similarity_search_with_score(...)`
- `as_retriever()`

## Known Database Examples

### Neon / Postgres

```python
from turborag.adapters.backends import build_neon_fetch_records
from turborag.adapters.compat import ExistingRAGAdapter

fetch_records = build_neon_fetch_records(
    dsn="postgresql://user:pass@host/db",
    table="public.chunks",
    id_column="chunk_id",
    text_column="text",
)

adapter = ExistingRAGAdapter.from_embeddings(
    embeddings=embeddings_matrix,
    ids=ids,
    query_embedder=embedder,
    fetch_records=fetch_records,
    bits=3,
)
```

### Supabase (Python client)

```python
from supabase import create_client
from turborag.adapters.backends import build_supabase_fetch_records

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
fetch_records = build_supabase_fetch_records(supabase, table="chunks")
```

### Pinecone

```python
from pinecone import Pinecone
from turborag.adapters.backends import build_pinecone_fetch_records

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("your-index")
fetch_records = build_pinecone_fetch_records(index, namespace="prod")
```

### Qdrant

```python
from qdrant_client import QdrantClient
from turborag.adapters.backends import build_qdrant_fetch_records

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
fetch_records = build_qdrant_fetch_records(client, collection_name="chunks")
```

### Chroma

```python
import chromadb
from turborag.adapters.backends import build_chroma_fetch_records

chroma = chromadb.PersistentClient(path="./chroma")
collection = chroma.get_collection("chunks")
fetch_records = build_chroma_fetch_records(collection)
```

## What Changes In Application Code

### Minimal-change path

Usually one boundary changes:

```python
# before
# db = ExistingVectorStore(...)

# after
from turborag.adapters.langchain import TurboVectorStore

db = TurboVectorStore.from_existing_records(
    ids=ids,
    embeddings=embeddings_matrix,
    embedding=embedder,
    resolver=fetch_records,
    bits=3,
)
```

The rest of the application can continue calling retrieval methods on `db`.

If you use `from_existing_records(...)`, keep writing new records through your current database path and then add the matching embeddings to TurboRAG. That preserves the no-database-migration model.

## Migration Strategy For Existing Production Systems

### Stage 1

Build TurboRAG offline from the embeddings and chunk IDs you already have.

### Stage 2

Run TurboRAG in shadow mode beside the current retriever.

### Stage 3

Compare top-k overlap, latency, and answer quality.

### Stage 4

Flip retrieval traffic to TurboRAG while keeping the old database and hydration path unchanged.

## Recommended Current Deployment Shapes

### Embedded mode

Use TurboRAG inside the application process when:

- the app is already Python,
- latency matters,
- and the team is comfortable linking retrieval directly into app runtime.

### HTTP sidecar mode

Use `turborag serve` when:

- the app is not Python,
- the team wants process isolation,
- or they want TurboRAG deployed separately from the main application.

Both modes keep the current metadata database in place.

## Current Limitation

The compatibility layer and import CLI are implemented, but full framework-native adapters for LangChain and LlamaIndex are still at an early stage. The current adapter is designed to make adoption easy now, while leaving room for stricter framework integration later.
