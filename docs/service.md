# Service API

TurboRAG can run as a local or remote HTTP sidecar over an existing index:

```bash
turborag serve --index ./turborag_sidecar --host 0.0.0.0 --port 8080
```

If you also want text-query support, start it with a local embedding model:

```bash
turborag serve \
  --index ./turborag_sidecar \
  --model sentence-transformers/all-MiniLM-L6-v2
```

## Endpoints

### `GET /health`

Returns a minimal readiness payload:

```json
{
  "status": "ok",
  "index_path": "./turborag_sidecar",
  "index_size": 100000
}
```

### `GET /index`

Returns the sidecar configuration and whether local text queries are enabled.

### `POST /query`

Accepts exactly one of `query_vector` or `query_text`.

Vector query example:

```bash
curl -X POST http://localhost:8080/query \
  -H 'content-type: application/json' \
  -d '{"query_vector":[0.1,0.2,0.3],"top_k":5}'
```

Text query example:

```bash
curl -X POST http://localhost:8080/query \
  -H 'content-type: application/json' \
  -d '{"query_text":"What changed in capex guidance?","top_k":5}'
```

Response shape:

```json
{
  "count": 2,
  "results": [
    {
      "chunk_id": "chunk-12",
      "text": "Capital expenditure guidance was raised...",
      "score": 0.8421,
      "source_doc": "q3_call.pdf",
      "page_num": 18,
      "graph_path": null,
      "explanation": "Retrieved from TurboRAG using the existing application record store."
    }
  ]
}
```

### `POST /ingest`

Appends new embedding-level records to the existing sidecar and refreshes `records.jsonl`.

```bash
curl -X POST http://localhost:8080/ingest \
  -H 'content-type: application/json' \
  -d '{
    "records": [
      {
        "chunk_id": "chunk-999",
        "text": "New chunk text",
        "embedding": [0.1, 0.2, 0.3],
        "source_doc": "new.pdf",
        "page_num": 4,
        "metadata": {"source": "manual"}
      }
    ]
  }'
```

## Current Scope

The service is intentionally aligned with the sidecar workflow that exists today:

- it serves existing indexes,
- it ingests embedding-level records,
- it hydrates results from `records.jsonl`,
- and it reuses the same retrieval path as the Python adapters.

It does not yet implement the richer raw-document ingest or graph visualisation endpoints described in the PDF.

For how this fits into a production RAG stack with an existing database, see [docs/current-rag-rollout.md](file:///Users/ratnamshah/turborag/docs/current-rag-rollout.md).
