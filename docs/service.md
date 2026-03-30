# Service API

TurboRAG can run as a local or remote HTTP sidecar over an existing index:

```bash
turborag serve --index ./turborag_sidecar --host 0.0.0.0 --port 8080
```

With text-query support and multiple workers:

```bash
turborag serve \
  --index ./turborag_sidecar \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --workers 4 \
  --cors-origins "http://localhost:3000,https://app.example.com"
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

### `GET /metrics`

Returns latency histograms and error counts:

```json
{
  "uptime_seconds": 3600.1,
  "errors": 2,
  "endpoints": {
    "/query": {
      "count": 1500,
      "total_ms": 45000.0,
      "avg_ms": 30.0,
      "min_ms": 5.2,
      "max_ms": 120.3
    }
  }
}
```

### `POST /query`

Accepts exactly one of `query_vector` or `query_text`.

Vector query:

```bash
curl -X POST http://localhost:8080/query \
  -H 'content-type: application/json' \
  -d '{"query_vector":[0.1,0.2,0.3],"top_k":5}'
```

Text query (requires `--model`):

```bash
curl -X POST http://localhost:8080/query \
  -H 'content-type: application/json' \
  -d '{"query_text":"What changed in capex guidance?","top_k":5}'
```

Response:

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
      "explanation": null
    }
  ]
}
```

### `POST /query/batch`

Batch multiple vector queries in a single request:

```bash
curl -X POST http://localhost:8080/query/batch \
  -H 'content-type: application/json' \
  -d '{
    "queries": [
      {"query_vector": [0.1, 0.2, 0.3]},
      {"query_vector": [0.4, 0.5, 0.6]}
    ],
    "top_k": 5
  }'
```

Response:

```json
{
  "batch_count": 2,
  "results": [
    {"count": 5, "results": [...]},
    {"count": 5, "results": [...]}
  ]
}
```

### `POST /ingest`

Appends embedding-level records to the index and refreshes `records.jsonl`:

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

### `POST /ingest-text`

Ingest raw text with automatic chunking and embedding (requires `--model`):

```bash
curl -X POST http://localhost:8080/ingest-text \
  -H 'content-type: application/json' \
  -d '{
    "text": "Full document text to chunk and index...",
    "source_doc": "report.md",
    "chunk_config": {
      "chunk_size": 512,
      "chunk_overlap": 64
    }
  }'
```

## Production Features

- **CORS**: Enabled by default (`*`). Restrict with `--cors-origins`.
- **Metrics**: Hit `GET /metrics` for latency histograms per endpoint.
- **Request tracking**: Pass `X-Request-Id` header — it's logged with every request.
- **Structured logging**: Use `--log-level` and `--log-format json` for production log ingestion.
- **Multi-worker**: Use `--workers N` for multi-process serving.

## Docker

```bash
docker build -t turborag .
docker run -p 8080:8080 -v ./my_index:/data/index turborag \
  turborag serve --index /data/index --host 0.0.0.0
```

## Integration Guide

For how this fits into a production RAG stack with an existing database, see [Current RAG Rollout](current-rag-rollout.md).
