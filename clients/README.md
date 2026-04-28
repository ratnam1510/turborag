# TurboRAG Client SDKs

Language-agnostic clients for the TurboRAG HTTP API. Each is a thin typed wrapper over the same sidecar service contract.

| Language | Package | Install |
|---|---|---|
| **Python** | `turborag` | `pip install turborag` |
| **TypeScript/Node.js** | `turborag` | `npm install turborag` |
| **Go** | `github.com/ratnam1510/turborag/clients/go` | `go get github.com/ratnam1510/turborag/clients/go` |
| **Ruby** | `turborag` | `gem install turborag` |
| **Rust** | `turborag` | `cargo add turborag` |

All clients talk to the same HTTP API — start a server with:

```bash
# Docker (no Python needed)
docker run -p 8080:8080 -v ./my_index:/data/index turborag \
  turborag serve --index /data/index --host 0.0.0.0

# Or pip
pip install turborag[serve]
turborag serve --index ./my_index --port 8080
```

## What Each Client Covers

Every client wraps all HTTP endpoints:

- `POST /query` — vector and text search
- `POST /query/batch` — batch vector search
- `POST /ingest` — add records with embeddings
- `POST /ingest-text` — raw text ingestion with auto-chunking
- `GET /health` — health check
- `GET /index` — index config and stats
- `GET /metrics` — latency and error metrics

## Adding A New Language

Each client is ~100 lines wrapping `POST`/`GET` with JSON serialization. To add a new language:

1. Create `clients/<language>/`
2. Implement the 7 API methods above
3. Keep dependencies minimal and use the HTTP sidecar contract
4. Match the type signatures from the TypeScript client
