# turborag (Node.js / TypeScript)

Typed client for the [TurboRAG](https://github.com/ratnam1510/turborag) compressed vector retrieval API. Zero dependencies — uses native `fetch`.

## Install

```bash
npm install turborag
```

## Prerequisites

Start a TurboRAG server:

```bash
# With Docker (no Python needed)
docker run -p 8080:8080 -v ./my_index:/data/index turborag \
  turborag serve --index /data/index --host 0.0.0.0

# Or with pip
pip install turborag[serve]
turborag serve --index ./my_index --port 8080
```

## Usage

```typescript
import { TurboRAG } from "turborag";

const client = new TurboRAG("http://localhost:8080");

// Query by vector
const { results } = await client.query({
  vector: [0.1, 0.2, 0.3],
  topK: 5,
});

// Query IDs only (application hydrates from existing DB)
const idOnly = await client.queryIds({
  vector: [0.1, 0.2, 0.3],
  topK: 5,
});

for (const r of results) {
  console.log(r.chunk_id, r.score, r.text);
}

// Query by text (requires --model on the server)
const textResults = await client.queryText({
  text: "What changed in capex guidance?",
  topK: 5,
});

// Batch query
const batch = await client.queryBatch({
  queries: [
    { vector: [0.1, 0.2, 0.3] },
    { vector: [0.4, 0.5, 0.6] },
  ],
  topK: 5,
});

// Batch IDs only
const batchIds = await client.queryBatchIds({
  queries: [
    { vector: [0.1, 0.2, 0.3] },
    { vector: [0.4, 0.5, 0.6] },
  ],
  topK: 5,
});

// Ingest records
await client.ingest({
  records: [
    {
      chunk_id: "c1",
      text: "Capital expenditure guidance increased.",
      embedding: [0.1, 0.2, 0.3],
      source_doc: "q3_call.pdf",
    },
  ],
});

// Ingest raw text (auto-chunked, requires --model)
await client.ingestText({
  text: "Full document text to chunk and index...",
  sourceDoc: "report.md",
});

// Health & metrics
const health = await client.health();
const info = await client.index();
const metrics = await client.metrics();
```

## API

| Method | Description |
|---|---|
| `query({ vector, topK, hydrate })` | Search by embedding vector (`hydrate: false` returns ID-only hits) |
| `queryIds({ vector, topK })` | Search by vector and return ID-only hits |
| `queryText({ text, topK })` | Search by text (requires `--model`) |
| `queryBatch({ queries, topK, hydrate })` | Batch vector search (`hydrate: false` returns ID-only hits) |
| `queryBatchIds({ queries, topK })` | Batch search with ID-only hits |
| `ingest({ records })` | Add records with embeddings |
| `ingestText({ text, sourceDoc })` | Ingest raw text |
| `health()` | Health check |
| `index()` | Index config and stats |
| `metrics()` | Latency and error metrics |
