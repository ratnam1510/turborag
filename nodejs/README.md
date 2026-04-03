# TurboRAG Node.js SDK

Node.js client for TurboRAG - a production-grade compressed vector retrieval engine.

## Installation

```bash
npm install turborag
```

## Quick Start

```javascript
const { createClient } = require('turborag');

const client = createClient({
    baseUrl: 'http://localhost:8080',
    timeout: 30000,
    defaultTopK: 5
});

const results = await client.queryByVector([0.1, 0.2, 0.3, ...], 10);
console.log(results);
```

## Configuration

### From Environment Variables

```javascript
const { createClientFromEnv } = require('turborag');

const client = createClientFromEnv();
```

Environment variables:
- `TURBORAG_API_URL` or `TURBORAG_URL`: Server URL (default: `http://localhost:8080`)
- `TURBORAG_TIMEOUT_MS` or `TURBORAG_TIMEOUT`: Request timeout in ms (default: `30000`)
- `TURBORAG_TOP_K`: Default number of results (default: `5`)

### Constructor Options

```javascript
const client = new TurboRAGClient({
    baseUrl: 'http://localhost:8080',
    timeout: 30000,
    defaultTopK: 5,
    headers: {
        'Authorization': 'Bearer token'
    }
});
```

## API Reference

### Health Check

```javascript
const health = await client.health();
```

### Index Information

```javascript
const index = await client.getIndex();
const index = await client.describe();
```

### Metrics

```javascript
const metrics = await client.getMetrics();
```

### Query by Vector

```javascript
const results = await client.queryByVector(
    [0.1, 0.2, 0.3],
    5
);
```

### Query by Text

Requires the TurboRAG server to be started with `--model`.

```javascript
const results = await client.queryByText(
    'What is the capital of France?',
    10
);
```

### Generic Query

```javascript
const results = await client.query({
    query_vector: [0.1, 0.2, 0.3],
    top_k: 5
});

const results = await client.query({
    query_text: 'search query',
    top_k: 10
});
```

### Batch Query

```javascript
const results = await client.queryBatch([
    { query_vector: [0.1, 0.2, 0.3] },
    { query_vector: [0.4, 0.5, 0.6] }
], 5);
```

### Ingest Records

```javascript
const result = await client.ingest([{
    chunk_id: 'doc-1-chunk-0',
    text: 'Document text content',
    embedding: [0.1, 0.2, 0.3, ...],
    source_doc: 'document.pdf',
    page_num: 1,
    metadata: { category: 'finance' }
}]);
```

### Ingest Text

Requires the TurboRAG server to be started with `--model`.

```javascript
const result = await client.ingestText(
    'Full document text to chunk and index...',
    'document.md',
    { chunk_size: 512, chunk_overlap: 64 }
);
```

## Response Types

### Query Response

```javascript
{
    count: 5,
    results: [{
        chunkId: 'chunk-123',
        text: 'Chunk content...',
        score: 0.95,
        sourceDoc: 'document.pdf',
        pageNum: 5,
        graphPath: null,
        explanation: null
    }]
}
```

### Batch Query Response

```javascript
{
    batchCount: 2,
    results: [
        { count: 5, results: [...] },
        { count: 5, results: [...] }
    ]
}
```

### Ingest Response

```javascript
{
    added: 10,
    indexSize: 1000,
    recordsSnapshot: '/path/to/records.jsonl'
}
```

## Error Handling

```javascript
const { 
    TurboRAGError, 
    ValidationError, 
    ConnectionError, 
    TimeoutError 
} = require('turborag');

try {
    await client.queryByVector([0.1, 0.2]);
} catch (error) {
    if (error instanceof ValidationError) {
        console.error('Invalid input:', error.message);
    } else if (error instanceof ConnectionError) {
        console.error('Cannot connect to server:', error.message);
    } else if (error instanceof TimeoutError) {
        console.error('Request timed out:', error.message);
    } else if (error instanceof TurboRAGError) {
        console.error('Server error:', error.message, error.statusCode);
    }
}
```

## Request Tracking

Pass a request ID for logging and debugging:

```javascript
const results = await client.queryByVector(
    [0.1, 0.2, 0.3],
    5,
    'request-123'
);
```

## TypeScript Support

The package includes TypeScript type definitions:

```typescript
import { 
    TurboRAGClient, 
    createClient,
    QueryResponse,
    RetrievalResult,
    IngestRecord
} from 'turborag';

const client = createClient({ baseUrl: 'http://localhost:8080' });

const results: QueryResponse = await client.queryByVector([0.1, 0.2], 5);
```

## Running Tests

```bash
npm test

TURBORAG_INTEGRATION_TEST=1 npm run test:integration
```

## License

MIT
