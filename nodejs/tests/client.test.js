const { describe, it, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert');
const {
    TurboRAGClient,
    createClient,
    createClientFromEnv,
    TurboRAGError,
    ValidationError,
    ConnectionError,
    TimeoutError
} = require('../src/index');

describe('TurboRAGClient', () => {
    describe('constructor', () => {
        it('should create client with default options', () => {
            const client = new TurboRAGClient();
            assert.strictEqual(client.baseUrl, 'http://localhost:8080');
            assert.strictEqual(client.timeout, 30000);
            assert.strictEqual(client.defaultTopK, 5);
        });

        it('should accept baseUrl option', () => {
            const client = new TurboRAGClient({ baseUrl: 'http://example.com:9000' });
            assert.strictEqual(client.baseUrl, 'http://example.com:9000');
        });

        it('should strip trailing slashes from baseUrl', () => {
            const client = new TurboRAGClient({ baseUrl: 'http://example.com/' });
            assert.strictEqual(client.baseUrl, 'http://example.com');
        });

        it('should accept apiUrl as alias for baseUrl', () => {
            const client = new TurboRAGClient({ apiUrl: 'http://api.example.com' });
            assert.strictEqual(client.baseUrl, 'http://api.example.com');
        });

        it('should accept timeout option', () => {
            const client = new TurboRAGClient({ timeout: 5000 });
            assert.strictEqual(client.timeout, 5000);
        });

        it('should accept timeoutMs as alias for timeout', () => {
            const client = new TurboRAGClient({ timeoutMs: 10000 });
            assert.strictEqual(client.timeout, 10000);
        });

        it('should accept defaultTopK option', () => {
            const client = new TurboRAGClient({ defaultTopK: 10 });
            assert.strictEqual(client.defaultTopK, 10);
        });

        it('should accept custom headers', () => {
            const client = new TurboRAGClient({
                headers: { 'Authorization': 'Bearer token' }
            });
            assert.strictEqual(client.headers['Authorization'], 'Bearer token');
            assert.strictEqual(client.headers['Content-Type'], 'application/json');
        });
    });

    describe('fromEnv', () => {
        it('should create client from environment variables', () => {
            const env = {
                TURBORAG_API_URL: 'http://env.example.com:8080',
                TURBORAG_TIMEOUT_MS: '15000',
                TURBORAG_TOP_K: '20'
            };
            const client = TurboRAGClient.fromEnv(env);
            assert.strictEqual(client.baseUrl, 'http://env.example.com:8080');
            assert.strictEqual(client.timeout, 15000);
            assert.strictEqual(client.defaultTopK, 20);
        });

        it('should use TURBORAG_URL as fallback', () => {
            const env = {
                TURBORAG_URL: 'http://fallback.example.com'
            };
            const client = TurboRAGClient.fromEnv(env);
            assert.strictEqual(client.baseUrl, 'http://fallback.example.com');
        });

        it('should use defaults when env vars not set', () => {
            const client = TurboRAGClient.fromEnv({});
            assert.strictEqual(client.baseUrl, 'http://localhost:8080');
            assert.strictEqual(client.timeout, 30000);
            assert.strictEqual(client.defaultTopK, 5);
        });
    });

    describe('createClient', () => {
        it('should create client instance', () => {
            const client = createClient({ baseUrl: 'http://test.com' });
            assert(client instanceof TurboRAGClient);
            assert.strictEqual(client.baseUrl, 'http://test.com');
        });
    });

    describe('createClientFromEnv', () => {
        it('should create client from env', () => {
            const client = createClientFromEnv({
                TURBORAG_API_URL: 'http://from-env.com'
            });
            assert(client instanceof TurboRAGClient);
            assert.strictEqual(client.baseUrl, 'http://from-env.com');
        });
    });
});

describe('Error classes', () => {
    describe('TurboRAGError', () => {
        it('should create error with message', () => {
            const error = new TurboRAGError('Test error');
            assert.strictEqual(error.message, 'Test error');
            assert.strictEqual(error.name, 'TurboRAGError');
            assert.strictEqual(error.statusCode, null);
            assert.strictEqual(error.details, null);
        });

        it('should create error with status code', () => {
            const error = new TurboRAGError('Test', 500);
            assert.strictEqual(error.statusCode, 500);
        });

        it('should create error with details', () => {
            const error = new TurboRAGError('Test', 400, { field: 'value' });
            assert.deepStrictEqual(error.details, { field: 'value' });
        });

        it('should be instance of Error', () => {
            const error = new TurboRAGError('Test');
            assert(error instanceof Error);
        });
    });

    describe('ValidationError', () => {
        it('should create validation error', () => {
            const error = new ValidationError('Invalid input');
            assert.strictEqual(error.name, 'ValidationError');
            assert.strictEqual(error.statusCode, 400);
        });

        it('should be instance of TurboRAGError', () => {
            const error = new ValidationError('Test');
            assert(error instanceof TurboRAGError);
        });
    });

    describe('ConnectionError', () => {
        it('should create connection error', () => {
            const error = new ConnectionError('Cannot connect');
            assert.strictEqual(error.name, 'ConnectionError');
            assert.strictEqual(error.statusCode, null);
        });

        it('should be instance of TurboRAGError', () => {
            const error = new ConnectionError('Test');
            assert(error instanceof TurboRAGError);
        });
    });

    describe('TimeoutError', () => {
        it('should create timeout error', () => {
            const error = new TimeoutError('Request timed out');
            assert.strictEqual(error.name, 'TimeoutError');
            assert.strictEqual(error.statusCode, null);
        });

        it('should be instance of TurboRAGError', () => {
            const error = new TimeoutError('Test');
            assert(error instanceof TurboRAGError);
        });
    });
});

describe('TurboRAGClient methods with mocked fetch', () => {
    let client;
    let originalFetch;
    let mockResponses;

    beforeEach(() => {
        client = new TurboRAGClient({ baseUrl: 'http://test.local:8080' });
        originalFetch = global.fetch;
        mockResponses = [];

        global.fetch = async (url, options) => {
            const response = mockResponses.shift();
            if (!response) {
                throw new Error('No mock response configured');
            }
            return response(url, options);
        };
    });

    const mockJsonResponse = (data, status = 200) => {
        return () => ({
            ok: status >= 200 && status < 300,
            status,
            headers: {
                get: (name) => name.toLowerCase() === 'content-type' ? 'application/json' : null
            },
            json: async () => data
        });
    };

    const mockErrorResponse = (message, status = 400) => {
        return () => ({
            ok: false,
            status,
            headers: {
                get: () => 'application/json'
            },
            json: async () => ({ detail: message })
        });
    };

    describe('health', () => {
        it('should call GET /health', async () => {
            mockResponses.push(mockJsonResponse({
                status: 'ok',
                index_path: './index',
                index_size: 100
            }));

            const result = await client.health();
            assert.strictEqual(result.status, 'ok');
            assert.strictEqual(result.index_size, 100);
        });
    });

    describe('getIndex', () => {
        it('should call GET /index', async () => {
            mockResponses.push(mockJsonResponse({
                index_path: './index',
                dim: 384,
                bits: 4,
                index_size: 1000,
                text_query_enabled: true
            }));

            const result = await client.getIndex();
            assert.strictEqual(result.dim, 384);
            assert.strictEqual(result.text_query_enabled, true);
        });
    });

    describe('describe', () => {
        it('should be alias for getIndex', async () => {
            mockResponses.push(mockJsonResponse({
                index_path: './index',
                dim: 384
            }));

            const result = await client.describe();
            assert.strictEqual(result.dim, 384);
        });
    });

    describe('getMetrics', () => {
        it('should call GET /metrics', async () => {
            mockResponses.push(mockJsonResponse({
                uptime_seconds: 3600,
                errors: 2,
                endpoints: {
                    '/query': { count: 100, avg_ms: 25 }
                }
            }));

            const result = await client.getMetrics();
            assert.strictEqual(result.uptime_seconds, 3600);
            assert.strictEqual(result.errors, 2);
        });
    });

    describe('query', () => {
        it('should query by vector', async () => {
            mockResponses.push(mockJsonResponse({
                count: 2,
                results: [
                    { chunk_id: 'a', text: 'text a', score: 0.95, source_doc: 'doc.pdf', page_num: 1 },
                    { chunk_id: 'b', text: 'text b', score: 0.85 }
                ]
            }));

            const result = await client.query({
                query_vector: [0.1, 0.2, 0.3],
                top_k: 2
            });

            assert.strictEqual(result.count, 2);
            assert.strictEqual(result.results[0].chunkId, 'a');
            assert.strictEqual(result.results[0].score, 0.95);
            assert.strictEqual(result.results[0].sourceDoc, 'doc.pdf');
            assert.strictEqual(result.results[0].pageNum, 1);
            assert.strictEqual(result.results[1].chunkId, 'b');
            assert.strictEqual(result.results[1].pageNum, null);
        });

        it('should query by text', async () => {
            mockResponses.push(mockJsonResponse({
                count: 1,
                results: [
                    { chunk_id: 'c', text: 'text c', score: 0.9 }
                ]
            }));

            const result = await client.query({
                query_text: 'test query',
                top_k: 5
            });

            assert.strictEqual(result.count, 1);
            assert.strictEqual(result.results[0].chunkId, 'c');
        });

        it('should include request ID header', async () => {
            let capturedHeaders;
            mockResponses.push((url, options) => {
                capturedHeaders = options.headers;
                return mockJsonResponse({ count: 0, results: [] })();
            });

            await client.query({ query_vector: [1.0] }, 'req-123');
            assert.strictEqual(capturedHeaders['X-Request-Id'], 'req-123');
        });

        it('should throw on error response', async () => {
            mockResponses.push(mockErrorResponse('Invalid query', 400));

            await assert.rejects(
                client.query({ query_vector: [1.0] }),
                (error) => {
                    assert(error instanceof TurboRAGError);
                    assert.strictEqual(error.statusCode, 400);
                    return true;
                }
            );
        });
    });

    describe('queryByVector', () => {
        it('should call query with vector', async () => {
            mockResponses.push(mockJsonResponse({
                count: 1,
                results: [{ chunk_id: 'a', score: 0.9 }]
            }));

            const result = await client.queryByVector([0.1, 0.2], 3);
            assert.strictEqual(result.count, 1);
        });

        it('should use default topK', async () => {
            mockResponses.push(mockJsonResponse({ count: 0, results: [] }));
            await client.queryByVector([1.0]);
        });

        it('should reject empty vector', async () => {
            await assert.rejects(
                client.queryByVector([]),
                /non-empty array/
            );
        });
    });

    describe('queryByText', () => {
        it('should call query with text', async () => {
            mockResponses.push(mockJsonResponse({
                count: 1,
                results: [{ chunk_id: 'a', text: 'match', score: 0.9 }]
            }));

            const result = await client.queryByText('search term', 5);
            assert.strictEqual(result.count, 1);
        });

        it('should reject empty text', async () => {
            await assert.rejects(
                client.queryByText(''),
                /non-empty string/
            );
        });

        it('should reject whitespace text', async () => {
            await assert.rejects(
                client.queryByText('   '),
                /non-empty string/
            );
        });
    });

    describe('queryBatch', () => {
        it('should call POST /query/batch', async () => {
            mockResponses.push(mockJsonResponse({
                batch_count: 2,
                results: [
                    { count: 1, results: [{ chunk_id: 'a', score: 0.9 }] },
                    { count: 1, results: [{ chunk_id: 'b', score: 0.8 }] }
                ]
            }));

            const result = await client.queryBatch([
                { query_vector: [0.1, 0.2] },
                { query_vector: [0.3, 0.4] }
            ], 5);

            assert.strictEqual(result.batchCount, 2);
            assert.strictEqual(result.results[0].results[0].chunkId, 'a');
            assert.strictEqual(result.results[1].results[0].chunkId, 'b');
        });
    });

    describe('ingest', () => {
        it('should call POST /ingest', async () => {
            mockResponses.push(mockJsonResponse({
                added: 2,
                index_size: 100,
                records_snapshot: './records.jsonl'
            }));

            const result = await client.ingest([
                { chunk_id: 'x', text: 'text x', embedding: [1.0, 2.0] },
                { chunk_id: 'y', text: 'text y', embedding: [3.0, 4.0] }
            ]);

            assert.strictEqual(result.added, 2);
            assert.strictEqual(result.indexSize, 100);
            assert.strictEqual(result.recordsSnapshot, './records.jsonl');
        });
    });

    describe('ingestText', () => {
        it('should call POST /ingest-text', async () => {
            mockResponses.push(mockJsonResponse({
                added: 3,
                chunks: [
                    { chunk_id: 'c1', text: 'chunk 1...' },
                    { chunk_id: 'c2', text: 'chunk 2...' },
                    { chunk_id: 'c3', text: 'chunk 3...' }
                ],
                index_size: 103
            }));

            const result = await client.ingestText(
                'A long document text...',
                'document.md',
                { chunk_size: 256, chunk_overlap: 32 }
            );

            assert.strictEqual(result.added, 3);
            assert.strictEqual(result.chunks.length, 3);
            assert.strictEqual(result.chunks[0].chunkId, 'c1');
            assert.strictEqual(result.indexSize, 103);
        });

        it('should work without optional parameters', async () => {
            mockResponses.push(mockJsonResponse({
                added: 1,
                chunks: [{ chunk_id: 'c1', text: 'text' }],
                index_size: 1
            }));

            const result = await client.ingestText('Some text');
            assert.strictEqual(result.added, 1);
        });
    });

    afterEach(() => {
        global.fetch = originalFetch;
    });
});
