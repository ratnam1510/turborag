const { describe, it, before, after } = require('node:test');
const assert = require('node:assert');
const { TurboRAGClient, ConnectionError, TurboRAGError } = require('../src/index');

describe('Integration Tests (requires running TurboRAG server)', { skip: !process.env.TURBORAG_INTEGRATION_TEST }, () => {
    let client;
    const testChunkId = `test-chunk-${Date.now()}`;

    before(() => {
        client = TurboRAGClient.fromEnv();
    });

    describe('Health and Index endpoints', () => {
        it('should check health', async () => {
            const health = await client.health();
            assert.strictEqual(health.status, 'ok');
            assert(typeof health.index_size === 'number');
        });

        it('should get index info', async () => {
            const index = await client.getIndex();
            assert(typeof index.dim === 'number');
            assert(typeof index.bits === 'number');
            assert(typeof index.index_size === 'number');
        });

        it('should get metrics', async () => {
            const metrics = await client.getMetrics();
            assert(typeof metrics.uptime_seconds === 'number');
            assert(typeof metrics.errors === 'number');
            assert(typeof metrics.endpoints === 'object');
        });
    });

    describe('Query endpoints', () => {
        it('should query by vector', async () => {
            const index = await client.getIndex();
            const dim = index.dim;
            const vector = Array(dim).fill(0.1);

            const result = await client.queryByVector(vector, 5);
            assert(typeof result.count === 'number');
            assert(Array.isArray(result.results));
        });

        it('should query batch', async () => {
            const index = await client.getIndex();
            const dim = index.dim;
            const vector1 = Array(dim).fill(0.1);
            const vector2 = Array(dim).fill(0.2);

            const result = await client.queryBatch([
                { query_vector: vector1 },
                { query_vector: vector2 }
            ], 3);

            assert.strictEqual(result.batchCount, 2);
            assert.strictEqual(result.results.length, 2);
        });
    });

    describe('Ingest endpoint', () => {
        it('should ingest records', async () => {
            const index = await client.getIndex();
            const dim = index.dim;
            const embedding = Array(dim).fill(0.5);

            const result = await client.ingest([{
                chunk_id: testChunkId,
                text: 'Integration test record',
                embedding,
                source_doc: 'test.txt',
                page_num: 1,
                metadata: { test: true }
            }]);

            assert(result.added >= 1);
            assert(typeof result.indexSize === 'number');
        });

        it('should find ingested record', async () => {
            const index = await client.getIndex();
            const dim = index.dim;
            const embedding = Array(dim).fill(0.5);

            const result = await client.queryByVector(embedding, 10);
            const found = result.results.some(r => r.chunkId === testChunkId);
            assert(found, 'Should find the ingested test record');
        });
    });

    describe('Error handling', () => {
        it('should handle validation errors', async () => {
            await assert.rejects(
                client.query({ query_vector: [], top_k: 5 }),
                (error) => {
                    assert(error instanceof Error);
                    return true;
                }
            );
        });

        it('should handle server errors gracefully', async () => {
            await assert.rejects(
                client.query({ query_text: 'test', query_vector: [1.0] }),
                (error) => {
                    assert(error instanceof Error);
                    return true;
                }
            );
        });
    });
});

describe('Connection error handling', () => {
    it('should throw ConnectionError for unreachable server', async () => {
        const client = new TurboRAGClient({
            baseUrl: 'http://localhost:59999',
            timeout: 1000
        });

        await assert.rejects(
            client.health(),
            (error) => {
                assert(error instanceof TurboRAGError);
                return true;
            }
        );
    });
});
