const { TurboRAGClient, TurboRAGError } = require('./src/index');

const TURBORAG_URL = 'http://127.0.0.1:8080';
const DIM = 384;

function generateVector(seed = 0) {
    const vec = [];
    for (let i = 0; i < DIM; i++) {
        vec.push(Math.sin(seed + i * 0.1) * 0.5);
    }
    const norm = Math.sqrt(vec.reduce((a, b) => a + b * b, 0));
    return vec.map(v => v / norm);
}

async function runIntegrationTests() {
    const client = new TurboRAGClient({
        baseUrl: TURBORAG_URL,
        timeout: 10000
    });

    const results = {
        passed: 0,
        failed: 0,
        tests: []
    };

    async function test(name, fn) {
        const start = Date.now();
        try {
            await fn();
            const duration = Date.now() - start;
            results.passed++;
            results.tests.push({ name, status: 'PASS', duration });
            console.log(`✓ ${name} (${duration}ms)`);
        } catch (error) {
            const duration = Date.now() - start;
            results.failed++;
            results.tests.push({ name, status: 'FAIL', error: error.message, duration });
            console.log(`✗ ${name} (${duration}ms): ${error.message}`);
        }
    }

    console.log('\n=== TurboRAG Node.js SDK Integration Tests ===\n');
    console.log(`Server: ${TURBORAG_URL}`);
    console.log(`Vector Dimension: ${DIM}\n`);

    await test('Health check', async () => {
        const health = await client.health();
        if (health.status !== 'ok') throw new Error('Status not ok');
        if (health.index_size !== 1000) throw new Error(`Expected 1000 vectors, got ${health.index_size}`);
    });

    await test('Get index info', async () => {
        const index = await client.getIndex();
        if (index.dim !== DIM) throw new Error(`Expected dim ${DIM}, got ${index.dim}`);
        if (index.bits !== 4) throw new Error(`Expected bits 4, got ${index.bits}`);
        if (index.index_size !== 1000) throw new Error(`Expected 1000 vectors`);
    });

    await test('Describe (alias)', async () => {
        const desc = await client.describe();
        if (desc.dim !== DIM) throw new Error('describe() failed');
    });

    await test('Get metrics', async () => {
        const metrics = await client.getMetrics();
        if (typeof metrics.uptime_seconds !== 'number') throw new Error('No uptime');
        if (typeof metrics.endpoints !== 'object') throw new Error('No endpoints');
    });

    await test('Query by vector - single', async () => {
        const vector = generateVector(42);
        const response = await client.queryByVector(vector, 5);
        if (response.count !== 5) throw new Error(`Expected 5 results, got ${response.count}`);
        if (!response.results[0].chunkId) throw new Error('Missing chunkId');
        if (typeof response.results[0].score !== 'number') throw new Error('Missing score');
    });

    await test('Query by vector - top_k=10', async () => {
        const vector = generateVector(100);
        const response = await client.queryByVector(vector, 10);
        if (response.count !== 10) throw new Error(`Expected 10 results, got ${response.count}`);
    });

    await test('Query by vector - top_k=1', async () => {
        const vector = generateVector(200);
        const response = await client.queryByVector(vector, 1);
        if (response.count !== 1) throw new Error(`Expected 1 result, got ${response.count}`);
    });

    await test('Query with request ID', async () => {
        const vector = generateVector(300);
        const response = await client.queryByVector(vector, 3, 'test-request-123');
        if (response.count !== 3) throw new Error('Query with request ID failed');
    });

    await test('Batch query - 2 queries', async () => {
        const queries = [
            { query_vector: generateVector(1) },
            { query_vector: generateVector(2) }
        ];
        const response = await client.queryBatch(queries, 5);
        if (response.batchCount !== 2) throw new Error(`Expected 2 batches, got ${response.batchCount}`);
        if (response.results[0].count !== 5) throw new Error('First batch wrong count');
        if (response.results[1].count !== 5) throw new Error('Second batch wrong count');
    });

    await test('Batch query - 5 queries', async () => {
        const queries = Array(5).fill(0).map((_, i) => ({ query_vector: generateVector(i * 10) }));
        const response = await client.queryBatch(queries, 3);
        if (response.batchCount !== 5) throw new Error('5 batch queries failed');
    });

    const newChunkId = `integration-test-${Date.now()}`;
    await test('Ingest single record', async () => {
        const response = await client.ingest([{
            chunk_id: newChunkId,
            text: 'Integration test document for Node.js SDK',
            embedding: generateVector(999),
            source_doc: 'integration-test.txt',
            page_num: 1,
            metadata: { test: true, sdk: 'nodejs' }
        }]);
        if (response.added !== 1) throw new Error(`Expected added=1, got ${response.added}`);
        if (response.indexSize !== 1001) throw new Error(`Expected indexSize=1001, got ${response.indexSize}`);
    });

    await test('Query finds ingested record', async () => {
        const vector = generateVector(999);
        const response = await client.queryByVector(vector, 5);
        const found = response.results.some(r => r.chunkId === newChunkId);
        if (!found) throw new Error('Ingested record not found in search results');
    });

    await test('Ingest multiple records', async () => {
        const records = Array(5).fill(0).map((_, i) => ({
            chunk_id: `batch-ingest-${Date.now()}-${i}`,
            text: `Batch ingested document ${i}`,
            embedding: generateVector(1000 + i)
        }));
        const response = await client.ingest(records);
        if (response.added !== 5) throw new Error(`Expected added=5, got ${response.added}`);
    });

    await test('Validation error - empty vector', async () => {
        try {
            await client.queryByVector([], 5);
            throw new Error('Should have thrown validation error');
        } catch (e) {
            if (!e.message.includes('non-empty')) throw new Error('Wrong error message');
        }
    });

    await test('Validation error - invalid top_k', async () => {
        try {
            await client.query({ query_vector: generateVector(0), top_k: 0 });
            throw new Error('Should have thrown validation error');
        } catch (e) {
            if (!e.message.includes('between 1 and 1000')) throw new Error('Wrong error for top_k=0');
        }
    });

    await test('Validation error - top_k too large', async () => {
        try {
            await client.query({ query_vector: generateVector(0), top_k: 1001 });
            throw new Error('Should have thrown validation error');
        } catch (e) {
            if (!e.message.includes('between 1 and 1000')) throw new Error('Wrong error for top_k=1001');
        }
    });

    console.log('\n=== Results ===');
    console.log(`Passed: ${results.passed}`);
    console.log(`Failed: ${results.failed}`);
    console.log(`Total:  ${results.passed + results.failed}\n`);

    return results;
}

runIntegrationTests()
    .then(results => {
        const output = {
            summary: {
                passed: results.passed,
                failed: results.failed,
                total: results.passed + results.failed,
                success_rate: ((results.passed / (results.passed + results.failed)) * 100).toFixed(1) + '%'
            },
            tests: results.tests
        };
        console.log(JSON.stringify(output, null, 2));
        process.exit(results.failed > 0 ? 1 : 0);
    })
    .catch(err => {
        console.error('Test runner error:', err);
        process.exit(1);
    });
