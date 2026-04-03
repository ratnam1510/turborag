const { TurboRAGClient } = require('./src/index');

const TURBORAG_URL = 'http://127.0.0.1:8080';
const DIM = 384;
const NUM_QUERIES = 100;
const TOP_K = 10;

function generateVector(seed = 0) {
    const vec = [];
    for (let i = 0; i < DIM; i++) {
        vec.push(Math.sin(seed + i * 0.1) * 0.5);
    }
    const norm = Math.sqrt(vec.reduce((a, b) => a + b * b, 0));
    return vec.map(v => v / norm);
}

async function runBenchmark() {
    const client = new TurboRAGClient({
        baseUrl: TURBORAG_URL,
        timeout: 30000
    });

    console.log('\n=== TurboRAG Node.js SDK Performance Benchmark ===\n');

    const index = await client.getIndex();
    console.log(`Index: ${index.index_size} vectors, ${index.dim} dimensions, ${index.bits}-bit quantization`);
    console.log(`Queries: ${NUM_QUERIES}, top_k: ${TOP_K}\n`);

    const queries = Array(NUM_QUERIES).fill(0).map((_, i) => generateVector(i * 7));

    console.log('--- Single Query Benchmark ---');
    const singleStart = Date.now();
    for (const q of queries) {
        await client.queryByVector(q, TOP_K);
    }
    const singleElapsed = Date.now() - singleStart;
    const singleQPS = (NUM_QUERIES / (singleElapsed / 1000)).toFixed(1);
    console.log(`Time: ${singleElapsed}ms for ${NUM_QUERIES} queries`);
    console.log(`QPS: ${singleQPS} queries/second`);
    console.log(`Avg latency: ${(singleElapsed / NUM_QUERIES).toFixed(2)}ms\n`);

    console.log('--- Batch Query Benchmark ---');
    const batchQueries = queries.map(q => ({ query_vector: q }));
    const batchStart = Date.now();
    const batchResponse = await client.queryBatch(batchQueries, TOP_K);
    const batchElapsed = Date.now() - batchStart;
    const batchQPS = (NUM_QUERIES / (batchElapsed / 1000)).toFixed(1);
    console.log(`Time: ${batchElapsed}ms for ${NUM_QUERIES} queries (batched)`);
    console.log(`QPS: ${batchQPS} queries/second`);
    console.log(`Avg latency: ${(batchElapsed / NUM_QUERIES).toFixed(2)}ms\n`);

    console.log('--- Ingest Benchmark ---');
    const ingestRecords = Array(100).fill(0).map((_, i) => ({
        chunk_id: `benchmark-${Date.now()}-${i}`,
        text: `Benchmark document ${i}`,
        embedding: generateVector(10000 + i)
    }));
    const ingestStart = Date.now();
    await client.ingest(ingestRecords);
    const ingestElapsed = Date.now() - ingestStart;
    console.log(`Time: ${ingestElapsed}ms for 100 records`);
    console.log(`Rate: ${(100 / (ingestElapsed / 1000)).toFixed(1)} records/second\n`);

    const results = {
        single_query: {
            total_queries: NUM_QUERIES,
            elapsed_ms: singleElapsed,
            qps: parseFloat(singleQPS),
            avg_latency_ms: singleElapsed / NUM_QUERIES
        },
        batch_query: {
            total_queries: NUM_QUERIES,
            elapsed_ms: batchElapsed,
            qps: parseFloat(batchQPS),
            avg_latency_ms: batchElapsed / NUM_QUERIES
        },
        ingest: {
            records: 100,
            elapsed_ms: ingestElapsed,
            rate: 100 / (ingestElapsed / 1000)
        }
    };

    console.log('=== Results JSON ===');
    console.log(JSON.stringify(results, null, 2));

    return results;
}

runBenchmark().catch(console.error);
