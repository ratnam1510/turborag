const { TurboRAG } = require("./dist/index.js");

const baseUrl = process.env.TURBORAG_URL || "http://127.0.0.1:8080";
const timeout = Number.parseInt(process.env.TURBORAG_TIMEOUT_MS || "30000", 10);
const queryCount = Number.parseInt(process.env.TURBORAG_NUM_QUERIES || "100", 10);
const topK = Number.parseInt(process.env.TURBORAG_TOP_K || "10", 10);
const ingestCount = Number.parseInt(process.env.TURBORAG_INGEST_RECORDS || "100", 10);

function makeVector(dim, seed) {
  const values = [];
  for (let i = 0; i < dim; i += 1) {
    values.push(Math.sin(seed + i * 0.07) + Math.cos(seed * 0.11 + i * 0.03));
  }
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)) || 1;
  return values.map((value) => value / norm);
}

async function main() {
  const client = new TurboRAG(baseUrl, { timeout });
  const index = await client.getIndex();
  const queries = Array.from({ length: queryCount }, (_value, i) => makeVector(index.dim, i + 1));

  console.log("\n=== TurboRAG TypeScript Client Benchmark ===\n");
  console.log(`Base URL: ${baseUrl}`);
  console.log(`Index size: ${index.index_size}`);
  console.log(`Dimension: ${index.dim}`);
  console.log(`Bits: ${index.bits}`);
  console.log(`Queries: ${queryCount}`);
  console.log(`topK: ${topK}\n`);

  const singleStart = performance.now();
  for (const vector of queries) {
    await client.queryByVector(vector, topK);
  }
  const singleElapsed = performance.now() - singleStart;

  const batchStart = performance.now();
  const batchResponse = await client.queryBatch({
    queries: queries.map((vector) => ({ vector })),
    topK,
  });
  const batchElapsed = performance.now() - batchStart;

  const records = Array.from({ length: ingestCount }, (_value, i) => ({
    chunk_id: `node-bench-${Date.now()}-${i}`,
    text: `Benchmark record ${i}`,
    embedding: makeVector(index.dim, 10_000 + i),
    metadata: { benchmark: true, index: i },
  }));

  const ingestStart = performance.now();
  const ingestResponse = await client.ingest({ records });
  const ingestElapsed = performance.now() - ingestStart;

  const results = {
    baseUrl,
    index: {
      size: index.index_size,
      dim: index.dim,
      bits: index.bits,
    },
    singleQuery: {
      queries: queryCount,
      elapsedMs: Number(singleElapsed.toFixed(2)),
      qps: Number((queryCount / (singleElapsed / 1000)).toFixed(2)),
      avgLatencyMs: Number((singleElapsed / queryCount).toFixed(2)),
    },
    batchQuery: {
      queries: queryCount,
      batchCount: batchResponse.batch_count,
      elapsedMs: Number(batchElapsed.toFixed(2)),
      qps: Number((queryCount / (batchElapsed / 1000)).toFixed(2)),
      avgLatencyMs: Number((batchElapsed / queryCount).toFixed(2)),
    },
    ingest: {
      records: ingestCount,
      added: ingestResponse.added,
      indexSize: ingestResponse.index_size,
      elapsedMs: Number(ingestElapsed.toFixed(2)),
      recordsPerSecond: Number((ingestCount / (ingestElapsed / 1000)).toFixed(2)),
    },
  };

  console.log(JSON.stringify(results, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
