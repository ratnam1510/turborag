const { TurboRAG } = require("./dist/index.js");

const baseUrl = process.env.TURBORAG_URL || "http://127.0.0.1:8080";
const timeout = Number.parseInt(process.env.TURBORAG_TIMEOUT_MS || "120000", 10);
const queryCount = Number.parseInt(process.env.TURBORAG_NUM_QUERIES || "50", 10);
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

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function getMemoryUsage() {
  const mem = process.memoryUsage();
  return {
    heapUsed: mem.heapUsed,
    heapTotal: mem.heapTotal,
    rss: mem.rss,
    external: mem.external,
  };
}

async function main() {
  // Force GC if available
  if (global.gc) global.gc();
  
  const memStart = getMemoryUsage();
  
  const client = new TurboRAG(baseUrl, { timeout });
  const index = await client.getIndex();
  const queries = Array.from({ length: queryCount }, (_value, i) => makeVector(index.dim, i + 1));

  console.log("\n=== TurboRAG TypeScript Client Benchmark ===\n");
  console.log(`Base URL: ${baseUrl}`);
  console.log(`Index size: ${index.index_size}`);
  console.log(`Dimension: ${index.dim}`);
  console.log(`Bits: ${index.bits}`);
  console.log(`Queries: ${queryCount}`);
  console.log(`topK: ${topK}`);
  console.log(`\nMemory at start:`);
  console.log(`  Heap used: ${formatBytes(memStart.heapUsed)}`);
  console.log(`  RSS: ${formatBytes(memStart.rss)}\n`);

  let maxHeapUsed = memStart.heapUsed;
  let maxRss = memStart.rss;

  // Track memory during single queries
  const singleStart = performance.now();
  for (const vector of queries) {
    await client.queryByVector(vector, topK);
    const mem = getMemoryUsage();
    maxHeapUsed = Math.max(maxHeapUsed, mem.heapUsed);
    maxRss = Math.max(maxRss, mem.rss);
  }
  const singleElapsed = performance.now() - singleStart;
  const memAfterSingle = getMemoryUsage();

  // Track memory during batch query
  const batchStart = performance.now();
  const batchResponse = await client.queryBatch({
    queries: queries.map((vector) => ({ vector })),
    topK,
  });
  const batchElapsed = performance.now() - batchStart;
  const memAfterBatch = getMemoryUsage();
  maxHeapUsed = Math.max(maxHeapUsed, memAfterBatch.heapUsed);
  maxRss = Math.max(maxRss, memAfterBatch.rss);

  // Track memory during ingest
  const records = Array.from({ length: ingestCount }, (_value, i) => ({
    chunk_id: `node-bench-${Date.now()}-${i}`,
    text: `Benchmark record ${i}`,
    embedding: makeVector(index.dim, 10_000 + i),
    metadata: { benchmark: true, index: i },
  }));

  const ingestStart = performance.now();
  const ingestResponse = await client.ingest({ records });
  const ingestElapsed = performance.now() - ingestStart;
  const memAfterIngest = getMemoryUsage();
  maxHeapUsed = Math.max(maxHeapUsed, memAfterIngest.heapUsed);
  maxRss = Math.max(maxRss, memAfterIngest.rss);

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
    memory: {
      startHeapUsedMB: Number((memStart.heapUsed / (1024 * 1024)).toFixed(2)),
      startRssMB: Number((memStart.rss / (1024 * 1024)).toFixed(2)),
      maxHeapUsedMB: Number((maxHeapUsed / (1024 * 1024)).toFixed(2)),
      maxRssMB: Number((maxRss / (1024 * 1024)).toFixed(2)),
      finalHeapUsedMB: Number((memAfterIngest.heapUsed / (1024 * 1024)).toFixed(2)),
      finalRssMB: Number((memAfterIngest.rss / (1024 * 1024)).toFixed(2)),
      heapGrowthMB: Number(((memAfterIngest.heapUsed - memStart.heapUsed) / (1024 * 1024)).toFixed(2)),
    },
  };

  console.log(JSON.stringify(results, null, 2));
  
  console.log("\n=== Memory Summary ===");
  console.log(`Start heap: ${formatBytes(memStart.heapUsed)}`);
  console.log(`Max heap:   ${formatBytes(maxHeapUsed)}`);
  console.log(`Final heap: ${formatBytes(memAfterIngest.heapUsed)}`);
  console.log(`Max RSS:    ${formatBytes(maxRss)}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
