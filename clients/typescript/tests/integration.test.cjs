const test = require("node:test");
const assert = require("node:assert/strict");

const { TurboRAG, ConnectionError, TurboRAGError } = require("../dist/index.js");

const runIntegration = process.env.TURBORAG_INTEGRATION_TEST === "1";

function makeVector(dim, seed) {
  const values = [];
  for (let i = 0; i < dim; i += 1) {
    values.push(Math.sin(seed + i * 0.07) + Math.cos(seed * 0.11 + i * 0.03));
  }
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)) || 1;
  return values.map((value) => value / norm);
}

test("integration: health, index, query, and ingest work against a live server", { skip: !runIntegration }, async () => {
  const baseUrl = process.env.TURBORAG_URL || "http://127.0.0.1:8080";
  const client = new TurboRAG(baseUrl, { timeout: 10_000 });

  const health = await client.health();
  assert.equal(health.status, "ok");

  const index = await client.index();
  assert.equal(typeof index.dim, "number");
  assert.equal(typeof index.index_size, "number");

  const queryVector = makeVector(index.dim, 7);
  const query = await client.query({ vector: queryVector, topK: 5 });
  assert.equal(typeof query.count, "number");
  assert(Array.isArray(query.results));

  const chunkId = `node-integration-${Date.now()}`;
  const ingest = await client.ingest({
    records: [
      {
        chunk_id: chunkId,
        text: "Node integration test record",
        embedding: makeVector(index.dim, 99),
        source_doc: "integration.txt",
        metadata: { suite: "node" },
      },
    ],
  });
  assert.equal(ingest.added, 1);

  const lookup = await client.query({ vector: makeVector(index.dim, 99), topK: 10 });
  assert(lookup.results.some((result) => result.chunk_id === chunkId));
});

test("integration: unreachable service raises ConnectionError", async () => {
  const client = new TurboRAG("http://127.0.0.1:59999", { timeout: 250 });
  await assert.rejects(() => client.health(), (error) => {
    assert(error instanceof Error);
    return error instanceof ConnectionError || error instanceof TurboRAGError;
  });
});
