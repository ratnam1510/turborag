const test = require("node:test");
const assert = require("node:assert/strict");

const {
  TurboRAG,
  TurboRAGClient,
  TurboRAGError,
  ValidationError,
  ConnectionError,
  TimeoutError,
  createClient,
  createClientFromEnv,
} = require("../dist/index.js");

function jsonResponse(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  };
}

function textResponse(body, status) {
  return {
    ok: false,
    status,
    json: async () => {
      throw new Error("not json");
    },
    text: async () => body,
  };
}

test("constructor trims trailing slash and preserves custom headers", async () => {
  const originalFetch = global.fetch;
  let seenUrl;
  let seenHeaders;

  global.fetch = async (url, options) => {
    seenUrl = url;
    seenHeaders = options.headers;
    return jsonResponse({ status: "ok", index_path: "./idx", index_size: 3 });
  };

  try {
    const client = new TurboRAG("http://localhost:8080/", {
      timeout: 1234,
      headers: { Authorization: "Bearer token" },
    });

    await client.health();

    assert.equal(client.baseUrl, "http://localhost:8080");
    assert.equal(client.timeout, 1234);
    assert.equal(seenUrl, "http://localhost:8080/health");
    assert.equal(seenHeaders.Authorization, "Bearer token");
    assert.equal(seenHeaders["Content-Type"], "application/json");
  } finally {
    global.fetch = originalFetch;
  }
});

test("constructor also accepts an options object", async () => {
  const originalFetch = global.fetch;
  let seenUrl;

  global.fetch = async (url) => {
    seenUrl = url;
    return jsonResponse({ status: "ok", index_path: "./idx", index_size: 1 });
  };

  try {
    const client = new TurboRAG({
      baseUrl: "http://localhost:8081/",
      timeoutMs: 500,
      defaultTopK: 9,
    });
    await client.health();

    assert.equal(client.baseUrl, "http://localhost:8081");
    assert.equal(client.timeout, 500);
    assert.equal(client.defaultTopK, 9);
    assert.equal(seenUrl, "http://localhost:8081/health");
  } finally {
    global.fetch = originalFetch;
  }
});

test("fromEnv and factory helpers create the expected client", async () => {
  const originalFetch = global.fetch;
  const urls = [];

  global.fetch = async (url) => {
    urls.push(url);
    return jsonResponse({ status: "ok", index_path: "./idx", index_size: 1 });
  };

  try {
    const envClient = TurboRAG.fromEnv({
      TURBORAG_API_URL: "http://localhost:9090/",
      TURBORAG_TIMEOUT_MS: "2222",
      TURBORAG_TOP_K: "11",
    });
    const aliasClient = createClient({ baseUrl: "http://localhost:9091" });
    const envAliasClient = createClientFromEnv({ TURBORAG_URL: "http://localhost:9092" });

    assert(envClient instanceof TurboRAG);
    assert(aliasClient instanceof TurboRAG);
    assert(envAliasClient instanceof TurboRAG);
    assert.equal(TurboRAGClient, TurboRAG);
    assert.equal(envClient.defaultTopK, 11);
    assert.equal(envClient.timeout, 2222);

    await envClient.health();
    await aliasClient.health();
    await envAliasClient.health();

    assert.deepEqual(urls, [
      "http://localhost:9090/health",
      "http://localhost:9091/health",
      "http://localhost:9092/health",
    ]);
  } finally {
    global.fetch = originalFetch;
  }
});

test("health performs a GET request", async () => {
  const originalFetch = global.fetch;

  global.fetch = async (url, options) => {
    assert.equal(url, "http://localhost:8080/health");
    assert.equal(options.method, "GET");
    return jsonResponse({ status: "ok", index_path: "./idx", index_size: 42 });
  };

  try {
    const client = new TurboRAG("http://localhost:8080");
    const result = await client.health();

    assert.deepEqual(result, {
      status: "ok",
      index_path: "./idx",
      index_size: 42,
    });
  } finally {
    global.fetch = originalFetch;
  }
});

test("index and metrics call the expected endpoints", async () => {
  const originalFetch = global.fetch;
  const calls = [];

  global.fetch = async (url, options) => {
    calls.push({ url, method: options.method });
    if (url.endsWith("/index")) {
      return jsonResponse({
        index_path: "./idx",
        dim: 384,
        bits: 4,
        shard_size: 1000,
        normalize: true,
        value_range: 1,
        index_size: 1000,
        records_loaded: 1000,
        records_snapshot: "./idx/records.jsonl",
        text_query_enabled: false,
      });
    }
    return jsonResponse({
      uptime_seconds: 1.5,
      errors: 0,
      endpoints: { "/query": { count: 2, total_ms: 3, avg_ms: 1.5, min_ms: 1, max_ms: 2 } },
    });
  };

  try {
    const client = new TurboRAG("http://localhost:8080");
    const index = await client.index();
    const metrics = await client.metrics();

    assert.equal(index.dim, 384);
    assert.equal(metrics.errors, 0);
    assert.deepEqual(calls, [
      { url: "http://localhost:8080/index", method: "GET" },
      { url: "http://localhost:8080/metrics", method: "GET" },
    ]);
  } finally {
    global.fetch = originalFetch;
  }
});

test("getIndex, describe, and getMetrics alias the original methods", async () => {
  const originalFetch = global.fetch;
  const calls = [];

  global.fetch = async (url) => {
    calls.push(url);
    if (url.endsWith("/index")) {
      return jsonResponse({ index_path: "./idx", dim: 384, bits: 4, shard_size: 1000, normalize: true, value_range: 1, index_size: 1000, records_loaded: 1000, records_snapshot: null, text_query_enabled: false });
    }
    return jsonResponse({ uptime_seconds: 1, errors: 0, endpoints: {} });
  };

  try {
    const client = new TurboRAG("http://localhost:8080");
    await client.getIndex();
    await client.describe();
    await client.getMetrics();
    assert.deepEqual(calls, [
      "http://localhost:8080/index",
      "http://localhost:8080/index",
      "http://localhost:8080/metrics",
    ]);
  } finally {
    global.fetch = originalFetch;
  }
});

test("query serializes vector requests correctly", async () => {
  const originalFetch = global.fetch;

  global.fetch = async (url, options) => {
    assert.equal(url, "http://localhost:8080/query");
    assert.equal(options.method, "POST");
    assert.deepEqual(JSON.parse(options.body), {
      query_vector: [0.1, 0.2, 0.3],
      top_k: 7,
    });
    return jsonResponse({
      count: 1,
      results: [
        {
          chunk_id: "chunk-1",
          text: "alpha",
          score: 0.99,
          source_doc: "doc.md",
          page_num: 2,
          graph_path: null,
          explanation: null,
        },
      ],
    });
  };

  try {
    const client = new TurboRAG("http://localhost:8080");
    const result = await client.query({ vector: [0.1, 0.2, 0.3], topK: 7 });

    assert.equal(result.count, 1);
    assert.equal(result.results[0].chunk_id, "chunk-1");
    assert.equal(result.results[0].source_doc, "doc.md");
  } finally {
    global.fetch = originalFetch;
  }
});

test("queryByVector sends X-Request-Id and uses default topK", async () => {
  const originalFetch = global.fetch;

  global.fetch = async (_url, options) => {
    assert.equal(options.headers["X-Request-Id"], "req-123");
    assert.deepEqual(JSON.parse(options.body), {
      query_vector: [0.5, 0.6],
      top_k: 8,
    });
    return jsonResponse({ count: 0, results: [] });
  };

  try {
    const client = new TurboRAG("http://localhost:8080", { defaultTopK: 8 });
    const result = await client.queryByVector([0.5, 0.6], undefined, "req-123");
    assert.equal(result.count, 0);
  } finally {
    global.fetch = originalFetch;
  }
});

test("query can disable hydration for ID-only sidecar responses", async () => {
  const originalFetch = global.fetch;

  global.fetch = async (_url, options) => {
    assert.deepEqual(JSON.parse(options.body), {
      query_vector: [0.1, 0.2, 0.3],
      top_k: 5,
      hydrate: false,
    });
    return jsonResponse({
      count: 1,
      results: [
        {
          chunk_id: "chunk-1",
          text: "",
          score: 0.9,
          source_doc: null,
          page_num: null,
          graph_path: null,
          explanation: "hydrate externally",
        },
      ],
    });
  };

  try {
    const client = new TurboRAG("http://localhost:8080");
    const result = await client.query({ vector: [0.1, 0.2, 0.3], hydrate: false });
    assert.equal(result.count, 1);
    assert.equal(result.results[0].chunk_id, "chunk-1");
    assert.equal(result.results[0].text, "");
  } finally {
    global.fetch = originalFetch;
  }
});

test("queryText serializes text requests correctly", async () => {
  const originalFetch = global.fetch;

  global.fetch = async (_url, options) => {
    assert.deepEqual(JSON.parse(options.body), {
      query_text: "capital guidance",
      top_k: 4,
    });
    return jsonResponse({ count: 0, results: [] });
  };

  try {
    const client = new TurboRAG("http://localhost:8080");
    const result = await client.queryText({ text: "capital guidance", topK: 4 });
    assert.equal(result.count, 0);
  } finally {
    global.fetch = originalFetch;
  }
});

test("queryByText uses the direct helper surface", async () => {
  const originalFetch = global.fetch;

  global.fetch = async (_url, options) => {
    assert.deepEqual(JSON.parse(options.body), {
      query_text: "search text",
      top_k: 5,
    });
    return jsonResponse({ count: 0, results: [] });
  };

  try {
    const client = new TurboRAG("http://localhost:8080");
    await client.queryByText("search text", 5);
  } finally {
    global.fetch = originalFetch;
  }
});

test("queryBatch maps vectors to query_vector payloads", async () => {
  const originalFetch = global.fetch;

  global.fetch = async (_url, options) => {
    assert.deepEqual(JSON.parse(options.body), {
      queries: [
        { query_vector: [1, 0] },
        { query_vector: [0, 1] },
      ],
      top_k: 3,
    });
    return jsonResponse({
      batch_count: 2,
      results: [
        { count: 1, results: [{ chunk_id: "a", text: "A", score: 1.0 }] },
        { count: 1, results: [{ chunk_id: "b", text: "B", score: 0.8 }] },
      ],
    });
  };

  try {
    const client = new TurboRAG("http://localhost:8080");
    const result = await client.queryBatch({
      queries: [{ vector: [1, 0] }, { vector: [0, 1] }],
      topK: 3,
    });

    assert.equal(result.batch_count, 2);
    assert.equal(result.results[1].results[0].chunk_id, "b");
  } finally {
    global.fetch = originalFetch;
  }
});

test("queryBatch can disable hydration", async () => {
  const originalFetch = global.fetch;

  global.fetch = async (_url, options) => {
    assert.deepEqual(JSON.parse(options.body), {
      queries: [{ query_vector: [1, 0] }],
      top_k: 2,
      hydrate: false,
    });
    return jsonResponse({
      batch_count: 1,
      results: [{ count: 1, results: [{ chunk_id: "a", text: "", score: 1.0 }] }],
    });
  };

  try {
    const client = new TurboRAG("http://localhost:8080");
    const result = await client.queryBatch({
      queries: [{ vector: [1, 0] }],
      topK: 2,
      hydrate: false,
    });
    assert.equal(result.batch_count, 1);
    assert.equal(result.results[0].results[0].chunk_id, "a");
  } finally {
    global.fetch = originalFetch;
  }
});

test("queryIds and queryBatchIds convenience helpers set hydrate=false", async () => {
  const originalFetch = global.fetch;
  const payloads = [];

  global.fetch = async (_url, options) => {
    payloads.push(JSON.parse(options.body));
    if (payloads.length === 1) {
      return jsonResponse({ count: 0, results: [] });
    }
    return jsonResponse({ batch_count: 0, results: [] });
  };

  try {
    const client = new TurboRAG("http://localhost:8080");
    await client.queryIds({ vector: [0.2, 0.3], topK: 4 });
    await client.queryBatchIds({ queries: [{ vector: [0.4, 0.6] }], topK: 3 });

    assert.deepEqual(payloads, [
      { query_vector: [0.2, 0.3], top_k: 4, hydrate: false },
      { queries: [{ query_vector: [0.4, 0.6] }], top_k: 3, hydrate: false },
    ]);
  } finally {
    global.fetch = originalFetch;
  }
});

test("invalid inputs raise ValidationError", async () => {
  const client = new TurboRAG("http://localhost:8080");

  await assert.rejects(() => client.query({ vector: [], topK: 5 }), ValidationError);
  await assert.rejects(() => client.queryText({ text: "   " }), ValidationError);
  await assert.rejects(() => client.queryBatch({ queries: [], topK: 1 }), ValidationError);
  await assert.rejects(
    () => client.ingest({ records: [{ chunk_id: "", text: "x", embedding: [1] }] }),
    ValidationError,
  );
  await assert.rejects(
    () => client.ingestText({ text: "doc", chunkConfig: { chunk_overlap: -1 } }),
    ValidationError,
  );
});

test("ingest serializes records payloads", async () => {
  const originalFetch = global.fetch;

  global.fetch = async (_url, options) => {
    assert.deepEqual(JSON.parse(options.body), {
      records: [
        {
          chunk_id: "c1",
          text: "hello",
          embedding: [0.2, 0.4],
          source_doc: "doc.txt",
        },
      ],
    });
    return jsonResponse({
      added: 1,
      index_size: 10,
      records_snapshot: "./index/records.jsonl",
    });
  };

  try {
    const client = new TurboRAG("http://localhost:8080");
    const result = await client.ingest({
      records: [
        {
          chunk_id: "c1",
          text: "hello",
          embedding: [0.2, 0.4],
          source_doc: "doc.txt",
        },
      ],
    });

    assert.equal(result.added, 1);
    assert.equal(result.records_snapshot, "./index/records.jsonl");
  } finally {
    global.fetch = originalFetch;
  }
});

test("ingestText serializes optional chunk config", async () => {
  const originalFetch = global.fetch;

  global.fetch = async (_url, options) => {
    assert.deepEqual(JSON.parse(options.body), {
      text: "full text",
      source_doc: "report.md",
      chunk_config: { chunk_size: 256, chunk_overlap: 32 },
    });
    return jsonResponse({
      added: 2,
      chunks: [
        { chunk_id: "c1", text: "part 1" },
        { chunk_id: "c2", text: "part 2" },
      ],
      index_size: 12,
    });
  };

  try {
    const client = new TurboRAG("http://localhost:8080");
    const result = await client.ingestText({
      text: "full text",
      sourceDoc: "report.md",
      chunkConfig: { chunk_size: 256, chunk_overlap: 32 },
    });

    assert.equal(result.added, 2);
    assert.equal(result.chunks[0].chunk_id, "c1");
  } finally {
    global.fetch = originalFetch;
  }
});

test("non-2xx GET responses raise TurboRAGError", async () => {
  const originalFetch = global.fetch;
  global.fetch = async () => textResponse("boom", 503);

  try {
    const client = new TurboRAG("http://localhost:8080");
    await assert.rejects(() => client.health(), (error) => {
      assert(error instanceof TurboRAGError);
      assert.equal(error.status, 503);
      assert.equal(error.body, "boom");
      return true;
    });
  } finally {
    global.fetch = originalFetch;
  }
});

test("non-2xx POST responses raise TurboRAGError", async () => {
  const originalFetch = global.fetch;
  global.fetch = async () => textResponse("bad query", 400);

  try {
    const client = new TurboRAG("http://localhost:8080");
    await assert.rejects(() => client.query({ vector: [1] }), (error) => {
      assert(error instanceof TurboRAGError);
      assert.equal(error.status, 400);
      assert.match(error.message, /bad query/);
      return true;
    });
  } finally {
    global.fetch = originalFetch;
  }
});

test("failed error-body reads still surface a TurboRAGError", async () => {
  const originalFetch = global.fetch;
  global.fetch = async () => ({
    ok: false,
    status: 502,
    json: async () => {
      throw new Error("not used");
    },
    text: async () => {
      throw new Error("broken body");
    },
  });

  try {
    const client = new TurboRAG("http://localhost:8080");
    await assert.rejects(() => client.index(), (error) => {
      assert(error instanceof TurboRAGError);
      assert.equal(error.status, 502);
      assert.equal(error.body, "");
      return true;
    });
  } finally {
    global.fetch = originalFetch;
  }
});

test("abort errors raise TimeoutError", async () => {
  const originalFetch = global.fetch;
  global.fetch = async () => {
    const error = new Error("aborted");
    error.name = "AbortError";
    throw error;
  };

  try {
    const client = new TurboRAG("http://localhost:8080", { timeout: 5 });
    await assert.rejects(() => client.health(), TimeoutError);
  } finally {
    global.fetch = originalFetch;
  }
});

test("network errors raise ConnectionError", async () => {
  const originalFetch = global.fetch;
  global.fetch = async () => {
    throw new TypeError("fetch failed");
  };

  try {
    const client = new TurboRAG("http://localhost:8080");
    await assert.rejects(() => client.health(), ConnectionError);
  } finally {
    global.fetch = originalFetch;
  }
});
