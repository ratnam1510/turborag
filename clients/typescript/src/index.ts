/**
 * TurboRAG TypeScript/Node.js Client
 *
 * Thin typed wrapper over the TurboRAG HTTP API.
 * Zero runtime dependencies — uses native fetch.
 *
 * Backward compatible with the existing `TurboRAG` API while also exposing
 * the stronger convenience surface from the PR review (`fromEnv`,
 * `queryByVector`, aliases like `getIndex`, request IDs, and richer errors).
 *
 * @example
 * ```ts
 * import { TurboRAG } from "turborag";
 *
 * const client = new TurboRAG("http://localhost:8080");
 * const results = await client.query({ vector: [0.1, 0.2, 0.3], topK: 5 });
 * ```
 */

export interface QueryResult {
  chunk_id: string;
  text: string;
  score: number;
  source_doc: string | null;
  page_num: number | null;
  graph_path: string[] | null;
  explanation: string | null;
}

export interface QueryResponse {
  count: number;
  results: QueryResult[];
}

export interface QueryParams {
  vector: number[];
  topK?: number;
  hydrate?: boolean;
}

export interface BatchQueryResponse {
  batch_count: number;
  results: { count: number; results: QueryResult[] }[];
}

export interface BatchQueryParams {
  queries: { vector: number[] }[];
  topK?: number;
  hydrate?: boolean;
}

export interface IngestRecord {
  chunk_id: string;
  text: string;
  embedding: number[];
  source_doc?: string;
  page_num?: number;
  section?: string;
  metadata?: Record<string, unknown>;
}

export interface IngestResponse {
  added: number;
  index_size: number;
  records_snapshot: string | null;
}

export interface ChunkConfig {
  chunk_size?: number;
  chunk_overlap?: number;
}

export interface IngestTextResponse {
  added: number;
  chunks: { chunk_id: string; text: string }[];
  index_size: number;
}

export interface HealthResponse {
  status: string;
  index_path: string;
  index_size: number;
}

export interface IndexInfo {
  index_path: string;
  dim: number;
  bits: number;
  shard_size: number;
  normalize: boolean;
  value_range: number;
  index_size: number;
  records_loaded: number;
  records_snapshot: string | null;
  text_query_enabled: boolean;
  hydration_source?: "local_snapshot" | "external_backend" | "hybrid" | "id_only";
  allow_unhydrated?: boolean;
}

export interface MetricsResponse {
  uptime_seconds: number;
  errors: number;
  endpoints: Record<
    string,
    {
      count: number;
      total_ms: number;
      avg_ms: number;
      min_ms: number | null;
      max_ms: number | null;
    }
  >;
}

export interface TurboRAGOptions {
  timeout?: number;
  timeoutMs?: number;
  defaultTopK?: number;
  topK?: number;
  headers?: Record<string, string>;
}

export interface TurboRAGClientOptions extends TurboRAGOptions {
  baseUrl?: string;
  apiUrl?: string;
}

type EnvMap = Record<string, string | undefined>;

function readEnv(): EnvMap {
  const scope = globalThis as typeof globalThis & {
    process?: { env?: EnvMap };
  };
  return scope.process?.env ?? {};
}

export class TurboRAGError extends Error {
  status: number | null;
  body: string;
  details: unknown;

  constructor(message: string, status: number | null = null, body = "", details: unknown = null) {
    super(message);
    this.name = "TurboRAGError";
    this.status = status;
    this.body = body;
    this.details = details;
  }
}

export class ValidationError extends TurboRAGError {
  constructor(message: string, details: unknown = null) {
    super(message, 400, "", details);
    this.name = "ValidationError";
  }
}

export class ConnectionError extends TurboRAGError {
  constructor(message: string, details: unknown = null) {
    super(message, null, "", details);
    this.name = "ConnectionError";
  }
}

export class TimeoutError extends TurboRAGError {
  constructor(message: string, details: unknown = null) {
    super(message, null, "", details);
    this.name = "TimeoutError";
  }
}

export class TurboRAG {
  private baseUrl: string;
  private timeout: number;
  private defaultTopK: number;
  private headers: Record<string, string>;

  constructor(baseUrl: string, options?: TurboRAGOptions);
  constructor(options?: TurboRAGClientOptions);
  constructor(
    baseUrlOrOptions: string | TurboRAGClientOptions = "http://localhost:8080",
    options: TurboRAGOptions = {},
  ) {
    const config =
      typeof baseUrlOrOptions === "string"
        ? {
            baseUrl: baseUrlOrOptions,
            ...options,
          }
        : {
            ...baseUrlOrOptions,
          };

    this.baseUrl = (config.baseUrl ?? config.apiUrl ?? "http://localhost:8080").replace(/\/+$/, "");
    this.timeout = config.timeout ?? config.timeoutMs ?? 30_000;
    this.defaultTopK = config.defaultTopK ?? config.topK ?? 5;
    this.headers = {
      "Content-Type": "application/json",
      ...config.headers,
    };
  }

  static fromEnv(env: EnvMap = readEnv()): TurboRAG {
    return new TurboRAG({
      baseUrl: env.TURBORAG_API_URL ?? env.TURBORAG_URL ?? "http://localhost:8080",
      timeout:
        env.TURBORAG_TIMEOUT_MS !== undefined || env.TURBORAG_TIMEOUT !== undefined
          ? Number.parseInt(env.TURBORAG_TIMEOUT_MS ?? env.TURBORAG_TIMEOUT ?? "30000", 10)
          : 30_000,
      defaultTopK:
        env.TURBORAG_TOP_K !== undefined ? Number.parseInt(env.TURBORAG_TOP_K, 10) : 5,
    });
  }

  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>("GET", "/health");
  }

  async index(): Promise<IndexInfo> {
    return this.request<IndexInfo>("GET", "/index");
  }

  async getIndex(): Promise<IndexInfo> {
    return this.index();
  }

  async describe(): Promise<IndexInfo> {
    return this.index();
  }

  async metrics(): Promise<MetricsResponse> {
    return this.request<MetricsResponse>("GET", "/metrics");
  }

  async getMetrics(): Promise<MetricsResponse> {
    return this.metrics();
  }

  async query(
    params: QueryParams,
    requestId?: string,
  ): Promise<QueryResponse> {
    this.assertVector(params.vector, "vector");
    const topK = this.resolveTopK(params.topK);
    if (params.hydrate !== undefined && typeof params.hydrate !== "boolean") {
      throw new ValidationError("hydrate must be a boolean");
    }

    const payload: Record<string, unknown> = {
      query_vector: params.vector,
      top_k: topK,
    };
    if (params.hydrate !== undefined) {
      payload.hydrate = params.hydrate;
    }

    return this.request<QueryResponse>(
      "POST",
      "/query",
      payload,
      requestId,
    );
  }

  async queryByVector(
    vector: number[],
    topK?: number,
    requestId?: string,
    hydrate?: boolean,
  ): Promise<QueryResponse> {
    return this.query({ vector, topK, hydrate }, requestId);
  }

  async queryText(
    params: {
      text: string;
      topK?: number;
    },
    requestId?: string,
  ): Promise<QueryResponse> {
    this.assertText(params.text, "text");
    const topK = this.resolveTopK(params.topK);
    return this.request<QueryResponse>(
      "POST",
      "/query",
      {
        query_text: params.text,
        top_k: topK,
      },
      requestId,
    );
  }

  async queryByText(text: string, topK?: number, requestId?: string): Promise<QueryResponse> {
    return this.queryText({ text, topK }, requestId);
  }

  async queryBatch(
    params: BatchQueryParams,
    requestId?: string,
  ): Promise<BatchQueryResponse> {
    if (!Array.isArray(params.queries) || params.queries.length === 0) {
      throw new ValidationError("queries must be a non-empty array");
    }
    const topK = this.resolveTopK(params.topK);
    if (params.hydrate !== undefined && typeof params.hydrate !== "boolean") {
      throw new ValidationError("hydrate must be a boolean");
    }

    const queries = params.queries.map((query, index) => {
      this.assertVector(query.vector, `queries[${index}].vector`);
      return { query_vector: query.vector };
    });

    const payload: Record<string, unknown> = {
      queries,
      top_k: topK,
    };
    if (params.hydrate !== undefined) {
      payload.hydrate = params.hydrate;
    }

    return this.request<BatchQueryResponse>(
      "POST",
      "/query/batch",
      payload,
      requestId,
    );
  }

  async queryIds(
    params: {
      vector: number[];
      topK?: number;
    },
    requestId?: string,
  ): Promise<QueryResponse> {
    return this.query({ ...params, hydrate: false }, requestId);
  }

  async queryBatchIds(
    params: {
      queries: { vector: number[] }[];
      topK?: number;
    },
    requestId?: string,
  ): Promise<BatchQueryResponse> {
    return this.queryBatch({ ...params, hydrate: false }, requestId);
  }

  async ingest(
    params: {
      records: IngestRecord[];
    },
    requestId?: string,
  ): Promise<IngestResponse> {
    if (!Array.isArray(params.records) || params.records.length === 0) {
      throw new ValidationError("records must be a non-empty array");
    }
    for (const [index, record] of params.records.entries()) {
      if (!record || typeof record.chunk_id !== "string" || record.chunk_id.length === 0) {
        throw new ValidationError(`records[${index}].chunk_id must be a non-empty string`);
      }
      if (typeof record.text !== "string") {
        throw new ValidationError(`records[${index}].text must be a string`);
      }
      this.assertVector(record.embedding, `records[${index}].embedding`);
    }

    return this.request<IngestResponse>(
      "POST",
      "/ingest",
      { records: params.records },
      requestId,
    );
  }

  async ingestText(
    params: {
      text: string;
      sourceDoc?: string;
      chunkConfig?: ChunkConfig;
    },
    requestId?: string,
  ): Promise<IngestTextResponse> {
    this.assertText(params.text, "text");
    if (
      params.chunkConfig?.chunk_size !== undefined &&
      (!Number.isInteger(params.chunkConfig.chunk_size) || params.chunkConfig.chunk_size <= 0)
    ) {
      throw new ValidationError("chunkConfig.chunk_size must be a positive integer");
    }
    if (
      params.chunkConfig?.chunk_overlap !== undefined &&
      (!Number.isInteger(params.chunkConfig.chunk_overlap) || params.chunkConfig.chunk_overlap < 0)
    ) {
      throw new ValidationError("chunkConfig.chunk_overlap must be a non-negative integer");
    }

    return this.request<IngestTextResponse>(
      "POST",
      "/ingest-text",
      {
        text: params.text,
        source_doc: params.sourceDoc,
        chunk_config: params.chunkConfig,
      },
      requestId,
    );
  }

  private resolveTopK(topK?: number): number {
    const value = topK ?? this.defaultTopK;
    if (!Number.isInteger(value) || value < 1 || value > 1000) {
      throw new ValidationError("topK must be an integer between 1 and 1000");
    }
    return value;
  }

  private assertVector(vector: number[], label: string): void {
    if (!Array.isArray(vector) || vector.length === 0) {
      throw new ValidationError(`${label} must be a non-empty array`);
    }
    if (!vector.every((value) => typeof value === "number" && Number.isFinite(value))) {
      throw new ValidationError(`${label} must contain only numeric values`);
    }
  }

  private assertText(text: string, label: string): void {
    if (typeof text !== "string" || text.trim().length === 0) {
      throw new ValidationError(`${label} must be a non-empty string`);
    }
  }

  private async request<T>(
    method: "GET" | "POST",
    path: string,
    body?: unknown,
    requestId?: string,
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    const headers = { ...this.headers };
    if (requestId) {
      headers["X-Request-Id"] = requestId;
    }

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        const bodyText = await response.text().catch(() => "");
        throw new TurboRAGError(`TurboRAG HTTP ${response.status}: ${bodyText}`, response.status, bodyText);
      }

      return (await response.json()) as T;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof TurboRAGError) {
        throw error;
      }

      if (error instanceof ValidationError) {
        throw error;
      }

      if (error && typeof error === "object" && "name" in error && error.name === "AbortError") {
        throw new TimeoutError(`Request to ${path} timed out after ${this.timeout}ms`, error);
      }

      if (error instanceof Error) {
        throw new ConnectionError(`Failed to reach TurboRAG at ${this.baseUrl}: ${error.message}`, error);
      }

      throw new TurboRAGError("Unknown TurboRAG client error", null, "", error);
    }
  }
}

export const TurboRAGClient = TurboRAG;

export function createClient(
  baseUrlOrOptions: string | TurboRAGClientOptions = "http://localhost:8080",
  options?: TurboRAGOptions,
): TurboRAG {
  return typeof baseUrlOrOptions === "string"
    ? new TurboRAG(baseUrlOrOptions, options)
    : new TurboRAG(baseUrlOrOptions);
}

export function createClientFromEnv(env?: EnvMap): TurboRAG {
  return TurboRAG.fromEnv(env);
}

export default TurboRAG;
