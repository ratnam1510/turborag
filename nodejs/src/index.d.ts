export interface TurboRAGClientOptions {
    baseUrl?: string;
    apiUrl?: string;
    timeout?: number;
    timeoutMs?: number;
    defaultTopK?: number;
    topK?: number;
    headers?: Record<string, string>;
}

export interface HealthResponse {
    status: string;
    index_path: string;
    index_size: number;
}

export interface IndexResponse {
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
}

export interface MetricsResponse {
    uptime_seconds: number;
    errors: number;
    endpoints: Record<string, {
        count: number;
        total_ms: number;
        avg_ms: number;
        min_ms: number | null;
        max_ms: number | null;
    }>;
}

export interface QueryOptions {
    query_text?: string;
    query_vector?: number[];
    top_k?: number;
}

export interface BatchQuery {
    query_vector: number[];
}

export interface RetrievalResult {
    chunkId: string;
    text: string | null;
    score: number;
    sourceDoc: string | null;
    pageNum: number | null;
    graphPath: string[] | null;
    explanation: string | null;
}

export interface QueryResponse {
    count: number;
    results: RetrievalResult[];
}

export interface BatchQueryResponse {
    batchCount: number;
    results: QueryResponse[];
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
    indexSize: number;
    recordsSnapshot: string | null;
}

export interface ChunkConfig {
    chunk_size?: number;
    chunk_overlap?: number;
}

export interface IngestTextResponse {
    added: number;
    chunks: Array<{
        chunkId: string;
        text: string;
    }>;
    indexSize: number;
}

export declare class TurboRAGError extends Error {
    name: string;
    statusCode: number | null;
    details: unknown;
    constructor(message: string, statusCode?: number | null, details?: unknown);
}

export declare class ValidationError extends TurboRAGError {
    name: string;
    constructor(message: string, details?: unknown);
}

export declare class ConnectionError extends TurboRAGError {
    name: string;
    constructor(message: string, details?: unknown);
}

export declare class TimeoutError extends TurboRAGError {
    name: string;
    constructor(message: string, details?: unknown);
}

export declare class TurboRAGClient {
    baseUrl: string;
    timeout: number;
    defaultTopK: number;
    headers: Record<string, string>;

    constructor(options?: TurboRAGClientOptions);
    static fromEnv(env?: NodeJS.ProcessEnv): TurboRAGClient;

    health(): Promise<HealthResponse>;
    getIndex(): Promise<IndexResponse>;
    describe(): Promise<IndexResponse>;
    getMetrics(): Promise<MetricsResponse>;

    query(options: QueryOptions, requestId?: string | null): Promise<QueryResponse>;
    queryByVector(vector: number[], topK?: number | null, requestId?: string | null): Promise<QueryResponse>;
    queryByText(text: string, topK?: number | null, requestId?: string | null): Promise<QueryResponse>;
    queryBatch(queries: BatchQuery[], topK?: number | null, requestId?: string | null): Promise<BatchQueryResponse>;

    ingest(records: IngestRecord[], requestId?: string | null): Promise<IngestResponse>;
    ingestText(text: string, sourceDoc?: string | null, chunkConfig?: ChunkConfig | null, requestId?: string | null): Promise<IngestTextResponse>;
}

export declare function createClient(options?: TurboRAGClientOptions): TurboRAGClient;
export declare function createClientFromEnv(env?: NodeJS.ProcessEnv): TurboRAGClient;

export declare function validateQueryPayload(payload: unknown): {
    queryText: string | null;
    queryVector: number[] | null;
    topK: number;
};

export declare function validateBatchQueryPayload(payload: unknown): {
    queries: BatchQuery[];
    topK: number;
};

export declare function validateIngestPayload(payload: unknown): {
    records: IngestRecord[];
};

export declare function validateIngestTextPayload(payload: unknown): {
    text: string;
    sourceDoc: string | null;
    chunkConfig: ChunkConfig;
};
