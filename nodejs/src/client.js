const { TurboRAGError, ValidationError, ConnectionError, TimeoutError } = require('./errors');
const {
    validateQueryPayload,
    validateBatchQueryPayload,
    validateIngestPayload,
    validateIngestTextPayload
} = require('./validators');

class TurboRAGClient {
    constructor(options = {}) {
        this.baseUrl = (options.baseUrl || options.apiUrl || 'http://localhost:8080').replace(/\/+$/, '');
        this.timeout = options.timeout || options.timeoutMs || 30000;
        this.defaultTopK = options.defaultTopK || options.topK || 5;
        this.headers = {
            'Content-Type': 'application/json',
            ...(options.headers || {})
        };
    }

    static fromEnv(env = process.env) {
        return new TurboRAGClient({
            baseUrl: env.TURBORAG_API_URL || env.TURBORAG_URL || 'http://localhost:8080',
            timeout: parseInt(env.TURBORAG_TIMEOUT_MS || env.TURBORAG_TIMEOUT || '30000', 10),
            defaultTopK: parseInt(env.TURBORAG_TOP_K || '5', 10)
        });
    }

    async _request(method, path, body = null, requestId = null) {
        const url = `${this.baseUrl}${path}`;
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        const headers = { ...this.headers };
        if (requestId) {
            headers['X-Request-Id'] = requestId;
        }

        try {
            const options = {
                method,
                headers,
                signal: controller.signal
            };

            if (body !== null) {
                options.body = JSON.stringify(body);
            }

            const response = await fetch(url, options);
            clearTimeout(timeoutId);

            let data;
            const contentType = response.headers.get('content-type') || '';
            if (contentType.includes('application/json')) {
                data = await response.json();
            } else {
                const text = await response.text();
                try {
                    data = JSON.parse(text);
                } catch {
                    data = { detail: text };
                }
            }

            if (!response.ok) {
                const message = data.detail || data.message || `HTTP ${response.status}`;
                throw new TurboRAGError(message, response.status, data);
            }

            return data;
        } catch (error) {
            clearTimeout(timeoutId);

            if (error instanceof TurboRAGError) {
                throw error;
            }

            if (error.name === 'AbortError') {
                throw new TimeoutError(`Request to ${path} timed out after ${this.timeout}ms`);
            }

            if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND' || error.cause?.code === 'ECONNREFUSED') {
                throw new ConnectionError(`Failed to connect to TurboRAG at ${this.baseUrl}: ${error.message}`);
            }

            throw new TurboRAGError(`Request failed: ${error.message}`, null, { originalError: error.message });
        }
    }

    async health() {
        return this._request('GET', '/health');
    }

    async getIndex() {
        return this._request('GET', '/index');
    }

    async describe() {
        return this._request('GET', '/index');
    }

    async getMetrics() {
        return this._request('GET', '/metrics');
    }

    async query(options, requestId = null) {
        const validated = validateQueryPayload(options);
        const payload = { top_k: validated.topK };

        if (validated.queryText !== null) {
            payload.query_text = validated.queryText;
        } else {
            payload.query_vector = validated.queryVector;
        }

        const response = await this._request('POST', '/query', payload, requestId);
        return this._normalizeQueryResponse(response);
    }

    async queryByVector(vector, topK = null, requestId = null) {
        if (!Array.isArray(vector) || vector.length === 0) {
            throw new ValidationError('vector must be a non-empty array');
        }
        return this.query({
            query_vector: vector,
            top_k: topK || this.defaultTopK
        }, requestId);
    }

    async queryByText(text, topK = null, requestId = null) {
        if (typeof text !== 'string' || text.trim().length === 0) {
            throw new ValidationError('text must be a non-empty string');
        }
        return this.query({
            query_text: text,
            top_k: topK || this.defaultTopK
        }, requestId);
    }

    async queryBatch(queries, topK = null, requestId = null) {
        const validated = validateBatchQueryPayload({
            queries,
            top_k: topK || this.defaultTopK
        });

        const payload = {
            queries: validated.queries,
            top_k: validated.topK
        };

        const response = await this._request('POST', '/query/batch', payload, requestId);
        return this._normalizeBatchResponse(response);
    }

    async ingest(records, requestId = null) {
        const validated = validateIngestPayload({ records });
        const response = await this._request('POST', '/ingest', validated, requestId);
        return this._normalizeIngestResponse(response);
    }

    async ingestText(text, sourceDoc = null, chunkConfig = null, requestId = null) {
        const validated = validateIngestTextPayload({
            text,
            source_doc: sourceDoc,
            chunk_config: chunkConfig
        });

        const payload = {
            text: validated.text
        };
        if (validated.sourceDoc) {
            payload.source_doc = validated.sourceDoc;
        }
        if (validated.chunkConfig) {
            payload.chunk_config = validated.chunkConfig;
        }

        const response = await this._request('POST', '/ingest-text', payload, requestId);
        return this._normalizeIngestTextResponse(response);
    }

    _normalizeQueryResponse(response) {
        return {
            count: response.count || 0,
            results: (response.results || []).map(r => this._normalizeResult(r))
        };
    }

    _normalizeBatchResponse(response) {
        return {
            batchCount: response.batch_count || 0,
            results: (response.results || []).map(batch => ({
                count: batch.count || 0,
                results: (batch.results || []).map(r => this._normalizeResult(r))
            }))
        };
    }

    _normalizeResult(result) {
        return {
            chunkId: result.chunk_id,
            text: result.text || null,
            score: typeof result.score === 'number' ? result.score : parseFloat(result.score),
            sourceDoc: result.source_doc || null,
            pageNum: result.page_num !== undefined && result.page_num !== null ? parseInt(result.page_num, 10) : null,
            graphPath: result.graph_path || null,
            explanation: result.explanation || null
        };
    }

    _normalizeIngestResponse(response) {
        return {
            added: response.added || 0,
            indexSize: response.index_size || 0,
            recordsSnapshot: response.records_snapshot || null
        };
    }

    _normalizeIngestTextResponse(response) {
        return {
            added: response.added || 0,
            chunks: (response.chunks || []).map(c => ({
                chunkId: c.chunk_id,
                text: c.text
            })),
            indexSize: response.index_size || 0
        };
    }
}

function createClient(options = {}) {
    return new TurboRAGClient(options);
}

function createClientFromEnv(env = process.env) {
    return TurboRAGClient.fromEnv(env);
}

module.exports = {
    TurboRAGClient,
    createClient,
    createClientFromEnv
};
