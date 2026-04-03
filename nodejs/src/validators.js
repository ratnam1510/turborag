function validateQueryPayload(payload) {
    if (!payload || typeof payload !== 'object') {
        throw new Error('Query payload must be an object');
    }

    const allowedKeys = new Set(['query_text', 'query_vector', 'top_k']);
    const unknownKeys = Object.keys(payload).filter(k => !allowedKeys.has(k));
    if (unknownKeys.length > 0) {
        throw new Error(`Unknown query fields: ${unknownKeys.join(', ')}`);
    }

    const hasText = typeof payload.query_text === 'string' && payload.query_text.trim().length > 0;
    const hasVector = payload.query_vector !== undefined;

    if (hasVector && (!Array.isArray(payload.query_vector) || payload.query_vector.length === 0)) {
        throw new Error('query_vector must be a non-empty JSON array');
    }

    if (hasText === (hasVector && payload.query_vector.length > 0)) {
        throw new Error('Provide exactly one of query_text or query_vector');
    }

    if (hasVector) {
        if (!payload.query_vector.every(v => typeof v === 'number' && !isNaN(v))) {
            throw new Error('query_vector must contain only numeric values');
        }
    }

    const topK = payload.top_k !== undefined ? parseInt(payload.top_k, 10) : 5;
    if (isNaN(topK) || topK <= 0 || topK > 1000) {
        throw new Error('top_k must be between 1 and 1000');
    }

    return {
        queryText: hasText ? payload.query_text : null,
        queryVector: hasVector ? payload.query_vector : null,
        topK
    };
}

function validateBatchQueryPayload(payload) {
    if (!payload || typeof payload !== 'object') {
        throw new Error('Batch query payload must be an object');
    }

    if (!Array.isArray(payload.queries) || payload.queries.length === 0) {
        throw new Error('queries must be a non-empty array');
    }

    const topK = payload.top_k !== undefined ? parseInt(payload.top_k, 10) : 5;
    if (isNaN(topK) || topK <= 0 || topK > 1000) {
        throw new Error('top_k must be between 1 and 1000');
    }

    const validatedQueries = payload.queries.map((q, i) => {
        if (!q || typeof q !== 'object') {
            throw new Error(`Query ${i + 1} must be an object with query_vector`);
        }
        if (!Array.isArray(q.query_vector) || q.query_vector.length === 0) {
            throw new Error(`Query ${i + 1} requires a non-empty query_vector`);
        }
        if (!q.query_vector.every(v => typeof v === 'number' && !isNaN(v))) {
            throw new Error(`Query ${i + 1} query_vector must contain only numeric values`);
        }
        return { query_vector: q.query_vector.map(v => parseFloat(v)) };
    });

    return { queries: validatedQueries, topK };
}

function validateIngestPayload(payload) {
    if (!payload || typeof payload !== 'object') {
        throw new Error('Ingest payload must be an object');
    }

    const allowedKeys = new Set(['records']);
    const unknownKeys = Object.keys(payload).filter(k => !allowedKeys.has(k));
    if (unknownKeys.length > 0) {
        throw new Error(`Unknown ingest fields: ${unknownKeys.join(', ')}`);
    }

    if (!Array.isArray(payload.records) || payload.records.length === 0) {
        throw new Error('At least one record is required');
    }

    const allowedRecordKeys = new Set(['chunk_id', 'text', 'embedding', 'source_doc', 'page_num', 'section', 'metadata']);

    const validatedRecords = payload.records.map((item, index) => {
        if (!item || typeof item !== 'object') {
            throw new Error(`Record ${index + 1} must be an object`);
        }

        const unknownRecordKeys = Object.keys(item).filter(k => !allowedRecordKeys.has(k));
        if (unknownRecordKeys.length > 0) {
            throw new Error(`Record ${index + 1} contains unknown fields: ${unknownRecordKeys.join(', ')}`);
        }

        if (typeof item.chunk_id !== 'string' || item.chunk_id.length === 0) {
            throw new Error(`Record ${index + 1} requires a non-empty chunk_id`);
        }
        if (typeof item.text !== 'string') {
            throw new Error(`Record ${index + 1} requires text`);
        }
        if (!Array.isArray(item.embedding) || item.embedding.length === 0) {
            throw new Error(`Record ${index + 1} requires a non-empty embedding array`);
        }
        if (!item.embedding.every(v => typeof v === 'number' && !isNaN(v))) {
            throw new Error(`Record ${index + 1} embedding must contain only numeric values`);
        }
        if (item.metadata !== undefined && (typeof item.metadata !== 'object' || item.metadata === null)) {
            throw new Error(`Record ${index + 1} metadata must be an object`);
        }

        return {
            chunk_id: item.chunk_id,
            text: item.text,
            embedding: item.embedding.map(v => parseFloat(v)),
            source_doc: item.source_doc || null,
            page_num: item.page_num !== undefined ? parseInt(item.page_num, 10) : null,
            section: item.section || null,
            metadata: item.metadata || {}
        };
    });

    return { records: validatedRecords };
}

function validateIngestTextPayload(payload) {
    if (!payload || typeof payload !== 'object') {
        throw new Error('Ingest text payload must be an object');
    }

    if (typeof payload.text !== 'string' || payload.text.trim().length === 0) {
        throw new Error('text is required and must be non-empty');
    }

    const chunkConfig = payload.chunk_config || {};
    const chunkSize = chunkConfig.chunk_size !== undefined ? parseInt(chunkConfig.chunk_size, 10) : 512;
    const chunkOverlap = chunkConfig.chunk_overlap !== undefined ? parseInt(chunkConfig.chunk_overlap, 10) : 64;

    if (isNaN(chunkSize) || chunkSize <= 0) {
        throw new Error('chunk_size must be a positive integer');
    }
    if (isNaN(chunkOverlap) || chunkOverlap < 0) {
        throw new Error('chunk_overlap must be a non-negative integer');
    }

    return {
        text: payload.text,
        sourceDoc: payload.source_doc || null,
        chunkConfig: { chunk_size: chunkSize, chunk_overlap: chunkOverlap }
    };
}

module.exports = {
    validateQueryPayload,
    validateBatchQueryPayload,
    validateIngestPayload,
    validateIngestTextPayload
};
