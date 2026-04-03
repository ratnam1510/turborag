const { describe, it, beforeEach } = require('node:test');
const assert = require('node:assert');
const {
    validateQueryPayload,
    validateBatchQueryPayload,
    validateIngestPayload,
    validateIngestTextPayload
} = require('../src/validators');

describe('validateQueryPayload', () => {
    it('should validate query_vector payload', () => {
        const result = validateQueryPayload({
            query_vector: [0.1, 0.2, 0.3],
            top_k: 5
        });
        assert.deepStrictEqual(result.queryVector, [0.1, 0.2, 0.3]);
        assert.strictEqual(result.queryText, null);
        assert.strictEqual(result.topK, 5);
    });

    it('should validate query_text payload', () => {
        const result = validateQueryPayload({
            query_text: 'test query',
            top_k: 10
        });
        assert.strictEqual(result.queryText, 'test query');
        assert.strictEqual(result.queryVector, null);
        assert.strictEqual(result.topK, 10);
    });

    it('should default top_k to 5', () => {
        const result = validateQueryPayload({ query_vector: [1.0] });
        assert.strictEqual(result.topK, 5);
    });

    it('should reject null payload', () => {
        assert.throws(() => validateQueryPayload(null), /must be an object/);
    });

    it('should reject non-object payload', () => {
        assert.throws(() => validateQueryPayload('string'), /must be an object/);
    });

    it('should reject unknown fields', () => {
        assert.throws(() => validateQueryPayload({
            query_vector: [1.0],
            unknown_field: true
        }), /Unknown query fields/);
    });

    it('should reject both query_text and query_vector', () => {
        assert.throws(() => validateQueryPayload({
            query_text: 'test',
            query_vector: [1.0]
        }), /exactly one/);
    });

    it('should reject neither query_text nor query_vector', () => {
        assert.throws(() => validateQueryPayload({
            top_k: 5
        }), /exactly one/);
    });

    it('should reject empty query_text', () => {
        assert.throws(() => validateQueryPayload({
            query_text: '   '
        }), /exactly one/);
    });

    it('should reject empty query_vector', () => {
        assert.throws(() => validateQueryPayload({
            query_vector: []
        }), /must be a non-empty JSON array/);
    });

    it('should reject non-numeric query_vector values', () => {
        assert.throws(() => validateQueryPayload({
            query_vector: [1.0, 'bad', 3.0]
        }), /numeric values/);
    });

    it('should reject top_k less than 1', () => {
        assert.throws(() => validateQueryPayload({
            query_vector: [1.0],
            top_k: 0
        }), /between 1 and 1000/);
    });

    it('should reject top_k greater than 1000', () => {
        assert.throws(() => validateQueryPayload({
            query_vector: [1.0],
            top_k: 1001
        }), /between 1 and 1000/);
    });
});

describe('validateBatchQueryPayload', () => {
    it('should validate batch query payload', () => {
        const result = validateBatchQueryPayload({
            queries: [
                { query_vector: [0.1, 0.2] },
                { query_vector: [0.3, 0.4] }
            ],
            top_k: 3
        });
        assert.strictEqual(result.queries.length, 2);
        assert.deepStrictEqual(result.queries[0].query_vector, [0.1, 0.2]);
        assert.strictEqual(result.topK, 3);
    });

    it('should default top_k to 5', () => {
        const result = validateBatchQueryPayload({
            queries: [{ query_vector: [1.0] }]
        });
        assert.strictEqual(result.topK, 5);
    });

    it('should reject empty queries array', () => {
        assert.throws(() => validateBatchQueryPayload({
            queries: []
        }), /non-empty array/);
    });

    it('should reject missing queries', () => {
        assert.throws(() => validateBatchQueryPayload({}), /non-empty array/);
    });

    it('should reject query without query_vector', () => {
        assert.throws(() => validateBatchQueryPayload({
            queries: [{}]
        }), /requires a non-empty query_vector/);
    });

    it('should reject non-numeric vector values in batch', () => {
        assert.throws(() => validateBatchQueryPayload({
            queries: [{ query_vector: ['bad'] }]
        }), /numeric values/);
    });
});

describe('validateIngestPayload', () => {
    it('should validate ingest payload', () => {
        const result = validateIngestPayload({
            records: [{
                chunk_id: 'chunk-1',
                text: 'Test text',
                embedding: [0.1, 0.2, 0.3],
                source_doc: 'test.pdf',
                page_num: 5,
                metadata: { key: 'value' }
            }]
        });
        assert.strictEqual(result.records.length, 1);
        assert.strictEqual(result.records[0].chunk_id, 'chunk-1');
        assert.strictEqual(result.records[0].text, 'Test text');
        assert.deepStrictEqual(result.records[0].embedding, [0.1, 0.2, 0.3]);
        assert.strictEqual(result.records[0].source_doc, 'test.pdf');
        assert.strictEqual(result.records[0].page_num, 5);
    });

    it('should handle minimal record', () => {
        const result = validateIngestPayload({
            records: [{
                chunk_id: 'chunk-1',
                text: 'Test',
                embedding: [1.0]
            }]
        });
        assert.strictEqual(result.records[0].source_doc, null);
        assert.strictEqual(result.records[0].page_num, null);
        assert.deepStrictEqual(result.records[0].metadata, {});
    });

    it('should reject empty records', () => {
        assert.throws(() => validateIngestPayload({
            records: []
        }), /At least one record/);
    });

    it('should reject unknown top-level fields', () => {
        assert.throws(() => validateIngestPayload({
            records: [{ chunk_id: 'a', text: 'b', embedding: [1] }],
            unknown: true
        }), /Unknown ingest fields/);
    });

    it('should reject unknown record fields', () => {
        assert.throws(() => validateIngestPayload({
            records: [{
                chunk_id: 'a',
                text: 'b',
                embedding: [1],
                unknown_field: true
            }]
        }), /unknown fields/);
    });

    it('should reject empty chunk_id', () => {
        assert.throws(() => validateIngestPayload({
            records: [{
                chunk_id: '',
                text: 'b',
                embedding: [1]
            }]
        }), /non-empty chunk_id/);
    });

    it('should reject missing text', () => {
        assert.throws(() => validateIngestPayload({
            records: [{
                chunk_id: 'a',
                embedding: [1]
            }]
        }), /requires text/);
    });

    it('should reject empty embedding', () => {
        assert.throws(() => validateIngestPayload({
            records: [{
                chunk_id: 'a',
                text: 'b',
                embedding: []
            }]
        }), /non-empty embedding/);
    });

    it('should reject non-numeric embedding values', () => {
        assert.throws(() => validateIngestPayload({
            records: [{
                chunk_id: 'a',
                text: 'b',
                embedding: ['bad']
            }]
        }), /numeric values/);
    });

    it('should reject non-object metadata', () => {
        assert.throws(() => validateIngestPayload({
            records: [{
                chunk_id: 'a',
                text: 'b',
                embedding: [1],
                metadata: 'string'
            }]
        }), /metadata must be an object/);
    });
});

describe('validateIngestTextPayload', () => {
    it('should validate ingest text payload', () => {
        const result = validateIngestTextPayload({
            text: 'Some document text',
            source_doc: 'doc.md',
            chunk_config: {
                chunk_size: 256,
                chunk_overlap: 32
            }
        });
        assert.strictEqual(result.text, 'Some document text');
        assert.strictEqual(result.sourceDoc, 'doc.md');
        assert.strictEqual(result.chunkConfig.chunk_size, 256);
        assert.strictEqual(result.chunkConfig.chunk_overlap, 32);
    });

    it('should use default chunk config', () => {
        const result = validateIngestTextPayload({
            text: 'Some text'
        });
        assert.strictEqual(result.chunkConfig.chunk_size, 512);
        assert.strictEqual(result.chunkConfig.chunk_overlap, 64);
    });

    it('should reject empty text', () => {
        assert.throws(() => validateIngestTextPayload({
            text: ''
        }), /non-empty/);
    });

    it('should reject whitespace-only text', () => {
        assert.throws(() => validateIngestTextPayload({
            text: '   '
        }), /non-empty/);
    });

    it('should reject missing text', () => {
        assert.throws(() => validateIngestTextPayload({}), /non-empty/);
    });
});
