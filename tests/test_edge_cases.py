"""Edge-case robustness tests for core TurboRAG modules."""
from __future__ import annotations

import json

import numpy as np
import pytest

from turborag.compress import (
    SUPPORTED_BITS,
    bytes_per_vector,
    dequantize_qjl,
    generate_rotation,
    normalize_rows,
    quantize_qjl,
)
from turborag.index import TurboIndex
from turborag.types import ChunkRecord, RetrievalResult


# ---------------------------------------------------------------------------
# compress.py edge cases
# ---------------------------------------------------------------------------

class TestCompressEdgeCases:

    def test_quantize_1d_input(self):
        """quantize_qjl should accept a 1-D vector and reshape internally."""
        vec = np.array([0.5, -0.3, 0.1], dtype=np.float32)
        packed = quantize_qjl(vec, bits=3)
        assert packed.ndim == 2
        assert packed.shape[0] == 1

    def test_quantize_clipping_preserves_extreme_values(self):
        """Values outside [-value_range, value_range] are clipped, not NaN."""
        extreme = np.array([[10.0, -10.0, 0.0]], dtype=np.float32)
        packed = quantize_qjl(extreme, bits=3, value_range=1.0)
        restored = dequantize_qjl(packed, dim=3, bits=3, value_range=1.0)
        assert np.all(np.isfinite(restored))
        assert restored[0, 0] == pytest.approx(1.0, abs=0.01)
        assert restored[0, 1] == pytest.approx(-1.0, abs=0.01)

    def test_unsupported_bits_raises(self):
        with pytest.raises(ValueError, match="bits must be"):
            quantize_qjl(np.zeros((1, 4), dtype=np.float32), bits=5)

    def test_normalize_rows_zero_vector(self):
        """A zero vector should not produce NaN after normalization."""
        vectors = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        result = normalize_rows(vectors)
        assert np.all(np.isfinite(result))

    def test_bytes_per_vector_all_supported_bits(self):
        for bits in SUPPORTED_BITS:
            result = bytes_per_vector(dim=100, bits=bits)
            assert result > 0

    def test_generate_rotation_small_dim(self):
        rot = generate_rotation(dim=3, seed=0)
        assert rot.shape == (3, 3)
        identity = rot @ rot.T
        np.testing.assert_allclose(identity, np.eye(3, dtype=np.float32), atol=1e-5)


# ---------------------------------------------------------------------------
# index.py edge cases
# ---------------------------------------------------------------------------

class TestIndexEdgeCases:

    def test_search_empty_index_returns_empty(self):
        index = TurboIndex(dim=4, bits=3)
        result = index.search(np.zeros(4, dtype=np.float32), k=5)
        assert result == []

    def test_search_k_zero_returns_empty(self):
        index = TurboIndex(dim=4, bits=3)
        rng = np.random.default_rng(0)
        index.add(rng.normal(size=(3, 4)).astype(np.float32), ["a", "b", "c"])
        assert index.search(np.zeros(4, dtype=np.float32), k=0) == []

    def test_add_duplicate_ids_raises(self):
        index = TurboIndex(dim=4, bits=3)
        rng = np.random.default_rng(1)
        index.add(rng.normal(size=(2, 4)).astype(np.float32), ["a", "b"])
        with pytest.raises(ValueError, match="duplicate"):
            index.add(rng.normal(size=(1, 4)).astype(np.float32), ["a"])

    def test_add_mismatched_dim_raises(self):
        index = TurboIndex(dim=4, bits=3)
        with pytest.raises(ValueError, match="shape"):
            index.add(np.zeros((2, 5), dtype=np.float32), ["a", "b"])

    def test_add_mismatched_length_raises(self):
        index = TurboIndex(dim=4, bits=3)
        with pytest.raises(ValueError, match="same length"):
            index.add(np.zeros((2, 4), dtype=np.float32), ["a", "b", "c"])

    def test_add_non_unique_within_batch_raises(self):
        index = TurboIndex(dim=4, bits=3)
        with pytest.raises(ValueError, match="unique"):
            index.add(np.zeros((2, 4), dtype=np.float32), ["a", "a"])

    def test_search_2d_query_accepted(self):
        """A (1, dim) query matrix should be accepted."""
        index = TurboIndex(dim=4, bits=3)
        rng = np.random.default_rng(2)
        index.add(rng.normal(size=(3, 4)).astype(np.float32), ["a", "b", "c"])
        query = rng.normal(size=(1, 4)).astype(np.float32)
        result = index.search(query, k=2)
        assert len(result) == 2

    def test_len_tracks_additions(self):
        index = TurboIndex(dim=4, bits=3)
        assert len(index) == 0
        rng = np.random.default_rng(3)
        index.add(rng.normal(size=(5, 4)).astype(np.float32), [f"id-{i}" for i in range(5)])
        assert len(index) == 5

    def test_dim_must_be_positive(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            TurboIndex(dim=0, bits=3)

    def test_sharding_works(self):
        """When shard_size < total vectors, multiple shards are created."""
        index = TurboIndex(dim=4, bits=3, shard_size=2)
        rng = np.random.default_rng(4)
        index.add(rng.normal(size=(5, 4)).astype(np.float32), [f"id-{i}" for i in range(5)])
        assert len(index._shards) == 3  # 2+2+1
        assert len(index) == 5
        results = index.search(rng.normal(size=4).astype(np.float32), k=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# types.py edge cases
# ---------------------------------------------------------------------------

class TestTypesEdgeCases:

    def test_chunk_record_defaults(self):
        record = ChunkRecord(chunk_id="x", text="hello")
        assert record.source_doc is None
        assert record.page_num is None
        assert record.section is None
        assert record.metadata == {}

    def test_retrieval_result_defaults(self):
        result = RetrievalResult(chunk_id="x", text="hello", score=0.5)
        assert result.graph_path is None
        assert result.explanation is None

    def test_chunk_record_metadata_isolation(self):
        """Each ChunkRecord should get its own metadata dict."""
        r1 = ChunkRecord(chunk_id="a", text="a")
        r2 = ChunkRecord(chunk_id="b", text="b")
        r1.metadata["key"] = "val"
        assert "key" not in r2.metadata


# ---------------------------------------------------------------------------
# ingest edge cases (lightweight, no file I/O)
# ---------------------------------------------------------------------------

class TestIngestEdgeCases:

    def test_empty_jsonl_raises(self, tmp_path):
        from turborag.ingest import load_dataset

        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="no records found"):
            load_dataset(empty)

    def test_missing_embedding_raises(self, tmp_path):
        from turborag.ingest import load_dataset

        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps({"chunk_id": "a", "text": "hello"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing embedding"):
            load_dataset(bad)
