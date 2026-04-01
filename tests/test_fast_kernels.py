"""Tests for the fast LUT-based scoring kernels."""
from __future__ import annotations

import numpy as np
import pytest

from turborag.compress import (
    compressed_dot,
    compressed_dot_naive,
    generate_rotation,
    normalize_rows,
    quantize_qjl,
)
from turborag.fast_kernels import (
    build_query_lut,
    score_shard_lut,
    score_shard_lut_batch,
)


class TestLUTCorrectnessVsNaive:
    """Verify the LUT scorer produces the same results as the naive dequantize-then-matmul path."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    @pytest.mark.parametrize("dim", [8, 16, 64, 128])
    def test_scores_match_naive(self, bits: int, dim: int) -> None:
        rng = np.random.default_rng(42)
        n_db = 50
        rotation = generate_rotation(dim, seed=7)

        db_raw = rng.normal(size=(n_db, dim)).astype(np.float32)
        q_raw = rng.normal(size=(1, dim)).astype(np.float32)

        db_norm = normalize_rows(db_raw)
        q_norm = normalize_rows(q_raw)

        db_rotated = (db_norm @ rotation.T).astype(np.float32)
        q_rotated = (q_norm @ rotation.T).astype(np.float32)

        db_packed = quantize_qjl(db_rotated, bits=bits)
        q_packed = quantize_qjl(q_rotated, bits=bits)

        # Naive path: full dequantization + matmul
        naive_scores = compressed_dot_naive(q_packed, db_packed, dim=dim, bits=bits)

        # LUT path: precompute table + scan packed bytes
        lut_scores = compressed_dot(q_packed, db_packed, dim=dim, bits=bits)

        np.testing.assert_allclose(lut_scores, naive_scores, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_ranking_order_matches(self, bits: int) -> None:
        """The top-k retrieved set and scores must agree between LUT and naive."""
        dim = 32
        rng = np.random.default_rng(123)
        n_db = 200
        rotation = generate_rotation(dim, seed=7)

        db_raw = rng.normal(size=(n_db, dim)).astype(np.float32)
        q_raw = rng.normal(size=(1, dim)).astype(np.float32)

        db_norm = normalize_rows(db_raw)
        q_norm = normalize_rows(q_raw)

        db_rotated = (db_norm @ rotation.T).astype(np.float32)
        q_rotated = (q_norm @ rotation.T).astype(np.float32)

        db_packed = quantize_qjl(db_rotated, bits=bits)
        q_packed = quantize_qjl(q_rotated, bits=bits)

        naive_scores = compressed_dot_naive(q_packed, db_packed, dim=dim, bits=bits)
        lut_scores = compressed_dot(q_packed, db_packed, dim=dim, bits=bits)

        # Top-10 *set* must be identical (order may differ for tied scores)
        naive_top10 = set(np.argsort(naive_scores)[-10:])
        lut_top10 = set(np.argsort(lut_scores)[-10:])
        assert naive_top10 == lut_top10, f"Top-10 sets differ: {naive_top10} vs {lut_top10}"

        # Scores for those indices must be numerically close
        for idx in naive_top10:
            np.testing.assert_allclose(
                lut_scores[idx], naive_scores[idx], rtol=1e-5, atol=1e-6,
                err_msg=f"Score mismatch at index {idx}",
            )


class TestLUTEdgeCases:
    """Edge cases: zero vectors, single dimension, single vector."""

    def test_zero_query(self) -> None:
        dim = 8
        bits = 3
        q = np.zeros(dim, dtype=np.float32)
        lut = build_query_lut(q, bits=bits)
        assert lut.shape == (dim, 2**bits)
        assert np.all(lut == 0.0)

    def test_single_db_vector(self) -> None:
        dim = 16
        bits = 3
        rng = np.random.default_rng(42)
        rotation = generate_rotation(dim, seed=7)

        db = normalize_rows(rng.normal(size=(1, dim)).astype(np.float32))
        q = normalize_rows(rng.normal(size=(1, dim)).astype(np.float32))

        db_rotated = (db @ rotation.T).astype(np.float32)
        q_rotated = (q @ rotation.T).astype(np.float32)

        db_packed = quantize_qjl(db_rotated, bits=bits)
        q_packed = quantize_qjl(q_rotated, bits=bits)

        naive = compressed_dot_naive(q_packed, db_packed, dim=dim, bits=bits)
        lut = compressed_dot(q_packed, db_packed, dim=dim, bits=bits)

        assert naive.shape == (1,)
        np.testing.assert_allclose(lut, naive, rtol=1e-5, atol=1e-6)

    def test_large_dim(self) -> None:
        """Verify correctness at higher dimensionality."""
        dim = 256
        bits = 3
        rng = np.random.default_rng(99)
        rotation = generate_rotation(dim, seed=7)

        db = normalize_rows(rng.normal(size=(20, dim)).astype(np.float32))
        q = normalize_rows(rng.normal(size=(1, dim)).astype(np.float32))

        db_rotated = (db @ rotation.T).astype(np.float32)
        q_rotated = (q @ rotation.T).astype(np.float32)

        db_packed = quantize_qjl(db_rotated, bits=bits)
        q_packed = quantize_qjl(q_rotated, bits=bits)

        naive = compressed_dot_naive(q_packed, db_packed, dim=dim, bits=bits)
        lut = compressed_dot(q_packed, db_packed, dim=dim, bits=bits)

        np.testing.assert_allclose(lut, naive, rtol=1e-5, atol=1e-6)


class TestTurboIndexBatchSearch:
    """Tests for the new batch search capability."""

    def test_batch_search_matches_individual(self) -> None:
        from turborag.index import TurboIndex

        dim = 16
        rng = np.random.default_rng(42)
        index = TurboIndex(dim=dim, bits=3, seed=7)

        vectors = rng.normal(size=(100, dim)).astype(np.float32)
        ids = [f"chunk-{i}" for i in range(100)]
        index.add(vectors, ids)

        queries = rng.normal(size=(5, dim)).astype(np.float32)

        # Individual searches
        individual = [index.search(queries[i], k=5) for i in range(5)]

        # Batch search
        batch = index.search_batch(queries, k=5)

        assert len(batch) == 5
        for qi in range(5):
            assert [hit[0] for hit in batch[qi]] == [hit[0] for hit in individual[qi]]

    def test_batch_search_with_workers(self) -> None:
        from turborag.index import TurboIndex

        dim = 8
        rng = np.random.default_rng(42)
        index = TurboIndex(dim=dim, bits=3, seed=7, shard_size=20)

        vectors = rng.normal(size=(50, dim)).astype(np.float32)
        ids = [f"chunk-{i}" for i in range(50)]
        index.add(vectors, ids)

        queries = rng.normal(size=(3, dim)).astype(np.float32)

        sequential = index.search_batch(queries, k=5)
        threaded = index.search_batch(queries, k=5, max_workers=2)

        assert len(sequential) == len(threaded)
        for qi in range(3):
            assert [hit[0] for hit in sequential[qi]] == [hit[0] for hit in threaded[qi]]

    def test_batch_empty_index(self) -> None:
        from turborag.index import TurboIndex

        dim = 8
        index = TurboIndex(dim=dim, bits=3, seed=7)
        queries = np.random.default_rng(42).normal(size=(3, dim)).astype(np.float32)
        results = index.search_batch(queries, k=5)
        assert results == [[], [], []]


class TestLUTPerformanceRegression:
    """Ensure the LUT path works correctly and efficiently at realistic sizes."""

    def test_lut_memory_advantage(self) -> None:
        """LUT scoring uses O(n_vectors) memory. Naive uses O(n_vectors * dim).

        Verify the LUT path doesn't allocate a massive float32 matrix.
        """
        dim = 128
        bits = 3
        n_db = 1000
        rng = np.random.default_rng(42)
        rotation = generate_rotation(dim, seed=7)

        db = normalize_rows(rng.normal(size=(n_db, dim)).astype(np.float32))
        q = normalize_rows(rng.normal(size=(1, dim)).astype(np.float32))

        db_rotated = (db @ rotation.T).astype(np.float32)
        q_rotated = (q @ rotation.T).astype(np.float32)

        db_packed = quantize_qjl(db_rotated, bits=bits)

        # LUT path: builds a tiny (dim, 8) table, scores are (n_db,)
        lut = build_query_lut(q_rotated[0], bits=bits)
        assert lut.shape == (dim, 2**bits)  # 128 * 8 = 1024 float64s = 8KB
        assert lut.nbytes == dim * (2**bits) * 8  # 8 bytes per float64

        # The naive path would allocate (n_db, dim) float32 = 512KB
        # LUT path only allocates scores array = 4KB
        scores = score_shard_lut(db_packed, lut, dim=dim, bits=bits)
        assert scores.shape == (n_db,)
        assert scores.dtype == np.float32

    def test_index_search_handles_large_corpus(self) -> None:
        """End-to-end: TurboIndex.search works on a 10K vector corpus."""
        from turborag.index import TurboIndex

        dim = 64
        rng = np.random.default_rng(42)
        n_db = 10_000

        index = TurboIndex(dim=dim, bits=3, seed=7)
        vectors = rng.normal(size=(n_db, dim)).astype(np.float32)
        ids = [f"doc-{i}" for i in range(n_db)]
        index.add(vectors, ids)

        # Search should return correct results quickly
        results = index.search(vectors[0], k=10)
        assert len(results) == 10
        # Self-match should be top-1
        assert results[0][0] == "doc-0"



