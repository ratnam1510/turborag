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
    topk_shard_lut,
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

    def test_batch_search_uses_native_weighted_batch_scorer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from turborag import _cscore_wrapper as cscore_wrapper
        from turborag.index import TurboIndex

        dim = 16
        rng = np.random.default_rng(123)
        index = TurboIndex(dim=dim, bits=3, seed=7)

        vectors = rng.normal(size=(256, dim)).astype(np.float32)
        ids = [f"chunk-{i}" for i in range(256)]
        index.add(vectors, ids)

        queries = rng.normal(size=(4, dim)).astype(np.float32)
        individual = [index.search(queries[i], k=5) for i in range(len(queries))]

        real_batch = cscore_wrapper.score_3bit_weighted_batch_topk_c
        batch_calls = 0

        def counted_batch(*args, **kwargs):
            nonlocal batch_calls
            batch_calls += 1
            return real_batch(*args, **kwargs)

        def fail_single(*args, **kwargs):
            raise AssertionError("search_batch should use the native weighted batch scorer")

        monkeypatch.setattr(cscore_wrapper, "score_3bit_weighted_batch_topk_c", counted_batch)
        monkeypatch.setattr(cscore_wrapper, "score_3bit_weighted_topk_c", fail_single)

        batch = index.search_batch(queries, k=5)

        assert batch_calls == 1
        assert len(batch) == len(individual)
        for batch_hits, individual_hits in zip(batch, individual, strict=False):
            assert [hit[0] for hit in batch_hits] == [hit[0] for hit in individual_hits]


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


class TestTopKScoring:
    def test_auto_exact_threads_caps_at_eight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from turborag._cscore_wrapper import _resolve_exact_threads

        monkeypatch.delenv("TURBORAG_EXACT_THREADS", raising=False)
        monkeypatch.setattr("os.cpu_count", lambda: 10)

        assert _resolve_exact_threads(None) == 8

    def test_topk_matches_full_scores_for_3bit(self) -> None:
        dim = 384
        bits = 3
        rng = np.random.default_rng(42)
        rotation = generate_rotation(dim, seed=7)

        db = normalize_rows(rng.normal(size=(500, dim)).astype(np.float32))
        q = normalize_rows(rng.normal(size=(1, dim)).astype(np.float32))

        db_rotated = (db @ rotation.T).astype(np.float32)
        q_rotated = (q @ rotation.T).astype(np.float32)

        db_packed = quantize_qjl(db_rotated, bits=bits)
        lut = build_query_lut(q_rotated[0], bits=bits)

        scores = score_shard_lut(db_packed, lut, dim=dim, bits=bits)
        expected_idx = np.argpartition(scores, -10)[-10:]
        expected_idx = expected_idx[np.argsort(scores[expected_idx])[::-1]]

        top_idx, top_scores = topk_shard_lut(db_packed, lut, dim=dim, bits=bits, k=10)

        np.testing.assert_array_equal(top_idx, expected_idx.astype(np.int32))
        np.testing.assert_allclose(top_scores, scores[expected_idx], rtol=1e-6, atol=1e-6)

    def test_topk_matches_full_scores_for_2bit_and_4bit(self) -> None:
        rng = np.random.default_rng(123)

        for bits in (2, 4):
            dim = 64
            rotation = generate_rotation(dim, seed=7)
            db = normalize_rows(rng.normal(size=(120, dim)).astype(np.float32))
            q = normalize_rows(rng.normal(size=(1, dim)).astype(np.float32))

            db_rotated = (db @ rotation.T).astype(np.float32)
            q_rotated = (q @ rotation.T).astype(np.float32)

            db_packed = quantize_qjl(db_rotated, bits=bits)
            lut = build_query_lut(q_rotated[0], bits=bits)

            scores = score_shard_lut(db_packed, lut, dim=dim, bits=bits)
            expected_idx = np.argpartition(scores, -7)[-7:]
            expected_idx = expected_idx[np.argsort(scores[expected_idx])[::-1]]

            top_idx, top_scores = topk_shard_lut(db_packed, lut, dim=dim, bits=bits, k=7)

            np.testing.assert_array_equal(top_idx, expected_idx.astype(np.int32))
            np.testing.assert_allclose(top_scores, scores[expected_idx], rtol=1e-6, atol=1e-6)

    def test_threaded_native_topk_matches_single_thread(self) -> None:
        from turborag._cscore_wrapper import build_fused_lut_c, score_fused_3bit_topk_c

        dim = 384
        bits = 3
        rng = np.random.default_rng(99)
        rotation = generate_rotation(dim, seed=7)

        db = normalize_rows(rng.normal(size=(4000, dim)).astype(np.float32))
        q = normalize_rows(rng.normal(size=(1, dim)).astype(np.float32))

        db_rotated = (db @ rotation.T).astype(np.float32)
        q_rotated = (q @ rotation.T).astype(np.float32)

        db_packed = quantize_qjl(db_rotated, bits=bits)
        lut = build_query_lut(q_rotated[0], bits=bits)
        fused3 = build_fused_lut_c(lut, dim, bits)
        assert fused3 is not None

        single = score_fused_3bit_topk_c(db_packed, fused3, lut, dim, 10, num_threads=1)
        threaded = score_fused_3bit_topk_c(db_packed, fused3, lut, dim, 10, num_threads=4)

        assert single is not None
        assert threaded is not None

        np.testing.assert_array_equal(threaded[0], single[0])
        np.testing.assert_allclose(threaded[1], single[1], rtol=1e-6, atol=1e-6)

    def test_float32_native_topk_matches_double_path(self) -> None:
        from turborag._cscore_wrapper import (
            build_fused_lut_c,
            build_fused_lut_6bit_f32_c,
            build_fused_lut_f32_c,
            score_fused_3bit_topk_c,
            score_fused_3bit_topk_6bit_f32_c,
            score_fused_3bit_topk_f32_c,
        )

        dim = 384
        bits = 3
        rng = np.random.default_rng(1234)
        rotation = generate_rotation(dim, seed=7)

        db = normalize_rows(rng.normal(size=(5000, dim)).astype(np.float32))
        q = normalize_rows(rng.normal(size=(1, dim)).astype(np.float32))

        db_rotated = (db @ rotation.T).astype(np.float32)
        q_rotated = (q @ rotation.T).astype(np.float32)
        db_packed = quantize_qjl(db_rotated, bits=bits)
        lut = build_query_lut(q_rotated[0], bits=bits)
        fused3 = build_fused_lut_c(lut, dim, bits)
        fused6_f32 = build_fused_lut_6bit_f32_c(lut.astype(np.float32), dim)
        fused3_f32 = build_fused_lut_f32_c(lut.astype(np.float32), dim, bits)
        assert fused3 is not None
        assert fused6_f32 is not None
        assert fused3_f32 is not None

        exact = score_fused_3bit_topk_c(db_packed, fused3, lut, dim, 20, num_threads=1)
        fused6 = score_fused_3bit_topk_6bit_f32_c(db_packed, fused6_f32, dim, 20)
        approx = score_fused_3bit_topk_f32_c(db_packed, fused3_f32, lut.astype(np.float32), dim, 20)

        assert exact is not None
        assert fused6 is not None
        assert approx is not None
        np.testing.assert_array_equal(fused6[0], exact[0])
        np.testing.assert_allclose(fused6[1], exact[1], rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(approx[0], exact[0])
        np.testing.assert_allclose(approx[1], exact[1], rtol=1e-4, atol=1e-4)

    def test_weighted_scorer_matches_lut_exact(self) -> None:
        """Weighted integer scorer must produce identical top-k as LUT scorer."""
        from turborag._cscore_wrapper import (
            build_fused_lut_6bit_f32_c,
            score_3bit_weighted_topk_c,
            score_fused_3bit_topk_6bit_f32_c,
        )
        from turborag.fast_kernels import build_query_weights_f32

        dim = 384
        bits = 3
        rng = np.random.default_rng(99)
        rotation = generate_rotation(dim, seed=7)

        db = normalize_rows(rng.normal(size=(10000, dim)).astype(np.float32))
        db_rotated = (db @ rotation.T).astype(np.float32)
        db_packed = quantize_qjl(db_rotated, bits=bits)

        for qi in range(20):
            q = normalize_rows(rng.normal(size=(1, dim)).astype(np.float32))
            q_rotated = (q @ rotation.T).astype(np.float32)[0]

            # LUT path (6-bit fused)
            lut_f32 = build_query_lut(q_rotated, bits=bits).astype(np.float32)
            fused6 = build_fused_lut_6bit_f32_c(lut_f32, dim)
            assert fused6 is not None
            lut_result = score_fused_3bit_topk_6bit_f32_c(db_packed, fused6, dim, 10)
            assert lut_result is not None

            # Weighted path
            weights, bias = build_query_weights_f32(q_rotated, value_range=1.0)
            weighted_result = score_3bit_weighted_topk_c(db_packed, weights, bias, dim=dim, k=10)
            assert weighted_result is not None

            # Same top-k IDs
            assert set(int(i) for i in lut_result[0]) == set(int(i) for i in weighted_result[0]), \
                f"Query {qi}: weighted scorer returned different top-k than LUT"

    def test_weighted_scorer_threaded_matches_single(self) -> None:
        """Threaded weighted scorer must match single-threaded."""
        from turborag._cscore_wrapper import score_3bit_weighted_topk_c
        from turborag.fast_kernels import build_query_weights_f32

        dim = 384
        rng = np.random.default_rng(42)
        rotation = generate_rotation(dim, seed=7)

        db = normalize_rows(rng.normal(size=(50000, dim)).astype(np.float32))
        db_rotated = (db @ rotation.T).astype(np.float32)
        db_packed = quantize_qjl(db_rotated, bits=3)

        q = normalize_rows(rng.normal(size=(1, dim)).astype(np.float32))
        q_rotated = (q @ rotation.T).astype(np.float32)[0]
        weights, bias = build_query_weights_f32(q_rotated, value_range=1.0)

        single = score_3bit_weighted_topk_c(db_packed, weights, bias, dim=dim, k=10, num_threads=1)
        threaded = score_3bit_weighted_topk_c(db_packed, weights, bias, dim=dim, k=10, num_threads=8)

        assert single is not None
        assert threaded is not None
        np.testing.assert_array_equal(threaded[0], single[0])
        np.testing.assert_allclose(threaded[1], single[1], rtol=1e-5, atol=1e-5)

    def test_weighted_batch_scorer_matches_single_query_path(self) -> None:
        from turborag._cscore_wrapper import (
            score_3bit_weighted_batch_topk_c,
            score_3bit_weighted_topk_c,
        )
        from turborag.fast_kernels import build_query_weights_f32

        dim = 384
        rng = np.random.default_rng(1234)
        rotation = generate_rotation(dim, seed=7)

        db = normalize_rows(rng.normal(size=(12000, dim)).astype(np.float32))
        db_rotated = (db @ rotation.T).astype(np.float32)
        db_packed = quantize_qjl(db_rotated, bits=3)

        queries = normalize_rows(rng.normal(size=(6, dim)).astype(np.float32))
        rotated_queries = (queries @ rotation.T).astype(np.float32)
        weights_batch = []
        biases = []
        for rotated in rotated_queries:
            weights, bias = build_query_weights_f32(rotated, value_range=1.0)
            weights_batch.append(weights)
            biases.append(bias)

        batch = score_3bit_weighted_batch_topk_c(
            db_packed,
            np.ascontiguousarray(weights_batch, dtype=np.float32),
            np.ascontiguousarray(biases, dtype=np.float32),
            dim=dim,
            k=10,
            num_threads=1,
        )
        assert batch is not None

        for qi, rotated in enumerate(rotated_queries):
            weights, bias = build_query_weights_f32(rotated, value_range=1.0)
            single = score_3bit_weighted_topk_c(
                db_packed,
                weights,
                bias,
                dim=dim,
                k=10,
                num_threads=1,
            )
            assert single is not None
            np.testing.assert_array_equal(batch[qi][0], single[0])
            np.testing.assert_allclose(batch[qi][1], single[1], rtol=1e-5, atol=1e-5)
