"""Tests for metadata filter evaluation."""
from __future__ import annotations

import numpy as np
import pytest

from turborag.filters import match, match_mask, validate_filters


class TestMatch:
    def test_equality_implicit(self) -> None:
        assert match({"color": "red"}, {"color": "red"})
        assert not match({"color": "blue"}, {"color": "red"})

    def test_equality_explicit(self) -> None:
        assert match({"x": 5}, {"x": {"$eq": 5}})
        assert not match({"x": 5}, {"x": {"$eq": 6}})

    def test_ne(self) -> None:
        assert match({"x": 5}, {"x": {"$ne": 6}})
        assert not match({"x": 5}, {"x": {"$ne": 5}})

    def test_gt(self) -> None:
        assert match({"score": 90}, {"score": {"$gt": 80}})
        assert not match({"score": 80}, {"score": {"$gt": 80}})

    def test_gte(self) -> None:
        assert match({"score": 80}, {"score": {"$gte": 80}})
        assert not match({"score": 79}, {"score": {"$gte": 80}})

    def test_lt(self) -> None:
        assert match({"price": 10}, {"price": {"$lt": 20}})
        assert not match({"price": 20}, {"price": {"$lt": 20}})

    def test_lte(self) -> None:
        assert match({"price": 20}, {"price": {"$lte": 20}})
        assert not match({"price": 21}, {"price": {"$lte": 20}})

    def test_in(self) -> None:
        assert match({"status": "active"}, {"status": {"$in": ["active", "pending"]}})
        assert not match({"status": "deleted"}, {"status": {"$in": ["active", "pending"]}})

    def test_nin(self) -> None:
        assert match({"status": "active"}, {"status": {"$nin": ["deleted", "banned"]}})
        assert not match({"status": "deleted"}, {"status": {"$nin": ["deleted", "banned"]}})

    def test_exists_true(self) -> None:
        assert match({"a": 1}, {"a": {"$exists": True}})
        assert not match({}, {"a": {"$exists": True}})

    def test_exists_false(self) -> None:
        assert match({}, {"a": {"$exists": False}})
        assert not match({"a": 1}, {"a": {"$exists": False}})

    def test_compound_and(self) -> None:
        meta = {"category": "finance", "year": 2024, "score": 95}
        assert match(meta, {"category": "finance", "year": {"$gte": 2024}})
        assert not match(meta, {"category": "finance", "year": {"$gt": 2024}})

    def test_range(self) -> None:
        assert match({"price": 50}, {"price": {"$gte": 10, "$lt": 100}})
        assert not match({"price": 100}, {"price": {"$gte": 10, "$lt": 100}})

    def test_missing_field_returns_false(self) -> None:
        assert not match({}, {"x": 5})
        assert not match({"y": 5}, {"x": 5})

    def test_string_comparison(self) -> None:
        assert match({"date": "2024-06"}, {"date": {"$gt": "2024-01"}})
        assert not match({"date": "2024-01"}, {"date": {"$gt": "2024-06"}})

    def test_incomparable_types_return_false(self) -> None:
        assert not match({"x": "hello"}, {"x": {"$gt": 5}})

    def test_none_value(self) -> None:
        assert match({"x": None}, {"x": None})
        assert not match({"x": None}, {"x": 5})

    def test_bool_value(self) -> None:
        assert match({"active": True}, {"active": True})
        assert not match({"active": False}, {"active": True})


class TestMatchMask:
    def test_basic_mask(self) -> None:
        metadata = [
            {"category": "a"},
            {"category": "b"},
            {"category": "a"},
            None,
            {"category": "a"},
        ]
        mask = match_mask(metadata, {"category": "a"})
        expected = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_empty_list(self) -> None:
        mask = match_mask([], {"x": 1})
        assert len(mask) == 0

    def test_all_none(self) -> None:
        mask = match_mask([None, None], {"x": 1})
        np.testing.assert_array_equal(mask, [False, False])

    def test_all_match(self) -> None:
        metadata = [{"x": 1}, {"x": 1}, {"x": 1}]
        mask = match_mask(metadata, {"x": 1})
        np.testing.assert_array_equal(mask, [True, True, True])

    def test_complex_filter(self) -> None:
        metadata = [
            {"category": "finance", "year": 2024},
            {"category": "tech", "year": 2024},
            {"category": "finance", "year": 2023},
            {"category": "finance", "year": 2025},
        ]
        mask = match_mask(metadata, {"category": "finance", "year": {"$gte": 2024}})
        np.testing.assert_array_equal(mask, [True, False, False, True])


class TestValidateFilters:
    def test_valid_operators(self) -> None:
        validate_filters({"x": {"$gt": 5, "$lt": 10}})
        validate_filters({"x": {"$in": [1, 2]}})
        validate_filters({"x": {"$exists": True}})

    def test_invalid_operator(self) -> None:
        with pytest.raises(ValueError, match="Unknown filter operator"):
            validate_filters({"x": {"$regex": "abc"}})

    def test_bare_value_is_valid(self) -> None:
        validate_filters({"x": 5})
        validate_filters({"x": "hello"})


class TestFilteredSearch:
    """Integration test: filtered search through TurboIndex."""

    def test_search_with_filters(self) -> None:
        from turborag import TurboIndex

        rng = np.random.default_rng(42)
        n, dim = 1000, 32
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        ids = [f"doc-{i}" for i in range(n)]
        metadata = [{"category": "a" if i % 3 == 0 else "b", "score": i} for i in range(n)]

        index = TurboIndex(dim=dim, bits=3, seed=7)
        index.add(vectors, ids, metadata=metadata)

        query = rng.standard_normal(dim).astype(np.float32)

        # Search with category filter
        results = index.search(query, k=5, filters={"category": "a"})
        assert len(results) == 5
        for cid, _ in results:
            idx = int(cid.split("-")[1])
            assert idx % 3 == 0, f"Expected category 'a' (idx%3==0), got idx={idx}"

    def test_search_with_range_filter(self) -> None:
        from turborag import TurboIndex

        rng = np.random.default_rng(42)
        n, dim = 500, 16
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        ids = [f"doc-{i}" for i in range(n)]
        metadata = [{"year": 2020 + (i % 5)} for i in range(n)]

        index = TurboIndex(dim=dim, bits=3, seed=7)
        index.add(vectors, ids, metadata=metadata)

        query = rng.standard_normal(dim).astype(np.float32)
        results = index.search(query, k=10, filters={"year": {"$gte": 2023}})
        assert len(results) == 10
        for cid, _ in results:
            idx = int(cid.split("-")[1])
            year = 2020 + (idx % 5)
            assert year >= 2023

    def test_search_no_matches(self) -> None:
        from turborag import TurboIndex

        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, 16)).astype(np.float32)
        ids = [f"doc-{i}" for i in range(100)]
        metadata = [{"x": 1}] * 100

        index = TurboIndex(dim=16, bits=3, seed=7)
        index.add(vectors, ids, metadata=metadata)

        results = index.search(rng.standard_normal(16).astype(np.float32), k=5, filters={"x": 999})
        assert results == []

    def test_search_without_metadata_ignores_filter(self) -> None:
        from turborag import TurboIndex

        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, 16)).astype(np.float32)
        ids = [f"doc-{i}" for i in range(100)]

        index = TurboIndex(dim=16, bits=3, seed=7)
        index.add(vectors, ids)  # no metadata

        results = index.search(rng.standard_normal(16).astype(np.float32), k=5, filters={"x": 1})
        assert results == []

    def test_save_load_preserves_metadata(self) -> None:
        import tempfile
        from turborag import TurboIndex

        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, 16)).astype(np.float32)
        ids = [f"doc-{i}" for i in range(50)]
        metadata = [{"tag": "a" if i < 25 else "b"} for i in range(50)]

        index = TurboIndex(dim=16, bits=3, seed=7)
        index.add(vectors, ids, metadata=metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            index.save(tmpdir)
            loaded = TurboIndex.open(tmpdir)

            query = rng.standard_normal(16).astype(np.float32)
            results = loaded.search(query, k=5, filters={"tag": "a"})
            assert len(results) > 0
            for cid, _ in results:
                idx = int(cid.split("-")[1])
                assert idx < 25

    def test_delete_removes_metadata(self) -> None:
        from turborag import TurboIndex

        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, 16)).astype(np.float32)
        ids = [f"doc-{i}" for i in range(100)]
        metadata = [{"category": "keep" if i >= 50 else "delete"} for i in range(100)]

        index = TurboIndex(dim=16, bits=3, seed=7)
        index.add(vectors, ids, metadata=metadata)

        # Delete first 50
        index.delete([f"doc-{i}" for i in range(50)])

        # Search for "delete" category should return nothing
        query = rng.standard_normal(16).astype(np.float32)
        results = index.search(query, k=5, filters={"category": "delete"})
        assert results == []

    def test_batch_search_with_filters(self) -> None:
        from turborag import TurboIndex

        rng = np.random.default_rng(42)
        n, dim = 500, 16
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        ids = [f"doc-{i}" for i in range(n)]
        metadata = [{"group": i % 3} for i in range(n)]

        index = TurboIndex(dim=dim, bits=3, seed=7)
        index.add(vectors, ids, metadata=metadata)

        queries = rng.standard_normal((3, dim)).astype(np.float32)
        results = index.search_batch(queries, k=5, filters={"group": 0})
        assert len(results) == 3
        for batch in results:
            assert len(batch) == 5
            for cid, _ in batch:
                idx = int(cid.split("-")[1])
                assert idx % 3 == 0
