"""Tests for index delete/update operations and exception hierarchy."""
from __future__ import annotations

import numpy as np
import pytest

from turborag.exceptions import DuplicateIDError, IDNotFoundError
from turborag.index import TurboIndex


class TestIndexDeleteUpdate:
    """Tests for delete() and update() operations."""

    def test_delete_single(self) -> None:
        dim = 8
        rng = np.random.default_rng(42)
        index = TurboIndex(dim=dim, bits=3, seed=7)
        vectors = rng.normal(size=(5, dim)).astype(np.float32)
        ids = ["a", "b", "c", "d", "e"]
        index.add(vectors, ids)

        assert len(index) == 5
        assert "c" in index

        removed = index.delete(["c"])
        assert removed == 1
        assert len(index) == 4
        assert "c" not in index

        # Remaining vectors still searchable
        results = index.search(vectors[0], k=5)
        result_ids = [r[0] for r in results]
        assert "c" not in result_ids
        assert "a" in result_ids

    def test_delete_multiple(self) -> None:
        dim = 8
        rng = np.random.default_rng(42)
        index = TurboIndex(dim=dim, bits=3, seed=7)
        vectors = rng.normal(size=(10, dim)).astype(np.float32)
        ids = [f"v-{i}" for i in range(10)]
        index.add(vectors, ids)

        removed = index.delete(["v-2", "v-5", "v-8"])
        assert removed == 3
        assert len(index) == 7

    def test_delete_nonexistent_returns_zero(self) -> None:
        dim = 8
        index = TurboIndex(dim=dim, bits=3, seed=7)
        rng = np.random.default_rng(42)
        index.add(rng.normal(size=(3, dim)).astype(np.float32), ["a", "b", "c"])

        removed = index.delete(["x", "y", "z"])
        assert removed == 0
        assert len(index) == 3

    def test_delete_allows_readd(self) -> None:
        dim = 8
        rng = np.random.default_rng(42)
        index = TurboIndex(dim=dim, bits=3, seed=7)
        vectors = rng.normal(size=(3, dim)).astype(np.float32)
        index.add(vectors, ["a", "b", "c"])

        index.delete(["b"])
        assert len(index) == 2

        new_vec = rng.normal(size=(1, dim)).astype(np.float32)
        index.add(new_vec, ["b"])  # should not raise
        assert len(index) == 3
        assert "b" in index

    def test_update_replaces_vector(self) -> None:
        dim = 8
        rng = np.random.default_rng(42)
        index = TurboIndex(dim=dim, bits=3, seed=7)
        vectors = rng.normal(size=(5, dim)).astype(np.float32)
        ids = ["a", "b", "c", "d", "e"]
        index.add(vectors, ids)

        # Search with original vector "a" — should find itself
        results_before = index.search(vectors[0], k=1)
        assert results_before[0][0] == "a"

        # Update "a" with a very different vector
        new_vector = rng.normal(size=(1, dim)).astype(np.float32) * 100
        index.update(new_vector, ["a"])
        assert len(index) == 5

    def test_update_nonexistent_raises(self) -> None:
        dim = 8
        index = TurboIndex(dim=dim, bits=3, seed=7)
        rng = np.random.default_rng(42)
        index.add(rng.normal(size=(3, dim)).astype(np.float32), ["a", "b", "c"])

        with pytest.raises(IDNotFoundError):
            index.update(
                rng.normal(size=(1, dim)).astype(np.float32),
                ["nonexistent"],
            )

    def test_repr(self) -> None:
        index = TurboIndex(dim=16, bits=3, seed=7)
        rng = np.random.default_rng(42)
        index.add(rng.normal(size=(5, 16)).astype(np.float32), [f"v-{i}" for i in range(5)])
        r = repr(index)
        assert "dim=16" in r
        assert "bits=3" in r
        assert "size=5" in r

    def test_contains(self) -> None:
        dim = 8
        index = TurboIndex(dim=dim, bits=3, seed=7)
        rng = np.random.default_rng(42)
        index.add(rng.normal(size=(2, dim)).astype(np.float32), ["x", "y"])

        assert "x" in index
        assert "y" in index
        assert "z" not in index


class TestDomainExceptions:
    """Verify the exception hierarchy works correctly."""

    def test_duplicate_id_error_is_index_error(self) -> None:
        from turborag.exceptions import IndexError as TurboIndexError
        err = DuplicateIDError("chunk-1")
        assert isinstance(err, TurboIndexError)
        assert err.chunk_id == "chunk-1"
        assert "chunk-1" in str(err)

    def test_id_not_found_error(self) -> None:
        err = IDNotFoundError("chunk-99")
        assert err.chunk_id == "chunk-99"
        assert "chunk-99" in str(err)

    def test_turborag_error_hierarchy(self) -> None:
        from turborag.exceptions import TurboRAGError, IngestError, ChunkingError
        err = ChunkingError("pdf broken")
        assert isinstance(err, IngestError)
        assert isinstance(err, TurboRAGError)
