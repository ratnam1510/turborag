"""Tests for turborag.hybrid – HybridRetriever modes, embedder dispatch, edge cases."""
from __future__ import annotations

import json

import numpy as np
import pytest

networkx = pytest.importorskip("networkx")

from turborag.embeddings import Embedder
from turborag.hybrid import HybridRetriever
from turborag.index import TurboIndex
from turborag.types import ChunkRecord, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 6


def _make_chunks(n: int = 4) -> dict[str, ChunkRecord]:
    return {
        f"c{i}": ChunkRecord(chunk_id=f"c{i}", text=f"text for chunk {i}", source_doc="doc.pdf", page_num=i)
        for i in range(n)
    }


def _make_index(chunks: dict[str, ChunkRecord], rng: np.random.Generator | None = None) -> tuple[TurboIndex, np.ndarray]:
    """Build a small TurboIndex and return (index, vectors) for the given chunks."""
    if rng is None:
        rng = np.random.default_rng(42)
    ids = list(chunks)
    vectors = rng.normal(size=(len(ids), DIM)).astype(np.float32)
    index = TurboIndex(dim=DIM, bits=4, seed=7)
    index.add(vectors, ids)
    return index, vectors


def _make_graph(edges: list[tuple[str, str]], node_chunk_map: dict[str, list[str]] | None = None):
    """Build a simple networkx graph with optional chunk_ids on nodes."""
    import networkx as nx

    g = nx.Graph()
    node_chunk_map = node_chunk_map or {}
    all_nodes = set()
    for src, tgt in edges:
        all_nodes.update([src, tgt])
    for node in all_nodes:
        g.add_node(node, chunk_ids=json.dumps(node_chunk_map.get(node, [])))
    for src, tgt in edges:
        g.add_edge(src, tgt)
    return g


class SimpleEmbedder:
    """Minimal embedder satisfying the Embedder protocol via embed()."""

    def __init__(self, vectors: np.ndarray, chunks: dict[str, ChunkRecord]) -> None:
        self._lookup: dict[str, np.ndarray] = {}
        for i, cid in enumerate(chunks):
            self._lookup[chunks[cid].text] = vectors[i]

    def embed(self, text: str) -> np.ndarray:
        # Return first matching vector or zeros
        for key, vec in self._lookup.items():
            if key in text or text in key:
                return vec
        return np.zeros(DIM, dtype=np.float32)


class EmbedQueryOnlyEmbedder:
    """Embedder that only has embed_query (LangChain-style), not embed()."""

    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector


# ---------------------------------------------------------------------------
# Embedder dispatch
# ---------------------------------------------------------------------------

class TestEmbedderDispatch:

    def test_embed_method_is_preferred(self):
        """An embedder with .embed() should be called via that method."""
        chunks = _make_chunks(2)
        index, vectors = _make_index(chunks)
        graph = _make_graph([])
        embedder = SimpleEmbedder(vectors, chunks)

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=embedder, chunks=chunks,
        )
        results = retriever.query("text for chunk 0", k=2, mode="dense")
        assert len(results) > 0

    def test_embed_query_fallback(self):
        """An embedder with only embed_query() should still work."""
        chunks = _make_chunks(2)
        index, vectors = _make_index(chunks)
        graph = _make_graph([])
        embedder = EmbedQueryOnlyEmbedder(vectors[0])

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=embedder, chunks=chunks,
        )
        results = retriever.query("anything", k=2, mode="dense")
        assert len(results) > 0

    def test_callable_fallback(self):
        """A bare callable should work as embedder."""
        chunks = _make_chunks(2)
        index, vectors = _make_index(chunks)
        graph = _make_graph([])

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=lambda text: vectors[0],
            chunks=chunks,
        )
        results = retriever.query("anything", k=2, mode="dense")
        assert len(results) > 0

    def test_runtime_checkable_protocol(self):
        """SimpleEmbedder should satisfy the Embedder runtime protocol."""
        chunks = _make_chunks(1)
        _, vectors = _make_index(chunks)
        embedder = SimpleEmbedder(vectors, chunks)
        assert isinstance(embedder, Embedder)

    def test_embed_query_only_does_not_satisfy_protocol(self):
        """EmbedQueryOnlyEmbedder lacks embed(), so it shouldn't match."""
        embedder = EmbedQueryOnlyEmbedder(np.zeros(DIM, dtype=np.float32))
        assert not isinstance(embedder, Embedder)


# ---------------------------------------------------------------------------
# Query modes
# ---------------------------------------------------------------------------

class TestQueryModes:

    def test_dense_mode_returns_results(self):
        chunks = _make_chunks(4)
        index, vectors = _make_index(chunks)
        graph = _make_graph([])
        embedder = SimpleEmbedder(vectors, chunks)

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=embedder, chunks=chunks,
        )
        results = retriever.query("text for chunk 0", k=2, mode="dense")
        assert len(results) >= 1
        assert results[0].chunk_id == "c0"

    def test_graph_mode_returns_results(self):
        chunks = _make_chunks(4)
        index, vectors = _make_index(chunks)
        graph = _make_graph(
            [("Alpha", "Beta")],
            node_chunk_map={"Alpha": ["c0", "c1"], "Beta": ["c2"]},
        )
        embedder = SimpleEmbedder(vectors, chunks)

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=embedder, chunks=chunks,
        )
        # query text contains entity name "Alpha"
        results = retriever.query("Alpha stuff", k=10, mode="graph")
        result_ids = [r.chunk_id for r in results]
        assert "c0" in result_ids or "c1" in result_ids

    def test_hybrid_mode_merges_scores(self):
        chunks = _make_chunks(4)
        index, vectors = _make_index(chunks)
        graph = _make_graph(
            [("Alpha", "Beta")],
            node_chunk_map={"Alpha": ["c0"]},
        )
        embedder = SimpleEmbedder(vectors, chunks)

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=embedder, chunks=chunks,
        )
        # "Alpha" triggers graph path; dense also returns c0
        results = retriever.query("Alpha text for chunk 0", k=4, mode="hybrid")
        assert len(results) >= 1

    def test_invalid_mode_raises(self):
        chunks = _make_chunks(1)
        index, vectors = _make_index(chunks)
        graph = _make_graph([])
        embedder = SimpleEmbedder(vectors, chunks)

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=embedder, chunks=chunks,
        )
        with pytest.raises(ValueError, match="mode must be"):
            retriever.query("x", mode="invalid")


# ---------------------------------------------------------------------------
# Graph expansion
# ---------------------------------------------------------------------------

class TestGraphExpansion:

    def test_bfs_respects_depth(self):
        chunks = _make_chunks(4)
        index, vectors = _make_index(chunks)
        # A -> B -> C  (depth-limited)
        graph = _make_graph(
            [("A", "B"), ("B", "C")],
            node_chunk_map={"A": ["c0"], "B": ["c1"], "C": ["c2"]},
        )
        embedder = SimpleEmbedder(vectors, chunks)

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=embedder, chunks=chunks, graph_depth=1,
        )
        results = retriever.query("A query", k=10, mode="graph")
        result_ids = {r.chunk_id for r in results}
        # At depth 1: A and B reachable, C should not be
        assert "c0" in result_ids
        assert "c1" in result_ids
        assert "c2" not in result_ids

    def test_graph_mode_no_matching_entities_returns_empty(self):
        chunks = _make_chunks(2)
        index, vectors = _make_index(chunks)
        # Use entity names that will NOT appear as substrings of the query
        graph = _make_graph(
            [("Xylophone123", "Zeppelin456")],
            node_chunk_map={"Xylophone123": ["c0"]},
        )
        embedder = SimpleEmbedder(vectors, chunks)

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=embedder, chunks=chunks,
        )
        results = retriever.query("no match here", k=5, mode="graph")
        assert results == []

    def test_malformed_chunk_ids_on_node_skipped(self):
        """Nodes with corrupt chunk_ids JSON should not crash retrieval."""
        import networkx as nx

        chunks = _make_chunks(2)
        index, vectors = _make_index(chunks)
        g = nx.Graph()
        g.add_node("Alpha", chunk_ids="NOT VALID JSON")
        embedder = SimpleEmbedder(vectors, chunks)

        retriever = HybridRetriever(
            index=index, graph=g, community_summaries={},
            embedder=embedder, chunks=chunks,
        )
        results = retriever.query("Alpha query", k=5, mode="graph")
        assert results == []  # gracefully returns empty, no crash


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------

class TestExplain:

    def test_explain_with_explanation(self):
        result = RetrievalResult(chunk_id="c0", text="t", score=1.0, explanation="Custom note")
        chunks = _make_chunks(1)
        index, vectors = _make_index(chunks)
        graph = _make_graph([])
        embedder = SimpleEmbedder(vectors, chunks)

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=embedder, chunks=chunks,
        )
        assert retriever.explain(result) == "Custom note"

    def test_explain_with_graph_path(self):
        result = RetrievalResult(chunk_id="c0", text="t", score=1.0, graph_path=["A", "B"])
        chunks = _make_chunks(1)
        index, vectors = _make_index(chunks)
        graph = _make_graph([])
        embedder = SimpleEmbedder(vectors, chunks)

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=embedder, chunks=chunks,
        )
        assert "A -> B" in retriever.explain(result)

    def test_explain_fallback(self):
        result = RetrievalResult(chunk_id="c0", text="t", score=1.0)
        chunks = _make_chunks(1)
        index, vectors = _make_index(chunks)
        graph = _make_graph([])
        embedder = SimpleEmbedder(vectors, chunks)

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=embedder, chunks=chunks,
        )
        assert "dense" in retriever.explain(result).lower()


# ---------------------------------------------------------------------------
# Reranker integration
# ---------------------------------------------------------------------------

class TestReranker:

    def test_reranker_reorders_results(self):
        chunks = _make_chunks(3)
        index, vectors = _make_index(chunks)
        graph = _make_graph([])
        embedder = SimpleEmbedder(vectors, chunks)

        # Reranker always favours c2
        def reranker(query: str, records: list[ChunkRecord]) -> list[float]:
            return [10.0 if r.chunk_id == "c2" else 0.1 for r in records]

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=embedder, chunks=chunks, reranker=reranker,
        )
        results = retriever.query("text for chunk 0", k=3, mode="dense")
        assert results[0].chunk_id == "c2"
