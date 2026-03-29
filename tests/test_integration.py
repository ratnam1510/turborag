"""End-to-end integration test: ingest → graph extraction → hybrid retrieval.

This exercises the full pipeline using mocked LLM components so no external
services are required.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

networkx = pytest.importorskip("networkx")

from turborag.graph import GraphBuilder
from turborag.hybrid import HybridRetriever
from turborag.index import TurboIndex
from turborag.ingest import build_sidecar_index, load_dataset, write_records_snapshot, load_records_snapshot
from turborag.types import ChunkRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8


class FakeLLM:
    """Deterministic LLM stub that returns canned extraction payloads
    based on keywords found in the prompt text."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)

        # Chunk about "quantum computing at ACME"
        if "quantum" in prompt.lower():
            return json.dumps({
                "entities": [
                    {"name": "Quantum Computing", "type": "CONCEPT", "description": "A computing paradigm"},
                    {"name": "ACME Corp", "type": "ORG", "description": "A technology company"},
                ],
                "relationships": [
                    {"source": "ACME Corp", "target": "Quantum Computing", "relation": "researches", "weight": 0.9},
                ],
            })

        # Chunk about "neural networks"
        if "neural" in prompt.lower():
            return json.dumps({
                "entities": [
                    {"name": "Neural Networks", "type": "CONCEPT", "description": "A machine learning technique"},
                    {"name": "ACME Corp", "type": "ORG", "description": "A technology company"},
                ],
                "relationships": [
                    {"source": "ACME Corp", "target": "Neural Networks", "relation": "develops", "weight": 0.8},
                ],
            })

        # Chunk about "climate research"
        if "climate" in prompt.lower():
            return json.dumps({
                "entities": [
                    {"name": "Climate Research", "type": "CONCEPT", "description": "Study of climate"},
                    {"name": "GreenTech", "type": "ORG", "description": "An environmental org"},
                ],
                "relationships": [
                    {"source": "GreenTech", "target": "Climate Research", "relation": "funds", "weight": 0.7},
                ],
            })

        # Summary prompts
        if "communit" in prompt.lower() or "summarise" in prompt.lower():
            return "A community of technology and research entities."

        return json.dumps({"entities": [], "relationships": []})


class SimpleEmbedder:
    """Produces deterministic vectors from chunk text using a hash-based scheme."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(hash(text) % (2**31))
        return rng.normal(size=self.dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Full pipeline: ingest data → build graph → build index → hybrid query."""

    def _build_chunks_and_embeddings(self, embedder: SimpleEmbedder):
        """Create a small corpus of chunks with embeddings."""
        texts = [
            "ACME Corp has invested heavily in quantum computing research over the past decade.",
            "Neural networks developed at ACME Corp have achieved state-of-the-art results.",
            "GreenTech is funding climate research to combat global warming.",
            "The annual report covers financial performance and strategic direction.",
        ]
        chunks: dict[str, ChunkRecord] = {}
        vectors = []
        for i, text in enumerate(texts):
            cid = f"chunk-{i}"
            chunks[cid] = ChunkRecord(chunk_id=cid, text=text, source_doc="corpus.pdf", page_num=i + 1)
            vectors.append(embedder.embed(text))
        matrix = np.vstack(vectors)
        return chunks, matrix

    def test_full_pipeline(self, tmp_path):
        """Ingest → graph build → index → hybrid retrieval end-to-end."""
        embedder = SimpleEmbedder()
        chunks, matrix = self._build_chunks_and_embeddings(embedder)
        ids = list(chunks.keys())

        # --- Step 1: Build the compressed index ---
        index = TurboIndex(dim=DIM, bits=4, seed=42)
        index.add(matrix, ids)
        index.save(str(tmp_path / "index"))

        # Verify persistence round-trip
        loaded_index = TurboIndex.open(str(tmp_path / "index"))
        assert len(loaded_index) == len(ids)

        # --- Step 2: Build the entity graph ---
        llm = FakeLLM()
        builder = GraphBuilder(llm_client=llm, cache_dir=str(tmp_path / "cache"))
        for cid, chunk in chunks.items():
            builder.add_chunk(cid, chunk.text)

        graph = builder.build()
        assert graph.number_of_nodes() > 0
        summaries = builder.summarise_communities()
        builder.close()

        # Verify expected entities exist
        assert "ACME Corp" in graph.nodes
        assert "Quantum Computing" in graph.nodes
        assert "Neural Networks" in graph.nodes

        # --- Step 3: Hybrid retrieval ---
        retriever = HybridRetriever(
            index=loaded_index,
            graph=graph,
            community_summaries=summaries,
            embedder=embedder,
            chunks=chunks,
        )

        # Dense-only query
        dense_results = retriever.query("quantum computing research", k=4, mode="dense")
        assert len(dense_results) >= 1
        assert all(r.chunk_id in chunks for r in dense_results)

        # Graph-only query – should find ACME Corp via token-boundary match
        graph_results = retriever.query("Tell me about ACME Corp", k=4, mode="graph")
        graph_ids = {r.chunk_id for r in graph_results}
        # ACME Corp is linked to chunk-0 and chunk-1
        assert "chunk-0" in graph_ids or "chunk-1" in graph_ids

        # Hybrid query – merges both signals
        hybrid_results = retriever.query("ACME Corp quantum computing", k=4, mode="hybrid")
        assert len(hybrid_results) >= 1
        # The quantum chunk should rank high in hybrid
        hybrid_ids = [r.chunk_id for r in hybrid_results]
        assert "chunk-0" in hybrid_ids

        # Verify explanations
        for result in hybrid_results:
            explanation = retriever.explain(result)
            assert isinstance(explanation, str) and len(explanation) > 0

    def test_graph_entity_detection_respects_token_boundaries(self, tmp_path):
        """Ensure the improved entity detection does not false-positive on partial matches."""
        embedder = SimpleEmbedder()
        chunks, matrix = self._build_chunks_and_embeddings(embedder)
        ids = list(chunks.keys())

        index = TurboIndex(dim=DIM, bits=4, seed=42)
        index.add(matrix, ids)

        llm = FakeLLM()
        builder = GraphBuilder(llm_client=llm)
        for cid, chunk in chunks.items():
            builder.add_chunk(cid, chunk.text)
        graph = builder.build()

        retriever = HybridRetriever(
            index=index, graph=graph, community_summaries={},
            embedder=embedder, chunks=chunks,
        )

        # "ACMECorp" (no space) should NOT match the entity "ACME Corp" with the
        # new token-boundary matcher, since tokens are ["acme", "corp"] vs ["acmecorp"]
        results_no_match = retriever.query("ACMECorp does something", k=4, mode="graph")
        assert results_no_match == []

        # "ACME Corp" (with space) SHOULD match
        results_match = retriever.query("ACME Corp does something", k=4, mode="graph")
        assert len(results_match) > 0

    def test_ingest_and_sidecar_round_trip(self, tmp_path):
        """Verify the ingest helpers can build a sidecar and records survive round-trip."""
        dataset_path = tmp_path / "data.jsonl"
        rows = [
            {"chunk_id": "a", "text": "alpha text", "embedding": [1.0, 0.0, 0.0, 0.0]},
            {"chunk_id": "b", "text": "beta text", "embedding": [0.0, 1.0, 0.0, 0.0]},
            {"chunk_id": "c", "text": "gamma text", "embedding": [0.0, 0.0, 1.0, 0.0]},
        ]
        dataset_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        dataset = load_dataset(dataset_path)
        result = build_sidecar_index(dataset, tmp_path / "sidecar", bits=4)

        # Records snapshot round trip
        records = load_records_snapshot(result.records_path)
        assert set(records.keys()) == {"a", "b", "c"}
        assert records["a"].text == "alpha text"

        # Index search works
        loaded_index = TurboIndex.open(str(result.index_path))
        hits = loaded_index.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=2)
        assert hits[0][0] == "a"
