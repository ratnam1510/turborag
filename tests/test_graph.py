"""Tests for turborag.graph – GraphBuilder hardening and core behaviour."""
from __future__ import annotations

import json

import pytest

networkx = pytest.importorskip("networkx")

from turborag.graph import ENTITY_PROMPT, GraphBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeLLM:
    """Deterministic LLM stub that returns canned entity-extraction JSON."""

    def __init__(self, responses: dict[str, str] | None = None, default: str | None = None) -> None:
        self._responses = responses or {}
        self._default = default
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        for key, value in self._responses.items():
            if key in prompt:
                return value
        if self._default is not None:
            return self._default
        return json.dumps({"entities": [], "relationships": []})


def _simple_payload(entities, relationships=None):
    return json.dumps({"entities": entities, "relationships": relationships or []})


CHUNK_A_RESPONSE = _simple_payload(
    [
        {"name": "Alice", "type": "PERSON", "description": "A researcher"},
        {"name": "ACME", "type": "ORG", "description": "A company"},
    ],
    [{"source": "Alice", "target": "ACME", "relation": "works_at", "weight": 0.9}],
)

CHUNK_B_RESPONSE = _simple_payload(
    [
        {"name": "Alice", "type": "PERSON", "description": ""},
        {"name": "WidgetX", "type": "PRODUCT", "description": "A product"},
    ],
    [{"source": "Alice", "target": "WidgetX", "relation": "invented", "weight": 0.7}],
)


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------

class TestGraphBuilderBasics:

    def test_add_chunk_creates_nodes_and_edges(self):
        llm = FakeLLM(default=CHUNK_A_RESPONSE)
        builder = GraphBuilder(llm_client=llm)
        builder.add_chunk("c1", "Alice works at ACME")

        assert "Alice" in builder.graph
        assert "ACME" in builder.graph
        assert builder.graph.has_edge("Alice", "ACME")
        assert json.loads(builder.graph.nodes["Alice"]["chunk_ids"]) == ["c1"]

    def test_multiple_chunks_merge_entity_chunk_ids(self):
        llm = FakeLLM(responses={"chunk-a": CHUNK_A_RESPONSE, "chunk-b": CHUNK_B_RESPONSE})
        builder = GraphBuilder(llm_client=llm)
        builder.add_chunk("c1", "chunk-a Alice")
        builder.add_chunk("c2", "chunk-b Alice again")

        chunk_ids = json.loads(builder.graph.nodes["Alice"]["chunk_ids"])
        assert sorted(chunk_ids) == ["c1", "c2"]

    def test_build_assigns_communities(self):
        llm = FakeLLM(default=CHUNK_A_RESPONSE)
        builder = GraphBuilder(llm_client=llm)
        builder.add_chunk("c1", "text")
        graph = builder.build()

        assert graph.number_of_nodes() > 0
        communities = builder.get_communities()
        assert len(communities) >= 1

    def test_build_on_empty_graph_is_safe(self):
        llm = FakeLLM()
        builder = GraphBuilder(llm_client=llm)
        graph = builder.build()
        assert graph.number_of_nodes() == 0
        assert builder.get_communities() == {}

    def test_summarise_communities_without_llm(self):
        llm = FakeLLM(default=CHUNK_A_RESPONSE)
        builder = GraphBuilder(llm_client=llm)
        builder.add_chunk("c1", "some text")
        builder.build()

        # Replace llm_client with None so fallback summary is used
        builder.llm_client = None
        summaries = builder.summarise_communities()
        assert len(summaries) >= 1
        for _cid, text in summaries.items():
            assert isinstance(text, str) and len(text) > 0

    def test_summarise_communities_with_llm(self):
        llm = FakeLLM(default=CHUNK_A_RESPONSE)
        builder = GraphBuilder(llm_client=llm)
        builder.add_chunk("c1", "some text")
        builder.build()

        # Now the LLM will provide a summary
        llm._default = "A community of entities."
        summaries = builder.summarise_communities()
        assert any("A community" in s for s in summaries.values())

    def test_entity_type_filtering(self):
        llm = FakeLLM(default=CHUNK_A_RESPONSE)
        builder = GraphBuilder(llm_client=llm, entity_types=["ORG"])
        builder.add_chunk("c1", "text")

        assert "ACME" in builder.graph
        assert "Alice" not in builder.graph

    def test_edge_weight_takes_max(self):
        """When two chunks produce the same edge, the max weight wins."""
        response_1 = _simple_payload(
            [{"name": "A", "type": "CONCEPT"}, {"name": "B", "type": "CONCEPT"}],
            [{"source": "A", "target": "B", "relation": "r1", "weight": 0.3}],
        )
        response_2 = _simple_payload(
            [{"name": "A", "type": "CONCEPT"}, {"name": "B", "type": "CONCEPT"}],
            [{"source": "A", "target": "B", "relation": "r2", "weight": 0.8}],
        )
        llm = FakeLLM(responses={"first": response_1, "second": response_2})
        builder = GraphBuilder(llm_client=llm)
        builder.add_chunk("c1", "first")
        builder.add_chunk("c2", "second")

        assert builder.graph["A"]["B"]["weight"] == pytest.approx(0.8)
        relations = set(json.loads(builder.graph["A"]["B"]["relations"]))
        assert relations == {"r1", "r2"}


# ---------------------------------------------------------------------------
# Malformed LLM output hardening
# ---------------------------------------------------------------------------

class TestMalformedLLMOutput:

    def test_invalid_json_does_not_crash(self):
        llm = FakeLLM(default="NOT VALID JSON {{{")
        builder = GraphBuilder(llm_client=llm)
        builder.add_chunk("c1", "text")  # should not raise
        assert builder.graph.number_of_nodes() == 0

    def test_non_dict_json_does_not_crash(self):
        llm = FakeLLM(default='"just a string"')
        builder = GraphBuilder(llm_client=llm)
        builder.add_chunk("c1", "text")
        assert builder.graph.number_of_nodes() == 0

    def test_entity_without_name_is_skipped(self):
        payload = _simple_payload([{"type": "PERSON", "description": "no name"}])
        llm = FakeLLM(default=payload)
        builder = GraphBuilder(llm_client=llm)
        builder.add_chunk("c1", "text")
        assert builder.graph.number_of_nodes() == 0

    def test_entity_non_dict_is_skipped(self):
        resp = json.dumps({"entities": ["not-a-dict", 42], "relationships": []})
        llm = FakeLLM(default=resp)
        builder = GraphBuilder(llm_client=llm)
        builder.add_chunk("c1", "text")
        assert builder.graph.number_of_nodes() == 0

    def test_relationship_non_dict_is_skipped(self):
        resp = json.dumps({
            "entities": [{"name": "A", "type": "CONCEPT"}],
            "relationships": ["not-a-dict"],
        })
        llm = FakeLLM(default=resp)
        builder = GraphBuilder(llm_client=llm)
        builder.add_chunk("c1", "text")
        assert builder.graph.number_of_nodes() == 1
        assert builder.graph.number_of_edges() == 0

    def test_relationship_bad_weight_uses_default(self):
        resp = json.dumps({
            "entities": [{"name": "A", "type": "CONCEPT"}, {"name": "B", "type": "CONCEPT"}],
            "relationships": [{"source": "A", "target": "B", "relation": "r", "weight": "not-a-number"}],
        })
        llm = FakeLLM(default=resp)
        builder = GraphBuilder(llm_client=llm)
        builder.add_chunk("c1", "text")
        assert builder.graph.has_edge("A", "B")
        assert builder.graph["A"]["B"]["weight"] == pytest.approx(0.5)

    def test_relationship_missing_source_or_target_skipped(self):
        resp = json.dumps({
            "entities": [{"name": "A", "type": "CONCEPT"}],
            "relationships": [
                {"source": "A", "target": "", "relation": "r"},
                {"source": "", "target": "A", "relation": "r"},
                {"target": "A", "relation": "r"},
            ],
        })
        llm = FakeLLM(default=resp)
        builder = GraphBuilder(llm_client=llm)
        builder.add_chunk("c1", "text")
        assert builder.graph.number_of_edges() == 0

    def test_no_llm_client_raises_on_add_chunk(self):
        builder = GraphBuilder(llm_client=None)
        with pytest.raises(ValueError, match="llm_client is required"):
            builder.add_chunk("c1", "text")


# ---------------------------------------------------------------------------
# Cache & context manager
# ---------------------------------------------------------------------------

class TestCacheAndClose:

    def test_cache_round_trip(self, tmp_path):
        llm = FakeLLM(default=CHUNK_A_RESPONSE)
        builder = GraphBuilder(llm_client=llm, cache_dir=str(tmp_path / "cache"))
        builder.add_chunk("c1", "text one")
        assert len(llm.calls) == 1

        # Second call with same chunk_id+text should hit cache
        builder2 = GraphBuilder(llm_client=llm, cache_dir=str(tmp_path / "cache"))
        builder2.add_chunk("c1", "text one")
        assert len(llm.calls) == 1  # no new calls

    def test_context_manager_closes_db(self, tmp_path):
        with GraphBuilder(llm_client=None, cache_dir=str(tmp_path / "cache")) as builder:
            assert builder._db is not None
        assert builder._db is None

    def test_close_is_idempotent(self, tmp_path):
        builder = GraphBuilder(llm_client=None, cache_dir=str(tmp_path / "cache"))
        builder.close()
        builder.close()  # should not raise
        assert builder._db is None

    def test_close_without_cache_is_noop(self):
        builder = GraphBuilder(llm_client=None)
        builder.close()  # no-op, no error

    def test_summary_cache_round_trip(self, tmp_path):
        llm = FakeLLM(default=CHUNK_A_RESPONSE)
        builder = GraphBuilder(llm_client=llm, cache_dir=str(tmp_path / "cache"))
        builder.add_chunk("c1", "text")
        builder.build()
        summaries_1 = builder.summarise_communities()

        # Second builder should read summary from cache
        builder2 = GraphBuilder(llm_client=llm, cache_dir=str(tmp_path / "cache"))
        builder2.add_chunk("c1", "text")
        builder2.build()
        call_count_before = len(llm.calls)
        summaries_2 = builder2.summarise_communities()
        assert summaries_1 == summaries_2


# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

def test_entity_prompt_has_placeholder():
    assert "{chunk_text}" in ENTITY_PROMPT
