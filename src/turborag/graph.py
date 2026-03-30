from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENTITY_PROMPT = """You are an entity extractor. Given the following text, extract all named entities and the relationships between them. Return ONLY valid JSON matching this schema:
{{
  "entities": [{{"name": str, "type": str, "description": str}}],
  "relationships": [{{"source": str, "target": str, "relation": str, "weight": float}}]
}}
Entity types: PERSON, ORG, CONCEPT, METRIC, DATE, PRODUCT, LOCATION
Relationship weight: 0.0 (weak) to 1.0 (strong/direct).
Text: {chunk_text}
"""

SUMMARY_PROMPT = """Summarise the following entity community in one concise paragraph. Focus on the major concepts and how they relate.
Community entities: {entities}
"""


class GraphBuilder:
    """Build a simple entity graph from chunk-level extraction responses.

    Supports use as a context manager for deterministic resource cleanup::

        with GraphBuilder(llm_client=client, cache_dir="/tmp/cache") as builder:
            builder.add_chunk("c1", "Some text")
            graph = builder.build()
    """

    def __init__(
        self,
        llm_client: Any,
        entity_types: list[str] | None = None,
        max_community_size: int = 10,
        cache_dir: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.entity_types = set(entity_types) if entity_types else None
        self.max_community_size = max_community_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._communities: dict[int, list[str]] = {}
        self._summaries: dict[int, str] = {}
        self.graph = self._make_graph()
        self._db: sqlite3.Connection | None = None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.cache_dir / "llm_cache.sqlite3")
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            self._db.commit()

    # -- Context manager support --------------------------------------------

    def __enter__(self) -> "GraphBuilder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.close()

    def close(self) -> None:
        """Close the underlying SQLite cache connection, if open."""
        if self._db is not None:
            try:
                self._db.close()
            except Exception:  # pragma: no cover – best-effort
                logger.debug("Ignoring error while closing cache database", exc_info=True)
            finally:
                self._db = None

    # -- Public API ---------------------------------------------------------

    def add_chunk(self, chunk_id: str, text: str) -> None:
        payload = self._extract_payload(chunk_id=chunk_id, text=text)
        entities = payload.get("entities", [])
        relationships = payload.get("relationships", [])

        for entity in entities:
            if not isinstance(entity, dict):
                logger.warning("Skipping non-dict entity in chunk %s: %r", chunk_id, entity)
                continue
            name = entity.get("name")
            if not name:
                logger.warning("Skipping entity without a name in chunk %s", chunk_id)
                continue
            entity_type = entity.get("type")
            if self.entity_types and entity_type not in self.entity_types:
                continue
            if name not in self.graph:
                self.graph.add_node(
                    name,
                    name=name,
                    type=entity_type,
                    description=entity.get("description", ""),
                    chunk_ids=json.dumps([chunk_id]),
                )
            else:
                existing_raw = self.graph.nodes[name].get("chunk_ids", "[]")
                try:
                    chunk_ids = set(json.loads(existing_raw))
                except (json.JSONDecodeError, TypeError):
                    chunk_ids = set()
                chunk_ids.add(chunk_id)
                self.graph.nodes[name]["chunk_ids"] = json.dumps(sorted(chunk_ids))
                if entity.get("description") and not self.graph.nodes[name].get("description"):
                    self.graph.nodes[name]["description"] = entity["description"]

        for relationship in relationships:
            if not isinstance(relationship, dict):
                logger.warning("Skipping non-dict relationship in chunk %s: %r", chunk_id, relationship)
                continue
            source = relationship.get("source")
            target = relationship.get("target")
            if not source or not target:
                continue
            # Only add edges between nodes that were actually accepted into the
            # graph (respects entity_types filtering and name validation above).
            if source not in self.graph or target not in self.graph:
                continue
            try:
                weight = float(relationship.get("weight", 0.5))
            except (TypeError, ValueError):
                weight = 0.5
            relation = relationship.get("relation", "related_to") or "related_to"
            if self.graph.has_edge(source, target):
                existing = self.graph[source][target]
                existing["weight"] = max(float(existing.get("weight", 0.0)), weight)
                try:
                    relations = set(json.loads(existing.get("relations", "[]")))
                except (json.JSONDecodeError, TypeError):
                    relations = set()
                relations.add(relation)
                existing["relations"] = json.dumps(sorted(relations))
            else:
                self.graph.add_edge(
                    source,
                    target,
                    weight=weight,
                    relations=json.dumps([relation]),
                )

    def build(self):
        if self.graph.number_of_nodes() == 0:
            self._communities = {}
            return self.graph

        communities = self._detect_communities()
        grouped: dict[int, list[str]] = {}
        for node, community_id in communities.items():
            grouped.setdefault(int(community_id), []).append(node)
            self.graph.nodes[node]["community"] = int(community_id)
        self._communities = {key: sorted(values) for key, values in grouped.items()}
        return self.graph

    def get_communities(self) -> dict[int, list[str]]:
        return dict(self._communities)

    def save(self, path: str | Path) -> None:
        """Persist the graph, communities, and summaries to disk.

        Writes:
        - ``graph.graphml`` — the entity-relationship graph
        - ``communities.json`` — community membership
        - ``summaries.json`` — community summaries (if generated)
        """
        nx = self._networkx()
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        nx.write_graphml(self.graph, str(save_dir / "graph.graphml"))

        (save_dir / "communities.json").write_text(
            json.dumps(self._communities, indent=2), encoding="utf-8"
        )
        if self._summaries:
            (save_dir / "summaries.json").write_text(
                json.dumps(self._summaries, indent=2), encoding="utf-8"
            )

        logger.info(
            "Saved graph with %d nodes, %d edges, %d communities to %s",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            len(self._communities),
            save_dir,
        )

    @classmethod
    def load(cls, path: str | Path, llm_client: Any = None) -> "GraphBuilder":
        """Load a previously saved graph from disk."""
        nx = cls._networkx_static()
        load_dir = Path(path)

        builder = cls(llm_client=llm_client)
        builder.graph = nx.read_graphml(str(load_dir / "graph.graphml"))

        communities_path = load_dir / "communities.json"
        if communities_path.exists():
            raw = json.loads(communities_path.read_text(encoding="utf-8"))
            builder._communities = {int(k): v for k, v in raw.items()}

        summaries_path = load_dir / "summaries.json"
        if summaries_path.exists():
            raw = json.loads(summaries_path.read_text(encoding="utf-8"))
            builder._summaries = {int(k): v for k, v in raw.items()}

        logger.info(
            "Loaded graph with %d nodes, %d edges from %s",
            builder.graph.number_of_nodes(),
            builder.graph.number_of_edges(),
            load_dir,
        )
        return builder

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the graph."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "communities": len(self._communities),
            "summaries_generated": len(self._summaries),
            "entity_types": sorted(set(
                str(self.graph.nodes[n].get("type", "unknown"))
                for n in self.graph.nodes
            )),
        }

    @staticmethod
    def _networkx_static():
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Graph support requires installing turborag[graph]") from exc
        return nx

    def summarise_communities(self) -> dict[int, str]:
        summaries: dict[int, str] = {}
        for community_id, entities in self._communities.items():
            cache_key = self._cache_key("summary", f"{community_id}:{','.join(entities)}")
            cached = self._cache_get(cache_key)
            if cached is not None:
                summaries[community_id] = cached
                continue

            if self.llm_client is None:
                summary = ", ".join(entities[: self.max_community_size])
            else:
                prompt = SUMMARY_PROMPT.format(entities=", ".join(entities[: self.max_community_size]))
                summary = self.llm_client.complete(prompt)
            summaries[community_id] = summary
            self._cache_set(cache_key, summary)

        self._summaries = summaries
        return dict(summaries)

    # -- Internal helpers ---------------------------------------------------

    def _extract_payload(self, chunk_id: str, text: str) -> dict[str, Any]:
        cache_key = self._cache_key("chunk", f"{chunk_id}:{text}")
        cached = self._cache_get(cache_key)
        if cached is not None:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Corrupt cache entry for chunk %s, re-extracting", chunk_id)

        if self.llm_client is None:
            raise ValueError("llm_client is required for entity extraction")

        prompt = ENTITY_PROMPT.format(chunk_text=text)
        response = self.llm_client.complete(prompt)

        try:
            payload = json.loads(response)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "LLM returned invalid JSON for chunk %s, treating as empty: %s",
                chunk_id,
                exc,
            )
            payload = {"entities": [], "relationships": []}

        if not isinstance(payload, dict):
            logger.warning(
                "LLM returned non-object JSON for chunk %s, treating as empty",
                chunk_id,
            )
            payload = {"entities": [], "relationships": []}

        self._cache_set(cache_key, json.dumps(payload))
        return payload

    def _detect_communities(self) -> dict[str, int]:
        try:
            import leidenalg
            import igraph as ig
        except ImportError:
            return self._connected_components()

        ig_graph = ig.Graph.from_networkx(self.graph)
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            seed=42,
        )
        nodes = list(self.graph.nodes())
        return {nodes[index]: membership for index, membership in enumerate(partition.membership)}

    def _connected_components(self) -> dict[str, int]:
        nx = self._networkx()
        communities: dict[str, int] = {}
        for community_id, nodes in enumerate(nx.connected_components(self.graph)):
            for node in nodes:
                communities[str(node)] = community_id
        return communities

    def _cache_key(self, prefix: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{prefix}:{digest}"

    def _cache_get(self, cache_key: str) -> str | None:
        if self._db is None:
            return None
        try:
            row = self._db.execute("SELECT payload FROM cache WHERE cache_key = ?", (cache_key,)).fetchone()
        except sqlite3.DatabaseError:
            logger.warning("Cache read failed for key %s", cache_key, exc_info=True)
            return None
        return None if row is None else str(row[0])

    def _cache_set(self, cache_key: str, payload: str) -> None:
        if self._db is None:
            return
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO cache (cache_key, payload) VALUES (?, ?)",
                (cache_key, payload),
            )
            self._db.commit()
        except sqlite3.DatabaseError:
            logger.warning("Cache write failed for key %s", cache_key, exc_info=True)

    def _make_graph(self):
        nx = self._networkx()
        return nx.Graph()

    @staticmethod
    def _networkx():
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Graph support requires installing turborag[graph]") from exc
        return nx
