from __future__ import annotations

import re
import unicodedata
from collections import deque
import json
import logging
from typing import Callable, Mapping

from .embeddings import Embedder
from .index import TurboIndex
from .types import ChunkRecord, RetrievalResult

logger = logging.getLogger(__name__)

# Pre-compiled pattern for tokenization: splits on non-alphanumeric boundaries
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    """Lowercase, strip accents, and collapse whitespace for matching."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().strip()


def _tokenize(text: str) -> list[str]:
    """Split normalized text into non-empty alphanumeric tokens."""
    return [tok for tok in _TOKEN_SPLIT.split(text) if tok]


def _phrase_match(entity_tokens: list[str], query_tokens: list[str]) -> bool:
    """Return True if *entity_tokens* appears as a contiguous sub-sequence of *query_tokens*.

    This respects token boundaries so that e.g. entity "Alpha" does not
    false-positive on query word "Alphabet".
    """
    if not entity_tokens:
        return False
    elen = len(entity_tokens)
    for i in range(len(query_tokens) - elen + 1):
        if query_tokens[i : i + elen] == entity_tokens:
            return True
    return False


class HybridRetriever:
    """Combine dense retrieval with graph-guided expansion."""

    def __init__(
        self,
        index: TurboIndex,
        graph,
        community_summaries: dict[int, str],
        embedder: Embedder | Callable[[str], object],
        chunks: Mapping[str, ChunkRecord],
        reranker: Callable[[str, list[ChunkRecord]], list[float]] | None = None,
        graph_depth: int = 2,
    ) -> None:
        self.index = index
        self.graph = graph
        self.community_summaries = community_summaries
        self.embedder = embedder
        self.chunks = dict(chunks)
        self.reranker = reranker
        self.graph_depth = graph_depth

    def query(self, text: str, k: int = 10, mode: str = "hybrid") -> list[RetrievalResult]:
        if mode not in {"dense", "graph", "hybrid"}:
            raise ValueError("mode must be one of 'dense', 'graph', or 'hybrid'")

        dense_results: dict[str, RetrievalResult] = {}
        graph_results: dict[str, RetrievalResult] = {}

        if mode in {"dense", "hybrid"}:
            vector = self._embed(text)
            for chunk_id, score in self.index.search(vector, k=k):
                chunk = self.chunks.get(chunk_id)
                if chunk is None:
                    continue
                dense_results[chunk_id] = RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=score,
                    source_doc=chunk.source_doc,
                    page_num=chunk.page_num,
                    explanation="Dense retrieval match from the compressed index.",
                )

        if mode in {"graph", "hybrid"}:
            graph_results = self._graph_candidates(text=text, k=k)

        merged = dense_results
        for chunk_id, result in graph_results.items():
            if chunk_id in merged:
                merged[chunk_id].score += result.score
                if result.graph_path:
                    merged[chunk_id].graph_path = result.graph_path
                    merged[chunk_id].explanation = result.explanation
            else:
                merged[chunk_id] = result

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        if self.reranker and ranked:
            reranked_scores = self.reranker(text, [self.chunks[result.chunk_id] for result in ranked])
            for result, score in zip(ranked, reranked_scores, strict=False):
                result.score = float(score)
            ranked.sort(key=lambda item: item.score, reverse=True)

        return ranked[:k]

    def explain(self, result: RetrievalResult) -> str:
        if result.explanation:
            return result.explanation
        if result.graph_path:
            return "Matched graph path: " + " -> ".join(result.graph_path)
        return "Matched by dense similarity search."

    def _graph_candidates(self, text: str, k: int) -> dict[str, RetrievalResult]:
        query_entities = self._detect_query_entities(text)
        if not query_entities:
            return {}

        results: dict[str, RetrievalResult] = {}
        for entity in query_entities:
            for path, node, depth in self._bfs_paths(entity, max_depth=self.graph_depth):
                raw_chunk_ids = self.graph.nodes[node].get("chunk_ids", "[]")
                try:
                    chunk_id_list = json.loads(raw_chunk_ids)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Malformed chunk_ids on graph node %r, skipping", node)
                    continue
                for chunk_id in chunk_id_list:
                    chunk = self.chunks.get(chunk_id)
                    if chunk is None:
                        continue
                    score = 1.0 / (depth + 1)
                    explanation = f"Matched graph path: {' -> '.join(path)}"
                    existing = results.get(chunk_id)
                    if existing is None or score > existing.score:
                        results[chunk_id] = RetrievalResult(
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            score=score,
                            source_doc=chunk.source_doc,
                            page_num=chunk.page_num,
                            graph_path=path,
                            explanation=explanation,
                        )

        ranked = sorted(results.values(), key=lambda item: item.score, reverse=True)
        return {result.chunk_id: result for result in ranked[:k]}

    def _detect_query_entities(self, text: str) -> list[str]:
        """Detect graph entities mentioned in *text* using normalized token-boundary matching.

        Unlike the previous naive substring check, this version:
        - normalises both the query and entity names (lowercase, strip accents)
        - tokenises on word boundaries so that entity "Alpha" does NOT match "Alphabet"
        - supports multi-word entity names as contiguous token sub-sequences
        """
        query_tokens = _tokenize(_normalize(text))
        if not query_tokens:
            return []

        matched: list[str] = []
        for node in self.graph.nodes:
            entity_tokens = _tokenize(_normalize(str(node)))
            if _phrase_match(entity_tokens, query_tokens):
                matched.append(str(node))
        return matched

    def _bfs_paths(self, start: str, max_depth: int):
        seen = {start}
        queue = deque([(start, [start], 0)])
        while queue:
            node, path, depth = queue.popleft()
            yield path, node, depth
            if depth >= max_depth:
                continue
            for neighbor in self.graph.neighbors(node):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, path + [str(neighbor)], depth + 1))

    def _embed(self, text: str):
        """Dispatch to the best available embedding method.

        Resolution order:
        1. ``embed(text)`` – the canonical :class:`~turborag.embeddings.Embedder` protocol
        2. ``embed_query(text)`` – LangChain / adapter convention
        3. bare callable – ``embedder(text)``
        """
        if hasattr(self.embedder, "embed"):
            return self.embedder.embed(text)
        if hasattr(self.embedder, "embed_query"):
            return self.embedder.embed_query(text)
        if callable(self.embedder):
            return self.embedder(text)
        raise TypeError(
            "embedder must implement embed(), embed_query(), or be callable"
        )
