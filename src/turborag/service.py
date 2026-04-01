from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .adapters.compat import ExistingRAGAdapter
from .exceptions import TurboRAGError
from .ingest import SNAPSHOT_FILE_NAME, load_records_snapshot, write_records_snapshot
from .index import TurboIndex
from .types import ChunkRecord, RetrievalResult

logger = logging.getLogger("turborag.service")

# ---------------------------------------------------------------------------
#  Metrics collector (lightweight, no external deps)
# ---------------------------------------------------------------------------

@dataclass
class _LatencyBucket:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def record(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        if ms < self.min_ms:
            self.min_ms = ms
        if ms > self.max_ms:
            self.max_ms = ms

    def as_dict(self) -> dict[str, Any]:
        avg = (self.total_ms / self.count) if self.count else 0.0
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(avg, 3),
            "min_ms": round(self.min_ms, 3) if self.count else None,
            "max_ms": round(self.max_ms, 3) if self.count else None,
        }


@dataclass
class Metrics:
    """Lightweight in-process metrics collector."""

    _buckets: dict[str, _LatencyBucket] = field(default_factory=dict)
    _error_count: int = 0
    _start_time: float = field(default_factory=time.monotonic)

    def record_latency(self, endpoint: str, ms: float) -> None:
        if endpoint not in self._buckets:
            self._buckets[endpoint] = _LatencyBucket()
        self._buckets[endpoint].record(ms)

    def record_error(self) -> None:
        self._error_count += 1

    def snapshot(self) -> dict[str, Any]:
        uptime = time.monotonic() - self._start_time
        return {
            "uptime_seconds": round(uptime, 1),
            "errors": self._error_count,
            "endpoints": {k: v.as_dict() for k, v in self._buckets.items()},
        }


# ---------------------------------------------------------------------------
#  TurboService
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TurboService:
    index_path: Path
    index: TurboIndex
    records: dict[str, ChunkRecord]
    query_embedder: Any | None = None

    @classmethod
    def open(
        cls,
        index_path: str | Path,
        *,
        query_embedder: Any | None = None,
    ) -> "TurboService":
        target = Path(index_path)
        index = TurboIndex.open(str(target))
        snapshot_path = target / SNAPSHOT_FILE_NAME
        records = load_records_snapshot(snapshot_path) if snapshot_path.exists() else {}
        return cls(index_path=target, index=index, records=records, query_embedder=query_embedder)

    @property
    def snapshot_path(self) -> Path:
        return self.index_path / SNAPSHOT_FILE_NAME

    def describe(self) -> dict[str, Any]:
        snapshot = self.snapshot_path if self.snapshot_path.exists() else None
        return {
            "index_path": str(self.index_path),
            "dim": self.index.dim,
            "bits": self.index.bits,
            "shard_size": self.index.shard_size,
            "normalize": self.index.normalize,
            "value_range": self.index.value_range,
            "index_size": len(self.index),
            "records_loaded": len(self.records),
            "records_snapshot": None if snapshot is None else str(snapshot),
            "text_query_enabled": self.query_embedder is not None,
        }

    def query(self, payload: dict[str, Any]) -> dict[str, Any]:
        query_text, query_vector, top_k = _validate_query_payload(payload)

        adapter = ExistingRAGAdapter(
            index=self.index,
            query_embedder=self.query_embedder or _MissingServiceQueryEmbedder(),
            fetch_records=self._fetch_records,
        )

        if query_text is not None:
            results = adapter.query(query_text, k=top_k)
        else:
            vector = np.asarray(query_vector, dtype=np.float32)
            results = adapter.query_by_vector(vector, k=top_k)

        return {
            "count": len(results),
            "results": [_serialize_result(result) for result in results],
        }

    def query_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle batch vector queries."""
        queries = payload.get("queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError("queries must be a non-empty list")

        top_k = int(payload.get("top_k", 5))
        if top_k <= 0 or top_k > 1000:
            raise ValueError("top_k must be between 1 and 1000")

        vectors = []
        for i, q in enumerate(queries):
            if not isinstance(q, dict):
                raise ValueError(f"Query {i+1} must be an object with query_vector")
            qv = q.get("query_vector")
            if not isinstance(qv, list) or not qv:
                raise ValueError(f"Query {i+1} requires a non-empty query_vector")
            vectors.append([float(v) for v in qv])

        query_matrix = np.asarray(vectors, dtype=np.float32)
        all_results = self.index.search_batch(query_matrix, k=top_k)

        batch_results = []
        for hits in all_results:
            result_list = []
            for chunk_id, score in hits:
                rec = self.records.get(chunk_id)
                if rec:
                    result_list.append(_serialize_result(RetrievalResult(
                        chunk_id=rec.chunk_id,
                        text=rec.text,
                        score=score,
                        source_doc=rec.source_doc,
                        page_num=rec.page_num,
                    )))
                else:
                    result_list.append({"chunk_id": chunk_id, "score": float(score)})
            batch_results.append({"count": len(result_list), "results": result_list})

        return {"batch_count": len(batch_results), "results": batch_results}

    def ingest_records(self, payload: dict[str, Any]) -> dict[str, Any]:
        records, embeddings = _validate_ingest_payload(payload)
        self.index.add(embeddings, [record.chunk_id for record in records])
        _write_index_config(self.index, self.index_path)

        for record in records:
            self.records[record.chunk_id] = record
        write_records_snapshot(self.records.values(), self.snapshot_path)

        return {
            "added": len(records),
            "index_size": len(self.index),
            "records_snapshot": str(self.snapshot_path),
        }

    def ingest_text(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ingest raw text documents with automatic chunking and embedding."""
        text = payload.get("text")
        source_doc = payload.get("source_doc")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text is required and must be non-empty")

        if self.query_embedder is None:
            raise RuntimeError(
                "Text ingestion requires an embedder. Start with --model."
            )

        from .chunker import ChunkConfig, chunk_text

        config_raw = payload.get("chunk_config", {})
        config = ChunkConfig(
            chunk_size=int(config_raw.get("chunk_size", 512)),
            chunk_overlap=int(config_raw.get("chunk_overlap", 64)),
        )

        chunks = chunk_text(text, source_doc=source_doc, config=config)
        if not chunks:
            return {"added": 0, "index_size": len(self.index)}

        texts = [c.text for c in chunks]
        embeddings = np.array([self.query_embedder.embed(t) for t in texts], dtype=np.float32)
        ids = [c.chunk_id for c in chunks]

        self.index.add(embeddings, ids)
        _write_index_config(self.index, self.index_path)

        for c in chunks:
            self.records[c.chunk_id] = c
        write_records_snapshot(self.records.values(), self.snapshot_path)

        return {
            "added": len(chunks),
            "chunks": [{"chunk_id": c.chunk_id, "text": c.text[:200]} for c in chunks],
            "index_size": len(self.index),
        }

    def _fetch_records(self, ids: Sequence[str]) -> Sequence[ChunkRecord]:
        return [self.records[chunk_id] for chunk_id in ids if chunk_id in self.records]


class _MissingServiceQueryEmbedder:
    def embed_query(self, text: str):
        raise RuntimeError(
            "Text queries are not enabled for this service. Start `turborag serve` with --model or send query_vector."
        )


# ---------------------------------------------------------------------------
#  Starlette App Factory
# ---------------------------------------------------------------------------

def create_app(
    index_path: str | Path,
    *,
    query_embedder: Any | None = None,
    cors_origins: list[str] | None = None,
) -> Starlette:
    service = TurboService.open(index_path, query_embedder=query_embedder)
    metrics = Metrics()
    _write_lock = threading.Lock()

    # --- Helpers ---

    def _request_id(request: Request) -> str:
        return request.headers.get("x-request-id", str(uuid.uuid4())[:8])

    async def _json_or_400(request: Request):
        try:
            return await request.json()
        except Exception:
            return None

    # --- Route handlers ---

    async def root(request: Request) -> JSONResponse:
        return JSONResponse(service.describe())

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "index_path": str(service.index_path),
            "index_size": len(service.index),
        })

    async def describe_index(request: Request) -> JSONResponse:
        return JSONResponse(service.describe())

    async def query(request: Request) -> JSONResponse:
        rid = _request_id(request)
        start = time.monotonic()
        try:
            payload = await _json_or_400(request)
            if payload is None:
                return JSONResponse({"detail": "Invalid JSON body"}, status_code=400)
            result = service.query(payload)
            elapsed = (time.monotonic() - start) * 1000
            metrics.record_latency("/query", elapsed)
            logger.info("query rid=%s k=%s latency=%.1fms", rid, payload.get("top_k", 5), elapsed)
            return JSONResponse(result)
        except (RuntimeError, ValueError, TurboRAGError) as exc:
            metrics.record_error()
            logger.warning("query rid=%s error=%s", rid, exc)
            return JSONResponse({"detail": str(exc)}, status_code=400)

    async def query_batch(request: Request) -> JSONResponse:
        rid = _request_id(request)
        start = time.monotonic()
        try:
            payload = await _json_or_400(request)
            if payload is None:
                return JSONResponse({"detail": "Invalid JSON body"}, status_code=400)
            result = service.query_batch(payload)
            elapsed = (time.monotonic() - start) * 1000
            metrics.record_latency("/query/batch", elapsed)
            logger.info("query_batch rid=%s n=%d latency=%.1fms", rid, result["batch_count"], elapsed)
            return JSONResponse(result)
        except (RuntimeError, ValueError, TurboRAGError) as exc:
            metrics.record_error()
            return JSONResponse({"detail": str(exc)}, status_code=400)

    async def ingest(request: Request) -> JSONResponse:
        rid = _request_id(request)
        start = time.monotonic()
        try:
            payload = await _json_or_400(request)
            if payload is None:
                return JSONResponse({"detail": "Invalid JSON body"}, status_code=400)
            with _write_lock:
                result = service.ingest_records(payload)
            elapsed = (time.monotonic() - start) * 1000
            metrics.record_latency("/ingest", elapsed)
            logger.info("ingest rid=%s added=%d latency=%.1fms", rid, result["added"], elapsed)
            return JSONResponse(result)
        except (RuntimeError, ValueError, TurboRAGError) as exc:
            metrics.record_error()
            return JSONResponse({"detail": str(exc)}, status_code=400)

    async def ingest_text(request: Request) -> JSONResponse:
        rid = _request_id(request)
        start = time.monotonic()
        try:
            payload = await _json_or_400(request)
            if payload is None:
                return JSONResponse({"detail": "Invalid JSON body"}, status_code=400)
            with _write_lock:
                result = service.ingest_text(payload)
            elapsed = (time.monotonic() - start) * 1000
            metrics.record_latency("/ingest-text", elapsed)
            logger.info("ingest_text rid=%s added=%d latency=%.1fms", rid, result["added"], elapsed)
            return JSONResponse(result)
        except (RuntimeError, ValueError, TurboRAGError) as exc:
            metrics.record_error()
            return JSONResponse({"detail": str(exc)}, status_code=400)

    async def metrics_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(metrics.snapshot())

    # --- Build app ---

    middleware = []
    origins = cors_origins or ["*"]
    middleware.append(
        Middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    )

    app = Starlette(
        routes=[
            Route("/", root, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
            Route("/index", describe_index, methods=["GET"]),
            Route("/query", query, methods=["POST"]),
            Route("/query/batch", query_batch, methods=["POST"]),
            Route("/ingest", ingest, methods=["POST"]),
            Route("/ingest-text", ingest_text, methods=["POST"]),
            Route("/metrics", metrics_endpoint, methods=["GET"]),
        ],
        middleware=middleware,
    )
    app.state.turborag = service
    app.state.metrics = metrics
    return app


# ---------------------------------------------------------------------------
#  Serialisation Helpers
# ---------------------------------------------------------------------------

def _serialize_result(result: RetrievalResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "text": result.text,
        "score": float(result.score),
        "source_doc": result.source_doc,
        "page_num": result.page_num,
        "graph_path": result.graph_path,
        "explanation": result.explanation,
    }


def _validate_query_payload(payload: dict[str, Any]) -> tuple[str | None, list[float] | None, int]:
    if not isinstance(payload, dict):
        raise ValueError("Query payload must be a JSON object.")

    allowed_keys = {"query_text", "query_vector", "top_k"}
    unknown = sorted(set(payload) - allowed_keys)
    if unknown:
        raise ValueError(f"Unknown query fields: {', '.join(unknown)}")

    query_text = payload.get("query_text")
    query_vector = payload.get("query_vector")
    top_k = int(payload.get("top_k", 5))
    if top_k <= 0 or top_k > 1000:
        raise ValueError("top_k must be between 1 and 1000.")

    has_text = isinstance(query_text, str) and bool(query_text.strip())
    has_vector = query_vector is not None
    if has_text == has_vector:
        raise ValueError("Provide exactly one of query_text or query_vector.")

    if has_vector:
        if not isinstance(query_vector, list) or not query_vector:
            raise ValueError("query_vector must be a non-empty JSON array.")
        if not all(isinstance(value, (int, float)) for value in query_vector):
            raise ValueError("query_vector must contain only numeric values.")
    else:
        query_vector = None

    return (query_text if has_text else None, query_vector, top_k)


def _validate_ingest_payload(payload: dict[str, Any]) -> tuple[list[ChunkRecord], np.ndarray]:
    if not isinstance(payload, dict):
        raise ValueError("Ingest payload must be a JSON object.")

    allowed_keys = {"records"}
    unknown = sorted(set(payload) - allowed_keys)
    if unknown:
        raise ValueError(f"Unknown ingest fields: {', '.join(unknown)}")

    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("At least one record is required.")

    records: list[ChunkRecord] = []
    embeddings: list[list[float]] = []
    for index, item in enumerate(raw_records, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Record {index} must be a JSON object.")

        allowed_record_keys = {
            "chunk_id",
            "text",
            "embedding",
            "source_doc",
            "page_num",
            "section",
            "metadata",
        }
        unknown_record_keys = sorted(set(item) - allowed_record_keys)
        if unknown_record_keys:
            raise ValueError(
                f"Record {index} contains unknown fields: {', '.join(unknown_record_keys)}"
            )

        chunk_id = item.get("chunk_id")
        text = item.get("text")
        embedding = item.get("embedding")
        metadata = item.get("metadata", {})

        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"Record {index} requires a non-empty chunk_id.")
        if not isinstance(text, str):
            raise ValueError(f"Record {index} requires text.")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Record {index} requires a non-empty embedding array.")
        if not all(isinstance(value, (int, float)) for value in embedding):
            raise ValueError(f"Record {index} embedding must contain only numeric values.")
        if not isinstance(metadata, dict):
            raise ValueError(f"Record {index} metadata must be an object.")

        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                text=text,
                source_doc=item.get("source_doc"),
                page_num=item.get("page_num"),
                section=item.get("section"),
                metadata=dict(metadata),
            )
        )
        embeddings.append([float(value) for value in embedding])

    matrix = np.asarray(embeddings, dtype=np.float32)
    return records, matrix


def _write_index_config(index: TurboIndex, index_path: Path) -> None:
    config = {
        "dim": index.dim,
        "bits": index.bits,
        "seed": index.seed,
        "shard_size": index.shard_size,
        "normalize": index.normalize,
        "value_range": index.value_range,
        "schema_version": 1,
        "size": len(index),
    }
    (index_path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
