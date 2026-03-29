from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .adapters.compat import ExistingRAGAdapter
from .ingest import SNAPSHOT_FILE_NAME, load_records_snapshot, write_records_snapshot
from .index import TurboIndex
from .types import ChunkRecord, RetrievalResult


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

    def _fetch_records(self, ids: Sequence[str]) -> Sequence[ChunkRecord]:
        return [self.records[chunk_id] for chunk_id in ids if chunk_id in self.records]


class _MissingServiceQueryEmbedder:
    def embed_query(self, text: str):
        raise RuntimeError(
            "Text queries are not enabled for this service. Start `turborag serve` with --model or send query_vector."
        )


def create_app(
    index_path: str | Path,
    *,
    query_embedder: Any | None = None,
) -> Starlette:
    service = TurboService.open(index_path, query_embedder=query_embedder)

    async def root(request: Request) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(service.describe())

    async def health(request: Request) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(
            {
                "status": "ok",
                "index_path": str(service.index_path),
                "index_size": len(service.index),
            }
        )

    async def describe_index(request: Request) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(service.describe())

    async def query(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            return JSONResponse(service.query(payload))
        except (RuntimeError, ValueError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)

    async def ingest(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            return JSONResponse(service.ingest_records(payload))
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)

    app = Starlette(
        routes=[
            Route("/", root, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
            Route("/index", describe_index, methods=["GET"]),
            Route("/query", query, methods=["POST"]),
            Route("/ingest", ingest, methods=["POST"]),
        ]
    )
    app.state.turborag = service
    return app


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
