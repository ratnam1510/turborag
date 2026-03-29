from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .adapters.compat import ExistingRAGAdapter, coerce_chunk_record
from .index import TurboIndex
from .types import ChunkRecord


FloatMatrix = NDArray[np.float32]
FetchRecords = Callable[[Sequence[str]], Sequence[ChunkRecord | Mapping[str, Any]]]
SNAPSHOT_FILE_NAME = "records.jsonl"


@dataclass(slots=True)
class ImportedDataset:
    """A dataset ready to be indexed into TurboRAG."""

    records: list[ChunkRecord]
    embeddings: FloatMatrix

    @property
    def ids(self) -> list[str]:
        return [record.chunk_id for record in self.records]

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


@dataclass(slots=True)
class SidecarBuildResult:
    """Summary of an imported sidecar index build."""

    index_path: Path
    records_path: Path | None
    count: int
    dim: int
    bits: int
    shard_size: int


def load_dataset(path: str | Path, format: str = "auto") -> ImportedDataset:
    source = Path(path)
    detected = _detect_format(source, format)
    if detected == "jsonl":
        return load_jsonl_dataset(source)
    if detected == "npz":
        return load_npz_dataset(source)
    raise ValueError(f"unsupported dataset format: {detected}")


def load_jsonl_dataset(path: str | Path) -> ImportedDataset:
    source = Path(path)
    records: list[ChunkRecord] = []
    embeddings: list[list[float]] = []

    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            embedding = payload.get("embedding")
            if embedding is None:
                raise ValueError(f"missing embedding on line {line_number} in {source}")
            records.append(coerce_chunk_record(payload))
            embeddings.append(list(embedding))

    if not records:
        raise ValueError(f"no records found in {source}")

    matrix = np.asarray(embeddings, dtype=np.float32)
    _validate_dataset(records, matrix)
    return ImportedDataset(records=records, embeddings=matrix)


def load_npz_dataset(path: str | Path) -> ImportedDataset:
    source = Path(path)
    with np.load(source, allow_pickle=True) as data:
        if "embeddings" not in data or "ids" not in data:
            raise ValueError("npz datasets must contain 'embeddings' and 'ids'")

        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        ids = [str(value) for value in np.asarray(data["ids"]).tolist()]
        texts = _optional_array(data, "texts", default=[""] * len(ids))
        source_docs = _optional_array(data, "source_docs", default=[None] * len(ids))
        page_nums = _optional_array(data, "page_nums", default=[None] * len(ids))
        sections = _optional_array(data, "sections", default=[None] * len(ids))
        metadatas = _optional_metadata(data, len(ids))

    records = [
        ChunkRecord(
            chunk_id=ids[index],
            text=str(texts[index]),
            source_doc=None if source_docs[index] is None else str(source_docs[index]),
            page_num=_coerce_optional_int(page_nums[index]),
            section=None if sections[index] is None else str(sections[index]),
            metadata=metadatas[index],
        )
        for index in range(len(ids))
    ]

    _validate_dataset(records, embeddings)
    return ImportedDataset(records=records, embeddings=embeddings)


def build_sidecar_index(
    dataset: ImportedDataset,
    index_path: str | Path,
    *,
    bits: int = 3,
    shard_size: int = 100_000,
    seed: int = 42,
    normalize: bool = True,
    value_range: float = 1.0,
    save_records: bool = True,
) -> SidecarBuildResult:
    target = Path(index_path)
    target.mkdir(parents=True, exist_ok=True)

    index = TurboIndex(
        dim=dataset.dim,
        bits=bits,
        shard_size=shard_size,
        seed=seed,
        normalize=normalize,
        value_range=value_range,
    )
    index.add(dataset.embeddings, dataset.ids)
    index.save(str(target))

    records_path: Path | None = None
    if save_records:
        records_path = target / SNAPSHOT_FILE_NAME
        write_records_snapshot(dataset.records, records_path)

    return SidecarBuildResult(
        index_path=target,
        records_path=records_path,
        count=len(dataset.records),
        dim=dataset.dim,
        bits=bits,
        shard_size=shard_size,
    )


def write_records_snapshot(records: Iterable[ChunkRecord], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "chunk_id": record.chunk_id,
                "text": record.text,
                "source_doc": record.source_doc,
                "page_num": record.page_num,
                "section": record.section,
                "metadata": record.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return target


def load_records_snapshot(path: str | Path) -> dict[str, ChunkRecord]:
    source = Path(path)
    records: dict[str, ChunkRecord] = {}
    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = coerce_chunk_record(json.loads(line))
            records[record.chunk_id] = record
    return records


def snapshot_fetch_records(snapshot_path: str | Path) -> FetchRecords:
    record_store = load_records_snapshot(snapshot_path)
    return lambda ids: [record_store[chunk_id] for chunk_id in ids if chunk_id in record_store]


def open_sidecar_adapter(
    index_path: str | Path,
    *,
    query_embedder: Any | None = None,
    fetch_records: FetchRecords | None = None,
) -> ExistingRAGAdapter:
    index_dir = Path(index_path)
    index = TurboIndex.open(str(index_dir))

    resolver = fetch_records
    if resolver is None:
        snapshot_path = index_dir / SNAPSHOT_FILE_NAME
        if snapshot_path.exists():
            resolver = snapshot_fetch_records(snapshot_path)
        else:
            raise FileNotFoundError(
                f"No record snapshot found at {snapshot_path}. Provide fetch_records explicitly."
            )

    embedder = query_embedder if query_embedder is not None else _MissingQueryEmbedder()
    return ExistingRAGAdapter(index=index, query_embedder=embedder, fetch_records=resolver)


class _MissingQueryEmbedder:
    def embed_query(self, text: str):
        raise RuntimeError(
            "This sidecar adapter was opened without a query embedder. Use query_by_vector(...) or provide query_embedder."
        )


def _validate_dataset(records: Sequence[ChunkRecord], embeddings: FloatMatrix) -> None:
    if not records:
        raise ValueError("dataset must contain at least one record")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")
    if embeddings.shape[0] != len(records):
        raise ValueError("record count must match embedding rows")
    ids = [record.chunk_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("chunk IDs must be unique")


def _detect_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".npz":
        return "npz"
    raise ValueError(f"could not infer format from file extension: {path}")


def _optional_array(data: Any, key: str, default: list[Any]) -> list[Any]:
    if key not in data:
        return default
    values = np.asarray(data[key], dtype=object).tolist()
    if len(values) != len(default):
        raise ValueError(f"optional array '{key}' must have length {len(default)}")
    return values


def _optional_metadata(data: Any, count: int) -> list[dict[str, Any]]:
    if "metadata_json" not in data:
        return [{} for _ in range(count)]
    raw_values = np.asarray(data["metadata_json"], dtype=object).tolist()
    if len(raw_values) != count:
        raise ValueError(f"optional array 'metadata_json' must have length {count}")
    return [json.loads(value) if value else {} for value in raw_values]


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if value == "":
        return None
    return int(value)
