import json

import numpy as np

from turborag.ingest import (
    SNAPSHOT_FILE_NAME,
    build_sidecar_index,
    load_dataset,
    open_sidecar_adapter,
)
from turborag.types import ChunkRecord


def test_jsonl_dataset_can_build_sidecar_and_query_by_vector(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {"chunk_id": "a", "text": "apple finance", "embedding": [2.0, 0.0, 1.0]},
        {"chunk_id": "b", "text": "banana inventory", "embedding": [0.0, 2.0, 0.0]},
        {"chunk_id": "c", "text": "finance banana", "embedding": [0.0, 1.0, 1.0]},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    dataset = load_dataset(dataset_path)
    result = build_sidecar_index(dataset, tmp_path / "index", bits=4)
    assert (result.index_path / SNAPSHOT_FILE_NAME).exists()

    adapter = open_sidecar_adapter(result.index_path)
    results = adapter.query_by_vector(np.array([2.0, 0.0, 1.0], dtype=np.float32), k=2)
    assert [item.chunk_id for item in results] == ["a", "c"]


def test_npz_dataset_loader_supports_metadata(tmp_path):
    source = tmp_path / "dataset.npz"
    np.savez(
        source,
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ids=np.asarray(["a", "b"]),
        texts=np.asarray(["alpha", "beta"]),
        source_docs=np.asarray(["doc1", "doc2"]),
        page_nums=np.asarray([1, 2]),
        metadata_json=np.asarray(
            [json.dumps({"kind": "x"}), json.dumps({"kind": "y"})]
        ),
    )

    dataset = load_dataset(source)
    assert dataset.ids == ["a", "b"]
    assert dataset.records[0] == ChunkRecord(
        chunk_id="a",
        text="alpha",
        source_doc="doc1",
        page_num=1,
        section=None,
        metadata={"kind": "x"},
    )


def test_open_sidecar_adapter_supports_records_backend_without_snapshot(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {"chunk_id": "a", "text": "apple finance", "embedding": [2.0, 0.0, 1.0]},
        {"chunk_id": "b", "text": "banana inventory", "embedding": [0.0, 2.0, 0.0]},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    dataset = load_dataset(dataset_path)
    result = build_sidecar_index(
        dataset, tmp_path / "index", bits=4, save_records=False
    )
    assert (result.index_path / SNAPSHOT_FILE_NAME).exists() is False

    external_records = {
        "a": {"chunk_id": "a", "text": "apple finance", "source_doc": "doc-a"},
        "b": {"chunk_id": "b", "text": "banana inventory", "source_doc": "doc-b"},
    }

    adapter = open_sidecar_adapter(
        result.index_path,
        records_backend=external_records,
    )
    results = adapter.query_by_vector(np.array([2.0, 0.0, 1.0], dtype=np.float32), k=1)
    assert results[0].chunk_id == "a"
    assert results[0].source_doc == "doc-a"


def test_open_sidecar_adapter_reads_adapter_config(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {"chunk_id": "a", "text": "apple finance", "embedding": [2.0, 0.0, 1.0]},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    dataset = load_dataset(dataset_path)
    result = build_sidecar_index(
        dataset, tmp_path / "index", bits=4, save_records=False
    )

    adapter_config = result.index_path / "adapter.json"
    adapter_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "neon",
                "options": {
                    "dsn": "${DATABASE_URL}",
                    "table": "public.chunks",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")

    class FakeCursor:
        description = [
            ("chunk_id",),
            ("text",),
            ("source_doc",),
            ("page_num",),
            ("section",),
            ("metadata",),
        ]

        def execute(self, _query, _params):
            return None

        def fetchall(self):
            return [("a", "apple finance", "doc-a", 1, None, {"k": 1})]

        def close(self):
            return None

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def close(self):
            return None

    import turborag.adapters.backends as backends

    monkeypatch.setattr(
        backends, "_load_psycopg_connect", lambda: lambda _dsn: FakeConnection()
    )

    adapter = open_sidecar_adapter(result.index_path)
    hits = adapter.query_by_vector(np.array([2.0, 0.0, 1.0], dtype=np.float32), k=1)
    assert hits[0].chunk_id == "a"
    assert hits[0].source_doc == "doc-a"
