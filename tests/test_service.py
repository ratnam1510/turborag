from __future__ import annotations

import json

import pytest

from click.testing import CliRunner
pytest.importorskip("starlette")
from starlette.testclient import TestClient

from turborag.cli import cli
from turborag.ingest import build_sidecar_index, load_dataset, load_records_snapshot
from turborag.service import create_app


class FakeQueryEmbedder:
    def embed_query(self, text: str):
        mapping = {
            "apple": [2.0, 0.0, 1.0],
            "banana": [0.0, 2.0, 0.0],
        }
        return mapping[text]


def _build_index(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {"chunk_id": "a", "text": "apple finance", "embedding": [2.0, 0.0, 1.0]},
        {"chunk_id": "b", "text": "banana inventory", "embedding": [0.0, 2.0, 0.0]},
        {"chunk_id": "c", "text": "finance banana", "embedding": [0.0, 1.0, 1.0]},
    ]
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    dataset = load_dataset(dataset_path)
    return build_sidecar_index(dataset, tmp_path / "index", bits=4)


def test_service_exposes_index_query_and_ingest(tmp_path):
    _build_index(tmp_path)

    app = create_app(tmp_path / "index", query_embedder=FakeQueryEmbedder())
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    describe = client.get("/index")
    assert describe.status_code == 200
    assert describe.json()["index_size"] == 3
    assert describe.json()["text_query_enabled"] is True

    vector_query = client.post("/query", json={"query_vector": [2.0, 0.0, 1.0], "top_k": 2})
    assert vector_query.status_code == 200
    payload = vector_query.json()
    assert [item["chunk_id"] for item in payload["results"]] == ["a", "c"]

    text_query = client.post("/query", json={"query_text": "apple", "top_k": 1})
    assert text_query.status_code == 200
    assert text_query.json()["results"][0]["chunk_id"] == "a"

    ingest_response = client.post(
        "/ingest",
        json={
            "records": [
                {
                    "chunk_id": "d",
                    "text": "durian finance",
                    "embedding": [3.0, 0.0, 1.0],
                    "source_doc": "fresh.pdf",
                    "page_num": 7,
                    "metadata": {"topic": "fruit"},
                }
            ]
        },
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["index_size"] == 4

    updated_query = client.post("/query", json={"query_vector": [3.0, 0.0, 1.0], "top_k": 1})
    assert updated_query.status_code == 200
    assert updated_query.json()["results"][0]["chunk_id"] == "d"

    snapshot = load_records_snapshot(tmp_path / "index" / "records.jsonl")
    assert snapshot["d"].source_doc == "fresh.pdf"


def test_service_rejects_ambiguous_query_payload(tmp_path):
    _build_index(tmp_path)
    client = TestClient(create_app(tmp_path / "index"))

    response = client.post("/query", json={"query_text": "apple", "query_vector": [1.0, 0.0, 0.0]})
    assert response.status_code == 400
    assert "exactly one" in response.json()["detail"].lower()


def test_service_rejects_text_queries_without_embedder(tmp_path):
    _build_index(tmp_path)
    client = TestClient(create_app(tmp_path / "index"))

    response = client.post("/query", json={"query_text": "apple"})
    assert response.status_code == 400
    assert "text queries are not enabled" in response.json()["detail"].lower()


def test_serve_command_invokes_uvicorn(monkeypatch, tmp_path):
    _build_index(tmp_path)

    captured: dict[str, object] = {}

    def fake_run(app, host: str, port: int):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["serve", "--index", str(tmp_path / "index"), "--host", "0.0.0.0", "--port", "9090"],
    )

    assert result.exit_code == 0, result.output
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9090
    assert captured["app"] is not None
