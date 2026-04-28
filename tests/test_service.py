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

    def embed(self, text: str):
        return self.embed_query(text)


def _build_index(tmp_path):
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
    return build_sidecar_index(dataset, tmp_path / "index", bits=4)


# ---------------------------------------------------------------------------
#  Core API tests
# ---------------------------------------------------------------------------


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

    vector_query = client.post(
        "/query", json={"query_vector": [2.0, 0.0, 1.0], "top_k": 2}
    )
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

    updated_query = client.post(
        "/query", json={"query_vector": [3.0, 0.0, 1.0], "top_k": 1}
    )
    assert updated_query.status_code == 200
    assert updated_query.json()["results"][0]["chunk_id"] == "d"

    snapshot = load_records_snapshot(tmp_path / "index" / "records.jsonl")
    assert snapshot["d"].source_doc == "fresh.pdf"


def test_service_rejects_ambiguous_query_payload(tmp_path):
    _build_index(tmp_path)
    client = TestClient(create_app(tmp_path / "index"))

    response = client.post(
        "/query", json={"query_text": "apple", "query_vector": [1.0, 0.0, 0.0]}
    )
    assert response.status_code == 400
    assert "exactly one" in response.json()["detail"].lower()


def test_service_rejects_text_queries_without_embedder(tmp_path):
    _build_index(tmp_path)
    client = TestClient(create_app(tmp_path / "index"))

    response = client.post("/query", json={"query_text": "apple"})
    assert response.status_code == 400
    assert "text queries are not enabled" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
#  CORS tests
# ---------------------------------------------------------------------------


def test_cors_headers_present(tmp_path):
    _build_index(tmp_path)
    client = TestClient(
        create_app(tmp_path / "index", cors_origins=["http://example.com"])
    )

    response = client.options(
        "/query",
        headers={
            "origin": "http://example.com",
            "access-control-request-method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://example.com"


def test_cors_wildcard_default(tmp_path):
    _build_index(tmp_path)
    client = TestClient(create_app(tmp_path / "index"))

    response = client.get("/health", headers={"origin": "http://anywhere.dev"})
    assert response.headers.get("access-control-allow-origin") == "*"


# ---------------------------------------------------------------------------
#  Metrics endpoint
# ---------------------------------------------------------------------------


def test_metrics_endpoint(tmp_path):
    _build_index(tmp_path)
    client = TestClient(create_app(tmp_path / "index"))

    # Fire a query to generate metrics
    client.post("/query", json={"query_vector": [1.0, 0.0, 0.0]})

    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_seconds" in data
    assert "endpoints" in data
    assert "/query" in data["endpoints"]
    assert data["endpoints"]["/query"]["count"] == 1


# ---------------------------------------------------------------------------
#  Batch query endpoint
# ---------------------------------------------------------------------------


def test_batch_query(tmp_path):
    _build_index(tmp_path)
    client = TestClient(create_app(tmp_path / "index"))

    response = client.post(
        "/query/batch",
        json={
            "queries": [
                {"query_vector": [2.0, 0.0, 1.0]},
                {"query_vector": [0.0, 2.0, 0.0]},
            ],
            "top_k": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["batch_count"] == 2
    assert len(data["results"]) == 2


def test_batch_query_rejects_empty(tmp_path):
    _build_index(tmp_path)
    client = TestClient(create_app(tmp_path / "index"))

    response = client.post("/query/batch", json={"queries": []})
    assert response.status_code == 400


def test_query_allows_unhydrated_id_only_response(tmp_path):
    _build_index(tmp_path)
    app = create_app(tmp_path / "index", load_snapshot=False, allow_unhydrated=True)
    client = TestClient(app)

    response = client.post(
        "/query",
        json={
            "query_vector": [2.0, 0.0, 1.0],
            "top_k": 1,
            "hydrate": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["results"][0]["chunk_id"] == "a"
    assert payload["results"][0]["text"] == ""


def test_query_filters_work_with_hydrated_results(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "chunk_id": "a",
            "text": "apple finance",
            "embedding": [2.0, 0.0, 1.0],
            "metadata": {"topic": "finance"},
        },
        {
            "chunk_id": "b",
            "text": "banana inventory",
            "embedding": [0.0, 2.0, 0.0],
            "metadata": {"topic": "inventory"},
        },
        {
            "chunk_id": "c",
            "text": "finance banana",
            "embedding": [0.0, 1.0, 1.0],
            "metadata": {"topic": "finance"},
        },
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    dataset = load_dataset(dataset_path)
    build_sidecar_index(dataset, tmp_path / "index", bits=4)

    client = TestClient(create_app(tmp_path / "index"))
    response = client.post(
        "/query",
        json={
            "query_vector": [2.0, 0.0, 1.0],
            "top_k": 2,
            "filters": {"topic": "finance"},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert [item["chunk_id"] for item in payload["results"]] == ["a", "c"]


def test_ingest_records_persists_metadata_for_filters(tmp_path):
    _build_index(tmp_path)
    client = TestClient(create_app(tmp_path / "index"))

    ingest_response = client.post(
        "/ingest",
        json={
            "records": [
                {
                    "chunk_id": "d",
                    "text": "durian finance",
                    "embedding": [3.0, 0.0, 1.0],
                    "metadata": {"topic": "finance"},
                },
                {
                    "chunk_id": "e",
                    "text": "elderberry legal",
                    "embedding": [0.0, 3.0, 0.0],
                    "metadata": {"topic": "legal"},
                },
            ]
        },
    )
    assert ingest_response.status_code == 200

    response = client.post(
        "/query",
        json={
            "query_vector": [3.0, 0.0, 1.0],
            "top_k": 2,
            "hydrate": False,
            "filters": {"topic": "finance"},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert [item["chunk_id"] for item in payload["results"]] == ["d"]


def test_service_uses_external_records_backend(tmp_path):
    _build_index(tmp_path)

    class ExternalBackend:
        def get(self, ids):
            docs = []
            metas = []
            for chunk_id in ids:
                docs.append(f"external-{chunk_id}")
                metas.append({"source_doc": "external-db", "section": "records"})
            return {"ids": [ids], "documents": [docs], "metadatas": [metas]}

    app = create_app(
        tmp_path / "index",
        load_snapshot=False,
        records_backend=ExternalBackend(),
    )
    client = TestClient(app)

    response = client.post(
        "/query",
        json={
            "query_vector": [2.0, 0.0, 1.0],
            "top_k": 1,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["chunk_id"] == "a"
    assert payload["results"][0]["text"] == "external-a"
    assert payload["results"][0]["source_doc"] == "external-db"

    describe = client.get("/index")
    assert describe.status_code == 200
    assert describe.json()["hydration_source"] == "external_backend"


def test_query_batch_allows_unhydrated_flag(tmp_path):
    _build_index(tmp_path)
    app = create_app(tmp_path / "index", load_snapshot=False, allow_unhydrated=True)
    client = TestClient(app)

    response = client.post(
        "/query/batch",
        json={
            "queries": [
                {"query_vector": [2.0, 0.0, 1.0]},
                {"query_vector": [0.0, 2.0, 0.0]},
            ],
            "top_k": 1,
            "hydrate": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["batch_count"] == 2
    assert payload["results"][0]["results"][0]["text"] == ""


# ---------------------------------------------------------------------------
#  Request ID tracking
# ---------------------------------------------------------------------------


def test_request_id_logging(tmp_path):
    _build_index(tmp_path)
    client = TestClient(create_app(tmp_path / "index"))

    # The request should succeed even with a custom request ID header
    response = client.post(
        "/query",
        json={"query_vector": [1.0, 0.0, 0.0]},
        headers={"x-request-id": "test-rid-123"},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
#  Error handling
# ---------------------------------------------------------------------------


def test_invalid_json_body(tmp_path):
    _build_index(tmp_path)
    client = TestClient(create_app(tmp_path / "index"))

    response = client.post(
        "/query", content=b"not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert "invalid json" in response.json()["detail"].lower()


def test_unknown_query_fields_rejected(tmp_path):
    _build_index(tmp_path)
    client = TestClient(create_app(tmp_path / "index"))

    response = client.post("/query", json={"query_vector": [1.0], "bad_field": True})
    assert response.status_code == 400
    assert "unknown" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
#  CLI serve command
# ---------------------------------------------------------------------------


def test_serve_command_invokes_uvicorn(monkeypatch, tmp_path):
    _build_index(tmp_path)

    captured: dict[str, object] = {}

    def fake_run(app, host: str, port: int, **kwargs):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["workers"] = kwargs.get("workers", 1)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "serve",
            "--index",
            str(tmp_path / "index"),
            "--host",
            "0.0.0.0",
            "--port",
            "9090",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9090
    assert captured["app"] is not None


def test_serve_workers_option(monkeypatch, tmp_path):
    _build_index(tmp_path)

    captured: dict[str, object] = {}

    def fake_run(app, **kwargs):
        captured.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["serve", "--index", str(tmp_path / "index"), "--workers", "4"],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("workers") == 4


def test_serve_supports_memory_flags(monkeypatch, tmp_path):
    _build_index(tmp_path)

    captured: dict[str, object] = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "serve",
            "--index",
            str(tmp_path / "index"),
            "--no-load-snapshot",
            "--require-hydrated",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("host") == "127.0.0.1"
    assert captured.get("port") == 8080
    app = captured["app"]
    assert app.state.turborag.allow_unhydrated is False
    assert app.state.turborag.records == {}


def test_serve_adapter_config_requires_env_value(monkeypatch, tmp_path):
    _build_index(tmp_path)

    adapter_config = tmp_path / "adapter.json"
    adapter_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "neon",
                "options": {"dsn": "${MISSING_DATABASE_URL}"},
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "serve",
            "--index",
            str(tmp_path / "index"),
            "--adapter-config",
            str(adapter_config),
        ],
    )
    assert result.exit_code != 0
    assert "adapter option 'dsn'" in result.output


def test_serve_uses_adapter_config_when_env_is_set(monkeypatch, tmp_path):
    _build_index(tmp_path)

    adapter_config = tmp_path / "adapter.json"
    adapter_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "neon",
                "options": {
                    "dsn": "${DATABASE_URL}",
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
            return [("a", "alpha", "doc-a", 1, None, {})]

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

    app = create_app(
        tmp_path / "index", adapter_config=adapter_config, load_snapshot=False
    )
    client = TestClient(app)
    response = client.post("/query", json={"query_vector": [2.0, 0.0, 1.0], "top_k": 1})
    assert response.status_code == 200
    assert response.json()["results"][0]["chunk_id"] == "a"
