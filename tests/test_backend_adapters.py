import json

from turborag.adapters.backends import (
    as_chunk_records,
    build_chroma_fetch_records,
    build_neon_fetch_records,
    build_pinecone_fetch_records,
    build_postgres_fetch_records,
    build_qdrant_fetch_records,
    build_supabase_fetch_records,
)


def test_build_postgres_fetch_records_with_connection_returns_ordered_rows():
    class FakeCursor:
        def __init__(self):
            self.description = [
                ("chunk_id",),
                ("text",),
                ("source_doc",),
                ("page_num",),
                ("section",),
                ("metadata",),
            ]
            self.query = None
            self.params = None

        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                ("b", "beta", "doc-b", 2, "s2", {"a": 2}),
                ("a", "alpha", "doc-a", 1, "s1", {"a": 1}),
            ]

        def close(self):
            return None

    class FakeConnection:
        def __init__(self):
            self.cursor_instance = FakeCursor()

        def cursor(self):
            return self.cursor_instance

    connection = FakeConnection()
    fetch_records = build_postgres_fetch_records(
        connection=connection, table="public.chunks"
    )

    rows = fetch_records(["a", "b", "missing"])
    assert [item["chunk_id"] for item in rows] == ["a", "b"]
    assert rows[0]["text"] == "alpha"
    assert 'FROM "public"."chunks"' in connection.cursor_instance.query


def test_build_neon_fetch_records_is_postgres_alias():
    class FakeCursor:
        def __init__(self):
            self.description = [
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
            return [("a", "alpha", None, None, None, {})]

        def close(self):
            return None

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    fetch_records = build_neon_fetch_records(connection=FakeConnection())
    rows = fetch_records(["a"])
    assert rows[0]["chunk_id"] == "a"


def test_build_supabase_fetch_records_reads_data_field():
    class FakeQuery:
        def __init__(self, rows):
            self.rows = rows

        def select(self, _select):
            return self

        def in_(self, _column, _ids):
            return self

        def execute(self):
            return type("Resp", (), {"data": self.rows})()

    class FakeSupabase:
        def __init__(self, rows):
            self.rows = rows

        def table(self, _table):
            return FakeQuery(self.rows)

    client = FakeSupabase(
        [
            {
                "chunk_id": "a",
                "text": "alpha",
                "metadata": {"source_doc": "doc-a", "page_num": 7},
            }
        ]
    )
    fetch_records = build_supabase_fetch_records(client)

    rows = fetch_records(["a", "b"])
    assert rows[0]["chunk_id"] == "a"
    assert rows[0]["source_doc"] == "doc-a"
    assert rows[0]["page_num"] == 7


def test_build_pinecone_fetch_records_reads_vectors_metadata():
    class FakeIndex:
        def fetch(self, ids):
            return {
                "vectors": {
                    ids[0]: {
                        "metadata": {
                            "text": "alpha",
                            "source_doc": "doc-a",
                            "page_num": 9,
                        }
                    }
                }
            }

    fetch_records = build_pinecone_fetch_records(FakeIndex())
    rows = fetch_records(["a", "missing"])
    assert rows[0]["chunk_id"] == "a"
    assert rows[0]["text"] == "alpha"
    assert rows[0]["page_num"] == 9


def test_build_qdrant_fetch_records_reads_point_payload():
    class Point:
        def __init__(self, pid, payload):
            self.id = pid
            self.payload = payload

    class FakeQdrant:
        def retrieve(self, **kwargs):
            assert kwargs["collection_name"] == "chunks"
            return [
                Point("a", {"text": "alpha", "source_doc": "doc-a", "page_num": 3}),
                Point("b", {"chunk_id": "b", "text": "beta"}),
            ]

    fetch_records = build_qdrant_fetch_records(FakeQdrant(), collection_name="chunks")
    rows = fetch_records(["b", "a"])
    assert [item["chunk_id"] for item in rows] == ["b", "a"]
    assert rows[1]["source_doc"] == "doc-a"


def test_build_chroma_fetch_records_reads_get_payload():
    class FakeCollection:
        def get(self, **kwargs):
            assert kwargs["ids"] == ["a", "b"]
            return {
                "ids": [["a", "b"]],
                "documents": [["alpha", "beta"]],
                "metadatas": [[{"source_doc": "doc-a"}, {"page_num": 2}]],
            }

    fetch_records = build_chroma_fetch_records(FakeCollection())
    rows = fetch_records(["a", "b"])
    assert rows[0]["chunk_id"] == "a"
    assert rows[0]["source_doc"] == "doc-a"
    assert rows[1]["page_num"] == 2


def test_as_chunk_records_normalizes_payloads():
    def fetch_records(_ids):
        return [
            {
                "chunk_id": "a",
                "text": "alpha",
                "metadata": json.dumps({"k": "v"}),
            },
            {
                "id": "b",
                "content": "beta",
            },
        ]

    records = as_chunk_records(fetch_records, ["a", "b"])
    assert [record.chunk_id for record in records] == ["a", "b"]
    assert records[1].text == "beta"
