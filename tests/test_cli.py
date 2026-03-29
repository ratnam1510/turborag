import json

from click.testing import CliRunner

from turborag.cli import cli


def test_cli_can_import_and_query_existing_index(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {"chunk_id": "a", "text": "apple finance", "embedding": [2.0, 0.0, 1.0]},
        {"chunk_id": "b", "text": "banana inventory", "embedding": [0.0, 2.0, 0.0]},
        {"chunk_id": "c", "text": "finance banana", "embedding": [0.0, 1.0, 1.0]},
    ]
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    runner = CliRunner()
    import_result = runner.invoke(
        cli,
        [
            "import-existing-index",
            "--input",
            str(dataset_path),
            "--index",
            str(tmp_path / "index"),
            "--bits",
            "4",
        ],
    )
    assert import_result.exit_code == 0, import_result.output

    query_result = runner.invoke(
        cli,
        [
            "query",
            "--index",
            str(tmp_path / "index"),
            "--query-vector",
            "[2.0, 0.0, 1.0]",
            "--top-k",
            "2",
        ],
    )
    assert query_result.exit_code == 0, query_result.output
    payload = json.loads(query_result.output)
    assert [item["chunk_id"] for item in payload] == ["a", "c"]
