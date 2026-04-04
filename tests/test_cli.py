import json

from click.testing import CliRunner

from turborag.cli import cli


def _parse_json_output(output: str):
    text = output.strip()
    first_object = text.find("{", 0)
    first_array = text.find("[\n", 0)
    starts = [idx for idx in (first_object, first_array) if idx >= 0]
    if not starts:
        first_object = text.find("{")
        first_array = text.find("[", 0)
        starts = [idx for idx in (first_object, first_array) if idx >= 0]
    if not starts:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    start = min(starts)
    return json.loads(text[start:])


def test_cli_can_import_and_query_existing_index(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {"chunk_id": "a", "text": "apple finance", "embedding": [2.0, 0.0, 1.0]},
        {"chunk_id": "b", "text": "banana inventory", "embedding": [0.0, 2.0, 0.0]},
        {"chunk_id": "c", "text": "finance banana", "embedding": [0.0, 1.0, 1.0]},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

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
    payload = _parse_json_output(query_result.output)
    assert [item["chunk_id"] for item in payload] == ["a", "c"]


def test_cli_query_ids_only_returns_compact_payload(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {"chunk_id": "a", "text": "apple finance", "embedding": [2.0, 0.0, 1.0]},
        {"chunk_id": "b", "text": "banana inventory", "embedding": [0.0, 2.0, 0.0]},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

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
            "1",
            "--ids-only",
        ],
    )
    assert query_result.exit_code == 0, query_result.output
    payload = _parse_json_output(query_result.output)
    assert payload[0]["chunk_id"] == "a"
    assert set(payload[0].keys()) == {"chunk_id", "score"}


def test_cli_adapt_set_and_show(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    runner = CliRunner()

    set_result = runner.invoke(
        cli,
        [
            "adapt",
            "set",
            "neon",
            "--index",
            str(index_dir),
            "--option",
            "dsn=${DATABASE_URL}",
            "--option",
            "table=public.chunks",
        ],
    )
    assert set_result.exit_code == 0, set_result.output
    payload = _parse_json_output(set_result.output)
    assert payload["backend"] == "neon"

    show_result = runner.invoke(
        cli,
        [
            "adapt",
            "show",
            "--index",
            str(index_dir),
        ],
    )
    assert show_result.exit_code == 0, show_result.output
    shown = _parse_json_output(show_result.output)
    assert shown["backend"] == "neon"
    assert shown["options"]["dsn"] == "${DATABASE_URL}"


def test_cli_adapt_set_validates_required_options(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "adapt",
            "set",
            "pinecone",
            "--index",
            str(index_dir),
            "--option",
            "api_key=${PINECONE_API_KEY}",
        ],
    )
    assert result.exit_code != 0
    assert "index_name" in result.output


def test_cli_adapt_remove(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = index_dir / "adapter.json"
    adapter_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "neon",
                "options": {"dsn": "${DATABASE_URL}"},
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "adapt",
            "remove",
            "--index",
            str(index_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert adapter_path.exists() is False


def test_cli_adapt_demo(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["adapt", "demo", "supabase"])
    assert result.exit_code == 0, result.output
    assert "turborag adapt set supabase" in result.output


def test_cli_adapt_supabase_reads_env_defaults(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "secret-key")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "adapt",
            "supabase",
            "--index",
            str(index_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["backend"] == "supabase"
    assert payload["options"]["url"] == "${SUPABASE_URL}"
    assert payload["options"]["key"] == "${SUPABASE_KEY}"


def test_cli_adapt_neon_reads_env_defaults(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "adapt",
            "neon",
            "--index",
            str(index_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["backend"] == "neon"
    assert payload["options"]["dsn"] == "${DATABASE_URL}"


def test_cli_adapt_qdrant_requires_url_or_path(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "adapt",
            "qdrant",
            "--index",
            str(index_dir),
        ],
    )
    assert result.exit_code != 0
    assert "Qdrant requires either" in result.output


def test_cli_adapt_auto_detects_supabase(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "secret-key")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "adapt",
            "--index",
            str(index_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["backend"] == "supabase"


def test_cli_adapt_auto_can_force_backend(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "adapt",
            "--index",
            str(index_dir),
            "--backend",
            "neon",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["backend"] == "neon"


def test_cli_adapt_auto_uses_current_directory_index(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "secret-key")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "adapt",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["backend"] == "supabase"
    assert (tmp_path / "adapter.json").exists()


def test_cli_adapt_auto_errors_without_detectable_backend(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_PATH", raising=False)
    monkeypatch.delenv("CHROMA_PATH", raising=False)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "adapt",
            "--index",
            str(index_dir),
        ],
    )
    assert result.exit_code != 0
    assert "Could not detect a backend" in result.output
