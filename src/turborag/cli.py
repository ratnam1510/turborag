from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import click
import numpy as np

from .adapters.config import (
    ADAPTER_CONFIG_SCHEMA_VERSION,
    default_adapter_config_path,
    normalize_adapter_backend,
    save_adapter_config,
    validate_adapter_config,
)
from .embeddings import SentenceTransformerEmbedder
from .ingest import build_sidecar_index, load_dataset, open_sidecar_adapter
from .index import TurboIndex


LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    default="INFO",
    show_default=True,
    envvar="TURBORAG_LOG_LEVEL",
    help="Set the logging verbosity.",
)
@click.option(
    "--log-format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Log output format.",
)
def cli(log_level: str, log_format: str) -> None:
    """TurboRAG command-line tools."""
    level = getattr(logging, log_level.upper())
    if log_format == "json":
        fmt = '{"time":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","msg":"%(message)s"}'
    else:
        fmt = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)


@cli.command("import-existing-index")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--index", "index_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--format",
    "input_format",
    type=click.Choice(["auto", "jsonl", "npz"]),
    default="auto",
    show_default=True,
)
@click.option("--bits", default=3, show_default=True, type=int)
@click.option("--shard-size", default=100_000, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option(
    "--no-normalize",
    is_flag=True,
    default=False,
    help="Disable query/document vector normalization before compression.",
)
@click.option(
    "--no-record-snapshot",
    is_flag=True,
    default=False,
    help="Skip writing records.jsonl alongside the sidecar index.",
)
def import_existing_index(
    input_path: Path,
    index_path: Path,
    input_format: str,
    bits: int,
    shard_size: int,
    seed: int,
    no_normalize: bool,
    no_record_snapshot: bool,
) -> None:
    """Build a TurboRAG sidecar index from existing embeddings."""
    click.echo(f"Loading dataset from {input_path}...")
    dataset = load_dataset(input_path, format=input_format)
    click.echo(f"  → {len(dataset.ids)} vectors, dim={dataset.dim}")

    with click.progressbar(length=100, label="Building index") as bar:
        result = build_sidecar_index(
            dataset,
            index_path,
            bits=bits,
            shard_size=shard_size,
            seed=seed,
            normalize=not no_normalize,
            save_records=not no_record_snapshot,
        )
        bar.update(100)

    payload = {
        "index_path": str(result.index_path),
        "records_path": None
        if result.records_path is None
        else str(result.records_path),
        "count": result.count,
        "dim": result.dim,
        "bits": result.bits,
        "shard_size": result.shard_size,
    }
    click.echo(json.dumps(payload, indent=2))


@cli.command("query")
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--query", "query_text", type=str, default=None, help="Query text to embed locally."
)
@click.option(
    "--model",
    "model_name",
    type=str,
    default=None,
    help="sentence-transformers model used for --query text.",
)
@click.option(
    "--query-vector",
    type=str,
    default=None,
    help="Inline JSON array representing a precomputed query vector.",
)
@click.option(
    "--query-vector-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a JSON file containing a query vector array.",
)
@click.option("--top-k", default=5, show_default=True, type=int)
@click.option(
    "--ids-only",
    is_flag=True,
    default=False,
    help="Return only chunk_id and score. Skips local record hydration and minimizes memory.",
)
def query(
    index_path: Path,
    query_text: str | None,
    model_name: str | None,
    query_vector: str | None,
    query_vector_file: Path | None,
    top_k: int,
    ids_only: bool,
) -> None:
    """Query a TurboRAG sidecar index using text or a precomputed vector."""
    mode_count = sum(
        value is not None for value in (query_text, query_vector, query_vector_file)
    )
    if mode_count != 1:
        raise click.UsageError(
            "Provide exactly one of --query, --query-vector, or --query-vector-file"
        )

    if ids_only and query_text is not None:
        if not model_name:
            raise click.UsageError("--model is required when using --query text")
        embedder = SentenceTransformerEmbedder(model_name=model_name)
        vector = np.asarray(embedder.embed_query(query_text), dtype=np.float32)
        hits = TurboIndex.open(str(index_path)).search(vector, k=top_k)
        payload = [{"chunk_id": chunk_id, "score": score} for chunk_id, score in hits]
        click.echo(json.dumps(payload, indent=2))
        return

    if ids_only:
        vector = _load_query_vector(
            query_vector=query_vector, query_vector_file=query_vector_file
        )
        hits = TurboIndex.open(str(index_path)).search(vector, k=top_k)
        payload = [{"chunk_id": chunk_id, "score": score} for chunk_id, score in hits]
        click.echo(json.dumps(payload, indent=2))
        return

    if query_text is not None:
        if not model_name:
            raise click.UsageError("--model is required when using --query text")
        embedder = SentenceTransformerEmbedder(model_name=model_name)
        adapter = open_sidecar_adapter(index_path, query_embedder=embedder)
        results = adapter.query(query_text, k=top_k)
    else:
        adapter = open_sidecar_adapter(index_path)
        vector = _load_query_vector(
            query_vector=query_vector, query_vector_file=query_vector_file
        )
        results = adapter.query_by_vector(vector, k=top_k)

    payload = [
        {
            "chunk_id": result.chunk_id,
            "score": result.score,
            "source_doc": result.source_doc,
            "page_num": result.page_num,
            "text": result.text,
            "explanation": result.explanation,
        }
        for result in results
    ]
    click.echo(json.dumps(payload, indent=2))


@cli.command("describe-index")
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def describe_index(index_path: Path) -> None:
    """Show the configuration of an existing TurboRAG sidecar index."""
    config_path = index_path / "config.json"
    if not config_path.exists():
        raise click.ClickException(f"missing config file: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["index_path"] = str(index_path)
    payload["records_snapshot"] = (
        str(index_path / "records.jsonl")
        if (index_path / "records.jsonl").exists()
        else None
    )

    from ._cscore_wrapper import is_available as c_kernel_available

    payload["c_kernel_available"] = c_kernel_available()

    click.echo(json.dumps(payload, indent=2))


@cli.command("benchmark")
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--queries",
    "queries_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Corpus embeddings dataset (JSONL/NPZ) used to build side-by-side baselines.",
)
@click.option(
    "--baseline",
    "baselines",
    multiple=True,
    type=click.Choice(["exact", "faiss-flat", "faiss-hnsw", "faiss-ivfpq"]),
    help="Optional side-by-side baseline backend. Can be passed multiple times.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write the JSON benchmark artifact.",
)
@click.option(
    "--k",
    default=10,
    show_default=True,
    type=int,
    help="Number of results per query for recall/MRR calculation.",
)
@click.option(
    "--json-output",
    is_flag=True,
    default=False,
    help="Emit full JSON report instead of human-readable summary.",
)
def benchmark(
    index_path: Path,
    queries_path: Path,
    dataset_path: Path | None,
    baselines: tuple[str, ...],
    output_path: Path | None,
    k: int,
    json_output: bool,
) -> None:
    """Run a retrieval benchmark against a TurboRAG index.

    The queries file is JSONL where each line has:
    query_id (str), query_vector (list[float]), relevant_ids (list[str]).
    """
    from .benchmark import (
        BenchmarkSuite,
        TurboIndexBackend,
        build_baselines,
        load_query_cases,
        write_benchmark_artifact,
    )
    from .ingest import load_dataset
    from .index import TurboIndex

    index = TurboIndex.open(str(index_path))
    cases = load_query_cases(queries_path)
    suite = BenchmarkSuite(cases)
    turbo_backend = TurboIndexBackend(index=index)

    if baselines and dataset_path is None:
        raise click.UsageError("--dataset is required when using --baseline")

    if baselines:
        dataset = load_dataset(dataset_path)
        comparison = suite.compare(
            [
                turbo_backend,
                *build_baselines(dataset, baselines, normalize=index.normalize),
            ],
            k=k,
        )
        payload = comparison.to_dict()
        if output_path is not None:
            write_benchmark_artifact(output_path, payload)
        if json_output:
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo(comparison.summary())
    else:
        report = suite.run(index, k=k)
        payload = report.to_dict()
        if output_path is not None:
            write_benchmark_artifact(output_path, payload)
        if json_output:
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo(report.summary())


@cli.command("serve")
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--host", default="127.0.0.1", show_default=True, type=str)
@click.option("--port", default=8080, show_default=True, type=int)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=int,
    help="Number of uvicorn workers.",
)
@click.option(
    "--model",
    "model_name",
    type=str,
    default=None,
    help="Optional sentence-transformers model used for text query support.",
)
@click.option(
    "--cors-origins",
    type=str,
    default="*",
    show_default=True,
    help="Comma-separated list of allowed CORS origins.",
)
@click.option(
    "--adapter-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to adapter config JSON generated by `turborag adapt`.",
)
@click.option(
    "--no-load-snapshot",
    is_flag=True,
    default=False,
    help="Do not load records.jsonl into memory at startup. Use this with external record stores or ID-only query mode.",
)
@click.option(
    "--require-hydrated",
    is_flag=True,
    default=False,
    help="Drop hits that cannot be hydrated from local snapshot or external backend.",
)
def serve(
    index_path: Path,
    host: str,
    port: int,
    workers: int,
    model_name: str | None,
    cors_origins: str,
    adapter_config: Path | None,
    no_load_snapshot: bool,
    require_hydrated: bool,
) -> None:
    """Serve a TurboRAG sidecar index over HTTP."""
    try:
        import starlette  # noqa: F401
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "Service support requires the 'serve' extra: pip install turborag[serve]"
        )

    query_embedder = None
    if model_name:
        try:
            query_embedder = SentenceTransformerEmbedder(model_name=model_name)
        except ImportError as exc:
            raise click.ClickException(str(exc)) from exc

    from .service import create_app

    origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
    try:
        app = create_app(
            index_path,
            query_embedder=query_embedder,
            adapter_config=adapter_config,
            load_snapshot=not no_load_snapshot,
            allow_unhydrated=not require_hydrated,
            cors_origins=origins,
        )
    except (ValueError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc

    uvicorn.run(app, host=host, port=port, workers=workers)


@cli.group("adapt", invoke_without_command=True)
@click.option(
    "--index",
    "auto_index_path",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Index path used by auto mode (when no backend subcommand is provided).",
)
@click.option(
    "--config",
    "auto_config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional config target path used by auto mode.",
)
@click.option(
    "--backend",
    "auto_backend",
    type=click.Choice(
        [
            "postgres",
            "postgresql",
            "neon",
            "supabase",
            "supabase_postgres",
            "pinecone",
            "qdrant",
            "chroma",
        ],
        case_sensitive=False,
    ),
    default=None,
    help="Force a backend in auto mode. If omitted, backend is detected from environment.",
)
@click.option(
    "--option",
    "auto_adapter_options",
    multiple=True,
    help="Optional key=value overrides used by auto mode.",
)
@click.pass_context
def adapt(
    ctx: click.Context,
    auto_index_path: Path | None,
    auto_config_path: Path | None,
    auto_backend: str | None,
    auto_adapter_options: tuple[str, ...],
) -> None:
    """Configure plug-and-play adapters for existing databases."""

    if ctx.invoked_subcommand is not None:
        return

    index_path = _resolve_auto_index_path(auto_index_path)
    backend, options_payload = _detect_auto_backend_and_options(auto_backend)
    options_payload = _merge_options(options_payload, auto_adapter_options)

    _save_adapter_payload(
        index_path=index_path,
        config_path=auto_config_path,
        backend=backend,
        options_payload=options_payload,
    )


def _save_adapter_payload(
    *,
    index_path: Path,
    config_path: Path | None,
    backend: str,
    options_payload: dict[str, Any],
) -> None:
    raw_payload = {
        "schema_version": ADAPTER_CONFIG_SCHEMA_VERSION,
        "backend": backend,
        "options": options_payload,
    }
    try:
        config_payload = validate_adapter_config(raw_payload, resolve_env=False)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    target = (
        config_path
        if config_path is not None
        else default_adapter_config_path(index_path)
    )
    save_adapter_config(config_payload, target)

    click.echo(
        json.dumps(
            {
                "status": "ok",
                "index_path": str(index_path),
                "adapter_config": str(target),
                "backend": config_payload["backend"],
                "options": config_payload["options"],
            },
            indent=2,
        )
    )


def _parse_kv_pairs(options: tuple[str, ...]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for option in options:
        if "=" not in option:
            raise click.UsageError(
                f"Invalid --option value {option!r}. Use key=value format."
            )
        key, raw_value = option.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            raise click.UsageError("Adapter option keys must not be empty")
        parsed[key] = value
    return parsed


def _merge_options(base: dict[str, Any], extra: tuple[str, ...]) -> dict[str, Any]:
    overrides = _parse_kv_pairs(extra)
    merged = dict(base)
    merged.update(overrides)
    return merged


def _resolve_auto_index_path(index_path: Path | None) -> Path:
    if index_path is not None:
        if not index_path.exists():
            raise click.ClickException(f"index path not found: {index_path}")
        return index_path

    cwd = Path.cwd()
    if (cwd / "config.json").exists():
        return cwd

    default_path = Path("./turborag_sidecar")
    if default_path.exists():
        return default_path

    raise click.ClickException(
        "Could not auto-detect index path. Run `turborag adapt --index <path>` or execute from an index directory."
    )


def _detect_auto_backend_and_options(
    forced_backend: str | None,
) -> tuple[str, dict[str, Any]]:
    if forced_backend is not None:
        backend = normalize_adapter_backend(forced_backend)
        return _default_options_for_backend_from_env(backend)

    detected = _detect_backend_candidates_from_env()
    if not detected:
        raise click.ClickException(
            "Could not detect a backend from environment. Set one of: SUPABASE_URL+SUPABASE_KEY, DATABASE_URL, PINECONE_API_KEY+PINECONE_INDEX_NAME, QDRANT_URL/QDRANT_PATH, CHROMA_PATH; or pass --backend."
        )

    return detected[0]


def _detect_backend_candidates_from_env() -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for backend in ("supabase", "pinecone", "qdrant", "neon", "chroma"):
        try:
            _, options = _default_options_for_backend_from_env(backend)
            candidates.append((backend, options))
        except click.ClickException:
            continue
    return candidates


def _default_options_for_backend_from_env(
    backend: str,
) -> tuple[str, dict[str, Any]]:
    normalized = normalize_adapter_backend(backend)

    if normalized == "supabase":
        return (
            normalized,
            {
                "url": _required_value_or_env_placeholder(
                    None, ("SUPABASE_URL",), field="url", backend="supabase"
                ),
                "key": _required_value_or_env_placeholder(
                    None, ("SUPABASE_KEY",), field="key", backend="supabase"
                ),
                "table": _optional_value_or_default(
                    None, ("SUPABASE_TABLE",), "chunks"
                ),
                "id_column": "chunk_id",
                "text_column": "text",
                "source_doc_column": "source_doc",
                "page_num_column": "page_num",
                "section_column": "section",
                "metadata_column": "metadata",
            },
        )

    if normalized in {"neon", "postgres", "supabase_postgres"}:
        return (
            normalized,
            {
                "dsn": _required_value_or_env_placeholder(
                    None,
                    ("DATABASE_URL", "NEON_DATABASE_URL", "POSTGRES_DSN"),
                    field="dsn",
                    backend=normalized,
                ),
                "table": _optional_value_or_default(None, ("NEON_TABLE",), "chunks"),
                "id_column": "chunk_id",
                "text_column": "text",
                "source_doc_column": "source_doc",
                "page_num_column": "page_num",
                "section_column": "section",
                "metadata_column": "metadata",
            },
        )

    if normalized == "pinecone":
        namespace_value = _optional_value_or_default(None, ("PINECONE_NAMESPACE",), "")
        payload: dict[str, Any] = {
            "api_key": _required_value_or_env_placeholder(
                None,
                ("PINECONE_API_KEY",),
                field="api_key",
                backend="pinecone",
            ),
            "index_name": _required_value_or_env_placeholder(
                None,
                ("PINECONE_INDEX_NAME",),
                field="index_name",
                backend="pinecone",
            ),
            "text_key": "text",
            "source_doc_key": "source_doc",
            "page_num_key": "page_num",
            "section_key": "section",
        }
        if namespace_value:
            payload["namespace"] = namespace_value
        return (
            normalized,
            payload,
        )

    if normalized == "qdrant":
        url_value = _optional_value_or_default(None, ("QDRANT_URL",), "")
        path_value = _optional_value_or_default(None, ("QDRANT_PATH",), "")
        if not url_value and not path_value:
            raise click.ClickException(
                "Qdrant requires either QDRANT_URL or QDRANT_PATH in auto mode."
            )
        payload: dict[str, Any] = {
            "collection_name": _optional_value_or_default(
                None, ("QDRANT_COLLECTION",), "chunks"
            ),
            "text_key": "text",
            "source_doc_key": "source_doc",
            "page_num_key": "page_num",
            "section_key": "section",
        }
        if url_value:
            payload["url"] = url_value
        if path_value:
            payload["path"] = path_value
        api_key = _optional_value_or_default(None, ("QDRANT_API_KEY",), "")
        if api_key:
            payload["api_key"] = api_key
        return normalized, payload

    if normalized == "chroma":
        chroma_path = _optional_value_or_default(None, ("CHROMA_PATH",), "")
        if not chroma_path and not Path("./chroma").exists():
            raise click.ClickException("Chroma auto mode requires CHROMA_PATH.")
        return (
            normalized,
            {
                "path": chroma_path or "./chroma",
                "collection_name": _optional_value_or_default(
                    None, ("CHROMA_COLLECTION",), "chunks"
                ),
            },
        )

    raise click.ClickException(f"Unsupported auto backend: {backend}")


def _required_value_or_env_placeholder(
    explicit_value: str | None,
    env_names: tuple[str, ...],
    *,
    field: str,
    backend: str,
) -> str:
    if explicit_value is not None and explicit_value.strip() != "":
        return explicit_value.strip()

    for env_name in env_names:
        value = os.getenv(env_name)
        if value is not None and value.strip() != "":
            return f"${{{env_name}}}"

    env_hint = ", ".join(env_names)
    raise click.ClickException(
        f"Missing {backend} option '{field}'. Provide --{field.replace('_', '-')} or set one of: {env_hint}"
    )


def _optional_value_or_default(
    explicit_value: str | None,
    env_names: tuple[str, ...],
    default_value: str,
) -> str:
    if explicit_value is not None and explicit_value.strip() != "":
        return explicit_value.strip()
    for env_name in env_names:
        value = os.getenv(env_name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default_value


@adapt.command("set")
@click.argument(
    "backend",
    type=click.Choice(
        [
            "postgres",
            "postgresql",
            "neon",
            "supabase",
            "supabase_postgres",
            "pinecone",
            "qdrant",
            "chroma",
        ],
        case_sensitive=False,
    ),
)
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="TurboRAG sidecar index directory.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for the adapter config file (defaults to <index>/adapter.json).",
)
@click.option(
    "--option",
    "adapter_options",
    multiple=True,
    help=(
        "Adapter option as key=value. Use env references like dsn=${DATABASE_URL}. "
        "Pass multiple --option flags."
    ),
)
def adapt_set(
    backend: str,
    index_path: Path,
    config_path: Path | None,
    adapter_options: tuple[str, ...],
) -> None:
    """Create or update an adapter config for a known backend."""

    normalized_backend = normalize_adapter_backend(backend)
    options_payload = _parse_kv_pairs(adapter_options)
    _save_adapter_payload(
        index_path=index_path,
        config_path=config_path,
        backend=normalized_backend,
        options_payload=options_payload,
    )


@adapt.command("supabase")
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
@click.option("--url", default=None, help="Supabase project URL.")
@click.option("--key", default=None, help="Supabase service or anon key.")
@click.option("--table", default=None, help="Chunk table name.")
@click.option("--id-column", default="chunk_id", show_default=True)
@click.option("--text-column", default="text", show_default=True)
@click.option("--source-doc-column", default="source_doc", show_default=True)
@click.option("--page-num-column", default="page_num", show_default=True)
@click.option("--section-column", default="section", show_default=True)
@click.option("--metadata-column", default="metadata", show_default=True)
@click.option("--option", "adapter_options", multiple=True)
def adapt_supabase(
    index_path: Path,
    config_path: Path | None,
    url: str | None,
    key: str | None,
    table: str | None,
    id_column: str,
    text_column: str,
    source_doc_column: str,
    page_num_column: str,
    section_column: str,
    metadata_column: str,
    adapter_options: tuple[str, ...],
) -> None:
    """Autoconfigure Supabase adapter from env or explicit flags."""

    options_payload = {
        "url": _required_value_or_env_placeholder(
            url, ("SUPABASE_URL",), field="url", backend="supabase"
        ),
        "key": _required_value_or_env_placeholder(
            key, ("SUPABASE_KEY",), field="key", backend="supabase"
        ),
        "table": _optional_value_or_default(table, ("SUPABASE_TABLE",), "chunks"),
        "id_column": id_column,
        "text_column": text_column,
        "source_doc_column": source_doc_column,
        "page_num_column": page_num_column,
        "section_column": section_column,
        "metadata_column": metadata_column,
    }
    options_payload = _merge_options(options_payload, adapter_options)
    _save_adapter_payload(
        index_path=index_path,
        config_path=config_path,
        backend="supabase",
        options_payload=options_payload,
    )


@adapt.command("neon")
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
@click.option("--dsn", default=None, help="Neon/Postgres DSN.")
@click.option("--table", default=None, help="Chunk table name.")
@click.option("--id-column", default="chunk_id", show_default=True)
@click.option("--text-column", default="text", show_default=True)
@click.option("--source-doc-column", default="source_doc", show_default=True)
@click.option("--page-num-column", default="page_num", show_default=True)
@click.option("--section-column", default="section", show_default=True)
@click.option("--metadata-column", default="metadata", show_default=True)
@click.option("--option", "adapter_options", multiple=True)
def adapt_neon(
    index_path: Path,
    config_path: Path | None,
    dsn: str | None,
    table: str | None,
    id_column: str,
    text_column: str,
    source_doc_column: str,
    page_num_column: str,
    section_column: str,
    metadata_column: str,
    adapter_options: tuple[str, ...],
) -> None:
    """Autoconfigure Neon adapter from env or explicit flags."""

    options_payload = {
        "dsn": _required_value_or_env_placeholder(
            dsn,
            ("DATABASE_URL", "NEON_DATABASE_URL", "POSTGRES_DSN"),
            field="dsn",
            backend="neon",
        ),
        "table": _optional_value_or_default(table, ("NEON_TABLE",), "chunks"),
        "id_column": id_column,
        "text_column": text_column,
        "source_doc_column": source_doc_column,
        "page_num_column": page_num_column,
        "section_column": section_column,
        "metadata_column": metadata_column,
    }
    options_payload = _merge_options(options_payload, adapter_options)
    _save_adapter_payload(
        index_path=index_path,
        config_path=config_path,
        backend="neon",
        options_payload=options_payload,
    )


@adapt.command("pinecone")
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
@click.option("--api-key", default=None)
@click.option("--index-name", default=None)
@click.option("--namespace", default=None)
@click.option("--text-key", default="text", show_default=True)
@click.option("--source-doc-key", default="source_doc", show_default=True)
@click.option("--page-num-key", default="page_num", show_default=True)
@click.option("--section-key", default="section", show_default=True)
@click.option("--option", "adapter_options", multiple=True)
def adapt_pinecone(
    index_path: Path,
    config_path: Path | None,
    api_key: str | None,
    index_name: str | None,
    namespace: str | None,
    text_key: str,
    source_doc_key: str,
    page_num_key: str,
    section_key: str,
    adapter_options: tuple[str, ...],
) -> None:
    """Autoconfigure Pinecone adapter from env or explicit flags."""

    options_payload = {
        "api_key": _required_value_or_env_placeholder(
            api_key,
            ("PINECONE_API_KEY",),
            field="api_key",
            backend="pinecone",
        ),
        "index_name": _required_value_or_env_placeholder(
            index_name,
            ("PINECONE_INDEX_NAME",),
            field="index_name",
            backend="pinecone",
        ),
        "text_key": text_key,
        "source_doc_key": source_doc_key,
        "page_num_key": page_num_key,
        "section_key": section_key,
    }
    namespace_value = _optional_value_or_default(
        namespace,
        ("PINECONE_NAMESPACE",),
        "",
    )
    if namespace_value:
        options_payload["namespace"] = namespace_value
    options_payload = _merge_options(options_payload, adapter_options)
    _save_adapter_payload(
        index_path=index_path,
        config_path=config_path,
        backend="pinecone",
        options_payload=options_payload,
    )


@adapt.command("qdrant")
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
@click.option("--url", default=None)
@click.option("--path", default=None)
@click.option("--api-key", default=None)
@click.option("--collection-name", default=None)
@click.option("--text-key", default="text", show_default=True)
@click.option("--source-doc-key", default="source_doc", show_default=True)
@click.option("--page-num-key", default="page_num", show_default=True)
@click.option("--section-key", default="section", show_default=True)
@click.option("--option", "adapter_options", multiple=True)
def adapt_qdrant(
    index_path: Path,
    config_path: Path | None,
    url: str | None,
    path: str | None,
    api_key: str | None,
    collection_name: str | None,
    text_key: str,
    source_doc_key: str,
    page_num_key: str,
    section_key: str,
    adapter_options: tuple[str, ...],
) -> None:
    """Autoconfigure Qdrant adapter from env or explicit flags."""

    options_payload = {
        "collection_name": _optional_value_or_default(
            collection_name,
            ("QDRANT_COLLECTION",),
            "chunks",
        ),
        "text_key": text_key,
        "source_doc_key": source_doc_key,
        "page_num_key": page_num_key,
        "section_key": section_key,
    }

    url_value = _optional_value_or_default(url, ("QDRANT_URL",), "")
    path_value = _optional_value_or_default(path, ("QDRANT_PATH",), "")
    if not url_value and not path_value:
        raise click.ClickException(
            "Qdrant requires either --url/`QDRANT_URL` or --path/`QDRANT_PATH`."
        )
    if url_value:
        options_payload["url"] = url_value
    if path_value:
        options_payload["path"] = path_value

    api_key_value = _optional_value_or_default(api_key, ("QDRANT_API_KEY",), "")
    if api_key_value:
        options_payload["api_key"] = api_key_value

    options_payload = _merge_options(options_payload, adapter_options)
    _save_adapter_payload(
        index_path=index_path,
        config_path=config_path,
        backend="qdrant",
        options_payload=options_payload,
    )


@adapt.command("chroma")
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
@click.option("--path", default=None)
@click.option("--collection-name", default=None)
@click.option("--option", "adapter_options", multiple=True)
def adapt_chroma(
    index_path: Path,
    config_path: Path | None,
    path: str | None,
    collection_name: str | None,
    adapter_options: tuple[str, ...],
) -> None:
    """Autoconfigure Chroma adapter from env or explicit flags."""

    options_payload = {
        "path": _optional_value_or_default(path, ("CHROMA_PATH",), "./chroma"),
        "collection_name": _optional_value_or_default(
            collection_name,
            ("CHROMA_COLLECTION",),
            "chunks",
        ),
    }
    options_payload = _merge_options(options_payload, adapter_options)
    _save_adapter_payload(
        index_path=index_path,
        config_path=config_path,
        backend="chroma",
        options_payload=options_payload,
    )


@adapt.command("demo")
@click.argument(
    "backend",
    type=click.Choice(
        ["neon", "supabase", "pinecone", "qdrant", "chroma", "postgres"],
        case_sensitive=False,
    ),
)
def adapt_demo(backend: str) -> None:
    """Print an example command for a known backend."""

    key = normalize_adapter_backend(backend)
    examples = {
        "neon": "turborag adapt set neon --index ./turborag_sidecar --option dsn=${DATABASE_URL} --option table=public.chunks",
        "postgres": "turborag adapt set postgres --index ./turborag_sidecar --option dsn=${DATABASE_URL} --option table=public.chunks",
        "supabase": "turborag adapt set supabase --index ./turborag_sidecar --option url=${SUPABASE_URL} --option key=${SUPABASE_KEY} --option table=chunks",
        "pinecone": "turborag adapt set pinecone --index ./turborag_sidecar --option api_key=${PINECONE_API_KEY} --option index_name=my-index --option namespace=prod",
        "qdrant": "turborag adapt set qdrant --index ./turborag_sidecar --option url=http://localhost:6333 --option collection_name=chunks",
        "chroma": "turborag adapt set chroma --index ./turborag_sidecar --option path=./chroma --option collection_name=chunks",
    }

    example = examples.get(key)
    if example is None:
        raise click.ClickException(f"No demo available for backend: {backend}")
    click.echo(example)


@adapt.command("show")
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
)
def adapt_show(index_path: Path, config_path: Path | None) -> None:
    """Show current adapter configuration for an index."""

    target = (
        config_path
        if config_path is not None
        else default_adapter_config_path(index_path)
    )
    if not target.exists():
        raise click.ClickException(f"adapter config not found: {target}")

    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["path"] = str(target)
    click.echo(json.dumps(payload, indent=2))


@adapt.command("remove")
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
def adapt_remove(index_path: Path, config_path: Path | None) -> None:
    """Remove adapter configuration for an index."""

    target = (
        config_path
        if config_path is not None
        else default_adapter_config_path(index_path)
    )
    if not target.exists():
        raise click.ClickException(f"adapter config not found: {target}")

    target.unlink()
    click.echo(json.dumps({"status": "ok", "removed": str(target)}, indent=2))


@cli.command("mcp")
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def mcp(index_path: Path) -> None:
    """Start a local MCP server over stdio for TurboRAG retrieval.

    This allows any MCP-capable client (e.g. Claude Desktop) to use
    TurboRAG as a tool without hosting a separate HTTP server.
    Requires: pip install turborag[mcp]
    """
    try:
        import mcp  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "MCP support requires the 'mcp' extra: pip install turborag[mcp]"
        )

    import asyncio
    from .mcp_server import run_server

    asyncio.run(run_server(str(index_path)))


def _load_query_vector(
    *, query_vector: str | None, query_vector_file: Path | None
) -> np.ndarray:
    if query_vector_file is not None:
        raw = query_vector_file.read_text(encoding="utf-8")
    elif query_vector is not None:
        raw = query_vector
    else:
        raise click.UsageError("A query vector source is required")

    payload: Any = json.loads(raw)
    vector = np.asarray(payload, dtype=np.float32)
    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector[0]
    if vector.ndim != 1:
        raise click.UsageError("Query vectors must be a one-dimensional JSON array")
    return vector


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
