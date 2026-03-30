from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import numpy as np

from .embeddings import SentenceTransformerEmbedder
from .ingest import build_sidecar_index, load_dataset, open_sidecar_adapter


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
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--index", "index_path", required=True, type=click.Path(path_type=Path))
@click.option("--format", "input_format", type=click.Choice(["auto", "jsonl", "npz"]), default="auto", show_default=True)
@click.option("--bits", default=3, show_default=True, type=int)
@click.option("--shard-size", default=100_000, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--no-normalize", is_flag=True, default=False, help="Disable query/document vector normalization before compression.")
@click.option("--no-record-snapshot", is_flag=True, default=False, help="Skip writing records.jsonl alongside the sidecar index.")
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
        "records_path": None if result.records_path is None else str(result.records_path),
        "count": result.count,
        "dim": result.dim,
        "bits": result.bits,
        "shard_size": result.shard_size,
    }
    click.echo(json.dumps(payload, indent=2))


@cli.command("query")
@click.option("--index", "index_path", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--query", "query_text", type=str, default=None, help="Query text to embed locally.")
@click.option("--model", "model_name", type=str, default=None, help="sentence-transformers model used for --query text.")
@click.option("--query-vector", type=str, default=None, help="Inline JSON array representing a precomputed query vector.")
@click.option("--query-vector-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Path to a JSON file containing a query vector array.")
@click.option("--top-k", default=5, show_default=True, type=int)
def query(
    index_path: Path,
    query_text: str | None,
    model_name: str | None,
    query_vector: str | None,
    query_vector_file: Path | None,
    top_k: int,
) -> None:
    """Query a TurboRAG sidecar index using text or a precomputed vector."""
    mode_count = sum(value is not None for value in (query_text, query_vector, query_vector_file))
    if mode_count != 1:
        raise click.UsageError("Provide exactly one of --query, --query-vector, or --query-vector-file")

    if query_text is not None:
        if not model_name:
            raise click.UsageError("--model is required when using --query text")
        embedder = SentenceTransformerEmbedder(model_name=model_name)
        adapter = open_sidecar_adapter(index_path, query_embedder=embedder)
        results = adapter.query(query_text, k=top_k)
    else:
        adapter = open_sidecar_adapter(index_path)
        vector = _load_query_vector(query_vector=query_vector, query_vector_file=query_vector_file)
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
@click.option("--index", "index_path", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def describe_index(index_path: Path) -> None:
    """Show the configuration of an existing TurboRAG sidecar index."""
    config_path = index_path / "config.json"
    if not config_path.exists():
        raise click.ClickException(f"missing config file: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["index_path"] = str(index_path)
    payload["records_snapshot"] = str(index_path / "records.jsonl") if (index_path / "records.jsonl").exists() else None

    from ._cscore_wrapper import is_available as c_kernel_available
    payload["c_kernel_available"] = c_kernel_available()

    click.echo(json.dumps(payload, indent=2))


@cli.command("benchmark")
@click.option("--index", "index_path", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--queries", "queries_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dataset", "dataset_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Corpus embeddings dataset (JSONL/NPZ) used to build side-by-side baselines.")
@click.option("--baseline", "baselines", multiple=True, type=click.Choice(["exact", "faiss-flat", "faiss-hnsw", "faiss-ivfpq"]), help="Optional side-by-side baseline backend. Can be passed multiple times.")
@click.option("--output", "output_path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Optional path to write the JSON benchmark artifact.")
@click.option("--k", default=10, show_default=True, type=int, help="Number of results per query for recall/MRR calculation.")
@click.option("--json-output", is_flag=True, default=False, help="Emit full JSON report instead of human-readable summary.")
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
    from .benchmark import BenchmarkSuite, TurboIndexBackend, build_baselines, load_query_cases, write_benchmark_artifact
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
            [turbo_backend, *build_baselines(dataset, baselines, normalize=index.normalize)],
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
@click.option("--index", "index_path", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--host", default="127.0.0.1", show_default=True, type=str)
@click.option("--port", default=8080, show_default=True, type=int)
@click.option("--workers", default=1, show_default=True, type=int, help="Number of uvicorn workers.")
@click.option("--model", "model_name", type=str, default=None, help="Optional sentence-transformers model used for text query support.")
@click.option("--cors-origins", type=str, default="*", show_default=True, help="Comma-separated list of allowed CORS origins.")
def serve(index_path: Path, host: str, port: int, workers: int, model_name: str | None, cors_origins: str) -> None:
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
    app = create_app(index_path, query_embedder=query_embedder, cors_origins=origins)
    uvicorn.run(app, host=host, port=port, workers=workers)


@cli.command("mcp")
@click.option("--index", "index_path", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
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


def _load_query_vector(*, query_vector: str | None, query_vector_file: Path | None) -> np.ndarray:
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
