"""Tests for turborag.benchmark – reports, baselines, comparison, and CLI integration."""
from __future__ import annotations

import json

import numpy as np
import pytest

from turborag.benchmark import (
    BenchmarkCase,
    BenchmarkComparison,
    BenchmarkSuite,
    ExactSearchBackend,
    TurboIndexBackend,
    available_baselines,
    build_baselines,
    load_query_cases,
)
from turborag.index import TurboIndex
from turborag.ingest import load_dataset


DIM = 6


def _build_index(rng: np.random.Generator, n: int = 20) -> tuple[TurboIndex, np.ndarray, list[str]]:
    vectors = rng.normal(size=(n, DIM)).astype(np.float32)
    ids = [f"doc-{i}" for i in range(n)]
    index = TurboIndex(dim=DIM, bits=4, seed=7)
    index.add(vectors, ids)
    return index, vectors, ids


def _write_dataset(path, vectors: np.ndarray, ids: list[str]) -> None:
    rows = [
        {"chunk_id": chunk_id, "text": f"text for {chunk_id}", "embedding": vector.astype(float).tolist()}
        for chunk_id, vector in zip(ids, vectors, strict=False)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class TestBenchmarkSuite:
    def test_perfect_recall(self):
        rng = np.random.default_rng(42)
        index, vectors, ids = _build_index(rng)

        cases = [BenchmarkCase(query_id="q0", query_vector=vectors[0], relevant_ids={"doc-0"})]
        report = BenchmarkSuite(cases).run(index, k=5)

        assert report.mean_recall == pytest.approx(1.0)
        assert report.mean_reciprocal_rank == pytest.approx(1.0)
        assert report.k == 5
        assert report.index_size == 20
        assert report.label == "turborag"
        assert report.elapsed_seconds >= 0

    def test_run_backend_exact_baseline(self):
        rng = np.random.default_rng(42)
        _index, vectors, ids = _build_index(rng)

        cases = [BenchmarkCase(query_id="q0", query_vector=vectors[0], relevant_ids={"doc-0"})]
        suite = BenchmarkSuite(cases)
        report = suite.run_backend(ExactSearchBackend(ids=ids, embeddings=vectors), k=5)

        assert report.label == "exact"
        assert report.bits is None
        assert report.mean_recall == pytest.approx(1.0)
        assert report.mean_reciprocal_rank == pytest.approx(1.0)

    def test_compare_reports_against_exact_reference(self):
        rng = np.random.default_rng(42)
        index, vectors, ids = _build_index(rng)

        cases = [
            BenchmarkCase(query_id=f"q{i}", query_vector=vectors[i], relevant_ids={f"doc-{i}"})
            for i in range(5)
        ]
        suite = BenchmarkSuite(cases)
        comparison = suite.compare(
            [
                TurboIndexBackend(index=index),
                ExactSearchBackend(ids=ids, embeddings=vectors),
            ],
            k=5,
        )

        assert isinstance(comparison, BenchmarkComparison)
        assert comparison.reference_label == "exact"
        rows = comparison.rows()
        assert len(rows) == 2
        assert all(row.reference_overlap is not None for row in rows)
        payload = comparison.to_dict()
        assert payload["reference_label"] == "exact"
        assert len(payload["comparison_rows"]) == 2

    def test_empty_suite_raises(self):
        with pytest.raises(ValueError, match="at least one case"):
            BenchmarkSuite(cases=[])


class TestBenchmarkReport:
    def test_summary_is_human_readable(self):
        rng = np.random.default_rng(42)
        index, vectors, ids = _build_index(rng)
        cases = [BenchmarkCase(query_id="q0", query_vector=vectors[0], relevant_ids={"doc-0"})]
        report = BenchmarkSuite(cases).run(index, k=5)

        summary = report.summary()
        assert "Recall@5" in summary
        assert "Reciprocal Rank" in summary
        assert "Queries/sec" in summary

    def test_comparison_summary_is_human_readable(self):
        rng = np.random.default_rng(42)
        index, vectors, ids = _build_index(rng)
        cases = [BenchmarkCase(query_id="q0", query_vector=vectors[0], relevant_ids={"doc-0"})]
        comparison = BenchmarkSuite(cases).compare(
            [TurboIndexBackend(index=index), ExactSearchBackend(ids=ids, embeddings=vectors)],
            k=5,
        )
        summary = comparison.summary()
        assert "Benchmark Comparison" in summary
        assert "Reference: exact" in summary


class TestLoadQueryCases:
    def test_load_from_jsonl(self, tmp_path):
        path = tmp_path / "queries.jsonl"
        lines = [
            json.dumps({"query_id": "q0", "query_vector": [1.0, 0.0], "relevant_ids": ["a", "b"]}),
            json.dumps({"query_id": "q1", "query_vector": [0.0, 1.0], "relevant_ids": ["c"]}),
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cases = load_query_cases(path)
        assert len(cases) == 2
        assert cases[0].query_id == "q0"
        assert cases[0].relevant_ids == {"a", "b"}
        np.testing.assert_array_equal(cases[0].query_vector, [1.0, 0.0])

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="no benchmark cases"):
            load_query_cases(path)


class TestBaselineBuilders:
    def test_build_exact_baseline(self, tmp_path):
        rng = np.random.default_rng(42)
        _index, vectors, ids = _build_index(rng, n=8)
        dataset_path = tmp_path / "dataset.jsonl"
        _write_dataset(dataset_path, vectors, ids)

        dataset = load_dataset(dataset_path)
        baselines = build_baselines(dataset, ["exact"])

        assert len(baselines) == 1
        assert baselines[0].label == "exact"

    def test_available_baselines_includes_exact(self):
        assert "exact" in available_baselines()

    def test_build_faiss_flat_when_available(self, tmp_path):
        pytest.importorskip("faiss")

        rng = np.random.default_rng(42)
        _index, vectors, ids = _build_index(rng, n=32)
        dataset_path = tmp_path / "dataset.jsonl"
        _write_dataset(dataset_path, vectors, ids)
        dataset = load_dataset(dataset_path)

        baselines = build_baselines(dataset, ["faiss-flat"])
        assert baselines[0].label == "faiss-flat"


class TestBenchmarkCLI:
    def test_benchmark_command(self, tmp_path):
        from click.testing import CliRunner
        from turborag.cli import cli

        dataset_path = tmp_path / "data.jsonl"
        rows = [
            {"chunk_id": f"doc-{i}", "text": f"text {i}", "embedding": [float(i == j) for j in range(4)]}
            for i in range(4)
        ]
        dataset_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        runner = CliRunner()
        import_result = runner.invoke(
            cli,
            ["import-existing-index", "--input", str(dataset_path), "--index", str(tmp_path / "index"), "--bits", "4"],
        )
        assert import_result.exit_code == 0, import_result.output

        queries_path = tmp_path / "queries.jsonl"
        queries = [
            {"query_id": "q0", "query_vector": [1.0, 0.0, 0.0, 0.0], "relevant_ids": ["doc-0"]},
            {"query_id": "q1", "query_vector": [0.0, 1.0, 0.0, 0.0], "relevant_ids": ["doc-1"]},
        ]
        queries_path.write_text("\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8")

        bench_result = runner.invoke(
            cli,
            ["benchmark", "--index", str(tmp_path / "index"), "--queries", str(queries_path), "--k", "3"],
        )
        assert bench_result.exit_code == 0, bench_result.output
        assert "Recall@3" in bench_result.output

        bench_json = runner.invoke(
            cli,
            [
                "benchmark",
                "--index",
                str(tmp_path / "index"),
                "--queries",
                str(queries_path),
                "--k",
                "3",
                "--json-output",
            ],
        )
        assert bench_json.exit_code == 0, bench_json.output
        payload = json.loads(bench_json.output)
        assert payload["k"] == 3
        assert payload["label"] == "turborag"

    def test_side_by_side_cli_writes_artifact(self, tmp_path):
        from click.testing import CliRunner
        from turborag.cli import cli

        dataset_path = tmp_path / "data.jsonl"
        rows = [
            {"chunk_id": f"doc-{i}", "text": f"text {i}", "embedding": [float(i == j) for j in range(4)]}
            for i in range(4)
        ]
        dataset_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        runner = CliRunner()
        import_result = runner.invoke(
            cli,
            ["import-existing-index", "--input", str(dataset_path), "--index", str(tmp_path / "index"), "--bits", "4"],
        )
        assert import_result.exit_code == 0, import_result.output

        queries_path = tmp_path / "queries.jsonl"
        queries = [
            {"query_id": "q0", "query_vector": [1.0, 0.0, 0.0, 0.0], "relevant_ids": ["doc-0"]},
            {"query_id": "q1", "query_vector": [0.0, 1.0, 0.0, 0.0], "relevant_ids": ["doc-1"]},
        ]
        queries_path.write_text("\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8")

        report_path = tmp_path / "report.json"
        result = runner.invoke(
            cli,
            [
                "benchmark",
                "--index",
                str(tmp_path / "index"),
                "--queries",
                str(queries_path),
                "--dataset",
                str(dataset_path),
                "--baseline",
                "exact",
                "--output",
                str(report_path),
                "--json-output",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["reference_label"] == "exact"
        assert report_path.exists()
        assert len(payload["reports"]) == 2
