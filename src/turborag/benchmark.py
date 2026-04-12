"""Retrieval benchmark and side-by-side comparison utilities."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .index import TurboIndex
from .ingest import ImportedDataset

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]

DEFAULT_BASELINES = ("exact", "faiss-flat", "faiss-hnsw", "faiss-ivfpq")
WARMUP_QUERIES = 5


class SearchBackend(Protocol):
    label: str
    index_size: int
    bits: int | None

    def search(self, query: FloatVector, k: int) -> list[tuple[str, float]]: ...


@dataclass(slots=True)
class BenchmarkCase:
    """A single evaluation query with its ground-truth relevant IDs."""

    query_id: str
    query_vector: FloatVector
    relevant_ids: set[str]


@dataclass(slots=True)
class CaseResult:
    """Per-query metrics from a benchmark run."""

    query_id: str
    recall: float
    reciprocal_rank: float
    retrieved_ids: list[str]


@dataclass(slots=True)
class BenchmarkReport:
    """Aggregate results of a benchmark suite run."""

    label: str
    k: int
    case_results: list[CaseResult]
    elapsed_seconds: float
    index_size: int
    bits: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def mean_recall(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(cr.recall for cr in self.case_results) / len(self.case_results)

    @property
    def mean_reciprocal_rank(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(cr.reciprocal_rank for cr in self.case_results) / len(self.case_results)

    @property
    def queries_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return float("inf")
        return len(self.case_results) / self.elapsed_seconds

    def summary(self) -> str:
        bits = "n/a" if self.bits is None else str(self.bits)
        lines = [
            f"Benchmark Report [{self.label}]  (k={self.k}, bits={bits}, "
            f"index_size={self.index_size}, queries={len(self.case_results)})",
            f"  Mean Recall@{self.k}: {self.mean_recall:.4f}",
            f"  Mean Reciprocal Rank: {self.mean_reciprocal_rank:.4f}",
            f"  Queries/sec: {self.queries_per_second:.2f}",
            f"  Wall time: {self.elapsed_seconds:.3f}s",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "k": self.k,
            "bits": self.bits,
            "index_size": self.index_size,
            "num_queries": len(self.case_results),
            "mean_recall": round(self.mean_recall, 6),
            "mean_reciprocal_rank": round(self.mean_reciprocal_rank, 6),
            "queries_per_second": None if math.isinf(self.queries_per_second) else round(self.queries_per_second, 6),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "extra": self.extra,
            "cases": [
                {
                    "query_id": cr.query_id,
                    "recall": round(cr.recall, 6),
                    "reciprocal_rank": round(cr.reciprocal_rank, 6),
                    "retrieved_ids": cr.retrieved_ids,
                }
                for cr in self.case_results
            ],
        }


@dataclass(slots=True)
class ComparisonRow:
    label: str
    mean_recall: float
    mean_reciprocal_rank: float
    queries_per_second: float
    elapsed_seconds: float
    bits: int | None = None
    reference_overlap: float | None = None
    delta_recall_vs_reference: float | None = None
    delta_mrr_vs_reference: float | None = None
    qps_ratio_vs_reference: float | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "bits": self.bits,
            "mean_recall": round(self.mean_recall, 6),
            "mean_reciprocal_rank": round(self.mean_reciprocal_rank, 6),
            "queries_per_second": None if math.isinf(self.queries_per_second) else round(self.queries_per_second, 6),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "reference_overlap": None if self.reference_overlap is None else round(self.reference_overlap, 6),
            "delta_recall_vs_reference": None if self.delta_recall_vs_reference is None else round(self.delta_recall_vs_reference, 6),
            "delta_mrr_vs_reference": None if self.delta_mrr_vs_reference is None else round(self.delta_mrr_vs_reference, 6),
            "qps_ratio_vs_reference": None if self.qps_ratio_vs_reference is None or math.isinf(self.qps_ratio_vs_reference) else round(self.qps_ratio_vs_reference, 6),
        }


@dataclass(slots=True)
class BenchmarkComparison:
    k: int
    reports: list[BenchmarkReport]
    reference_label: str | None = None

    def summary(self) -> str:
        rows = self.rows()
        lines = [
            f"Benchmark Comparison  (k={self.k}, queries={len(self.reports[0].case_results) if self.reports else 0})",
            f"Reference: {self.reference_label or 'none'}",
            "label           recall      mrr         qps         overlap     d_recall    d_mrr",
        ]
        for row in rows:
            overlap = "n/a" if row.reference_overlap is None else f"{row.reference_overlap:.4f}"
            delta_recall = "n/a" if row.delta_recall_vs_reference is None else f"{row.delta_recall_vs_reference:+.4f}"
            delta_mrr = "n/a" if row.delta_mrr_vs_reference is None else f"{row.delta_mrr_vs_reference:+.4f}"
            lines.append(
                f"{row.label:<15} {row.mean_recall:<10.4f} {row.mean_reciprocal_rank:<10.4f} "
                f"{row.queries_per_second:<11.2f} {overlap:<11} {delta_recall:<11} {delta_mrr}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "reference_label": self.reference_label,
            "reports": [report.to_dict() for report in self.reports],
            "comparison_rows": [row.to_dict() for row in self.rows()],
        }

    def rows(self) -> list[ComparisonRow]:
        if not self.reports:
            return []
        by_label = {report.label: report for report in self.reports}
        reference = by_label.get(self.reference_label) if self.reference_label else None
        rows: list[ComparisonRow] = []
        for report in self.reports:
            row = ComparisonRow(
                label=report.label,
                bits=report.bits,
                mean_recall=report.mean_recall,
                mean_reciprocal_rank=report.mean_reciprocal_rank,
                queries_per_second=report.queries_per_second,
                elapsed_seconds=report.elapsed_seconds,
            )
            if reference is not None:
                row.reference_overlap = _mean_reference_overlap(reference, report)
                row.delta_recall_vs_reference = report.mean_recall - reference.mean_recall
                row.delta_mrr_vs_reference = report.mean_reciprocal_rank - reference.mean_reciprocal_rank
                row.qps_ratio_vs_reference = _safe_ratio(report.queries_per_second, reference.queries_per_second)
            rows.append(row)
        return rows


@dataclass(slots=True)
class TurboIndexBackend:
    index: TurboIndex
    label: str = "turborag"
    mode: str = "auto"

    @property
    def index_size(self) -> int:
        return len(self.index)

    @property
    def bits(self) -> int | None:
        return self.index.bits

    def search(self, query: FloatVector, k: int) -> list[tuple[str, float]]:
        return self.index.search(query, k=k, mode=self.mode)


@dataclass(slots=True)
class ExactSearchBackend:
    ids: list[str]
    embeddings: FloatMatrix
    normalize: bool = True
    label: str = "exact"
    bits: int | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.embeddings, dtype=np.float32)
        self.embeddings = _normalize_rows(matrix) if self.normalize else matrix

    @property
    def index_size(self) -> int:
        return int(self.embeddings.shape[0])

    def search(self, query: FloatVector, k: int) -> list[tuple[str, float]]:
        vector = np.asarray(query, dtype=np.float32)
        if self.normalize:
            vector = _normalize_rows(vector.reshape(1, -1))[0]
        scores = self.embeddings @ vector
        local_k = min(k, len(scores))
        if local_k <= 0:
            return []
        if local_k == len(scores):
            indices = np.arange(len(scores))
        else:
            indices = np.argpartition(scores, -local_k)[-local_k:]
        ranked = sorted(
            ((self.ids[int(index)], float(scores[int(index)])) for index in indices),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:k]


@dataclass(slots=True)
class FaissSearchBackend:
    ids: list[str]
    index: object
    normalize: bool = True
    label: str = "faiss"
    bits: int | None = None

    @property
    def index_size(self) -> int:
        return len(self.ids)

    def search(self, query: FloatVector, k: int) -> list[tuple[str, float]]:
        vector = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self.normalize:
            vector = _normalize_rows(vector)
        distances, indices = self.index.search(vector, k)
        hits: list[tuple[str, float]] = []
        for index, score in zip(indices[0], distances[0], strict=False):
            if int(index) < 0:
                continue
            hits.append((self.ids[int(index)], float(score)))
        return hits


class BenchmarkSuite:
    """Run retrieval evaluation against TurboRAG and comparison backends."""

    def __init__(self, cases: Sequence[BenchmarkCase]) -> None:
        if not cases:
            raise ValueError("benchmark suite requires at least one case")
        self.cases = list(cases)

    def run(self, index: TurboIndex, k: int = 10) -> BenchmarkReport:
        return self.run_backend(TurboIndexBackend(index=index), k=k)

    def run_backend(self, backend: SearchBackend, k: int = 10) -> BenchmarkReport:
        case_results: list[CaseResult] = []

        warmup_count = min(WARMUP_QUERIES, len(self.cases))
        for case in self.cases[:warmup_count]:
            backend.search(case.query_vector, k=k)

        t0 = time.perf_counter()

        for case in self.cases:
            hits = backend.search(case.query_vector, k=k)
            retrieved_ids = [chunk_id for chunk_id, _score in hits]

            if case.relevant_ids:
                recall = len(set(retrieved_ids) & case.relevant_ids) / len(case.relevant_ids)
            else:
                recall = 0.0

            rr = 0.0
            for rank, chunk_id in enumerate(retrieved_ids, start=1):
                if chunk_id in case.relevant_ids:
                    rr = 1.0 / rank
                    break

            case_results.append(
                CaseResult(
                    query_id=case.query_id,
                    recall=recall,
                    reciprocal_rank=rr,
                    retrieved_ids=retrieved_ids,
                )
            )

        elapsed = time.perf_counter() - t0
        return BenchmarkReport(
            label=backend.label,
            k=k,
            case_results=case_results,
            elapsed_seconds=elapsed,
            index_size=backend.index_size,
            bits=backend.bits,
        )

    def compare(
        self,
        backends: Sequence[SearchBackend],
        *,
        k: int = 10,
        reference_label: str | None = None,
    ) -> BenchmarkComparison:
        if not backends:
            raise ValueError("at least one backend is required for comparison")
        reports = [self.run_backend(backend, k=k) for backend in backends]
        resolved_reference = reference_label or _default_reference_label(reports)
        return BenchmarkComparison(k=k, reports=reports, reference_label=resolved_reference)


def load_query_cases(path: str | Path) -> list[BenchmarkCase]:
    """Load benchmark cases from a JSONL file."""
    source = Path(path)
    cases: list[BenchmarkCase] = []
    with source.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("query_id")
            vec = obj.get("query_vector")
            rids = obj.get("relevant_ids")
            if qid is None or vec is None or rids is None:
                raise ValueError(
                    f"line {lineno}: each case needs query_id, query_vector, relevant_ids"
                )
            cases.append(
                BenchmarkCase(
                    query_id=str(qid),
                    query_vector=np.asarray(vec, dtype=np.float32),
                    relevant_ids=set(rids),
                )
            )
    if not cases:
        raise ValueError(f"no benchmark cases found in {source}")
    return cases


def write_benchmark_artifact(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def build_baselines(
    dataset: ImportedDataset,
    baseline_names: Sequence[str],
    *,
    normalize: bool = True,
) -> list[SearchBackend]:
    requested = list(baseline_names)
    if not requested:
        return []
    ids = dataset.ids
    embeddings = np.asarray(dataset.embeddings, dtype=np.float32)
    backends: list[SearchBackend] = []
    for name in requested:
        if name == "exact":
            backends.append(ExactSearchBackend(ids=ids, embeddings=embeddings, normalize=normalize))
            continue
        if name.startswith("faiss-"):
            backends.append(build_faiss_backend(ids=ids, embeddings=embeddings, baseline=name, normalize=normalize))
            continue
        raise ValueError(f"unsupported baseline: {name}")
    return backends


def build_faiss_backend(
    *,
    ids: Sequence[str],
    embeddings: FloatMatrix,
    baseline: str,
    normalize: bool = True,
) -> FaissSearchBackend:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            f"{baseline} requires the `faiss` Python module to be installed."
        ) from exc

    matrix = np.asarray(embeddings, dtype=np.float32)
    if normalize:
        matrix = _normalize_rows(matrix)

    dim = int(matrix.shape[1])
    if baseline == "faiss-flat":
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        return FaissSearchBackend(ids=list(ids), index=index, normalize=normalize, label=baseline)

    if baseline == "faiss-hnsw":
        index = faiss.IndexHNSWFlat(dim, min(32, max(8, dim // 8)), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 32
        index.add(matrix)
        return FaissSearchBackend(ids=list(ids), index=index, normalize=normalize, label=baseline)

    if baseline == "faiss-ivfpq":
        n = int(matrix.shape[0])
        if n < 8:
            raise ValueError("faiss-ivfpq requires at least 8 corpus vectors")
        nlist = max(1, min(int(math.sqrt(n)), max(1, n // 39), n))
        m = _pq_subquantizers(dim)
        bits_per_code = 4 if n < 9984 else 8
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits_per_code, faiss.METRIC_INNER_PRODUCT)
        index.train(matrix)
        index.add(matrix)
        index.nprobe = min(8, nlist)
        return FaissSearchBackend(ids=list(ids), index=index, normalize=normalize, label=baseline)

    raise ValueError(f"unsupported FAISS baseline: {baseline}")


def available_baselines() -> list[str]:
    names = ["exact"]
    try:
        import faiss  # noqa: F401
    except ImportError:
        return names
    return list(DEFAULT_BASELINES)


def _normalize_rows(matrix: FloatMatrix) -> FloatMatrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)


def _pq_subquantizers(dim: int) -> int:
    for candidate in (16, 12, 10, 8, 6, 4, 3, 2, 1):
        if candidate <= dim and dim % candidate == 0:
            return candidate
    return 1


def _default_reference_label(reports: Sequence[BenchmarkReport]) -> str:
    labels = {report.label for report in reports}
    if "exact" in labels:
        return "exact"
    return reports[0].label


def _mean_reference_overlap(reference: BenchmarkReport, report: BenchmarkReport) -> float:
    reference_cases = {case.query_id: case for case in reference.case_results}
    overlaps: list[float] = []
    for case in report.case_results:
        reference_case = reference_cases.get(case.query_id)
        if reference_case is None:
            continue
        ref_ids = reference_case.retrieved_ids
        if not ref_ids:
            overlaps.append(0.0)
            continue
        overlap = len(set(case.retrieved_ids) & set(ref_ids)) / len(ref_ids)
        overlaps.append(overlap)
    if not overlaps:
        return 0.0
    return sum(overlaps) / len(overlaps)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator
