# Benchmarking

## Latest Results

Latest local reproducible runs on `arm64` as of 2026-04-12. The exact row below is from the current native threaded exact path with the 12-bit half-group fused scorer.

### Small Scale (1K vectors, 128-dim, 100 queries, k=10, 4-bit)

| Backend | Recall@10 | MRR | QPS | Memory |
|---|---|---|---|---|
| TurboRAG 4-bit | 1.000 | 1.000 | 6,209 | 0.08 MB |
| Exact float32 | 1.000 | 1.000 | 26,774 | 0.49 MB |
| FAISS Flat | 1.000 | 1.000 | 32,384 | 0.49 MB |
| FAISS HNSW | 1.000 | 1.000 | 23,640 | 0.55 MB |
| FAISS IVF-PQ | 0.990 | 0.990 | 27,438 | < 0.49 MB |

### Large Scale (100K vectors, 384-dim, 200 queries, k=10, 3-bit)

| Backend | Recall@10 | MRR | QPS | Memory |
|---|---|---|---|---|
| TurboRAG exact | 1.000 | 1.000 | 66 | 18.3 MB |
| TurboRAG fast | 0.975 | 0.975 | 131 | 18.3 MB |
| Exact float32 | 1.000 | 1.000 | 73 | 146.5 MB |
| FAISS Flat | 1.000 | 1.000 | 60 | 146.5 MB |
| FAISS HNSW | 0.610 | 0.610 | 771 | 152.6 MB |

TurboRAG maintains perfect recall in exact mode at both scales. The `fast` mode uses a binary sketch head with POPCNT pre-filtering followed by LUT refine, reaching about 2x the throughput of current exact with 97.5% recall on the synthetic 100K fixture.
Repeated exact-only runs on the same 100K fixture now median at 70.98 QPS in the main benchmark environment.

---

TurboRAG now supports three benchmark modes:

- single-backend evaluation of TurboRAG itself,
- explicit TurboRAG `exact` / `fast` / `auto` mode benchmarking,
- side-by-side comparison against exact float search and optional FAISS baselines.

## Install

### From PyPI

```bash
pip install turborag
```

### From Source

```bash
git clone https://github.com/ratnam1510/turborag.git
cd turborag
pip install -e '.[all,dev]'
```

If the environment already has the `faiss` Python module, TurboRAG can also compare against:

- `faiss-flat`
- `faiss-hnsw`
- `faiss-ivfpq`

## Input Files

### Corpus dataset

For side-by-side comparison, provide the original corpus embeddings as JSONL or NPZ.

### Queries file

Each line of `queries.jsonl` must contain:

```json
{"query_id":"q1","query_vector":[0.1,0.2,0.3],"relevant_ids":["doc-1","doc-7"]}
```

## TurboRAG-Only Benchmark

```bash
turborag benchmark \
  --index ./turborag_sidecar \
  --queries ./queries.jsonl \
  --turborag-mode exact \
  --k 10
```

Exact mode uses the native threaded 3-bit scorer by default. Override its thread count with:

```bash
TURBORAG_EXACT_THREADS=8 turborag benchmark \
  --index ./turborag_sidecar \
  --queries ./queries.jsonl \
  --turborag-mode exact \
  --k 10
```

## Side-By-Side Benchmark

Exact baseline only:

```bash
turborag benchmark \
  --index ./turborag_sidecar \
  --queries ./queries.jsonl \
  --turborag-mode exact \
  --dataset ./corpus.jsonl \
  --baseline exact \
  --k 10
```

Exact plus FAISS baselines:

```bash
turborag benchmark \
  --index ./turborag_sidecar \
  --queries ./queries.jsonl \
  --turborag-mode fast \
  --dataset ./corpus.jsonl \
  --baseline exact \
  --baseline faiss-flat \
  --baseline faiss-hnsw \
  --baseline faiss-ivfpq \
  --k 10 \
  --json-output \
  --output ./benchmark-report.json
```

## What The Comparison Shows

For each backend:

- mean Recall@k,
- mean reciprocal rank,
- queries per second,
- wall time,
- and overlap against the reference backend.

If `exact` is present, it becomes the default reference backend.

## Reproducible Local Fixture

Generate a deterministic synthetic benchmark fixture:

```bash
python3 scripts/generate_benchmark_fixture.py --output-dir ./tmp-benchmark
```

That writes:

- `corpus.jsonl`
- `queries.jsonl`
- `manifest.json`

## One-Command Local Comparison

```bash
./scripts/benchmark_compare.sh
```

That script:

1. generates a synthetic benchmark fixture,
2. builds a TurboRAG sidecar index,
3. runs side-by-side benchmarking,
4. writes a JSON artifact.

## Honest Scope

What exists now:

- side-by-side comparison infrastructure,
- exact float baseline,
- optional FAISS baselines,
- JSON benchmark artifacts,
- reproducible synthetic local fixtures.

What still does not exist:

- published real-world benchmark results checked into the repo,
- automated CPU/GPU matrix benchmarking across large external datasets.
