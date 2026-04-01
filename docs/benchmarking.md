# Benchmarking

## Latest Results

### Small Scale (1K vectors, 128-dim, 100 queries, k=10, 4-bit)

| Backend | Recall@10 | MRR | QPS |
|---|---|---|---|
| TurboRAG 4-bit | 1.000 | 1.000 | 6,209 |
| Exact float32 | 1.000 | 1.000 | 26,774 |
| FAISS Flat | 1.000 | 1.000 | 32,384 |
| FAISS HNSW | 1.000 | 1.000 | 23,640 |
| FAISS IVF-PQ | 0.990 | 0.990 | 27,438 |

### Large Scale (100K vectors, 384-dim, 200 queries, k=10, 3-bit)

| Backend | Recall@10 | MRR | QPS |
|---|---|---|---|
| TurboRAG 3-bit | 1.000 | 1.000 | 65 |
| Exact float32 | 1.000 | 1.000 | 240 |
| FAISS Flat | 1.000 | 1.000 | 232 |
| FAISS HNSW | 0.645 | 0.645 | 1,928 |

TurboRAG maintains perfect recall at both scales. At 100K×384 3-bit, the fused byte-triplet C kernel achieves a 4.4x speedup over the previous implementation (15 → 65 QPS). FAISS HNSW trades recall (0.645) for throughput at large scale.

---

TurboRAG now supports two benchmark modes:

- single-backend evaluation of TurboRAG itself,
- side-by-side comparison against exact float search and optional FAISS baselines.

## Install

Core benchmark flow:

```bash
pip install -e '.[dev]'
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
  --k 10
```

## Side-By-Side Benchmark

Exact baseline only:

```bash
turborag benchmark \
  --index ./turborag_sidecar \
  --queries ./queries.jsonl \
  --dataset ./corpus.jsonl \
  --baseline exact \
  --k 10
```

Exact plus FAISS baselines:

```bash
turborag benchmark \
  --index ./turborag_sidecar \
  --queries ./queries.jsonl \
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
- the exact headline benchmark claims from the PDF,
- automated CPU/GPU matrix benchmarking across large external datasets.
