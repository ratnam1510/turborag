#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${1:-$(mktemp -d "${TMPDIR:-/tmp}/turborag-bench.XXXXXX")}"
INDEX_DIR="$WORK_DIR/index"
CORPUS_PATH="$WORK_DIR/corpus.jsonl"
QUERIES_PATH="$WORK_DIR/queries.jsonl"
REPORT_PATH="$WORK_DIR/benchmark-report.json"

mkdir -p "$WORK_DIR"

echo "==> Generating benchmark fixture in $WORK_DIR"
python3 "$ROOT_DIR/scripts/generate_benchmark_fixture.py" --output-dir "$WORK_DIR" >/dev/null

echo "==> Building TurboRAG sidecar index"
PYTHONPATH="$ROOT_DIR/src" python3 -m turborag.cli import-existing-index \
  --input "$CORPUS_PATH" \
  --index "$INDEX_DIR" \
  --bits 4 \
  >/dev/null

BASELINES=("exact")
if python3 - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("faiss") else 1)
PY
then
  BASELINES+=("faiss-flat" "faiss-hnsw" "faiss-ivfpq")
fi

echo "==> Running side-by-side benchmark"
ARGS=()
for baseline in "${BASELINES[@]}"; do
  ARGS+=(--baseline "$baseline")
done

PYTHONPATH="$ROOT_DIR/src" python3 -m turborag.cli benchmark \
  --index "$INDEX_DIR" \
  --queries "$QUERIES_PATH" \
  --dataset "$CORPUS_PATH" \
  "${ARGS[@]}" \
  --k 10 \
  --output "$REPORT_PATH"

echo "Benchmark artifact written to $REPORT_PATH"
