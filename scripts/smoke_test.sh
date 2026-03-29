#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/turborag-smoke.XXXXXX")"
DATASET_PATH="$WORK_DIR/dataset.jsonl"
INDEX_PATH="$WORK_DIR/index"
PORT="${TURBORAG_SMOKE_PORT:-18080}"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd python3
require_cmd curl

cat > "$DATASET_PATH" <<'EOF'
{"chunk_id":"a","text":"apple finance guidance","embedding":[2.0,0.0,1.0]}
{"chunk_id":"b","text":"banana inventory update","embedding":[0.0,2.0,0.0]}
{"chunk_id":"c","text":"finance and banana mix","embedding":[0.0,1.0,1.0]}
EOF

run_cli() {
  PYTHONPATH="$ROOT_DIR/src" python3 -m turborag.cli "$@"
}

assert_top_hit() {
  local json_payload="$1"
  local expected="$2"
  python3 - "$json_payload" "$expected" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
expected = sys.argv[2]
if isinstance(payload, list):
    top = payload[0]["chunk_id"]
else:
    top = payload["results"][0]["chunk_id"]
if top != expected:
    raise SystemExit(f"expected top hit {expected!r}, got {top!r}")
PY
}

echo "==> Building sidecar index"
run_cli import-existing-index --input "$DATASET_PATH" --index "$INDEX_PATH" --bits 4 >/dev/null

echo "==> Checking direct CLI query"
QUERY_OUTPUT="$(run_cli query --index "$INDEX_PATH" --query-vector '[2.0,0.0,1.0]' --top-k 2)"
assert_top_hit "$QUERY_OUTPUT" "a"

echo "==> Starting HTTP service on port $PORT"
PYTHONPATH="$ROOT_DIR/src" python3 -m turborag.cli serve --index "$INDEX_PATH" --host 127.0.0.1 --port "$PORT" >/dev/null 2>&1 &
SERVER_PID="$!"

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

echo "==> Checking service health"
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

echo "==> Checking service query"
SERVICE_QUERY_OUTPUT="$(curl -fsS \
  -X POST "http://127.0.0.1:$PORT/query" \
  -H 'content-type: application/json' \
  -d '{"query_vector":[2.0,0.0,1.0],"top_k":2}')"
assert_top_hit "$SERVICE_QUERY_OUTPUT" "a"

echo "==> Checking incremental ingest"
curl -fsS \
  -X POST "http://127.0.0.1:$PORT/ingest" \
  -H 'content-type: application/json' \
  -d '{"records":[{"chunk_id":"d","text":"durian finance","embedding":[3.0,0.0,1.0],"source_doc":"fresh.pdf","page_num":7,"metadata":{"topic":"fruit"}}]}' \
  >/dev/null

UPDATED_QUERY_OUTPUT="$(curl -fsS \
  -X POST "http://127.0.0.1:$PORT/query" \
  -H 'content-type: application/json' \
  -d '{"query_vector":[3.0,0.0,1.0],"top_k":1}')"
assert_top_hit "$UPDATED_QUERY_OUTPUT" "d"

echo "Smoke test passed."
