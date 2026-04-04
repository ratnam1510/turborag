# Import Existing RAG Data

## Goal

Let a team with an already-running RAG stack build a TurboRAG sidecar from the embeddings and chunk records they already have.

If you want the full production rollout sequence after import, read [docs/current-rag-rollout.md](file:///Users/ratnamshah/turborag/docs/current-rag-rollout.md).

## Install

### From PyPI

```bash
# CLI import and query
pip install turborag

# With HTTP sidecar serving after import
pip install turborag[serve]

# With plug-and-play known DB adapters
pip install turborag[adapters]
```

### From Source

```bash
git clone https://github.com/ratnam1510/turborag.git
cd turborag
pip install -e '.[all,dev]'
```

## Supported Import Formats

### JSONL

Each line should contain one record with at least:

- `chunk_id` or `id`
- `text`, `page_content`, or `content`
- `embedding`

Optional fields:

- `source_doc`
- `page_num`
- `section`
- `metadata`

Example:

```json
{"chunk_id":"c1","text":"Capital expenditure guidance increased.","embedding":[0.1,0.2,0.3],"source_doc":"q3.pdf","page_num":18,"metadata":{"ticker":"ACME"}}
```

### NPZ

The `.npz` file must contain:

- `embeddings`
- `ids`

Optional arrays:

- `texts`
- `source_docs`
- `page_nums`
- `sections`
- `metadata_json`

`metadata_json` should be an array of JSON strings.

## CLI Flow

### Import

```bash
turborag import-existing-index \
  --input ./dataset.jsonl \
  --index ./turborag_sidecar \
  --bits 3
```

### Query By Precomputed Vector

```bash
turborag query \
  --index ./turborag_sidecar \
  --query-vector '[0.1, 0.2, 0.3]' \
  --top-k 5
```

### Configure Existing DB Adapter (One-Time)

If your app hydrates from an existing database, configure adapter auto mode once:

```bash
turborag adapt --index ./turborag_sidecar
```

You can force a backend when needed:

```bash
turborag adapt --index ./turborag_sidecar --backend supabase
```

### Query IDs Only (No Local Hydration)

```bash
turborag query \
  --index ./turborag_sidecar \
  --query-vector '[0.1, 0.2, 0.3]' \
  --top-k 5 \
  --ids-only
```

### Query By Text With A Local Sentence-Transformers Model

```bash
turborag query \
  --index ./turborag_sidecar \
  --query 'What changed in capex guidance?' \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --top-k 5
```

## Python Flow

```python
from turborag.ingest import build_sidecar_index, load_dataset

dataset = load_dataset("./dataset.jsonl")
result = build_sidecar_index(dataset, "./turborag_sidecar", bits=3)
print(result.index_path)
```

## What Gets Written

The sidecar directory contains:

- the TurboRAG index files,
- `records.jsonl` for local hydration when you want the sidecar to be self-contained.

If you do not want TurboRAG to write a local record snapshot, disable it during import and keep using your existing database resolver in application code.

When no snapshot is loaded, TurboRAG can still run as an ID-only sidecar over the same chunk IDs.
