# Installation

TurboRAG can be installed from PyPI or from source. Both methods support the same optional extras.

## From PyPI

```bash
# Core (compression, indexing, CLI)
pip install turborag

# With HTTP service
pip install turborag[serve]

# With MCP agent server
pip install turborag[mcp]

# With local embedding support (sentence-transformers)
pip install turborag[embed]

# With graph retrieval (networkx, Leiden clustering)
pip install turborag[graph]

# With PDF/document ingestion (pdfminer, tiktoken)
pip install turborag[ingest]

# Everything
pip install turborag[all]

# Everything + dev/test dependencies
pip install turborag[all,dev]
```

## From Source (Git Clone)

```bash
git clone https://github.com/ratnam1510/turborag.git
cd turborag

# Core only
pip install -e .

# With specific extras
pip install -e '.[serve]'
pip install -e '.[mcp]'
pip install -e '.[embed]'
pip install -e '.[graph]'
pip install -e '.[ingest]'

# Everything
pip install -e '.[all]'

# Development (includes pytest)
pip install -e '.[all,dev]'
```

## Requirements

- Python ≥ 3.11
- A C compiler (gcc or clang) is recommended for the C scoring kernel — TurboRAG auto-compiles it on first use. If no compiler is available, it falls back to a pure Python implementation.

## Extras Reference

| Extra | What it adds | When you need it |
|---|---|---|
| *(core)* | numpy, scipy, click | Always installed |
| `serve` | starlette, uvicorn | Running `turborag serve` (HTTP API) |
| `mcp` | mcp | Running `turborag mcp` (MCP stdio server) |
| `embed` | sentence-transformers | Local text embedding with `--model` |
| `graph` | networkx, leidenalg, python-igraph | Graph-augmented retrieval |
| `ingest` | pdfminer.six, tiktoken | PDF extraction, token-aware chunking |
| `all` | All of the above | Full feature set |
| `dev` | pytest | Running the test suite |

## Verify Installation

```bash
# Check the CLI works
turborag --help

# Check C kernel availability
python -c "from turborag._cscore_wrapper import is_available; print('C kernel:', 'available' if is_available() else 'not available (using Python fallback)')"

# Run tests (requires dev extra)
pytest
```

## Docker

No local Python installation needed — run TurboRAG directly from Docker:

```bash
# Build
docker build -t turborag .

# Serve an index
docker run -p 8080:8080 -v ./my_index:/data/index turborag \
  turborag serve --index /data/index --host 0.0.0.0

# Run MCP server over stdio
docker run -i turborag turborag mcp --index /data/index

# Query an index
docker run -v ./my_index:/data/index turborag \
  turborag query --index /data/index --query-vector '[0.1, 0.2, 0.3]' --top-k 5
```

## What Each Component Needs

### CLI (import, query, describe, benchmark)

```bash
pip install turborag
```

### HTTP Service

```bash
pip install turborag[serve]
turborag serve --index ./my_index --host 0.0.0.0 --port 8080
```

Add `--model sentence-transformers/all-MiniLM-L6-v2` for text query support (requires `turborag[serve,embed]`).

### MCP Server (Claude Desktop, agent integration)

```bash
pip install turborag[mcp]
turborag mcp --index ./my_index
```

### Graph Retrieval

```bash
pip install turborag[graph]
```

### Document Chunking & PDF Ingestion

```bash
pip install turborag[ingest]
```

### Benchmarking with FAISS Baselines

```bash
pip install turborag[dev]
pip install faiss-cpu  # or faiss-gpu
turborag benchmark --index ./my_index --queries ./queries.jsonl --baseline exact --baseline faiss-flat
```
