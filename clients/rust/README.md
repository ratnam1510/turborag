# turborag (Rust)

Rust client for the [TurboRAG](https://github.com/ratnam1510/turborag) compressed vector retrieval HTTP API.

## Install

From crates.io:

```toml
[dependencies]
turborag = "0.5.1"
```

For local development against this repo:

```toml
[dependencies]
turborag = { path = "clients/rust" }
```

## Prerequisites

Start a TurboRAG server:

```bash
# With Docker (no Python needed)
docker run -p 8080:8080 -v ./my_index:/data/index turborag \
  turborag serve --index /data/index --host 0.0.0.0

# Or with pip
pip install 'turborag[serve]'
turborag serve --index ./my_index --port 8080
```

## Usage

```rust
use turborag::{QueryRequest, TurboRagClient};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = TurboRagClient::new("http://localhost:8080")?;

    let response = client.query(QueryRequest {
        vector: vec![0.1, 0.2, 0.3],
        top_k: Some(5),
        hydrate: Some(true),
        filters: None,
    })?;

    for result in response.results {
        println!("{} {:.4} {}", result.chunk_id, result.score, result.text);
    }

    Ok(())
}
```

## API

- `query` and `query_text` for vector or text search
- `query_batch` for batch vector search
- `ingest` and `ingest_text` for index updates
- `health`, `index`, and `metrics` for service inspection
