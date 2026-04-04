# turborag (Ruby)

Ruby client for the [TurboRAG](https://github.com/ratnam1510/turborag) compressed vector retrieval API. Zero dependencies — uses stdlib `net/http`.

## Install

```bash
gem install turborag
```

Or in your Gemfile:

```ruby
gem "turborag"
```

## Prerequisites

Start a TurboRAG server:

```bash
# With Docker (no Python needed)
docker run -p 8080:8080 -v ./my_index:/data/index turborag \
  turborag serve --index /data/index --host 0.0.0.0

# Or with pip
pip install turborag[serve]
turborag serve --index ./my_index --port 8080
```

## Usage

```ruby
require "turborag"

client = TurboRAG::Client.new("http://localhost:8080")

# Query by vector
results = client.query(vector: [0.1, 0.2, 0.3], top_k: 5)
results["results"].each do |r|
  puts "#{r["chunk_id"]}  #{r["score"]}  #{r["text"]}"
end

# Query by text (requires --model)
results = client.query_text(text: "What changed in capex guidance?", top_k: 5)

# Batch query
batch = client.query_batch(
  queries: [{ vector: [0.1, 0.2, 0.3] }, { vector: [0.4, 0.5, 0.6] }],
  top_k: 5
)

# Ingest
client.ingest(records: [
  {
    chunk_id: "c1",
    text: "Capital expenditure guidance increased.",
    embedding: [0.1, 0.2, 0.3],
    source_doc: "q3_call.pdf",
  }
])

# Ingest raw text (auto-chunked, requires --model)
client.ingest_text(text: "Full document text...", source_doc: "report.md")

# Health & metrics
health = client.health
info = client.index
metrics = client.metrics
```

## API

| Method | Description |
|---|---|
| `query(vector:, top_k:)` | Search by embedding vector |
| `query_text(text:, top_k:)` | Search by text (requires `--model`) |
| `query_batch(queries:, top_k:)` | Batch vector search |
| `ingest(records:)` | Add records with embeddings |
| `ingest_text(text:, source_doc:)` | Ingest raw text |
| `health` | Health check |
| `index` | Index config and stats |
| `metrics` | Latency and error metrics |
