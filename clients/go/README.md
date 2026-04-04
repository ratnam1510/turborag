# turborag (Go)

Go client for the [TurboRAG](https://github.com/ratnam1510/turborag) compressed vector retrieval API. Zero dependencies — uses `net/http`.

## Install

```bash
go get github.com/ratnam1510/turborag/clients/go
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

```go
package main

import (
	"context"
	"fmt"
	"log"

	turborag "github.com/ratnam1510/turborag/clients/go"
)

func main() {
	client := turborag.New("http://localhost:8080")
	ctx := context.Background()

	// Query by vector
	results, err := client.Query(ctx, []float64{0.1, 0.2, 0.3}, 5)
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range results.Results {
		fmt.Printf("%s  %.4f  %s\n", r.ChunkID, r.Score, r.Text)
	}

	// Query by text (requires --model)
	textResults, err := client.QueryText(ctx, "What changed in capex guidance?", 5)

	// Batch query
	batch, err := client.QueryBatch(ctx, [][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	}, 5)

	// Ingest
	_, err = client.Ingest(ctx, []turborag.IngestRecord{
		{
			ChunkID:   "c1",
			Text:      "Capital expenditure guidance increased.",
			Embedding: []float64{0.1, 0.2, 0.3},
			SourceDoc: "q3_call.pdf",
		},
	})

	// Health & info
	health, _ := client.Health(ctx)
	info, _ := client.Index(ctx)
	fmt.Println(health.Status, info.IndexSize)
}
```

## API

| Method | Description |
|---|---|
| `Query(ctx, vector, topK)` | Search by embedding vector |
| `QueryText(ctx, text, topK)` | Search by text (requires `--model`) |
| `QueryBatch(ctx, vectors, topK)` | Batch vector search |
| `Ingest(ctx, records)` | Add records with embeddings |
| `IngestText(ctx, text, sourceDoc)` | Ingest raw text |
| `Health(ctx)` | Health check |
| `Index(ctx)` | Index config and stats |
