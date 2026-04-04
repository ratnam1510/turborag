// TurboRAG Go Client Benchmark
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"time"

	turborag "github.com/ratnam1510/turborag/clients/go"
)

func makeVector(dim int, seed float64) []float64 {
	values := make([]float64, dim)
	var sumSq float64
	for i := 0; i < dim; i++ {
		val := math.Sin(seed+float64(i)*0.07) + math.Cos(seed*0.11+float64(i)*0.03)
		values[i] = val
		sumSq += val * val
	}
	norm := math.Sqrt(sumSq)
	if norm == 0 {
		norm = 1
	}
	for i := range values {
		values[i] /= norm
	}
	return values
}

func main() {
	baseURL := os.Getenv("TURBORAG_URL")
	if baseURL == "" {
		baseURL = "http://127.0.0.1:8080"
	}

	queryCount := 50
	if val := os.Getenv("TURBORAG_NUM_QUERIES"); val != "" {
		queryCount, _ = strconv.Atoi(val)
	}

	topK := 10
	if val := os.Getenv("TURBORAG_TOP_K"); val != "" {
		topK, _ = strconv.Atoi(val)
	}

	ingestCount := 100
	if val := os.Getenv("TURBORAG_INGEST_RECORDS"); val != "" {
		ingestCount, _ = strconv.Atoi(val)
	}

	client := turborag.New(baseURL)
	client.HTTPClient.Timeout = 120 * time.Second

	ctx := context.Background()

	index, err := client.Index(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting index: %v\n", err)
		os.Exit(1)
	}

	queries := make([][]float64, queryCount)
	for i := 0; i < queryCount; i++ {
		queries[i] = makeVector(index.Dim, float64(i+1))
	}

	fmt.Println("\n=== TurboRAG Go Client Benchmark ===\n")
	fmt.Printf("Base URL: %s\n", baseURL)
	fmt.Printf("Index size: %d\n", index.IndexSize)
	fmt.Printf("Dimension: %d\n", index.Dim)
	fmt.Printf("Bits: %d\n", index.Bits)
	fmt.Printf("Queries: %d\n", queryCount)
	fmt.Printf("topK: %d\n\n", topK)

	// Single query benchmark
	singleStart := time.Now()
	for _, vector := range queries {
		_, err := client.Query(ctx, vector, topK)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Query error: %v\n", err)
			os.Exit(1)
		}
	}
	singleElapsed := time.Since(singleStart)

	// Batch query benchmark
	batchStart := time.Now()
	batchResp, err := client.QueryBatch(ctx, queries, topK)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Batch query error: %v\n", err)
		os.Exit(1)
	}
	batchElapsed := time.Since(batchStart)

	// Ingest benchmark
	records := make([]turborag.IngestRecord, ingestCount)
	for i := 0; i < ingestCount; i++ {
		records[i] = turborag.IngestRecord{
			ChunkID:   fmt.Sprintf("go-bench-%d-%d", time.Now().UnixNano(), i),
			Text:      fmt.Sprintf("Benchmark record %d", i),
			Embedding: makeVector(index.Dim, float64(10000+i)),
			Metadata:  map[string]interface{}{"benchmark": true, "index": i},
		}
	}

	ingestStart := time.Now()
	ingestResp, err := client.Ingest(ctx, records)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Ingest error: %v\n", err)
		os.Exit(1)
	}
	ingestElapsed := time.Since(ingestStart)

	results := map[string]interface{}{
		"baseUrl": baseURL,
		"index": map[string]interface{}{
			"size": index.IndexSize,
			"dim":  index.Dim,
			"bits": index.Bits,
		},
		"singleQuery": map[string]interface{}{
			"queries":      queryCount,
			"elapsedMs":    float64(singleElapsed.Milliseconds()),
			"qps":          float64(queryCount) / singleElapsed.Seconds(),
			"avgLatencyMs": float64(singleElapsed.Milliseconds()) / float64(queryCount),
		},
		"batchQuery": map[string]interface{}{
			"queries":      queryCount,
			"batchCount":   batchResp.BatchCount,
			"elapsedMs":    float64(batchElapsed.Milliseconds()),
			"qps":          float64(queryCount) / batchElapsed.Seconds(),
			"avgLatencyMs": float64(batchElapsed.Milliseconds()) / float64(queryCount),
		},
		"ingest": map[string]interface{}{
			"records":          ingestCount,
			"added":            ingestResp.Added,
			"indexSize":        ingestResp.IndexSize,
			"elapsedMs":        float64(ingestElapsed.Milliseconds()),
			"recordsPerSecond": float64(ingestCount) / ingestElapsed.Seconds(),
		},
	}

	output, _ := json.MarshalIndent(results, "", "  ")
	fmt.Println(string(output))
}
