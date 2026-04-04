// Package turborag provides a Go client for the TurboRAG compressed vector retrieval API.
//
// Usage:
//
//	client := turborag.New("http://localhost:8080")
//	results, err := client.Query(ctx, []float64{0.1, 0.2, 0.3}, 5)
package turborag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client communicates with a TurboRAG HTTP server.
type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

// New creates a TurboRAG client with a 30-second timeout.
func New(baseURL string) *Client {
	return &Client{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// ── Types ───────────────────────────────────────────────────────────────

// QueryResult represents a single search result.
type QueryResult struct {
	ChunkID     string   `json:"chunk_id"`
	Text        string   `json:"text"`
	Score       float64  `json:"score"`
	SourceDoc   *string  `json:"source_doc"`
	PageNum     *int     `json:"page_num"`
	GraphPath   []string `json:"graph_path"`
	Explanation *string  `json:"explanation"`
}

// QueryResponse is the response from /query.
type QueryResponse struct {
	Count   int           `json:"count"`
	Results []QueryResult `json:"results"`
}

// BatchQueryResponse is the response from /query/batch.
type BatchQueryResponse struct {
	BatchCount int `json:"batch_count"`
	Results    []struct {
		Count   int           `json:"count"`
		Results []QueryResult `json:"results"`
	} `json:"results"`
}

// IngestRecord is a record to add to the index.
type IngestRecord struct {
	ChunkID   string                 `json:"chunk_id"`
	Text      string                 `json:"text"`
	Embedding []float64              `json:"embedding"`
	SourceDoc string                 `json:"source_doc,omitempty"`
	PageNum   *int                   `json:"page_num,omitempty"`
	Section   string                 `json:"section,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// IngestResponse is the response from /ingest.
type IngestResponse struct {
	Added           int    `json:"added"`
	IndexSize       int    `json:"index_size"`
	RecordsSnapshot string `json:"records_snapshot"`
}

// IngestTextResponse is the response from /ingest-text.
type IngestTextResponse struct {
	Added int `json:"added"`
	Chunks []struct {
		ChunkID string `json:"chunk_id"`
		Text    string `json:"text"`
	} `json:"chunks"`
	IndexSize int `json:"index_size"`
}

// HealthResponse is the response from /health.
type HealthResponse struct {
	Status    string `json:"status"`
	IndexPath string `json:"index_path"`
	IndexSize int    `json:"index_size"`
}

// IndexInfo is the response from /index.
type IndexInfo struct {
	IndexPath        string  `json:"index_path"`
	Dim              int     `json:"dim"`
	Bits             int     `json:"bits"`
	ShardSize        int     `json:"shard_size"`
	Normalize        bool    `json:"normalize"`
	ValueRange       float64 `json:"value_range"`
	IndexSize        int     `json:"index_size"`
	RecordsLoaded    int     `json:"records_loaded"`
	RecordsSnapshot  *string `json:"records_snapshot"`
	TextQueryEnabled bool    `json:"text_query_enabled"`
}

// ── Query ───────────────────────────────────────────────────────────────

// Query searches the index with an embedding vector.
func (c *Client) Query(ctx context.Context, vector []float64, topK int) (*QueryResponse, error) {
	body := map[string]interface{}{
		"query_vector": vector,
		"top_k":        topK,
	}
	var resp QueryResponse
	if err := c.post(ctx, "/query", body, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// QueryText searches the index with text (requires --model on the server).
func (c *Client) QueryText(ctx context.Context, text string, topK int) (*QueryResponse, error) {
	body := map[string]interface{}{
		"query_text": text,
		"top_k":      topK,
	}
	var resp QueryResponse
	if err := c.post(ctx, "/query", body, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// QueryBatch searches with multiple vectors in a single request.
func (c *Client) QueryBatch(ctx context.Context, vectors [][]float64, topK int) (*BatchQueryResponse, error) {
	queries := make([]map[string]interface{}, len(vectors))
	for i, v := range vectors {
		queries[i] = map[string]interface{}{"query_vector": v}
	}
	body := map[string]interface{}{
		"queries": queries,
		"top_k":   topK,
	}
	var resp BatchQueryResponse
	if err := c.post(ctx, "/query/batch", body, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ── Ingest ──────────────────────────────────────────────────────────────

// Ingest adds records with precomputed embeddings to the index.
func (c *Client) Ingest(ctx context.Context, records []IngestRecord) (*IngestResponse, error) {
	body := map[string]interface{}{"records": records}
	var resp IngestResponse
	if err := c.post(ctx, "/ingest", body, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// IngestText ingests raw text with automatic chunking (requires --model).
func (c *Client) IngestText(ctx context.Context, text string, sourceDoc string) (*IngestTextResponse, error) {
	body := map[string]interface{}{
		"text":       text,
		"source_doc": sourceDoc,
	}
	var resp IngestTextResponse
	if err := c.post(ctx, "/ingest-text", body, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ── Info ────────────────────────────────────────────────────────────────

// Health checks if the service is running.
func (c *Client) Health(ctx context.Context) (*HealthResponse, error) {
	var resp HealthResponse
	if err := c.get(ctx, "/health", &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Index returns the index configuration and statistics.
func (c *Client) Index(ctx context.Context) (*IndexInfo, error) {
	var resp IndexInfo
	if err := c.get(ctx, "/index", &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ── HTTP helpers ────────────────────────────────────────────────────────

func (c *Client) get(ctx context.Context, path string, out interface{}) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+path, nil)
	if err != nil {
		return fmt.Errorf("turborag: %w", err)
	}
	return c.do(req, out)
}

func (c *Client) post(ctx context.Context, path string, body interface{}, out interface{}) error {
	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("turborag: marshal: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("turborag: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	return c.do(req, out)
}

func (c *Client) do(req *http.Request, out interface{}) error {
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("turborag: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("turborag: read body: %w", err)
	}
	if resp.StatusCode >= 400 {
		return fmt.Errorf("turborag: HTTP %d: %s", resp.StatusCode, string(data))
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("turborag: decode: %w", err)
	}
	return nil
}
