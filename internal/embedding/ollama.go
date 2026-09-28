package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// OllamaEmbedder calls a locally-running Ollama instance to produce embeddings.
// It satisfies the Embedder interface and is the preferred production embedder
// because it runs fully offline with no API key or cloud bill.
//
// Default model: "nomic-embed-text" (768-dimensional, runs on CPU).
// Start Ollama and pull the model before use:
//
//	ollama pull nomic-embed-text
//	ollama serve          # defaults to localhost:11434
//
// Batch: Ollama does not expose a native batch endpoint; EmbedBatch calls
// /api/embeddings sequentially. Wrap with CachingEmbedder to avoid redundant
// round-trips for repeated tokens.
type OllamaEmbedder struct {
	client *http.Client
	base   string // e.g. "http://localhost:11434"
	model  string // e.g. "nomic-embed-text"
}

// NewOllamaEmbedder creates an OllamaEmbedder.
// base is the Ollama server URL (default "http://localhost:11434").
// model is the embedding model name (default "nomic-embed-text").
func NewOllamaEmbedder(base, model string, timeout time.Duration) *OllamaEmbedder {
	if base == "" {
		base = "http://localhost:11434"
	}
	if model == "" {
		model = "nomic-embed-text"
	}
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	return &OllamaEmbedder{
		client: &http.Client{Timeout: timeout},
		base:   base,
		model:  model,
	}
}

type ollamaEmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type ollamaEmbedResponse struct {
	Embedding []float32 `json:"embedding"`
}

// Embed returns an embedding vector for text.
func (o *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	body, err := json.Marshal(ollamaEmbedRequest{Model: o.model, Prompt: text})
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.base+"/api/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama offline or timeout: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama: HTTP %d", resp.StatusCode)
	}

	var res ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("ollama: decode response: %w", err)
	}
	return res.Embedding, nil
}

// EmbedBatch embeds each text in texts sequentially.
// Failures are non-fatal — a nil entry is returned for each failed text.
func (o *OllamaEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	for i, t := range texts {
		vec, err := o.Embed(ctx, t)
		if err != nil {
			// Non-fatal: let the caller decide whether to skip nil entries.
			results[i] = nil
			continue
		}
		results[i] = vec
	}
	return results, nil
}

// Dimensions returns the embedding size for the configured model.
// nomic-embed-text produces 768-dimensional vectors.
// mxbai-embed-large produces 1024-dimensional vectors.
// all-minilm produces 384-dimensional vectors.
// Returns 0 if the model is unknown — the engine handles this gracefully.
func (o *OllamaEmbedder) Dimensions() int {
	switch o.model {
	case "nomic-embed-text":
		return 768
	case "mxbai-embed-large":
		return 1024
	case "all-minilm":
		return 384
	default:
		return 0
	}
}

// Name identifies the embedding model for index-file compatibility checks
// (see Named).
func (o *OllamaEmbedder) Name() string {
	return "ollama:" + o.model
}
