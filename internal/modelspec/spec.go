// Package modelspec is the registry of embedding models ZENITH knows how to run.
// It has no dependencies (in particular no go:embed), so build tooling such as
// scripts/download_assets.go can read it before any model has been downloaded.
package modelspec

import (
	"fmt"
	"sort"
	"strings"
)

// Pooling is how per-token hidden states are reduced to one sentence vector.
type Pooling int

const (
	// PoolMean averages the hidden states of real (non-padding) tokens.
	PoolMean Pooling = iota
	// PoolCLS uses the hidden state of the leading [CLS] token.
	PoolCLS
)

// Spec describes an embedding model the engine knows how to run. The ID is
// what gets recorded in a saved index's header (as "onnx:<ID>"), so it must be
// stable: changing what an ID points at would silently invalidate old indexes.
type Spec struct {
	ID          string
	Description string
	Dims        int
	Pooling     Pooling
	// QueryPrefix / DocPrefix are prepended to text before tokenising, for
	// models trained with asymmetric instructions. Empty for symmetric models.
	QueryPrefix string
	DocPrefix   string
	// ModelURL is the quantized ONNX export the model is fetched from.
	ModelURL string
	// SizeMB is the approximate download size, for user-facing messages.
	SizeMB int
	// Languages is a short human-readable coverage note.
	Languages string
}

// IndexName is the embedder identity recorded in saved index files.
func (s Spec) IndexName() string { return "onnx:" + s.ID }

// All models here share the bert-base-uncased WordPiece vocabulary (30522
// entries), which is why one bundled vocab.txt serves all of them. Models with
// a different tokenizer (e.g. multilingual-e5-small, which needs SentencePiece)
// are deliberately not listed until a tokenizer for them exists.
var registry = []Spec{
	{
		ID:          "all-MiniLM-L6-v2",
		Description: "general-purpose English sentence embeddings: smaller and faster, weaker on out-of-domain text (the previous default)",
		Dims:        384, Pooling: PoolMean,
		ModelURL:  "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx",
		SizeMB:    23,
		Languages: "English",
	},
	{
		ID:          "gte-small",
		Description: "GTE-small: stronger English retrieval, +4.7 nDCG@10 over MiniLM in hybrid on SciFact (the default)",
		Dims:        384, Pooling: PoolMean,
		ModelURL:  "https://huggingface.co/Xenova/gte-small/resolve/main/onnx/model_quantized.onnx",
		SizeMB:    34,
		Languages: "English",
	},
	{
		ID:          "bge-small-en-v1.5",
		Description: "BGE-small: English retrieval with a query instruction prefix",
		Dims:        384, Pooling: PoolCLS,
		QueryPrefix: "Represent this sentence for searching relevant passages: ",
		ModelURL:    "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx",
		SizeMB:      34,
		Languages:   "English",
	},
}

// Models returns every registered model, sorted by ID.
func Models() []Spec {
	out := append([]Spec(nil), registry...)
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Lookup finds a model by ID (case-insensitive).
func Lookup(id string) (Spec, error) {
	for _, s := range registry {
		if strings.EqualFold(s.ID, id) {
			return s, nil
		}
	}
	ids := make([]string, 0, len(registry))
	for _, s := range registry {
		ids = append(ids, s.ID)
	}
	return Spec{}, fmt.Errorf("localembedder: unknown model %q (available: %s)", id, strings.Join(ids, ", "))
}

// DefaultID is the model bundled into release binaries and fetched by
// scripts/download_assets.go when no -model is given. Changing it changes what
// new indexes are built with; existing indexes keep working because the model
// they were built with is recorded in their header (and refused on mismatch).
const DefaultID = "gte-small"
