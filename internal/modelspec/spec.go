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
	// VocabURL is a model-specific vocab.txt, fetched alongside ModelURL when
	// non-empty. Empty for every model that shares the bundled bert-base-uncased
	// vocabulary; set for models (e.g. cased or multilingual ones) that need a
	// different one.
	VocabURL string
	// DenseURL is a small safetensors file holding a trained linear+tanh
	// projection (sentence-transformers' "Dense" module) applied after
	// pooling, fetched alongside ModelURL when non-empty. Most ONNX exports
	// only expose last_hidden_state with no such layer; when the upstream
	// sentence-transformers model ships one separately (e.g. LaBSE's
	// 2_Dense), wiring it in here closes that gap without re-exporting the
	// (much larger) base model.
	DenseURL string
	// Cased, when true, skips lowercasing before WordPiece tokenization. False
	// (the default) matches every existing model's bert-base-uncased vocabulary.
	Cased bool
	// SizeMB is the approximate download size, for user-facing messages.
	SizeMB int
	// Languages is a short human-readable coverage note.
	Languages string
}

// IndexName is the embedder identity recorded in saved index files.
func (s Spec) IndexName() string { return "onnx:" + s.ID }

// Models here mostly share the bert-base-uncased WordPiece vocabulary (30522
// entries), which is why one bundled vocab.txt serves all of them. Models with
// a different tokenizer family (e.g. multilingual-e5-small, which needs
// SentencePiece) are deliberately not listed until a tokenizer for them
// exists. labse (LaBSE) is the exception that fits: it is still WordPiece,
// just with its own (much larger, cased) vocab fetched via VocabURL — see
// Spec.VocabURL / Spec.Cased.
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
	{
		ID:          "labse",
		Description: "LaBSE: BERT-based cross-lingual sentence embeddings trained for translation ranking across 109 languages; opt-in, never the bundled default. The base ONNX export exposes last_hidden_state only, so the real sentence-transformers pipeline (CLS pool -> trained dense+tanh projection -> normalize) is reconstructed here: CLS pooling plus the small separate 2_Dense projection fetched via DenseURL, matching sentence-transformers/LaBSE's own module config (1_Pooling: cls, 2_Dense: Linear(768,768)+Tanh) -- not a guess. Smoke-tested, not BEIR-benchmarked; a hand-built real-Wikipedia cross-lingual check is in README.md",
		Dims:        768, Pooling: PoolCLS,
		ModelURL:  "https://huggingface.co/Xenova/LaBSE/resolve/main/onnx/model_quantized.onnx",
		VocabURL:  "https://huggingface.co/Xenova/LaBSE/resolve/main/vocab.txt",
		DenseURL:  "https://huggingface.co/sentence-transformers/LaBSE/resolve/main/2_Dense/model.safetensors",
		Cased:     true,
		SizeMB:    450,
		Languages: "109 languages (LaBSE)",
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

// RerankerSpec describes a cross-encoder that reorders a shortlist of hybrid
// search hits. Unlike Spec, a reranker jointly encodes (query, passage) pairs
// and outputs one relevance logit rather than independent vectors — it has no
// Dims/Pooling, and it is never bundled into the binary (opt-in only, fetched
// with `zenith models pull` like a non-default embedding model).
type RerankerSpec struct {
	ID          string
	Description string
	ModelURL    string
	SizeMB      int
}

// Rerankers share the embedding models' bert-base-uncased WordPiece vocabulary.
var rerankerRegistry = []RerankerSpec{
	{
		ID:          "ms-marco-MiniLM-L-6-v2",
		Description: "cross-encoder trained on MS MARCO; reorders the top hybrid hits, too slow to run over a whole corpus",
		ModelURL:    "https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model_quantized.onnx",
		SizeMB:      23,
	},
}

// RerankerModels returns every registered reranker, sorted by ID.
func RerankerModels() []RerankerSpec {
	out := append([]RerankerSpec(nil), rerankerRegistry...)
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// LookupReranker finds a reranker by ID (case-insensitive).
func LookupReranker(id string) (RerankerSpec, error) {
	for _, s := range rerankerRegistry {
		if strings.EqualFold(s.ID, id) {
			return s, nil
		}
	}
	ids := make([]string, 0, len(rerankerRegistry))
	for _, s := range rerankerRegistry {
		ids = append(ids, s.ID)
	}
	return RerankerSpec{}, fmt.Errorf("modelspec: unknown reranker %q (available: %s)", id, strings.Join(ids, ", "))
}

// DefaultRerankerID is used by WithReranker(true) / --rerank when no specific
// reranker model is named.
const DefaultRerankerID = "ms-marco-MiniLM-L-6-v2"
