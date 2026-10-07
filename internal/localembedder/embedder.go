// Package localembedder provides in-process sentence embeddings from ONNX
// models (all-MiniLM-L6-v2, gte-small, bge-small-en-v1.5 — see spec.go). One
// model and the onnxruntime library are embedded in the binary via go:embed
// and extracted to a temp directory on first use; other registered models are
// loaded from a models directory (see NewFromDir and `zenith models`).
//
// Requires CGo (CGO_ENABLED=1) and a C compiler to build. Without CGo, New()
// returns an error and the caller should fall back to a deterministic embedder.
//
//go:generate go run ../../scripts/download_assets.go
package localembedder

import (
	"context"
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/modelspec"
)

// modelIDBytes names the registry entry the bundled model.onnx corresponds to.
// scripts/download_assets.go writes it next to the model it downloads, so the
// two can never disagree.
//
//go:embed assets/model.id
var modelIDBytes []byte

// BundledID is the registry ID of the model compiled into this binary.
func BundledID() string {
	if id := strings.TrimSpace(string(modelIDBytes)); id != "" {
		return id
	}
	return modelspec.DefaultID
}

// Embedder implements embedding.Embedder using in-process ONNX inference.
// All methods are safe for concurrent use.
type Embedder struct {
	spec  Spec
	tok   *tokenizer
	model *onnxModel
	dense *denseLayer // optional trained projection applied after pooling (see spec.DenseURL)
}

// New loads the bundled ONNX model and returns a ready Embedder.
// Requires CGo (CGO_ENABLED=1). Returns an error if CGo is unavailable or
// the onnxruntime library cannot be initialised.
func New() (*Embedder, error) {
	spec, err := Lookup(BundledID())
	if err != nil {
		return nil, err
	}
	return newEmbedder(spec, modelBytes, vocabBytes, nil)
}

// NewByID returns the embedder for a registered model. The bundled model is
// used directly; any other model is read from modelsDir/<id>/model.onnx (the
// layout `zenith models pull <id>` produces). An empty id means the bundled one.
func NewByID(id, modelsDir string) (*Embedder, error) {
	if id == "" || strings.EqualFold(id, BundledID()) {
		return New()
	}
	spec, err := Lookup(id)
	if err != nil {
		return nil, err
	}
	dir := filepath.Join(modelsDir, spec.ID)
	model, err := os.ReadFile(filepath.Join(dir, "model.onnx"))
	if err != nil {
		return nil, fmt.Errorf("localembedder: model %q is not installed (%w) — run: zenith models pull %s", spec.ID, err, spec.ID)
	}
	vocab := vocabBytes // every registered model shares the BERT-uncased vocabulary
	if v, err := os.ReadFile(filepath.Join(dir, "vocab.txt")); err == nil {
		vocab = v
	}
	var dense []byte
	if spec.DenseURL != "" {
		dense, err = os.ReadFile(filepath.Join(dir, "dense.safetensors"))
		if err != nil {
			return nil, fmt.Errorf("localembedder: model %q is missing its trained dense projection (%w) — run: zenith models pull %s", spec.ID, err, spec.ID)
		}
	}
	return newEmbedder(spec, model, vocab, dense)
}

func newEmbedder(spec Spec, model, vocab, dense []byte) (*Embedder, error) {
	if err := unavailable(); err != nil {
		return nil, err
	}
	tok, err := newTokenizerFromBytes(vocab, spec.Cased)
	if err != nil {
		return nil, fmt.Errorf("localembedder: tokenizer: %w", err)
	}

	libPath, err := extractToTemp(ortLibBytes, ortLibFilename)
	if err != nil {
		return nil, fmt.Errorf("localembedder: extract ort lib: %w", err)
	}

	m, err := newOnnxModel(model, libPath, !spec.NoTokenTypeIDs)
	if err != nil {
		return nil, fmt.Errorf("localembedder: ort session: %w", err)
	}

	e := &Embedder{spec: spec, tok: tok, model: m}
	if len(dense) > 0 {
		d, err := parseDenseSafetensors(dense)
		if err != nil {
			return nil, fmt.Errorf("localembedder: dense projection: %w", err)
		}
		if d.in != e.hiddenDims() || d.out != spec.Dims {
			return nil, fmt.Errorf("localembedder: dense projection shape [%d,%d] does not match model dims (hidden %d, output %d)", d.out, d.in, e.hiddenDims(), spec.Dims)
		}
		e.dense = d
	}
	return e, nil
}

// hiddenDims is the transformer's native hidden size — what pooling reads
// from the raw ONNX output — which equals spec.Dims unless HiddenDims
// overrides it (a non-square Dense projection; see Spec.HiddenDims).
func (e *Embedder) hiddenDims() int {
	if e.spec.HiddenDims > 0 {
		return e.spec.HiddenDims
	}
	return e.spec.Dims
}

// Spec returns the model description this embedder runs.
func (e *Embedder) Spec() Spec { return e.spec }

// seqLenFor rounds n up to a multiple of 8 (efficient ONNX kernel shapes),
// capped at maxLen. The model has dynamic sequence axes — padding every input
// to a fixed 256 wastes 3–60× compute on typical passages and single words.
func seqLenFor(n int) int {
	const align = 8
	l := (n + align - 1) / align * align
	if l > maxLen {
		l = maxLen
	}
	return l
}

func (e *Embedder) pool(hidden []float32, mask []int64, seqLen int) []float32 {
	var vec []float32
	if e.spec.Pooling == PoolCLS {
		vec = append([]float32(nil), hidden[:e.hiddenDims()]...)
	} else {
		vec = meanPool(hidden, mask, seqLen, e.hiddenDims())
	}
	if e.dense != nil {
		vec = e.dense.apply(vec)
	}
	return l2Normalize(vec)
}

// Embed returns an L2-normalised vector for a document (or any symmetric text).
func (e *Embedder) Embed(_ context.Context, text string) ([]float32, error) {
	return e.embedOne(e.spec.DocPrefix + text)
}

// EmbedQuery embeds a search query. For models trained with a query
// instruction (bge) that prefix is applied; for symmetric models it is
// identical to Embed. The engine calls this on the search path when available
// (see embedding.QueryEmbedder).
func (e *Embedder) EmbedQuery(_ context.Context, text string) ([]float32, error) {
	return e.embedOne(e.spec.QueryPrefix + text)
}

func (e *Embedder) embedOne(text string) ([]float32, error) {
	ids := e.tok.encodeIDs(text, maxLen)
	seqLen := seqLenFor(len(ids))

	flatIDs := make([]int64, seqLen)
	flatMask := make([]int64, seqLen)
	flatTypeIDs := make([]int64, seqLen)
	copy(flatIDs, ids)
	for i := range ids {
		flatMask[i] = 1
	}

	hidden, err := e.model.infer(flatIDs, flatMask, flatTypeIDs, 1, seqLen, e.hiddenDims())
	if err != nil {
		return nil, err
	}
	return e.pool(hidden, flatMask, seqLen), nil
}

// EmbedBatch returns embeddings for all texts in a single ONNX forward pass.
// The batch is padded to the longest sequence it contains, not to maxLen.
func (e *Embedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	n := len(texts)
	encoded := make([][]int64, n)
	longest := 0
	for i, t := range texts {
		encoded[i] = e.tok.encodeIDs(e.spec.DocPrefix+t, maxLen)
		if len(encoded[i]) > longest {
			longest = len(encoded[i])
		}
	}
	seqLen := seqLenFor(longest)

	flatIDs := make([]int64, n*seqLen)
	flatMask := make([]int64, n*seqLen)
	flatTypeIDs := make([]int64, n*seqLen)
	for i, ids := range encoded {
		base := i * seqLen
		copy(flatIDs[base:], ids)
		for j := range ids {
			flatMask[base+j] = 1
		}
	}

	hidden, err := e.model.infer(flatIDs, flatMask, flatTypeIDs, n, seqLen, e.hiddenDims())
	if err != nil {
		return nil, err
	}

	chunkSize := seqLen * e.hiddenDims()
	result := make([][]float32, n)
	for i := range texts {
		chunk := hidden[i*chunkSize : (i+1)*chunkSize]
		result[i] = e.pool(chunk, flatMask[i*seqLen:(i+1)*seqLen], seqLen)
	}
	return result, nil
}

// Dimensions returns the model's output size.
func (e *Embedder) Dimensions() int { return e.spec.Dims }

// Name identifies the embedding model for index-file compatibility checks
// (see internal/embedding.Named).
func (e *Embedder) Name() string { return e.spec.IndexName() }

// Verify interfaces at compile time.
var (
	_ embedding.Embedder      = (*Embedder)(nil)
	_ embedding.QueryEmbedder = (*Embedder)(nil)
)
