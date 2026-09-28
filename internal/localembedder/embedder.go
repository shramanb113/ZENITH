// Package localembedder provides in-process sentence embeddings using the
// all-MiniLM-L6-v2 ONNX model. The model and onnxruntime library are embedded
// in the binary via go:embed and extracted to a temp directory on first use.
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

	"github.com/shramanb113/ZENITH/internal/embedding"
)

//go:embed assets/model.onnx
var modelBytes []byte

//go:embed assets/vocab.txt
var vocabBytes []byte

// Embedder implements embedding.Embedder using in-process ONNX inference.
// All methods are safe for concurrent use.
type Embedder struct {
	tok   *tokenizer
	model *onnxModel
}

// New loads the embedded ONNX model and returns a ready Embedder.
// Requires CGo (CGO_ENABLED=1). Returns an error if CGo is unavailable or
// the onnxruntime library cannot be initialised.
func New() (*Embedder, error) {
	tok, err := newTokenizerFromBytes(vocabBytes)
	if err != nil {
		return nil, fmt.Errorf("localembedder: tokenizer: %w", err)
	}

	libPath, err := extractToTemp(ortLibBytes, ortLibFilename)
	if err != nil {
		return nil, fmt.Errorf("localembedder: extract ort lib: %w", err)
	}

	m, err := newOnnxModel(modelBytes, libPath)
	if err != nil {
		return nil, fmt.Errorf("localembedder: ort session: %w", err)
	}

	return &Embedder{tok: tok, model: m}, nil
}

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

// Embed returns a 384-dimensional L2-normalised vector for text.
func (e *Embedder) Embed(_ context.Context, text string) ([]float32, error) {
	ids := e.tok.encodeIDs(text, maxLen)
	seqLen := seqLenFor(len(ids))

	flatIDs := make([]int64, seqLen)
	flatMask := make([]int64, seqLen)
	flatTypeIDs := make([]int64, seqLen)
	copy(flatIDs, ids)
	for i := range ids {
		flatMask[i] = 1
	}

	hidden, err := e.model.infer(flatIDs, flatMask, flatTypeIDs, 1, seqLen)
	if err != nil {
		return nil, err
	}
	vec := meanPool(hidden, flatMask, seqLen, hiddenSize)
	return l2Normalize(vec), nil
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
		encoded[i] = e.tok.encodeIDs(t, maxLen)
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

	hidden, err := e.model.infer(flatIDs, flatMask, flatTypeIDs, n, seqLen)
	if err != nil {
		return nil, err
	}

	chunkSize := seqLen * hiddenSize
	result := make([][]float32, n)
	for i := range texts {
		chunk := hidden[i*chunkSize : (i+1)*chunkSize]
		vec := meanPool(chunk, flatMask[i*seqLen:(i+1)*seqLen], seqLen, hiddenSize)
		result[i] = l2Normalize(vec)
	}
	return result, nil
}

// Dimensions returns 384 — the output size of all-MiniLM-L6-v2.
func (e *Embedder) Dimensions() int { return hiddenSize }

// Name identifies the embedding model for index-file compatibility checks
// (see internal/embedding.Named).
func (e *Embedder) Name() string { return "onnx:all-MiniLM-L6-v2" }

// Verify implements embedding.Embedder at compile time.
var _ embedding.Embedder = (*Embedder)(nil)
