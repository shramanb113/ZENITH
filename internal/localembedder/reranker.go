package localembedder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/shramanb113/ZENITH/internal/modelspec"
)

// Reranker scores (query, passage) pairs with a cross-encoder: it jointly
// encodes the pair and outputs one relevance logit, more accurate than
// comparing two independently-computed embedding vectors but too slow to run
// over a whole corpus. Callers use it to reorder a short list of hybrid
// search hits (see internal/reranker), never to search the whole index.
//
// Unlike the bundled Embedder, a Reranker is never compiled into the binary —
// it is opt-in, fetched with `zenith models pull <id>` into the same
// ~/.zenith/models layout NewByID reads from.
type Reranker struct {
	tok   *tokenizer
	model *rerankModel
}

// NewReranker loads a registered cross-encoder from modelsDir/<id>/model.onnx.
// An empty id uses modelspec.DefaultRerankerID. Requires a CGo build.
func NewReranker(id, modelsDir string) (*Reranker, error) {
	if err := unavailable(); err != nil {
		return nil, err
	}
	if id == "" {
		id = modelspec.DefaultRerankerID
	}
	spec, err := modelspec.LookupReranker(id)
	if err != nil {
		return nil, err
	}
	dir := filepath.Join(modelsDir, spec.ID)
	modelData, err := os.ReadFile(filepath.Join(dir, "model.onnx"))
	if err != nil {
		return nil, fmt.Errorf("localembedder: reranker %q is not installed (%w) — run: zenith models pull %s", spec.ID, err, spec.ID)
	}
	// Rerankers in the registry share the bundled BERT-uncased vocab, same as
	// every embedding model (see modelspec's registry comment).
	tok, err := newTokenizerFromBytes(vocabBytes, false)
	if err != nil {
		return nil, fmt.Errorf("localembedder: tokenizer: %w", err)
	}
	libPath, err := extractToTemp(ortLibBytes, ortLibFilename)
	if err != nil {
		return nil, fmt.Errorf("localembedder: extract ort lib: %w", err)
	}
	m, err := newRerankModel(modelData, libPath)
	if err != nil {
		return nil, fmt.Errorf("localembedder: ort session: %w", err)
	}
	return &Reranker{tok: tok, model: m}, nil
}

// Score returns one relevance logit for (query, doc) — higher means more
// relevant. It is a raw cross-encoder logit, not a probability.
func (r *Reranker) Score(ctx context.Context, query, doc string) (float32, error) {
	scores, err := r.ScoreBatch(ctx, query, []string{doc})
	if err != nil {
		return 0, err
	}
	return scores[0], nil
}

// ScoreBatch scores query against every doc in one ONNX forward pass, padded
// to the longest pair encoding in the batch.
func (r *Reranker) ScoreBatch(_ context.Context, query string, docs []string) ([]float32, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	n := len(docs)
	type row struct{ ids, mask, typ []int64 }
	rows := make([]row, n)
	longest := 0
	for i, doc := range docs {
		ids, mask, typ := r.tok.encodePair(query, doc, maxLen)
		rows[i] = row{ids, mask, typ}
		if len(ids) > longest {
			longest = len(ids)
		}
	}
	seqLen := seqLenFor(longest)

	flatIDs := make([]int64, n*seqLen)
	flatMask := make([]int64, n*seqLen)
	flatType := make([]int64, n*seqLen)
	for i, rw := range rows {
		base := i * seqLen
		copy(flatIDs[base:], rw.ids)
		copy(flatMask[base:], rw.mask)
		copy(flatType[base:], rw.typ)
	}

	return r.model.score(flatIDs, flatMask, flatType, n, seqLen)
}

// Close releases the ONNX session.
func (r *Reranker) Close() { r.model.close() }
