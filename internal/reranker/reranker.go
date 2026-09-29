// Package reranker reorders a hybrid search engine's top candidates with a
// cross-encoder. It sits above internal/index (for SearchResponse) and
// internal/localembedder (for the ONNX cross-encoder), so both pkg/zenith
// and cmd/zenith — which uses *index.Engine directly and never goes through
// pkg/zenith — can share the same merge logic.
package reranker

import (
	"context"
	"math"
	"sort"

	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/localembedder"
)

// TopN caps how many of the highest-scoring hybrid candidates get rescored.
// Cross-encoder inference jointly encodes each (query, doc) pair, which is
// too expensive to run over an entire candidate list.
const TopN = 50

// Scorer scores a batch of (query, doc) pairs, returning one raw relevance
// logit per doc in the same order. *localembedder.Reranker implements this;
// the interface lets tests substitute a fake without an ONNX runtime.
type Scorer interface {
	ScoreBatch(ctx context.Context, query string, docs []string) ([]float32, error)
}

// TextLookup fetches the original text stored under a candidate's ID, which
// may be a chunk-suffixed ID (e.g. (*index.Engine).GetText).
type TextLookup func(id string) (string, bool)

// Reranker wraps a loaded cross-encoder for repeated use across searches.
type Reranker struct {
	scorer Scorer
}

// New loads a registered cross-encoder from modelsDir/<id>/model.onnx ("" for
// the default). Requires a CGO build; the model must already be installed
// (see localembedder.NewReranker / `zenith models pull`).
func New(id, modelsDir string) (*Reranker, error) {
	r, err := localembedder.NewReranker(id, modelsDir)
	if err != nil {
		return nil, err
	}
	return &Reranker{scorer: r}, nil
}

// Close releases the underlying ONNX session.
func (rr *Reranker) Close() {
	if c, ok := rr.scorer.(interface{ Close() }); ok {
		c.Close()
	}
}

// Rerank reorders raw's top TopN candidates against query using rr's
// cross-encoder. See the package-level Rerank for the merge semantics.
func (rr *Reranker) Rerank(ctx context.Context, query string, raw []index.SearchResponse, textOf TextLookup) []index.SearchResponse {
	return Rerank(ctx, rr.scorer, query, raw, textOf)
}

// Rerank rescoures the highest-scoring min(TopN, len(raw)) candidates in raw
// against query using s, replacing each one's Score with a sigmoid-squashed
// version of the cross-encoder logit — raw logits range roughly ±15 and the
// rest of the pipeline assumes non-negative scores (see
// pkg/zenith.normaliseScore) — then returns every candidate in raw, rescored
// ones first (ordered by their new score), followed by the untouched tail in
// its original relative order. A document absent from raw is never added:
// this reorders quality, it never changes recall.
//
// On any failure — a nil scorer, an empty raw, a text lookup miss, or a
// scoring error — Rerank returns raw unchanged. Reranking degrades to a
// no-op, never a search failure.
func Rerank(ctx context.Context, s Scorer, query string, raw []index.SearchResponse, textOf TextLookup) []index.SearchResponse {
	if s == nil || len(raw) == 0 {
		return raw
	}

	ranked := make([]index.SearchResponse, len(raw))
	copy(ranked, raw)
	sort.SliceStable(ranked, func(i, j int) bool { return ranked[i].Score > ranked[j].Score })

	n := TopN
	if n > len(ranked) {
		n = len(ranked)
	}
	head, tail := ranked[:n], ranked[n:]

	docs := make([]string, n)
	for i, r := range head {
		text, ok := textOf(r.ID)
		if !ok {
			return raw
		}
		docs[i] = text
	}

	logits, err := s.ScoreBatch(ctx, query, docs)
	if err != nil || len(logits) != n {
		return raw
	}

	for i := range head {
		head[i].Score = sigmoid(float64(logits[i]))
	}
	sort.SliceStable(head, func(i, j int) bool { return head[i].Score > head[j].Score })

	out := make([]index.SearchResponse, 0, len(ranked))
	out = append(out, head...)
	out = append(out, tail...)
	return out
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
