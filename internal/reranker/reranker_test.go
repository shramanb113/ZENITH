package reranker

import (
	"context"
	"errors"
	"testing"

	"github.com/shramanb113/ZENITH/internal/index"
)

// fakeScorer scores docs by a caller-supplied lookup, so tests can assert
// exact reordering without an ONNX runtime.
type fakeScorer struct {
	logits map[string]float32 // doc text -> logit
	err    error
}

func (f *fakeScorer) ScoreBatch(_ context.Context, _ string, docs []string) ([]float32, error) {
	if f.err != nil {
		return nil, f.err
	}
	out := make([]float32, len(docs))
	for i, d := range docs {
		out[i] = f.logits[d]
	}
	return out, nil
}

func textLookup(m map[string]string) TextLookup {
	return func(id string) (string, bool) {
		t, ok := m[id]
		return t, ok
	}
}

func TestRerank_ReordersByLogit(t *testing.T) {
	raw := []index.SearchResponse{
		{ID: "a", Score: 0.9}, // hybrid winner, but the cross-encoder disagrees
		{ID: "b", Score: 0.5}, // hybrid loser, but the cross-encoder prefers it
	}
	texts := map[string]string{"a": "irrelevant text", "b": "relevant text"}
	s := &fakeScorer{logits: map[string]float32{
		"irrelevant text": -5,
		"relevant text":   5,
	}}

	got := Rerank(context.Background(), s, "query", raw, textLookup(texts))
	if len(got) != 2 || got[0].ID != "b" || got[1].ID != "a" {
		t.Fatalf("got %+v, want [b, a]", got)
	}
	if got[0].Score <= got[1].Score {
		t.Fatalf("expected b's rescored score > a's: %+v", got)
	}
	// Sigmoid-squashed: strictly between 0 and 1, not a raw logit.
	if got[0].Score <= 0 || got[0].Score >= 1 {
		t.Fatalf("expected sigmoid-squashed score in (0,1), got %v", got[0].Score)
	}
}

func TestRerank_LeavesTailUntouched(t *testing.T) {
	n := TopN + 3
	raw := make([]index.SearchResponse, n)
	texts := map[string]string{}
	logits := map[string]float32{}
	for i := 0; i < n; i++ {
		id := string(rune('a' + i))
		text := id + "-text"
		raw[i] = index.SearchResponse{ID: id, Score: float64(n - i)} // descending
		texts[id] = text
		logits[text] = 0 // all tied; ordering within head shouldn't matter here
	}

	got := Rerank(context.Background(), &fakeScorer{logits: logits}, "q", raw, textLookup(texts))
	if len(got) != n {
		t.Fatalf("got %d results, want %d", len(got), n)
	}
	// The tail (beyond TopN) must be exactly the original order, untouched.
	for i := TopN; i < n; i++ {
		if got[i].ID != raw[i].ID {
			t.Fatalf("tail[%d] = %q, want %q (tail must be untouched)", i, got[i].ID, raw[i].ID)
		}
	}
}

func TestRerank_FallsBackOnScorerError(t *testing.T) {
	raw := []index.SearchResponse{{ID: "a", Score: 0.9}, {ID: "b", Score: 0.5}}
	texts := map[string]string{"a": "x", "b": "y"}
	got := Rerank(context.Background(), &fakeScorer{err: errors.New("ort failure")}, "q", raw, textLookup(texts))
	if len(got) != 2 || got[0].ID != "a" || got[0].Score != 0.9 {
		t.Fatalf("expected raw unchanged on scorer error, got %+v", got)
	}
}

func TestRerank_FallsBackOnMissingText(t *testing.T) {
	raw := []index.SearchResponse{{ID: "a", Score: 0.9}, {ID: "b", Score: 0.5}}
	texts := map[string]string{"a": "x"} // "b" missing
	got := Rerank(context.Background(), &fakeScorer{}, "q", raw, textLookup(texts))
	if len(got) != 2 || got[0].ID != "a" || got[0].Score != 0.9 {
		t.Fatalf("expected raw unchanged when a candidate's text is missing, got %+v", got)
	}
}

func TestRerank_NilScorerIsNoop(t *testing.T) {
	raw := []index.SearchResponse{{ID: "a", Score: 0.9}}
	got := Rerank(context.Background(), nil, "q", raw, textLookup(nil))
	if len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("expected raw unchanged with nil scorer, got %+v", got)
	}
}

func TestRerank_EmptyRaw(t *testing.T) {
	got := Rerank(context.Background(), &fakeScorer{}, "q", nil, textLookup(nil))
	if len(got) != 0 {
		t.Fatalf("expected empty result for empty raw, got %+v", got)
	}
}
