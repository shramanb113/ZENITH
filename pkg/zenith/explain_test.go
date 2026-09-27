package zenith_test

import (
	"context"
	"math"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/shramanb113/ZENITH/pkg/zenith"
)

func openBM25(t *testing.T, docs map[string]string) *zenith.DB {
	t.Helper()
	db, err := zenith.Open(":memory:", zenith.WithBM25Only(), zenith.WithoutWordVectors())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := db.AddBatch(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	return db
}

func explain(t *testing.T, db *zenith.DB, q string) []zenith.Result {
	t.Helper()
	res, err := db.Search(context.Background(), q, zenith.Explain(), zenith.Limit(100))
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func find(res []zenith.Result, id string) *zenith.Result {
	for i := range res {
		if res[i].ID == id {
			return &res[i]
		}
	}
	return nil
}

func TestExplainExactTerm(t *testing.T) {
	db := openBM25(t, map[string]string{"0": "basement waterlogging every monsoon", "1": "lovely park nearby"})
	res := explain(t, db, "waterlogging")
	r := find(res, "0")
	if r == nil || r.Signals == nil {
		t.Fatalf("doc 0 missing or no signals: %+v", res)
	}
	if find(res, "1") != nil {
		t.Fatal("doc 1 has no signal and must not be returned")
	}
	if len(r.Signals.Terms) != 1 || r.Signals.Terms[0].Dist != 0 || r.Signals.Terms[0].Synonym ||
		r.Signals.Terms[0].Term != r.Signals.Terms[0].Matched {
		t.Fatalf("want one exact term hit, got %+v", r.Signals.Terms)
	}
	if r.Signals.Lexical <= 0 {
		t.Fatalf("Lexical = %v, want > 0", r.Signals.Lexical)
	}
	if len(r.Signals.QueryTerms) != 1 {
		t.Fatalf("QueryTerms = %v", r.Signals.QueryTerms)
	}
}

func TestExplainFuzzyTerm(t *testing.T) {
	db := openBM25(t, map[string]string{"0": "seapage in bedroom walls"})
	r := find(explain(t, db, "seepage"), "0")
	if r == nil || len(r.Signals.Terms) != 1 || r.Signals.Terms[0].Dist != 1 {
		t.Fatalf("want one fuzzy hit at distance 1, got %+v", r)
	}
	if r.Signals.Terms[0].Matched == r.Signals.Terms[0].Term {
		t.Fatal("fuzzy Matched must be the document's term, not the query's")
	}
}

func TestExplainSynonymTerm(t *testing.T) {
	// "latency" → "latenc", whose built-in synonyms include "delay".
	db := openBM25(t, map[string]string{"0": "huge delay today"})
	r := find(explain(t, db, "latency"), "0")
	if r == nil || len(r.Signals.Terms) != 1 || !r.Signals.Terms[0].Synonym || r.Signals.Terms[0].Matched != "delay" {
		t.Fatalf("want synonym hit on delay, got %+v", r)
	}
}

func TestExplainDevanagari(t *testing.T) {
	db := openBM25(t, map[string]string{"0": "बेसमेंट में पानी भर जाता है"})
	r := find(explain(t, db, "पानी भर"), "0")
	if r == nil || len(r.Signals.Terms) != 2 {
		t.Fatalf("want 2 Devanagari term hits, got %+v", r)
	}
}

func TestExplainNoHitsIsEmptyNotNil(t *testing.T) {
	db := openBM25(t, map[string]string{"0": "lovely park"})
	res := explain(t, db, "sewage")
	if res == nil || len(res) != 0 {
		t.Fatalf("want empty non-nil slice, got %#v", res)
	}
}

func TestSearchWithoutExplainHasNoSignals(t *testing.T) {
	db := openBM25(t, map[string]string{"0": "basement waterlogging"})
	res, err := db.Search(context.Background(), "waterlogging")
	if err != nil || len(res) != 1 || res[0].Signals != nil {
		t.Fatalf("plain Search must not fill Signals: %+v %v", res, err)
	}
}

// axisEmbedder maps texts containing "flood" to e0 and everything else to e1 (unit vectors),
// and counts every text it is asked to embed.
type axisEmbedder struct{ calls atomic.Int64 }

func (a *axisEmbedder) vec(text string) []float32 {
	v := make([]float32, 4)
	if strings.Contains(strings.ToLower(text), "flood") {
		v[0] = 1
	} else {
		v[1] = 1
	}
	return v
}
func (a *axisEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	a.calls.Add(1)
	return a.vec(text), nil
}
func (a *axisEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	a.calls.Add(int64(len(texts)))
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = a.vec(t)
	}
	return out, nil
}
func (a *axisEmbedder) Dimensions() int { return 4 }

func TestExplainSemanticIsRawCosine(t *testing.T) {
	emb := &axisEmbedder{}
	db, err := zenith.Open(":memory:", zenith.WithEmbedder(emb), zenith.WithoutWordVectors(), zenith.WithCacheSize(0))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := db.AddBatch(context.Background(), map[string]string{"0": "flood river bank", "1": "quiet street"}); err != nil {
		t.Fatal(err)
	}
	res := explain(t, db, "flood")
	r0 := find(res, "0")
	if r0 == nil || math.Abs(r0.Signals.Semantic-1) > 1e-3 {
		t.Fatalf("doc 0 semantic = %+v, want ≈1", r0)
	}
	if r1 := find(res, "1"); r1 != nil {
		t.Fatalf("doc 1 is orthogonal with no terms and must be absent, got %+v", r1)
	}
	for _, r := range res {
		if r.Score < 0 || r.Score > 1 {
			t.Fatalf("Score out of range: %v", r.Score)
		}
	}
}

func TestWithoutWordVectorsEmbedsOnlyDocuments(t *testing.T) {
	emb := &axisEmbedder{}
	db, err := zenith.Open(":memory:", zenith.WithEmbedder(emb), zenith.WithoutWordVectors(), zenith.WithCacheSize(0))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	docs := map[string]string{"0": "flood river bank today", "1": "quiet street with many trees"}
	if err := db.AddBatch(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	if got := emb.calls.Load(); got != int64(len(docs)) {
		t.Fatalf("embedded %d texts, want %d (documents only, no per-token word vectors)", got, len(docs))
	}
}
