package ranking

import (
	"math"
	"math/rand"
	"testing"
)

// TestQuery_BitIdenticalAcrossCalls pins down that a document's BM25 score is
// a pure function of the index and the query: repeated calls must return the
// same bits. The fused ranking compares scores exactly (see cmpFloat in
// rrf.go), so a last-bit difference between two otherwise identical engines
// can swap two near-tied documents and move both by a whole RRF rank step.
// Summing per-term contributions in map-iteration order broke this: float
// addition is not associative, and on arm64 the compiler fuses the
// accumulation into an FMA, which makes even a two-term sum order-dependent.
func TestQuery_BitIdenticalAcrossCalls(t *testing.T) {
	s := NewBM25Scorer(BM25Params{})
	randomCorpus(s, rand.New(rand.NewSource(3)), 400, 40)
	queries := [][]string{
		{"w1", "w2", "w3", "w5", "w8", "w13"},
		{"w0", "w4", "w9", "w16", "w25", "w36", "w4"},
		{"w2", "w3"},
	}
	for _, q := range queries {
		first := map[uint64]float64{}
		for _, r := range s.Query(q) {
			first[r.DocID] = r.Score
		}
		for call := 0; call < 200; call++ {
			res := s.Query(q)
			if len(res) != len(first) {
				t.Fatalf("query %v call %d: %d results, first call had %d", q, call, len(res), len(first))
			}
			for _, r := range res {
				if math.Float64bits(r.Score) != math.Float64bits(first[r.DocID]) {
					t.Fatalf("query %v call %d: doc %d scored %.17g, first call %.17g",
						q, call, r.DocID, r.Score, first[r.DocID])
				}
			}
		}
	}
}
