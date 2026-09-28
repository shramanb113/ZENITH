package ranking

import (
	"fmt"
	"math/rand"
	"testing"
)

// The lexical-only fast path (no vector list) must return exactly what the
// general full-sort path returns, including under heavy score ties, where the
// document name breaks the tie.
func TestRRF_LexicalFastPathMatchesFullSort(t *testing.T) {
	for _, tc := range []struct {
		name       string
		n          int
		distinct   int // number of distinct keyword scores (small = many ties)
		topN       int
		withVecTie bool
	}{
		{"few ties", 5000, 4000, 10, false},
		{"heavy ties", 20000, 30, 10, false},
		{"all tied", 3000, 1, 25, false},
		{"ties with vector tiebreak values", 8000, 20, 50, true},
		{"topN near n", 60, 10, 50, false},
		{"topN >= n (general path)", 40, 5, 100, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := rand.New(rand.NewSource(int64(tc.n)))
			ids := make([]uint64, tc.n)
			kw := make(map[uint64]float64, tc.n)
			vec := map[uint64]float64{}
			names := make(map[uint64]string, tc.n)
			for i := range ids {
				id := uint64(r.Int63())
				ids[i] = id
				kw[id] = float64(r.Intn(tc.distinct))
				names[id] = fmt.Sprintf("doc-%08d", r.Intn(1_000_000_000)) // unique enough; ties broken by name
				if tc.withVecTie {
					vec[id] = float64(r.Intn(3))
				}
			}
			lookup := func(id uint64) string { return names[id] }
			// Vector scores only break ties inside the keyword list here; the vector
			// LIST is empty, which is what selects the fast path.
			fast := NewWeightedRRFRanker(20, tc.topN, 1.0, 2.0).Score(ids, kw, nil, vec, lookup)
			full := NewWeightedRRFRanker(20, tc.n+1, 1.0, 2.0).Score(ids, kw, nil, vec, lookup) // topN > n: no shortcut
			if len(full) > tc.topN {
				full = full[:tc.topN]
			}
			if len(fast) != len(full) {
				t.Fatalf("len fast=%d full=%d", len(fast), len(full))
			}
			for i := range fast {
				if fast[i] != full[i] {
					t.Fatalf("rank %d: fast=%+v full=%+v", i, fast[i], full[i])
				}
			}
		})
	}
}
