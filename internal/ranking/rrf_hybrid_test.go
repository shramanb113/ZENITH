package ranking

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestRRF_HybridShortcutMatchesFullSort: the hybrid top-N shortcut must return
// exactly what the general path (sort the whole keyword list, fuse, take the top
// N) returns — same documents, same order, same scores — including under heavy
// score ties, vector-list documents that are absent from the keyword list, and
// vector-list documents ranked deep in it.
func TestRRF_HybridShortcutMatchesFullSort(t *testing.T) {
	for _, tc := range []struct {
		name       string
		n, nv      int
		distinctKw int // few distinct keyword scores = many ties
		distinctV  int
		overlap    float64 // fraction of vector docs also in the keyword list
		topN       int
	}{
		{"typical: 30k keyword, 150 vector", 30000, 150, 5000, 100000, 0.9, 10},
		{"heavy keyword ties", 20000, 150, 20, 100000, 0.9, 10},
		{"heavy ties everywhere", 12000, 200, 5, 4, 0.8, 10},
		{"all keyword tied", 5000, 100, 1, 100000, 1.0, 10},
		{"vector docs disjoint from keyword list", 8000, 150, 3000, 100000, 0.0, 10},
		{"vector docs all in keyword list", 8000, 150, 3000, 100000, 1.0, 10},
		{"larger topN", 9000, 300, 50, 3, 0.7, 60},
		{"tiny vector list", 6000, 1, 100, 100000, 1.0, 10},
		{"vector list near the cap", 20000, 4000, 200, 100000, 0.5, 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for seed := int64(1); seed <= 6; seed++ {
				r := rand.New(rand.NewSource(seed*7919 + int64(tc.n)))
				names := map[uint64]string{}
				newID := func() uint64 {
					id := uint64(len(names) + 1)
					names[id] = fmt.Sprintf("doc-%09d", r.Intn(1_000_000_000)) + fmt.Sprint(id) // unique
					return id
				}
				kwIDs := make([]uint64, tc.n)
				kw := make(map[uint64]float64, tc.n)
				for i := range kwIDs {
					id := newID()
					kwIDs[i] = id
					kw[id] = float64(r.Intn(tc.distinctKw)) + 1
				}
				vcIDs := make([]uint64, 0, tc.nv)
				vc := make(map[uint64]float64, tc.nv)
				for len(vcIDs) < tc.nv {
					var id uint64
					if r.Float64() < tc.overlap {
						id = kwIDs[r.Intn(len(kwIDs))]
						if _, dup := vc[id]; dup {
							continue
						}
					} else {
						id = newID()
					}
					vcIDs = append(vcIDs, id)
					vc[id] = float64(r.Intn(tc.distinctV)) + 1
				}
				lookup := func(id uint64) string { return names[id] }

				fast := NewWeightedRRFRanker(20, tc.topN, 1.0, 2.0)
				if _, ok := fast.hybridTop(kwIDs, kw, vcIDs, vc, lookup); !ok {
					t.Fatalf("seed %d: shortcut did not apply", seed)
				}
				got := fast.Score(kwIDs, kw, vcIDs, vc, lookup)
				// topN larger than both lists together: the shortcut is not eligible.
				ref := NewWeightedRRFRanker(20, tc.n+tc.nv+1, 1.0, 2.0).Score(kwIDs, kw, vcIDs, vc, lookup)
				if len(ref) > tc.topN {
					ref = ref[:tc.topN]
				}
				if len(got) != len(ref) {
					t.Fatalf("seed %d: len shortcut=%d full=%d", seed, len(got), len(ref))
				}
				for i := range got {
					if got[i] != ref[i] {
						t.Fatalf("seed %d rank %d: shortcut=%+v full=%+v", seed, i, got[i], ref[i])
					}
				}
			}
		})
	}
}

// The shortcut must step aside when its preconditions do not hold.
func TestRRF_HybridShortcutEligibility(t *testing.T) {
	r := NewWeightedRRFRanker(20, 10, 1, 2)
	names := func(id uint64) string { return fmt.Sprint(id) }
	ids := func(n int) []uint64 {
		s := make([]uint64, n)
		for i := range s {
			s[i] = uint64(i + 1)
		}
		return s
	}
	scores := func(s []uint64) map[uint64]float64 {
		m := map[uint64]float64{}
		for _, id := range s {
			m[id] = float64(id)
		}
		return m
	}
	kw := ids(100)
	if _, ok := r.hybridTop(kw, scores(kw), nil, nil, names); ok {
		t.Error("no vector list: lexical path applies instead")
	}
	v := ids(5)
	if _, ok := r.hybridTop(kw[:14], scores(kw[:14]), v, scores(v), names); ok {
		t.Error("keyword list no longer than topN+vectors: nothing to prune")
	}
	if _, ok := r.hybridTop(kw, scores(kw), v, map[uint64]float64{1: 1}, names); ok {
		t.Error("vectorScores inconsistent with vectorIDs")
	}
	big := ids(maxHybridShortcutVec + 1)
	if _, ok := r.hybridTop(ids(20000), scores(ids(20000)), big, scores(big), names); ok {
		t.Error("vector list above the cap")
	}
}
