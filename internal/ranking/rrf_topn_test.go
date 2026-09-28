package ranking

import (
	"cmp"
	"fmt"
	"math/rand"
	"sort"
	"testing"
)

// referenceRRF is the straightforward full-sort formulation the optimised
// Score must stay equivalent to.
func referenceRRF(r *RRFRanker, kwIDs []uint64, kwS map[uint64]float64, vcIDs []uint64, vS map[uint64]float64, names map[uint64]string) []ScoredResult {
	kw := append([]uint64(nil), kwIDs...)
	sort.SliceStable(kw, func(i, j int) bool {
		a, b := kw[i], kw[j]
		if d := cmpFloat(kwS[b], kwS[a]); d != 0 {
			return d < 0
		}
		if d := cmpFloat(vS[b], vS[a]); d != 0 {
			return d < 0
		}
		return names[a] < names[b]
	})
	vc := append([]uint64(nil), vcIDs...)
	sort.SliceStable(vc, func(i, j int) bool {
		a, b := vc[i], vc[j]
		if d := cmpFloat(vS[b], vS[a]); d != 0 {
			return d < 0
		}
		return names[a] < names[b]
	})
	sc := map[uint64]float64{}
	for i, id := range kw {
		sc[id] += r.wKw / (r.k + float64(i+1))
	}
	for i, id := range vc {
		sc[id] += r.wVec / (r.k + float64(i+1))
	}
	var out []ScoredResult
	for id, s := range sc {
		if s > 0 {
			out = append(out, ScoredResult{ID: names[id], Score: s})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if d := cmpFloat(out[i].Score, out[j].Score); d != 0 {
			return d > 0
		}
		return cmp.Compare(out[i].ID, out[j].ID) < 0
	})
	if r.topN > 0 && len(out) > r.topN {
		out = out[:r.topN]
	}
	return out
}

func TestRRFScore_MatchesReferenceFullSort(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := 0; trial < 60; trial++ {
		n := 1 + rng.Intn(400)
		names := map[uint64]string{}
		kwS := map[uint64]float64{}
		vS := map[uint64]float64{}
		var kwIDs, vcIDs []uint64
		for i := 0; i < n; i++ {
			id := uint64(i + 1)
			names[id] = fmt.Sprintf("doc-%03d", rng.Intn(n*2)) + fmt.Sprint(i) // unique
			if rng.Intn(3) > 0 {
				kwIDs = append(kwIDs, id)
				kwS[id] = float64(rng.Intn(8)) // deliberate ties
			}
			if rng.Intn(3) > 0 {
				vcIDs = append(vcIDs, id)
				vS[id] = float64(rng.Intn(8)) / 8
			}
		}
		for _, topN := range []int{1, 7, 50, 1000} {
			r := NewWeightedRRFRanker(20, topN, 1, 2)
			got := r.Score(kwIDs, kwS, vcIDs, vS, names)
			want := referenceRRF(r, kwIDs, kwS, vcIDs, vS, names)
			if len(got) != len(want) {
				t.Fatalf("trial %d topN %d: len %d vs %d", trial, topN, len(got), len(want))
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("trial %d topN %d pos %d: got %+v want %+v", trial, topN, i, got[i], want[i])
				}
			}
		}
	}
}
