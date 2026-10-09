package ranking

import (
	"math"
	"testing"
)

// mapLookup adapts a map to an IDLookup for tests; IDs missing from the map
// resolve to "".
func mapLookup(m map[uint64]string) IDLookup {
	return func(id uint64) string { return m[id] }
}

func makeMapping(ids ...string) ([]uint64, map[uint64]float64, map[uint64]string) {
	mapping := make(map[uint64]string, len(ids))
	idList := make([]uint64, len(ids))
	scores := make(map[uint64]float64, len(ids))
	for i, id := range ids {
		uid := uint64(i + 1)
		idList[i] = uid
		mapping[uid] = id
		scores[uid] = float64(len(ids) - i) // highest score for first element
	}
	return idList, scores, mapping
}

func TestRRFScore_BasicOrdering(t *testing.T) {
	ranker := NewRRFRanker(60, 0)

	kwIDs, kwScores, mapping := makeMapping("doc_a", "doc_b", "doc_c")
	vcIDs, vcScores, _ := makeMapping("doc_a", "doc_c", "doc_b")
	// doc_c appears at rank 2 in both lists — should outscore doc_b.

	results := ranker.Score(kwIDs, kwScores, vcIDs, vcScores, mapLookup(mapping))
	if len(results) == 0 {
		t.Fatal("expected non-empty results")
	}
	if results[0].ID != "doc_a" {
		t.Errorf("expected doc_a first, got %q", results[0].ID)
	}
}

func TestRRFScore_DocumentInBothLists(t *testing.T) {
	ranker := NewRRFRanker(60, 0)

	// shared_doc appears in both — its RRF score = 1/(60+1) + 1/(60+1)
	// unique_doc appears only in keyword list at rank 1 — score = 1/(60+1)
	kwIDs := []uint64{1, 2}
	kwScores := map[uint64]float64{1: 100, 2: 50}
	vcIDs := []uint64{2}
	vcScores := map[uint64]float64{2: 80}
	mapping := map[uint64]string{1: "unique_doc", 2: "shared_doc"}

	results := ranker.Score(kwIDs, kwScores, vcIDs, vcScores, mapLookup(mapping))

	var sharedScore, uniqueScore float64
	for _, r := range results {
		switch r.ID {
		case "shared_doc":
			sharedScore = r.Score
		case "unique_doc":
			uniqueScore = r.Score
		}
	}
	if sharedScore <= uniqueScore {
		t.Errorf("shared_doc (%.6f) should outscore unique_doc (%.6f)", sharedScore, uniqueScore)
	}
}

func TestRRFScore_DoesNotMutateInputSlices(t *testing.T) {
	ranker := NewRRFRanker(60, 0)
	kwIDs := []uint64{3, 1, 2}
	vcIDs := []uint64{2, 3, 1}
	kwScores := map[uint64]float64{1: 10, 2: 20, 3: 30}
	vcScores := map[uint64]float64{1: 5, 2: 15, 3: 25}
	mapping := map[uint64]string{1: "a", 2: "b", 3: "c"}

	kwCopy := make([]uint64, len(kwIDs))
	vcCopy := make([]uint64, len(vcIDs))
	copy(kwCopy, kwIDs)
	copy(vcCopy, vcIDs)

	ranker.Score(kwIDs, kwScores, vcIDs, vcScores, mapLookup(mapping))

	for i := range kwIDs {
		if kwIDs[i] != kwCopy[i] {
			t.Errorf("kwIDs mutated at index %d: was %d, now %d", i, kwCopy[i], kwIDs[i])
		}
	}
	for i := range vcIDs {
		if vcIDs[i] != vcCopy[i] {
			t.Errorf("vcIDs mutated at index %d: was %d, now %d", i, vcCopy[i], vcIDs[i])
		}
	}
}

func TestRRFScore_EmptyInputs(t *testing.T) {
	ranker := NewRRFRanker(60, 0)
	results := ranker.Score(nil, nil, nil, nil, nil)
	if len(results) != 0 {
		t.Errorf("expected empty results for empty inputs, got %v", results)
	}
}

func TestRRFScore_OnlyKeyword(t *testing.T) {
	ranker := NewRRFRanker(60, 0)
	kwIDs := []uint64{1, 2}
	kwScores := map[uint64]float64{1: 50, 2: 30}
	mapping := map[uint64]string{1: "alpha", 2: "beta"}
	results := ranker.Score(kwIDs, kwScores, nil, nil, mapLookup(mapping))
	if len(results) == 0 {
		t.Fatal("expected results from keyword-only input")
	}
	// Score for rank-1 doc: 1/(60+1) ≈ 0.01639
	expected := 1.0 / (60.0 + 1.0)
	if math.Abs(results[0].Score-expected) > 1e-9 {
		t.Errorf("rank-1 RRF score = %.9f, want %.9f", results[0].Score, expected)
	}
}

func TestRRFScore_TopNCap(t *testing.T) {
	ranker := NewRRFRanker(60, 3) // topN = 3
	ids := make([]uint64, 10)
	scores := make(map[uint64]float64, 10)
	mapping := make(map[uint64]string, 10)
	for i := range 10 {
		ids[i] = uint64(i + 1)
		scores[uint64(i+1)] = float64(10 - i)
		mapping[uint64(i+1)] = "doc"
	}
	results := ranker.Score(ids, scores, nil, nil, mapLookup(mapping))
	if len(results) > 3 {
		t.Errorf("expected at most 3 results with topN=3, got %d", len(results))
	}
}

func TestRRFScore_DeterministicOrder(t *testing.T) {
	ranker := NewRRFRanker(60, 0)
	kwIDs := []uint64{1, 2}
	kwScores := map[uint64]float64{1: 100, 2: 100}
	mapping := map[uint64]string{1: "z_doc", 2: "a_doc"}

	r1 := ranker.Score(kwIDs, kwScores, nil, nil, mapLookup(mapping))
	r2 := ranker.Score(kwIDs, kwScores, nil, nil, mapLookup(mapping))

	if len(r1) != len(r2) {
		t.Fatal("non-deterministic result length")
	}
	for i := range r1 {
		if r1[i].ID != r2[i].ID {
			t.Errorf("non-deterministic at index %d: %q vs %q", i, r1[i].ID, r2[i].ID)
		}
	}
}

// cmpFloat previously used an epsilon tolerance (1e-9) that broke
// transitivity (a≈b, b≈c, but a>c is possible) and mishandled equal Infs.
// It must now be an exact comparator so tiny-but-real differences (e.g.
// coverage scores scaled to sit below BM25's range) are preserved instead of
// being folded into a tie.
func TestCmpFloat_ExactAndTransitive(t *testing.T) {
	if got := cmpFloat(3e-10, 6e-10); got >= 0 {
		t.Errorf("cmpFloat(3e-10, 6e-10) = %d, want negative", got)
	}
	if got := cmpFloat(1.0, 1.0); got != 0 {
		t.Errorf("cmpFloat(1.0, 1.0) = %d, want 0", got)
	}
	if got := cmpFloat(math.Inf(1), math.Inf(1)); got != 0 {
		t.Errorf("cmpFloat(+Inf, +Inf) = %d, want 0", got)
	}
	if got := cmpFloat(math.Inf(1), 1.0); got <= 0 {
		t.Errorf("cmpFloat(+Inf, 1.0) = %d, want positive", got)
	}
}
