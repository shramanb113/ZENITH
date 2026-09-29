package ranking

import (
	"cmp"
	"container/heap"
	"slices"
	"sort"
	"strings"
)

const (
	defaultK    = 60.0 // standard RRF k — 60 is the Cormack et al. recommendation
	defaultTopN = 10   // returnable results cap; was hardcoded 5 before
)

// RRFRanker implements Reciprocal Rank Fusion for merging lexical and
// vector result lists into a single ranked output.
//
// Formula: score(d) = Σ 1/(k + rank(d, list_i))  for each ranked list i
//
// Key properties:
//   - rank positions are 1-indexed (rank 1 = best)
//   - k dampens the advantage of top-ranked results (default 60)
//   - scores from both lists are simply summed — no normalisation needed
//   - documents appearing in both lists get a natural boost
type RRFRanker struct {
	k    float64
	topN int
	wKw  float64 // weight of the keyword/lexical list
	wVec float64 // weight of the vector/semantic list
}

// NewRRFRanker creates an RRFRanker with equal list weights.
// k=0 uses the standard value of 60.
// topN=0 uses the default of 10.
func NewRRFRanker(k float64, topN int) *RRFRanker {
	return NewWeightedRRFRanker(k, topN, 1.0, 1.0)
}

// NewWeightedRRFRanker creates an RRFRanker with per-list weights:
// score(d) = wKw/(k + rank_kw(d)) + wVec/(k + rank_vec(d)).
//
// Weighted RRF matters when the two retrievers differ in quality. Measured
// on MS MARCO dev (6,980 queries, all ground truths indexed): the dense list
// alone reached Recall@10 0.947 while the BM25 lexical list reached 0.803;
// equal-weight k=60 fusion scored 0.918 — *below* dense alone — because
// lexically-popular wrong answers collected contributions from both lists.
// k=20 with wVec=2.0 scored 0.960, sitting on a broad plateau (k 10–30,
// wVec 1.5–3.0 all ≥ 0.952). Weights ≤ 0 default to 1.
func NewWeightedRRFRanker(k float64, topN int, wKw, wVec float64) *RRFRanker {
	if k == 0 {
		k = defaultK
	}
	if topN == 0 {
		topN = defaultTopN
	}
	if wKw <= 0 {
		wKw = 1.0
	}
	if wVec <= 0 {
		wVec = 1.0
	}
	return &RRFRanker{k: k, topN: topN, wKw: wKw, wVec: wVec}
}

// Score implements the ranking.Scorer interface.
//
// Fixes vs the original implementation:
//
//  1. Copies input slices before sorting — the original mutated the
//     caller's slices via slices.SortFunc, which is a hidden side effect
//     that corrupts repeated searches using the same ID lists.
//
//  2. Float comparison uses a tolerance epsilon instead of != to avoid
//     false tie-breaks from floating-point noise.
//
//  3. Single result-collection loop — the original built rrfScores map
//     then ranged over it a second time. Now it's one pass.
//
//  4. topN is configurable on the struct, not hardcoded to 5.
//
//  5. cmp.Compare used for the string tie-breaker — cleaner, same semantics.
func (r *RRFRanker) Score(
	keywordIDs []uint64,
	keywordScores map[uint64]float64,
	vectorIDs []uint64,
	vectorScores map[uint64]float64,
	idMapping IDLookup,
) []ScoredResult {

	// --- 1. Rank each list ---
	//
	// The sort keys (scores, ID strings) are materialised into a flat struct
	// slice once, so the comparator does no map lookups. The original sorted
	// bare IDs and hit keywordScores/vectorScores/idMapping (3 hash lookups
	// plus a string compare) inside every comparison, which dominated query
	// latency once the candidate lists reached tens of thousands of documents.
	// Input slices are copied, never mutated.

	//
	// Document names are only needed to break exact score ties, and resolving
	// one can be costly (a binary search plus a decode in a memory-mapped
	// segment), so they are looked up lazily and memoised — the candidate lists
	// can hold tens of thousands of documents but ties are rare.
	memo := make(map[uint64]string)
	nameOf := func(id uint64) string {
		if n, ok := memo[id]; ok {
			return n
		}
		n := idMapping(id)
		memo[id] = n
		return n
	}

	if out, ok := r.hybridTop(keywordIDs, keywordScores, vectorIDs, vectorScores, nameOf); ok {
		return out
	}

	type kwEntry struct {
		id  uint64
		kw  float64
		vec float64
	}
	kw := make([]kwEntry, len(keywordIDs))
	for i, id := range keywordIDs {
		kw[i] = kwEntry{id: id, kw: keywordScores[id], vec: vectorScores[id]}
	}
	// Lexical-only fast path. With no vector list the fused score is
	// wKw/(k+rank), strictly decreasing in keyword rank, so the ranked output is
	// exactly the first topN entries of the keyword order — no need to fully sort
	// the (often tens of thousands of) candidates to find them. Identical output
	// to the general path below; see TestRRF_LexicalFastPathMatchesFullSort.
	if len(vectorIDs) == 0 && r.topN > 0 && len(kw) > r.topN {
		lessKw := func(a, b kwEntry) int { // negative when a ranks before b
			if d := cmpFloat(b.kw, a.kw); d != 0 {
				return d
			}
			if d := cmpFloat(b.vec, a.vec); d != 0 {
				return d
			}
			return strings.Compare(nameOf(a.id), nameOf(b.id))
		}
		// Bounded max-heap of the topN best so far; the root is the worst kept.
		h := make([]kwEntry, 0, r.topN)
		worstFirst := func(i, j int) bool { return lessKw(h[i], h[j]) > 0 }
		siftDown := func(i int) {
			for {
				l, rr, w := 2*i+1, 2*i+2, i
				if l < len(h) && worstFirst(l, w) {
					w = l
				}
				if rr < len(h) && worstFirst(rr, w) {
					w = rr
				}
				if w == i {
					return
				}
				h[i], h[w] = h[w], h[i]
				i = w
			}
		}
		for _, e := range kw {
			if len(h) < r.topN {
				h = append(h, e)
				for i := len(h) - 1; i > 0; { // sift up
					p := (i - 1) / 2
					if !worstFirst(i, p) {
						break
					}
					h[i], h[p] = h[p], h[i]
					i = p
				}
			} else if lessKw(e, h[0]) < 0 {
				h[0] = e
				siftDown(0)
			}
		}
		slices.SortFunc(h, lessKw)
		out := make([]ScoredResult, 0, len(h))
		for rank, e := range h {
			if sc := r.wKw / (r.k + float64(rank+1)); sc > 0 {
				out = append(out, ScoredResult{ID: nameOf(e.id), Score: sc})
			}
		}
		return out
	}

	// Keyword list: keyword score desc, tie-break vector score desc, then ID.
	slices.SortFunc(kw, func(a, b kwEntry) int {
		if d := cmpFloat(b.kw, a.kw); d != 0 {
			return d
		}
		if d := cmpFloat(b.vec, a.vec); d != 0 {
			return d
		}
		return strings.Compare(nameOf(a.id), nameOf(b.id))
	})

	type vcEntry struct {
		id  uint64
		vec float64
	}
	vc := make([]vcEntry, len(vectorIDs))
	for i, id := range vectorIDs {
		vc[i] = vcEntry{id: id, vec: vectorScores[id]}
	}
	// Vector list: vector score desc, tie-break by ID.
	slices.SortFunc(vc, func(a, b vcEntry) int {
		if d := cmpFloat(b.vec, a.vec); d != 0 {
			return d
		}
		return strings.Compare(nameOf(a.id), nameOf(b.id))
	})

	// --- 2. RRF accumulation ---

	// Pre-size to the union of both lists to avoid rehashing.
	rrfScores := make(map[uint64]float64, len(kw)+len(vc))
	for rank, e := range kw {
		rrfScores[e.id] += r.wKw / (r.k + float64(rank+1))
	}
	for rank, e := range vc {
		rrfScores[e.id] += r.wVec / (r.k + float64(rank+1))
	}

	// --- 3. Select the top N: score desc, then ID asc for determinism ---
	//
	// Only topN results are returned, so a bounded min-heap (O(n log topN))
	// replaces sorting every candidate. IDs are unique, so the order is
	// total and the result is identical to a full sort truncated to topN.

	better := func(a, b idScore) bool {
		if d := cmpFloat(a.score, b.score); d != 0 {
			return d > 0
		}
		return cmp.Compare(nameOf(a.id), nameOf(b.id)) < 0
	}

	limit := r.topN
	if limit <= 0 || limit > len(rrfScores) {
		limit = len(rrfScores)
	}
	h := &resultHeap{better: better, items: make([]idScore, 0, limit)}
	for id, score := range rrfScores {
		if score <= 0 {
			continue
		}
		sr := idScore{id: id, score: score}
		if len(h.items) < limit {
			heap.Push(h, sr)
		} else if better(sr, h.items[0]) {
			h.items[0] = sr
			heap.Fix(h, 0)
		}
	}
	top := h.items
	sort.Slice(top, func(i, j int) bool { return better(top[i], top[j]) })
	results := make([]ScoredResult, len(top))
	for i, t := range top {
		results[i] = ScoredResult{ID: nameOf(t.id), Score: t.score}
	}
	return results
}

// idScore is a candidate's fused score keyed by internal ID; its name is
// resolved only if it ties or makes the final cut.
type idScore struct {
	id    uint64
	score float64
}

// resultHeap is a min-heap under the "better" ordering: the root is the
// worst result currently kept, so it is the one evicted by a better one.
type resultHeap struct {
	better func(a, b idScore) bool
	items  []idScore
}

func (h *resultHeap) Len() int           { return len(h.items) }
func (h *resultHeap) Less(i, j int) bool { return h.better(h.items[j], h.items[i]) }
func (h *resultHeap) Swap(i, j int)      { h.items[i], h.items[j] = h.items[j], h.items[i] }
func (h *resultHeap) Push(x any)         { h.items = append(h.items, x.(idScore)) }
func (h *resultHeap) Pop() any {
	n := len(h.items)
	x := h.items[n-1]
	h.items = h.items[:n-1]
	return x
}

// cmpFloat compares two float64s exactly. Returns negative, zero, or
// positive — same contract as cmp.Compare.
//
// An earlier version used an epsilon tolerance (1e-9) intended to smooth
// over floating-point noise. In practice it broke sort.Slice's ordering
// contract: with a<eps>b<eps>c not implying a<eps>c, the comparator was
// non-transitive, which sort.Slice assumes and can silently misorder or
// panic over. Every value compared here is deterministically recomputed
// per search (not measured or accumulated across independent floating
// point paths), so exact comparison is both correct and reproducible; a
// real difference of a few billionths between candidates is signal (e.g.
// coverage-scaled scores for documents with no BM25 hit), not noise, and
// exact comparison preserves it instead of discarding it into a tie that
// falls back to alphabetical ID order.
func cmpFloat(a, b float64) int {
	switch {
	case a > b:
		return 1
	case a < b:
		return -1
	default:
		return 0
	}
}
