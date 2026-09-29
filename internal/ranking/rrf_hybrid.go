package ranking

import (
	"slices"
	"sort"
	"strings"
)

// maxHybridShortcutVec bounds the vector list the shortcut handles. ANN hands
// rank fusion ~150 candidates; an exact scan can hand it the whole corpus, and
// there the general path is as good.
const maxHybridShortcutVec = 4096

// hybridTop is the exact top-topN of the fused list without sorting the whole
// keyword list. ok is false when the shortcut does not apply (the caller then
// takes the general path).
//
// The keyword list is typically tens of thousands of candidates (every prefix,
// phonetic and fuzzy match) but the vector list only a few hundred. A fused
// score is wKw/(k+rank_kw) + wVec/(k+rank_vec), so a document can reach the top
// only if
//
//   - it is in the vector list (a few hundred documents; their exact keyword
//     rank is the number of keyword entries that order before them, found in one
//     pass by binary-searching each entry against the sorted vector-list
//     documents), or
//   - it has no vector score, in which case its score is wKw/(k+rank_kw),
//     strictly decreasing in rank, so only the topN best such entries matter
//     (a bounded heap; the rank of each is its position among them plus the
//     vector-list documents that order before it).
//
// The keyword order is the one the general path sorts by: keyword score desc,
// vector score desc, name asc — a total order, so ranks are exact. The result
// is identical to the general path's (TestRRF_HybridShortcutMatchesFullSort).
//
// Requirements, all of which every caller meets: IDs are unique within each
// list, and vectorScores has an entry per vectorIDs entry and no others.
func (r *RRFRanker) hybridTop(
	keywordIDs []uint64,
	keywordScores map[uint64]float64,
	vectorIDs []uint64,
	vectorScores map[uint64]float64,
	nameOf func(uint64) string,
) ([]ScoredResult, bool) {
	nv := len(vectorIDs)
	if r.topN <= 0 || nv == 0 || nv > maxHybridShortcutVec ||
		len(vectorScores) != nv || len(keywordIDs) <= r.topN+nv {
		return nil, false
	}

	type entry struct {
		id      uint64
		kw, vec float64
	}
	// before reports whether a orders strictly before b in the keyword list.
	before := func(a, b entry) bool {
		if a.kw != b.kw {
			return a.kw > b.kw
		}
		if a.vec != b.vec {
			return a.vec > b.vec
		}
		return strings.Compare(nameOf(a.id), nameOf(b.id)) < 0
	}

	inVec := make(map[uint64]int, nv) // id -> index into vec
	vec := make([]entry, nv)
	for i, id := range vectorIDs {
		inVec[id] = i
		vec[i] = entry{id: id, vec: vectorScores[id]}
	}

	// Pass 1: keyword scores; find which vector-list documents are also in the
	// keyword list.
	kwScore := make([]float64, len(keywordIDs))
	var both []entry // vector-list documents that are in the keyword list
	inKw := make([]bool, nv)
	for i, id := range keywordIDs {
		s := keywordScores[id]
		kwScore[i] = s
		if j, ok := inVec[id]; ok {
			vec[j].kw = s
			inKw[j] = true
		}
	}
	for j := range vec {
		if inKw[j] {
			both = append(both, vec[j])
		}
	}
	slices.SortFunc(both, func(a, b entry) int {
		switch {
		case before(a, b):
			return -1
		case before(b, a):
			return 1
		}
		return 0
	})
	m := len(both)

	// Pass 2: the topN best keyword entries with no vector score, and, for each
	// both[j], how many such entries order before it.
	var minBoth entry
	if m > 0 {
		minBoth = both[m-1]
	}
	h := make([]entry, 0, r.topN) // max-heap under before: the root is the worst kept
	worse := func(i, j int) bool { return before(h[j], h[i]) }
	siftDown := func(i int) {
		for {
			l, rr, w := 2*i+1, 2*i+2, i
			if l < len(h) && worse(l, w) {
				w = l
			}
			if rr < len(h) && worse(rr, w) {
				w = rr
			}
			if w == i {
				return
			}
			h[i], h[w] = h[w], h[i]
			i = w
		}
	}
	diff := make([]int, m+1)
	for i, id := range keywordIDs {
		if _, ok := inVec[id]; ok {
			continue
		}
		e := entry{id: id, kw: kwScore[i]}
		if m > 0 && before(e, minBoth) {
			// e orders before both[j] for every j from the first one it precedes.
			j := sort.Search(m, func(j int) bool { return before(e, both[j]) })
			diff[j]++
		}
		if len(h) < r.topN {
			h = append(h, e)
			for c := len(h) - 1; c > 0; { // sift up
				p := (c - 1) / 2
				if !worse(c, p) {
					break
				}
				h[c], h[p] = h[p], h[c]
				c = p
			}
		} else if before(e, h[0]) {
			h[0] = e
			siftDown(0)
		}
	}
	slices.SortFunc(h, func(a, b entry) int {
		switch {
		case before(a, b):
			return -1
		case before(b, a):
			return 1
		}
		return 0
	})

	// Exact keyword ranks (1-based).
	kwRank := make(map[uint64]int, m+len(h))
	before0 := 0 // keyword-only entries ordering before both[j]
	for j := 0; j < m; j++ {
		before0 += diff[j]
		kwRank[both[j].id] = 1 + j + before0
	}
	for k, e := range h {
		j := sort.Search(m, func(j int) bool { return before(e, both[j]) }) // both[:j] order before e
		kwRank[e.id] = 1 + k + j
	}

	// Vector ranks: vector score desc, name asc.
	byVec := slices.Clone(vec)
	slices.SortFunc(byVec, func(a, b entry) int {
		if a.vec != b.vec {
			if a.vec > b.vec {
				return -1
			}
			return 1
		}
		return strings.Compare(nameOf(a.id), nameOf(b.id))
	})

	cand := make([]idScore, 0, len(h)+nv)
	seen := make(map[uint64]bool, len(h)+nv)
	add := func(id uint64, vecRank int) {
		var s float64
		if kr, ok := kwRank[id]; ok {
			s += r.wKw / (r.k + float64(kr))
		}
		if vecRank > 0 {
			s += r.wVec / (r.k + float64(vecRank))
		}
		if s > 0 && !seen[id] {
			seen[id] = true
			cand = append(cand, idScore{id: id, score: s})
		}
	}
	for rank, e := range byVec {
		add(e.id, rank+1)
	}
	for _, e := range h {
		add(e.id, 0)
	}
	slices.SortFunc(cand, func(a, b idScore) int {
		if d := cmpFloat(b.score, a.score); d != 0 {
			return d
		}
		return strings.Compare(nameOf(a.id), nameOf(b.id))
	})
	if len(cand) > r.topN {
		cand = cand[:r.topN]
	}
	out := make([]ScoredResult, len(cand))
	for i, c := range cand {
		out[i] = ScoredResult{ID: nameOf(c.id), Score: c.score}
	}
	return out, true
}
