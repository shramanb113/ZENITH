package analysis

import "sort"

// MAX_DISTANCE is the default maximum Levenshtein edit distance considered a
// fuzzy match when a caller doesn't supply its own threshold (e.g. via
// config.FuzzyMaxDist). It is a default, not a hard cap: Levenshtein itself
// always computes the exact distance, however large, so callers may pass any
// threshold they like.
const MAX_DISTANCE = 2

// Levenshtein computes the EXACT edit distance between s1 and s2, operating
// on runes (not bytes) so a single multi-byte character — an accented Latin
// letter, a Devanagari character, etc. — counts as one edit, not several.
//
// There is no early termination: BKTree relies on the triangle inequality to
// prune its search, which only holds if the distances used to place and
// search nodes are exact. Terms are short (index vocabulary, query tokens),
// so the O(n*m) DP cost here is negligible; capping it previously broke
// correctness for no measurable benefit.
func Levenshtein(s1, s2 string) int {
	r1, r2 := []rune(s1), []rune(s2)
	if len(r1) > len(r2) {
		r1, r2 = r2, r1
	}

	n, m := len(r1), len(r2)

	prevRow := make([]int, n+1)
	currRow := make([]int, n+1)

	for i := 0; i <= n; i++ {
		prevRow[i] = i
	}

	for j := 1; j <= m; j++ {
		currRow[0] = j
		for i := 1; i <= n; i++ {
			cost := 1
			if r1[i-1] == r2[j-1] {
				cost = 0
			}
			currRow[i] = min(prevRow[i]+1, currRow[i-1]+1, prevRow[i-1]+cost)
		}
		prevRow, currRow = currRow, prevRow
	}

	return prevRow[n]
}

// -----------------------------------------------------------------------------
// FuzzySearcher
// -----------------------------------------------------------------------------

// FuzzySearcher wraps a BKTree and exposes the fuzzy search interface used by
// the query pipeline. The engine (index/engine.go) holds a *BKTree directly
// for the hot lexicalPass path — FuzzySearcher is the higher-level API for
// components that shouldn't touch BKTree internals.
//
// Lifecycle:
//
//	fs := NewFuzzySearcher()
//	fs.Build(terms)                        // bulk load from existing index
//	fs.Add("kubernetes")                   // incremental as docs are indexed
//	matches := fs.Search("kubrnets", 2)    // query time
type FuzzySearcher struct {
	tree *BKTree
}

// NewFuzzySearcher creates an empty FuzzySearcher.
func NewFuzzySearcher() *FuzzySearcher {
	return &FuzzySearcher{tree: NewBKTree()}
}

// Add inserts a single term. Call this per-term during indexing.
// Safe to call after Build — the tree grows incrementally.
func (f *FuzzySearcher) Add(term string) {
	f.tree.Add(term)
}

// Build populates the fuzzy index from a slice of terms.
// Typically called once when loading an existing index from disk.
func (f *FuzzySearcher) Build(terms []string) {
	for _, t := range terms {
		f.tree.Add(t)
	}
}

// Search returns all indexed terms within maxDist edits of query,
// sorted ascending by edit distance (closest match first).
// Pass maxDist <= 0 to use MAX_DISTANCE.
func (f *FuzzySearcher) Search(query string, maxDist int) []FuzzyMatch {
	if maxDist <= 0 {
		maxDist = MAX_DISTANCE
	}
	results := f.tree.Search(query, maxDist)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})
	return results
}

// Size returns the number of terms in the fuzzy index.
func (f *FuzzySearcher) Size() int {
	return f.tree.Size()
}
