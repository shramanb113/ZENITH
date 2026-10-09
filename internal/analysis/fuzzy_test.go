package analysis

import "testing"

// ─── Levenshtein ─────────────────────────────────────────────────────────────

func TestLevenshtein(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"", "", 0},
		{"a", "a", 0},
		{"abc", "abc", 0},
		{"kitten", "sitting", 3},
		{"kubernetes", "kubrnetes", 1},
		{"kubernetes", "kubrnets", 2},
		{"abc", "xyz", 3},
		{"hello", "helo", 1},
		{"deploy", "depoly", 2}, // transposition costs 2 edits
	}
	for _, c := range cases {
		if dist := Levenshtein(c.a, c.b); dist != c.want {
			t.Errorf("Levenshtein(%q,%q) = %d, want %d", c.a, c.b, dist, c.want)
		}
	}
}

func TestLevenshteinSymmetric(t *testing.T) {
	pairs := [][2]string{
		{"kubernetes", "kubrnetes"},
		{"hello", "helo"},
		{"abc", "ab"},
		{"kitten", "sitting"},
	}
	for _, p := range pairs {
		d1 := Levenshtein(p[0], p[1])
		d2 := Levenshtein(p[1], p[0])
		if d1 != d2 {
			t.Errorf("Levenshtein not symmetric for (%q, %q): %d vs %d", p[0], p[1], d1, d2)
		}
	}
}

// TestLevenshteinExactBeyondOldCap verifies distances above the old
// hard-coded MAX_DISTANCE(2) are still computed exactly, not truncated.
func TestLevenshteinExactBeyondOldCap(t *testing.T) {
	if dist := Levenshtein("kitten", "sitting"); dist != 3 {
		t.Fatalf("Levenshtein(kitten,sitting) = %d, want 3", dist)
	}
	if dist := Levenshtein("abcdef", "uvwxyz"); dist != 6 {
		t.Fatalf("Levenshtein(abcdef,uvwxyz) = %d, want 6", dist)
	}
}

// TestLevenshteinCountsRunesNotBytes verifies a single multi-byte character
// counts as one edit, not several (H4).
func TestLevenshteinCountsRunesNotBytes(t *testing.T) {
	if dist := Levenshtein("café", "cafe"); dist != 1 {
		t.Fatalf(`Levenshtein("café","cafe") = %d, want 1`, dist)
	}
	// Deleting one Devanagari character should be exactly one edit.
	if dist := Levenshtein("पानी", "पानि"); dist != 1 {
		t.Fatalf("Levenshtein(पानी,पानि) = %d, want 1", dist)
	}
}

// ─── BKTree ───────────────────────────────────────────────────────────────────

func TestBKTreeAddAndSearch(t *testing.T) {
	tree := NewBKTree()
	words := []string{"kubernetes", "deploy", "cluster", "network", "service"}
	for _, w := range words {
		tree.Add(w)
	}
	if tree.Size() != len(words) {
		t.Errorf("size = %d, want %d", tree.Size(), len(words))
	}

	// Exact match
	results := tree.Search("kubernetes", 0)
	if len(results) != 1 || results[0].Word != "kubernetes" {
		t.Errorf("exact match failed: %v", results)
	}

	// One-edit typo
	results = tree.Search("kubrnetes", 1)
	found := false
	for _, r := range results {
		if r.Word == "kubernetes" && r.Distance == 1 {
			found = true
		}
	}
	if !found {
		t.Errorf("expected 'kubernetes' with distance 1 in results %v", results)
	}
}

func TestBKTreeNoDuplicates(t *testing.T) {
	tree := NewBKTree()
	for i := 0; i < 5; i++ {
		tree.Add("same")
	}
	if tree.Size() != 1 {
		t.Errorf("expected size 1 after 5 inserts of same word, got %d", tree.Size())
	}
}

func TestBKTreeEmptySearch(t *testing.T) {
	tree := NewBKTree()
	results := tree.Search("anything", 2)
	if results != nil {
		t.Errorf("expected nil from empty tree, got %v", results)
	}
}

func TestBKTreeSearchReturnsSortedByDistance(t *testing.T) {
	tree := NewBKTree()
	// "kub" is 7 edits from "kubernetes"; "kubernetx" is 1 edit
	for _, w := range []string{"kubernetes", "kubernetx", "kubernets"} {
		tree.Add(w)
	}
	results := tree.Search("kubernetes", 2)
	for i := 1; i < len(results); i++ {
		if results[i-1].Distance > results[i].Distance {
			t.Errorf("results not sorted by distance: %v", results)
		}
	}
}

// TestBKTreePruningFindsMatchesBeyondOldFakeDistance is a regression test for
// the fabricated-distance pruning bug: with words inserted whose true
// pairwise distance exceeds the old hard-coded MAX_DISTANCE(2), a query
// within maxDist of a word stored deep in the tree must still be found.
// Before the fix, Add() stored "cdef" under a synthetic key and Search()
// pruned with a synthetic effective distance, so this search returned nothing.
func TestBKTreePruningFindsMatchesBeyondOldFakeDistance(t *testing.T) {
	tree := NewBKTree()
	for _, w := range []string{"ab", "cdef"} {
		tree.Add(w)
	}
	results := tree.Search("cd", 2)
	found := false
	for _, r := range results {
		if r.Word == "cdef" {
			found = true
			if r.Distance != 2 {
				t.Errorf("distance for cdef = %d, want 2", r.Distance)
			}
		}
	}
	if !found {
		t.Errorf("expected 'cdef' (distance 2 from 'cd') in results %v", results)
	}
}

// TestBKTreeSupportsMaxDistAboveOldCap verifies FuzzyMaxDist > 2 (previously
// silently capped) now actually widens the search.
func TestBKTreeSupportsMaxDistAboveOldCap(t *testing.T) {
	tree := NewBKTree()
	tree.Add("kitten")
	results := tree.Search("sitting", 3)
	found := false
	for _, r := range results {
		if r.Word == "kitten" && r.Distance == 3 {
			found = true
		}
	}
	if !found {
		t.Errorf("expected 'kitten' at distance 3 with maxDist=3, got %v", results)
	}
}
