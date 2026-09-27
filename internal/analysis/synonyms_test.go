package analysis

import (
	"slices"
	"testing"
)

func TestSynonyms(t *testing.T) {
	// Known entry — must have synonyms.
	syns := Synonyms("search")
	if len(syns) == 0 {
		t.Error("Synonyms('search') returned empty slice")
	}
	// Unknown entry — must return nil.
	if Synonyms("xyzzy") != nil {
		t.Error("Synonyms for unknown term should be nil")
	}
}

func TestExpandWithSynonyms_NoDuplicates(t *testing.T) {
	tokens := []string{"search", "search"}
	expanded := ExpandWithSynonyms(tokens)
	seen := make(map[string]int)
	for _, tok := range expanded {
		seen[tok]++
	}
	for tok, count := range seen {
		if count > 1 {
			t.Errorf("duplicate token %q in expanded result", tok)
		}
	}
}

func TestExpandWithSynonyms_PreservesOriginal(t *testing.T) {
	tokens := []string{"search", "cluster"}
	expanded := ExpandWithSynonyms(tokens)
	for _, orig := range tokens {
		if !slices.Contains(expanded, orig) {
			t.Errorf("original token %q missing from expanded result %v", orig, expanded)
		}
	}
}

func TestExpandWithSynonyms_AddsSynonyms(t *testing.T) {
	// "search" maps to ["retriev","queri","find","lookup"]
	expanded := ExpandWithSynonyms([]string{"search"})
	if len(expanded) <= 1 {
		t.Errorf("expected synonyms added for 'search', got %v", expanded)
	}
}

func TestExpandWithSynonyms_Empty(t *testing.T) {
	got := ExpandWithSynonyms(nil)
	if len(got) != 0 {
		t.Errorf("expected empty for nil input, got %v", got)
	}
}

// TestSynonymKeysReachableFromNaturalWords is a regression test for H2. Each
// of these keys used to be stored unstemmed (or with a guessed-wrong stem),
// so a real query using the natural word could never reach it via
// ExpandWithSynonyms — query tokens are always stem()'s OUTPUT, never the raw
// word. Note this deliberately checks stem(naturalWord)==key, NOT
// stem(key)==key: Porter2 is not idempotent (e.g. stem("database")=="databas"
// but stem("databas")=="databa"), so re-stemming an already-correct key can
// change it — that is not a bug in the table.
func TestSynonymKeysReachableFromNaturalWords(t *testing.T) {
	cases := map[string]string{
		"kubernetes":  "kubernet",
		"clusters":    "cluster",
		"compaction":  "compact",
		"container":   "contain",
		"lexical":     "lexic",
		"memory":      "memori",
		"pipeline":    "pipelin",
		"query":       "queri",
		"replication": "replic",
		"semantic":    "semant",
		"sstable":     "sstabl",
		"embedding":   "embed",
		"embed":       "emb",
	}
	synMu.RLock()
	defer synMu.RUnlock()
	for natural, wantKey := range cases {
		if got := stem(natural); got != wantKey {
			t.Errorf("stem(%q) = %q, want %q (test assumption out of date)", natural, got, wantKey)
			continue
		}
		if _, ok := synonymMap[wantKey]; !ok {
			t.Errorf("synonymMap has no entry for %q, unreachable from query word %q", wantKey, natural)
		}
	}
}

func TestCommentIndexIgnoresHashInsideToken(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{"a -> b # comment", 7},
		{"# full line comment", 0},
		{"c# -> csharp", -1},
		{"a -> b", -1},
	}
	for _, c := range cases {
		if got := commentIndex(c.in); got != c.want {
			t.Errorf("commentIndex(%q) = %d, want %d", c.in, got, c.want)
		}
	}
}
