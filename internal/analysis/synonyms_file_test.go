package analysis

import (
	"slices"
	"strings"
	"testing"
)

func TestLoadSynonymsDirections(t *testing.T) {
	a := NewStandardAnalyzer()
	n, err := LoadSynonyms(strings.NewReader(`
# comment
qzpaani -> qzwater
qzseelan <-> qzseepage
पानीक्यू ↔ qzjal
`), a)
	if err != nil || n != 5 {
		t.Fatalf("n=%d err=%v, want 5 mappings", n, err)
	}
	if !slices.Contains(Synonyms(stem("qzpaani")), stem("qzwater")) || slices.Contains(Synonyms(stem("qzwater")), stem("qzpaani")) {
		t.Fatal("-> must be one-way")
	}
	if !slices.Contains(Synonyms(stem("qzseepage")), stem("qzseelan")) || !slices.Contains(Synonyms(stem("qzseelan")), stem("qzseepage")) {
		t.Fatal("<-> must be two-way and stemmed")
	}
	if !slices.Contains(Synonyms("पानीक्यू"), "qzjal") {
		t.Fatal("Devanagari side must load")
	}
}

func TestLoadSynonymsRejectsBadLines(t *testing.T) {
	a := NewStandardAnalyzer()
	for _, bad := range []string{"qzone two words -> qzx", "qzonly", "qza => qzb", "the -> qzb"} {
		if _, err := LoadSynonyms(strings.NewReader(bad), a); err == nil || !strings.Contains(err.Error(), "line 1") {
			t.Fatalf("%q: want a line-1 error, got %v", bad, err)
		}
	}
}

func TestLoadSynonymsDoesNotDuplicate(t *testing.T) {
	a := NewStandardAnalyzer()
	for i := 0; i < 2; i++ {
		if _, err := LoadSynonyms(strings.NewReader("qzdup -> qzdupx"), a); err != nil {
			t.Fatal(err)
		}
	}
	if got := Synonyms("qzdup"); len(got) != 1 {
		t.Fatalf("Synonyms(qzdup) = %v, want one entry", got)
	}
}
