package analysis

import (
	"math/rand"
	"sort"
	"testing"
)

func randWord(r *rand.Rand, alphabet []rune, minLen, maxLen int) string {
	n := minLen + r.Intn(maxLen-minLen+1)
	b := make([]rune, n)
	for i := range b {
		b[i] = alphabet[r.Intn(len(alphabet))]
	}
	return string(b)
}

// FuzzySearch must return exactly the BK-tree's set with identical distances,
// including for multi-byte runes (the automaton works on UTF-8 bytes).
func TestFSTFuzzySearch_MatchesBKTree(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	alphabet := []rune("abcdeéñ日本")
	seen := map[string]bool{}
	var terms []string
	bk := NewBKTree()
	for len(terms) < 3000 {
		w := randWord(r, alphabet, 2, 8)
		if !seen[w] {
			seen[w] = true
			terms = append(terms, w)
			bk.Add(w)
		}
	}
	d := NewFSTDictionary()
	if err := d.Build(terms); err != nil {
		t.Fatal(err)
	}

	for dist := 1; dist <= maxAutomatonDist; dist++ {
		for i := 0; i < 150; i++ {
			q := randWord(r, alphabet, 1, 8)
			got, ok, err := d.FuzzySearch(q, dist)
			if err != nil || !ok {
				t.Fatalf("FuzzySearch(%q,%d): ok=%v err=%v", q, dist, ok, err)
			}
			want := bk.Search(q, dist)
			gm := map[string]int{}
			for _, m := range got {
				gm[m.Word] = m.Distance
			}
			wm := map[string]int{}
			for _, m := range want {
				wm[m.Word] = m.Distance
			}
			if len(gm) != len(wm) {
				t.Fatalf("q=%q dist=%d: fst=%d matches, bk=%d", q, dist, len(gm), len(wm))
			}
			var keys []string
			for k := range wm {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				if gd, ok := gm[k]; !ok || gd != wm[k] {
					t.Fatalf("q=%q dist=%d word=%q: fst=%v(%v) bk=%d", q, dist, k, gd, ok, wm[k])
				}
			}
		}
	}
}

func TestFSTFuzzySearch_Unsupported(t *testing.T) {
	d := NewFSTDictionary()
	if _, ok, _ := d.FuzzySearch("x", 0); ok {
		t.Error("dist 0 should be unsupported")
	}
	if _, ok, _ := d.FuzzySearch("x", maxAutomatonDist+1); ok {
		t.Error("dist above max should be unsupported")
	}
	if _, ok, _ := d.FuzzySearch("x", 1); ok {
		t.Error("unbuilt FST should report ok=false")
	}
}
