package queryset

import (
	"math/rand"
	"strings"
	"testing"
)

func TestApply_Deterministic(t *testing.T) {
	q := "how to debug kubernetes memory leaks in production"
	for _, k := range []Kind{Typo, CodeMixed} {
		a := Apply(k, q, rand.New(rand.NewSource(5)))
		b := Apply(k, q, rand.New(rand.NewSource(5)))
		if a != b {
			t.Errorf("%s not deterministic: %q vs %q", k, a, b)
		}
	}
	if Apply(Clean, q, rand.New(rand.NewSource(1))) != q {
		t.Error("clean must be unchanged")
	}
}

func TestPerturb_AlwaysChangesEligibleQueryByOneEditPerWord(t *testing.T) {
	q := "distributed database replication consistency"
	orig := strings.Fields(q)
	for seed := int64(0); seed < 200; seed++ {
		got := strings.Fields(Perturb(q, rand.New(rand.NewSource(seed)), 0.35))
		if len(got) != len(orig) {
			t.Fatalf("seed %d changed word count: %v", seed, got)
		}
		changed := 0
		for i := range orig {
			if got[i] != orig[i] {
				changed++
				if d := lev(orig[i], got[i]); d < 1 || d > 2 { // transposition counts as 2 plain edits
					t.Fatalf("seed %d word %q -> %q has edit distance %d", seed, orig[i], got[i], d)
				}
				if []rune(got[i])[0] != []rune(orig[i])[0] {
					t.Fatalf("first letter changed: %q -> %q", orig[i], got[i])
				}
			}
		}
		if changed == 0 {
			t.Fatalf("seed %d left an eligible query untouched", seed)
		}
	}
}

func TestPerturb_LeavesShortAndNonWordQueriesAlone(t *testing.T) {
	for _, q := range []string{"a to of", "x1 y2 z3", "", "2024 2025"} {
		if got := Perturb(q, rand.New(rand.NewSource(1)), 1); got != q {
			t.Errorf("Perturb(%q) = %q, want unchanged", q, got)
		}
	}
}

func TestMix_KeepsEveryOriginalWordInOrder(t *testing.T) {
	q := "best pizza recipe for beginners"
	orig := strings.Fields(q)
	for seed := int64(0); seed < 100; seed++ {
		got := strings.Fields(Mix(q, rand.New(rand.NewSource(seed))))
		j := 0
		for _, w := range got {
			if j < len(orig) && w == orig[j] {
				j++
			}
		}
		if j != len(orig) {
			t.Fatalf("seed %d dropped or reordered original words: %v", seed, got)
		}
	}
}

func lev(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	prev := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur := make([]int, len(rb)+1)
		cur[0] = i
		for j := 1; j <= len(rb); j++ {
			c := 1
			if ra[i-1] == rb[j-1] {
				c = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+c)
		}
		prev = cur
	}
	return prev[len(rb)]
}
