package index

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

func TestExtractPhrases(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{`machine learning`, nil},
		{`"machine learning"`, []string{"machine learning"}},
		{`x "a b" y "c"`, []string{"a b", "c"}},
		{`"a b`, nil},                   // unpaired: plain text
		{`a "b" "c`, []string{"b"}},     // trailing unpaired quote ignored
		{`"" " " "ok"`, []string{"ok"}}, // empty pairs constrain nothing
		{`"a""b"`, []string{"a", "b"}},  // adjacent pairs
		{`say "naïve café" now`, []string{"naïve café"}},
	} {
		if got := extractPhrases(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("extractPhrases(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestContainsPhrase(t *testing.T) {
	seq := []string{"a", "b", "a", "b", "c"}
	for _, c := range []struct {
		phrase []string
		want   bool
	}{
		{[]string{"a", "b"}, true},
		{[]string{"b", "c"}, true},
		{[]string{"a", "b", "c"}, true}, // second occurrence of "a b"
		{[]string{"b", "a", "b"}, true},
		{[]string{"c", "a"}, false},
		{[]string{"a", "c"}, false},
		{[]string{"a", "b", "c", "d"}, false},
		{nil, true},
	} {
		if got := containsPhrase(seq, c.phrase); got != c.want {
			t.Errorf("containsPhrase(%v, %v) = %v, want %v", seq, c.phrase, got, c.want)
		}
	}
}

// phraseTestEngine gives every document the same positive vector, so a plain
// query returns every document through the vector list: a phrase query must
// still return only documents that contain the phrase.
func phraseTestEngine(t *testing.T, cacheSize int, semantic float64) *Engine {
	t.Helper()
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	cfg.QueryCacheSize = cacheSize
	cfg.QueryCacheSemanticThreshold = semantic
	e := NewEngine(cfg, &countingSearchEmbedder{vec: []float32{1, 0}},
		ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	e.SetANNThreshold(0)
	t.Cleanup(func() { e.Close() })
	return e
}

var phraseCorpus = map[string]string{
	"exact":     "machine learning",
	"inside":    "deep machine learning models for vision",
	"punct":     "Machine-Learning pipelines at scale",
	"later":     "learning about machines, then machine learning",
	"stemmed":   "machines learned quickly",
	"reversed":  "learning machine design",
	"gap":       "machine based learning",
	"onlyone":   "a machine in the shop",
	"unrelated": "kubernetes cluster scheduling",
	"nlp1":      "natural language processing toolkit for python",
	"nlp2":      "natural language toolkit processing",
	"nlp3":      "natural language based processing",
	"rep1":      "new york new york city",
	"rep2":      "new york city",
	"war1":      "the art of war",
	"war2":      "war art",
}

func addCorpus(t *testing.T, e *Engine, corpus map[string]string, attrs func(id string) Attrs) {
	t.Helper()
	ctx := context.Background()
	ids := make([]string, 0, len(corpus))
	for id := range corpus {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		var a Attrs
		if attrs != nil {
			a = attrs(id)
		}
		if err := e.AddWithVectorAttrs(ctx, id, corpus[id], e.EmbedText(ctx, corpus[id]), a); err != nil {
			t.Fatal(err)
		}
	}
}

func resultIDs(res []SearchResponse) []string {
	out := make([]string, len(res))
	for i, r := range res {
		out[i] = r.ID
	}
	sort.Strings(out)
	return out
}

func searchIDs(t *testing.T, e *Engine, q string, pred Predicate) []string {
	t.Helper()
	res, err := e.SearchWithFilter(context.Background(), q, pred)
	if err != nil {
		t.Fatal(err)
	}
	return resultIDs(res)
}

func sorted(ids ...string) []string {
	out := append([]string(nil), ids...)
	sort.Strings(out)
	if out == nil {
		out = []string{}
	}
	return out
}

func sameIDs(a, b []string) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	return reflect.DeepEqual(a, b)
}

func TestPhrase_MatchesOnlyConsecutiveInOrder(t *testing.T) {
	e := phraseTestEngine(t, 0, 0)
	addCorpus(t, e, phraseCorpus, nil)

	// Without quotes the near-misses are hits — and, through the vector list,
	// so is everything else. That is what the phrase has to cut down.
	plain := searchIDs(t, e, "machine learning", nil)
	for _, id := range []string{"reversed", "gap", "onlyone", "unrelated"} {
		if !contains(plain, id) {
			t.Fatalf("plain query lost %q (%v): the test no longer exercises the filter", id, plain)
		}
	}

	for _, c := range []struct {
		q    string
		want []string
	}{
		// Case, punctuation and stemming are analysed away; order and adjacency are not.
		{`"machine learning"`, sorted("exact", "inside", "punct", "later", "stemmed")},
		// "later" = "learning about machines, ...": "about" is a stop word, so
		// learn+machin are adjacent there.
		{`"learning machine"`, sorted("reversed", "later")},
		{`"Machine-Learning"`, sorted("exact", "inside", "punct", "later", "stemmed")},
		// Multi-word phrases.
		{`"natural language processing"`, sorted("nlp1")},
		{`"language processing toolkit"`, sorted("nlp1")},
		{`"natural language toolkit"`, sorted("nlp2")},
		{`"natural language"`, sorted("nlp1", "nlp2", "nlp3")},
		// A repeated term must line up at every position, not just once.
		{`"new york new york"`, sorted("rep1")},
		{`"york new"`, sorted("rep1")},
		{`"new york city"`, sorted("rep1", "rep2")},
		// Removed stop words hold no position (documented in phrase.go).
		{`"art of war"`, sorted("war1")},
		{`"art war"`, sorted("war1")},
		{`"war art"`, sorted("war2")},
		// One-term phrase = required exact term (no prefix/fuzzy/vector-only hits).
		{`"vision"`, sorted("inside")},
		// Several phrases must all hold; free terms only rank.
		{`"machine learning" "deep machine"`, sorted("inside")},
		{`"machine learning" pipelines`, sorted("exact", "inside", "punct", "later", "stemmed")},
		{`kubernetes "machine learning"`, sorted("exact", "inside", "punct", "later", "stemmed")},
		// Nothing contains the phrase: no results at all, not vector-only ones.
		{`"learning deep"`, sorted()},
		{`"machine kubernetes"`, sorted()},
		{`"nosuchterm"`, sorted()},
	} {
		if got := searchIDs(t, e, c.q, nil); !sameIDs(got, c.want) {
			t.Errorf("%s: got %v, want %v", c.q, got, c.want)
		}
	}

	// A phrase that analyses to nothing constrains nothing, and an unpaired
	// quote is plain text: both behave exactly like the unquoted query.
	for _, q := range []string{`"the of" machine learning`, `machine "learning`, `machine learning"`} {
		if got := searchIDs(t, e, q, nil); !sameIDs(got, plain) {
			t.Errorf("%s: got %v, want the plain result %v", q, got, plain)
		}
	}
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

func TestPhrase_WithAttributeFilter(t *testing.T) {
	e := phraseTestEngine(t, 0, 0)
	tenant := map[string]string{"exact": "a", "inside": "b", "punct": "a", "later": "b", "stemmed": "a"}
	addCorpus(t, e, phraseCorpus, func(id string) Attrs {
		if v, ok := tenant[id]; ok {
			return Attrs{"tenant": {Kind: AttrString, S: v}}
		}
		return Attrs{"tenant": {Kind: AttrString, S: "a"}}
	})
	if got, want := searchIDs(t, e, `"machine learning"`, tenantPred("a")), sorted("exact", "punct", "stemmed"); !sameIDs(got, want) {
		t.Fatalf("filtered phrase: got %v, want %v", got, want)
	}
	// Through the structured-filter path too (attribute index + query cache key).
	spec := &FilterSpec{Op: "eq", Field: "tenant", Value: &SpecValue{AttrValue{Kind: AttrString, S: "b"}}}
	filter, err := spec.Compile()
	if err != nil {
		t.Fatal(err)
	}
	res, err := e.SearchFiltered(context.Background(), `"machine learning"`, filter)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := resultIDs(res), sorted("inside", "later"); !sameIDs(got, want) {
		t.Fatalf("spec-filtered phrase: got %v, want %v", got, want)
	}
}

func TestPhrase_ExplainOnlyListsPhraseDocuments(t *testing.T) {
	e := phraseTestEngine(t, 0, 0)
	addCorpus(t, e, phraseCorpus, nil)
	_, hits, err := e.ExplainFiltered(context.Background(), `"machine learning"`, nil)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]string, len(hits))
	for i, h := range hits {
		got[i] = h.ID
	}
	sort.Strings(got)
	if want := sorted("exact", "inside", "punct", "later", "stemmed"); !sameIDs(got, want) {
		t.Fatalf("explain hits %v, want %v", got, want)
	}
	_, plainHits, err := e.ExplainFiltered(context.Background(), "machine learning", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(plainHits) <= len(hits) {
		t.Fatalf("plain explain returned %d hits, phrase %d: the filter is not exercised", len(plainHits), len(hits))
	}
}

// The semantic near-duplicate cache compares query embeddings. A phrase query
// and the same words unquoted embed (here: exactly) alike but have different
// answers, so they must never be served from each other's cache entries.
func TestPhrase_QueryCacheNeverMixesPhraseAndPlain(t *testing.T) {
	e := phraseTestEngine(t, 100, 0.5) // constant vectors: every query is a semantic near-duplicate
	addCorpus(t, e, phraseCorpus, nil)
	ref := phraseTestEngine(t, 0, 0)
	addCorpus(t, ref, phraseCorpus, nil)

	for _, q := range []string{
		"machine learning", `"machine learning"`, `"learning machine"`, "machine learning",
		`"machine learning"`, `"natural language"`, "natural language", `"learning machine"`,
	} {
		want := searchIDs(t, ref, q, nil)
		if got := searchIDs(t, e, q, nil); !sameIDs(got, want) {
			t.Fatalf("cached engine answered %s with %v, want %v", q, got, want)
		}
	}
}

// ---- layered equivalence ----

var phraseVocab = []string{"alpha", "beta", "gamma", "delta", "omega", "sigma"}

var phraseQueries = []string{
	`"alpha beta"`, `"beta alpha"`, `"alpha beta gamma"`, `"gamma gamma"`, `"delta" "sigma omega"`,
	`"omega sigma" alpha`, `alpha "beta delta"`, `"sigma"`, `"alpha beta" "beta gamma"`,
}

func phraseText(r *rand.Rand) string {
	n := 3 + r.Intn(10)
	w := make([]string, n)
	for i := range w {
		w[i] = phraseVocab[r.Intn(len(phraseVocab))]
	}
	return strings.Join(w, " ")
}

// phraseOracle is the expected result set of a phrase query: the live
// documents whose analysed text contains every phrase.
func phraseOracle(e *Engine, live map[string]string, q string) []string {
	phrases := e.analyzePhrases(q)
	out := []string{}
	for id, text := range live {
		seq := e.analyzedTerms(text)
		ok := true
		for _, ph := range phrases {
			if !containsPhrase(seq, ph) {
				ok = false
				break
			}
		}
		if ok {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}

func assertPhraseSame(t *testing.T, stage string, ref, got *Engine, live map[string]string) {
	t.Helper()
	ctx := context.Background()
	matched := 0
	for _, q := range phraseQueries {
		want, err := ref.Search(ctx, q)
		if err != nil {
			t.Fatal(err)
		}
		have, err := got.Search(ctx, q)
		if err != nil {
			t.Fatal(err)
		}
		if oracle := phraseOracle(ref, live, q); !sameIDs(resultIDs(want), oracle) {
			t.Fatalf("%s: %s: reference returned %v, oracle %v", stage, q, resultIDs(want), oracle)
		}
		matched += len(want)
		if len(want) != len(have) {
			t.Fatalf("%s: %s: %d results, want %d", stage, q, len(have), len(want))
		}
		for i := range want {
			if want[i].ID != have[i].ID || math.Abs(want[i].Score-have[i].Score) > 1e-9*math.Max(1, math.Abs(want[i].Score)) {
				t.Fatalf("%s: %s rank %d: got %s/%v want %s/%v", stage, q, i, have[i].ID, have[i].Score, want[i].ID, want[i].Score)
			}
		}
	}
	if len(live) > 50 && matched < 50 {
		t.Fatalf("%s: only %d phrase hits in total — the comparison is not exercising phrases", stage, matched)
	}
}

// phraseDiffEngine is diffEngine with the ranker's candidate cap at
// Config.MaxResults instead of 10, so a phrase query returns its whole
// document set and can be compared against the oracle.
func phraseDiffEngine() *Engine {
	cfg := config.DefaultConfig()
	cfg.WordVectors = true
	e := NewEngine(cfg, bagEmbedder{}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetANNThreshold(0)
	e.SetAutoCompact(false)
	return e
}

func mustLoadPhrase(t *testing.T, path string) *Engine {
	t.Helper()
	e := phraseDiffEngine()
	if err := e.Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { e.Close() })
	return e
}

// Phrase results must be identical whether the documents live in the delta, a
// frozen layer, segments (flushed, reloaded, compacted, exported), and must
// equal a brute-force scan of the live texts at every step.
func TestPhrase_LayeredMatchesInMemoryAndOracle(t *testing.T) {
	ctx := context.Background()
	r := rand.New(rand.NewSource(7))
	path := filepath.Join(t.TempDir(), "phrase.db")
	ref, eng := phraseDiffEngine(), phraseDiffEngine()
	t.Cleanup(func() { eng.Close() })
	live := map[string]string{}
	next := 0

	both := func(f func(e *Engine) error) {
		t.Helper()
		for _, e := range []*Engine{ref, eng} {
			if err := f(e); err != nil {
				t.Fatal(err)
			}
		}
	}
	addN := func(n int) {
		for i := 0; i < n; i++ {
			id := fmt.Sprintf("p-%d", next)
			next++
			text := phraseText(r)
			vec := ref.EmbedText(ctx, text)
			both(func(e *Engine) error { return e.AddWithVectorAttrs(ctx, id, text, vec, nil) })
			live[id] = text
		}
	}
	pick := func() string {
		ids := make([]string, 0, len(live))
		for id := range live {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		return ids[r.Intn(len(ids))]
	}
	mutate := func(removes, replaces int) {
		for i := 0; i < removes; i++ {
			id := pick()
			both(func(e *Engine) error { return e.Remove(ctx, id) })
			delete(live, id)
		}
		for i := 0; i < replaces; i++ {
			id := pick()
			text := phraseText(r)
			vec := ref.EmbedText(ctx, text)
			both(func(e *Engine) error { return e.AddWithVectorAttrs(ctx, id, text, vec, nil) })
			live[id] = text
		}
	}

	addN(150)
	assertPhraseSame(t, "delta", ref, eng, live)
	if err := eng.Save(path); err != nil {
		t.Fatal(err)
	}
	assertPhraseSame(t, "one segment", ref, eng, live)

	addN(50)
	mutate(20, 20)
	assertPhraseSame(t, "delta over segment", ref, eng, live)

	// A frozen layer (checkpoint begun, segment not yet written), mutated while frozen.
	if _, _, err := eng.BeginCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	assertPhraseSame(t, "frozen layer", ref, eng, live)
	addN(10)
	mutate(10, 10)
	assertPhraseSame(t, "frozen layer mutated", ref, eng, live)
	if err := eng.FinishCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	assertPhraseSame(t, "after checkpoint", ref, eng, live)
	if err := eng.Save(path); err != nil {
		t.Fatal(err)
	}
	assertPhraseSame(t, "reloaded", ref, mustLoadPhrase(t, path), live)

	if err := eng.Compact(); err != nil {
		t.Fatal(err)
	}
	if eng.SegmentCount() != 1 {
		t.Fatalf("segments after compaction = %d", eng.SegmentCount())
	}
	assertPhraseSame(t, "compacted", ref, eng, live)
	assertPhraseSame(t, "reloaded after compaction", ref, mustLoadPhrase(t, path), live)

	addN(20)
	mutate(10, 10)
	export := filepath.Join(t.TempDir(), "export.db")
	if err := eng.Save(export); err != nil {
		t.Fatal(err)
	}
	assertPhraseSame(t, "export copy", ref, mustLoadPhrase(t, export), live)
}
