package index

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/ann"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// The layered engine (mapped segments + in-memory delta) must be observably
// identical to a plain in-memory engine holding the same documents. These tests
// drive both through the same operations and compare results after every
// persistence step.

var diffVocab = []string{
	"kubernetes", "cluster", "docker", "container", "search", "index", "ranking",
	"score", "vector", "embedding", "database", "storage", "network", "latency",
	"throughput", "memory", "segment", "compaction", "durable", "crash", "recovery",
	"golang", "concurrency", "channel", "goroutine", "python", "rust", "compiler",
	"analyzer", "tokenizer", "phonetic", "fuzzy", "prefix", "posting", "filter",
}

var diffQueries = []string{
	"kubernetes cluster", "search ranking", "docker", "memry", "durable crash recovery",
	"golang concurrency", "nosuchtermanywhere", "vecter embeding", "prefix posting filter",
	"latency throughput memory",
}

func diffEngine() *Engine {
	cfg := config.DefaultConfig()
	cfg.WordVectors = true
	e := NewEngine(cfg, bagEmbedder{}, ranking.NewRRFRanker(0, 0), analysis.NewStandardAnalyzer())
	e.SetANNThreshold(0) // exact scan: results must match the reference bit for bit
	e.SetAutoCompact(false)
	return e
}

func diffText(r *rand.Rand) string {
	n := 4 + r.Intn(8)
	out := ""
	for i := 0; i < n; i++ {
		out += diffVocab[r.Intn(len(diffVocab))] + " "
	}
	return out
}

func tenantPred(name string) Predicate {
	return func(a Attrs) bool { v, ok := a["tenant"]; return ok && v.S == name }
}

func assertSame(t *testing.T, stage string, ref, got *Engine) {
	t.Helper()
	ctx := context.Background()
	if ref.Count() != got.Count() {
		t.Fatalf("%s: Count ref=%d got=%d", stage, ref.Count(), got.Count())
	}
	hits := 0
	for _, q := range diffQueries {
		for _, pred := range []Predicate{nil, tenantPred("a")} {
			want, err := ref.SearchWithFilter(ctx, q, pred)
			if err != nil {
				t.Fatal(err)
			}
			have, err := got.SearchWithFilter(ctx, q, pred)
			if err != nil {
				t.Fatal(err)
			}
			hits += len(want)
			if len(want) != len(have) {
				t.Fatalf("%s: query %q (filter=%v): %d results, want %d", stage, q, pred != nil, len(have), len(want))
			}
			for i := range want {
				if want[i].ID != have[i].ID || math.Abs(want[i].Score-have[i].Score) > 1e-9*math.Max(1, math.Abs(want[i].Score)) {
					// Diagnostic dump for a platform-specific divergence under
					// investigation (macOS/arm64 CI only, never reproduced on
					// amd64): capture the exact vector-candidate neighborhood
					// on both engines so a CI failure log carries enough detail
					// to pin down the cause. Remove once resolved.
					dumpVectorNeighborhood(t, stage+" ref", ref, q)
					dumpVectorNeighborhood(t, stage+" got", got, q)
					t.Fatalf("%s: query %q (filter=%v) rank %d: got %s/%v want %s/%v",
						stage, q, pred != nil, i, have[i].ID, have[i].Score, want[i].ID, want[i].Score)
				}
			}
		}
	}
	if ref.Count() > 20 && hits < 50 {
		t.Fatalf("%s: only %d total hits over %d queries — the comparison is not exercising ranking", stage, hits, len(diffQueries))
	}
}

// dumpVectorNeighborhood logs the top vector-candidate ranking for a query
// against a single engine, with full-precision scores, so a diverging
// assertSame failure carries enough detail to compare rank-for-rank against
// the other engine without a repro.
func dumpVectorNeighborhood(t *testing.T, tag string, e *Engine, query string) {
	t.Helper()
	ctx := context.Background()
	qv := e.EmbedText(ctx, query)
	type sc struct {
		id    uint64
		score float64
	}
	var scores []sc
	e.eachVector(func(id uint64, v []uint16) {
		scores = append(scores, sc{id: id, score: ann.DotF32F16(qv, v)})
	})
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
	t.Logf("--- %s: vector neighborhood for %q ---", tag, query)
	for i := 0; i < 10 && i < len(scores); i++ {
		t.Logf("  rank %d: id=%d name=%s score=%.17g", i+1, scores[i].id, e.origID(scores[i].id), scores[i].score)
	}
}

func mustLoad(t *testing.T, path string) *Engine {
	t.Helper()
	e := diffEngine()
	if err := e.Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { e.Close() })
	return e
}

func TestLayered_DifferentialAgainstInMemory(t *testing.T) {
	ctx := context.Background()
	r := rand.New(rand.NewSource(42))
	path := filepath.Join(t.TempDir(), "diff.db")

	ref, eng := diffEngine(), diffEngine()
	t.Cleanup(func() { eng.Close() })
	live := map[string]bool{}
	next := 0

	both := func(f func(e *Engine) error) {
		t.Helper()
		if err := f(ref); err != nil {
			t.Fatal(err)
		}
		if err := f(eng); err != nil {
			t.Fatal(err)
		}
	}
	addN := func(n int) {
		for i := 0; i < n; i++ {
			id := fmt.Sprintf("doc-%d", next)
			next++
			text := diffText(r)
			attrs := Attrs{"tenant": {Kind: AttrString, S: string(rune('a' + r.Intn(3)))}}
			vec := ref.EmbedText(ctx, text)
			both(func(e *Engine) error {
				return e.AddWithVectorAttrs(ctx, id, text, vec, attrs)
			})
			live[id] = true
		}
	}
	pickLive := func() string {
		for id := range live {
			return id
		}
		return ""
	}
	mutate := func(removes, replaces int) {
		for i := 0; i < removes; i++ {
			id := pickLive()
			if id == "" {
				return
			}
			both(func(e *Engine) error { return e.Remove(ctx, id) })
			delete(live, id)
		}
		for i := 0; i < replaces; i++ {
			id := pickLive()
			if id == "" {
				return
			}
			text := diffText(r)
			attrs := Attrs{"tenant": {Kind: AttrString, S: string(rune('a' + r.Intn(3)))}}
			vec := ref.EmbedText(ctx, text)
			both(func(e *Engine) error { return e.AddWithVectorAttrs(ctx, id, text, vec, attrs) })
		}
	}

	addN(150)
	if len(ref.vectors.GetVectors()) == 0 || len(ref.vectors.GetWordVectors()) == 0 {
		t.Fatal("test corpus has no vectors: the vector and word-vector paths are not being exercised")
	}
	assertSame(t, "all in delta", ref, eng)

	if err := eng.Save(path); err != nil {
		t.Fatal(err)
	}
	assertSame(t, "after first flush", ref, eng)

	addN(60)
	mutate(20, 20) // touches both segment docs and delta docs
	assertSame(t, "delta over segment", ref, eng)

	if err := eng.Save(path); err != nil {
		t.Fatal(err)
	}
	assertSame(t, "after second flush", ref, eng)
	assertSame(t, "reloaded (2 segments)", ref, mustLoad(t, path))

	addN(40)
	mutate(30, 30)
	if err := eng.Save(path); err != nil {
		t.Fatal(err)
	}
	if eng.SegmentCount() != 3 {
		t.Fatalf("segments = %d, want 3", eng.SegmentCount())
	}
	assertSame(t, "three segments", ref, eng)

	if err := eng.Compact(); err != nil {
		t.Fatal(err)
	}
	if eng.SegmentCount() != 1 {
		t.Fatalf("segments after compact = %d, want 1", eng.SegmentCount())
	}
	assertSame(t, "after compaction", ref, eng)
	assertSame(t, "reloaded after compaction", ref, mustLoad(t, path))

	// Keep mutating on top of the compacted segment, including unflushed changes.
	addN(25)
	mutate(15, 15)
	assertSame(t, "delta over compacted", ref, eng)
	if err := eng.Save(path); err != nil {
		t.Fatal(err)
	}
	assertSame(t, "final reload", ref, mustLoad(t, path))

	// Export to another path leaves the engine intact and yields an equal copy.
	export := filepath.Join(t.TempDir(), "export.db")
	if err := eng.Save(export); err != nil {
		t.Fatal(err)
	}
	if eng.DBPath() != filepath.Clean(path) {
		t.Fatalf("export rebound the engine to %s", eng.DBPath())
	}
	assertSame(t, "export copy", ref, mustLoad(t, export))
	assertSame(t, "engine after export", ref, eng)
}

// A flush with nothing new must not write a segment, and a Save→Load→Save
// cycle with no changes must be stable.
func TestLayered_EmptyFlushWritesNoSegment(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "e.db")
	e := diffEngine()
	t.Cleanup(func() { e.Close() })
	if err := e.Add(ctx, "a", "hello world"); err != nil {
		t.Fatal(err)
	}
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	if n := e.SegmentCount(); n != 1 {
		t.Fatalf("segments = %d after redundant Save, want 1", n)
	}
}

// Deletions of segment documents must survive a flush that contains nothing but
// deletions (they are recorded in the new segment's DELS section).
func TestLayered_DeleteOnlyFlushPersists(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "d.db")
	e := diffEngine()
	t.Cleanup(func() { e.Close() })
	for i := 0; i < 20; i++ {
		if err := e.Add(ctx, fmt.Sprintf("d%d", i), "shared token alpha"); err != nil {
			t.Fatal(err)
		}
	}
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	if err := e.Remove(ctx, "d3"); err != nil {
		t.Fatal(err)
	}
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	re := mustLoad(t, path)
	if re.Count() != 19 {
		t.Fatalf("reloaded Count = %d, want 19", re.Count())
	}
	if _, ok := re.GetText("d3"); ok {
		t.Fatal("deleted document reappeared after reload")
	}
}

// Searches and writes running while segments are flushed and compacted must
// never observe a torn state (run with -race). Every search must return
// results and no error; the mapped memory under a running search must stay valid.
func TestLayered_ConcurrentSearchDuringFlushAndCompact(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "conc.db")
	e := diffEngine()
	t.Cleanup(func() { e.Close() })
	r := rand.New(rand.NewSource(3))
	for i := 0; i < 200; i++ {
		if err := e.Add(ctx, fmt.Sprintf("seed-%d", i), diffText(r)+" anchorterm"); err != nil {
			t.Fatal(err)
		}
	}
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}

	stop := make(chan struct{})
	errs := make(chan error, 8)
	done := make(chan struct{})
	for w := 0; w < 4; w++ {
		go func(w int) {
			defer func() { done <- struct{}{} }()
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				res, err := e.Search(ctx, diffQueries[(i+w)%len(diffQueries)])
				if err != nil {
					errs <- err
					return
				}
				if (i+w)%len(diffQueries) == 0 && len(res) == 0 {
					errs <- fmt.Errorf("search returned nothing for %q mid-maintenance", diffQueries[0])
					return
				}
			}
		}(w)
	}

	for round := 0; round < 12; round++ {
		for i := 0; i < 15; i++ {
			id := fmt.Sprintf("live-%d-%d", round, i)
			if err := e.Add(ctx, id, diffText(r)+" anchorterm kubernetes cluster"); err != nil {
				t.Fatal(err)
			}
		}
		if round%3 == 0 {
			_ = e.Remove(ctx, fmt.Sprintf("seed-%d", round))
		}
		if err := e.Save(path); err != nil {
			t.Fatal(err)
		}
		if round%4 == 3 {
			if err := e.Compact(); err != nil {
				t.Fatal(err)
			}
		}
	}
	close(stop)
	for w := 0; w < 4; w++ {
		<-done
	}
	select {
	case err := <-errs:
		t.Fatal(err)
	default:
	}
	assertSame(t, "post-concurrency reload", e, mustLoad(t, path))
}

// The ANN graph aliases vectors that live in memory-mapped segments. Across
// flush (rebind to the new mapping), compaction (rebind again, then the old
// mappings are unmapped) and reload (rebuild) it must stay active, keep
// returning results close to an exact scan, and never read unmapped memory.
func TestLayered_ANNSurvivesFlushCompactAndReload(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "ann.db")
	docs := bagDocs(900, 11)

	exact, fast := newBagEngine(0), newBagEngine(100)
	fast.SetAutoCompact(false)
	t.Cleanup(func() { fast.Close() })

	check := func(stage string, f *Engine) {
		t.Helper()
		if !f.ANNActive() {
			t.Fatalf("%s: ANN is not active", stage)
		}
		r := rand.New(rand.NewSource(5))
		var overlap float64
		const queries = 40
		for i := 0; i < queries; i++ {
			q := docs[r.Intn(len(docs))].Text
			a, _ := exact.Search(ctx, q)
			b, err := f.Search(ctx, q)
			if err != nil {
				t.Fatal(err)
			}
			ea, eb := topResultIDs(a, 10), topResultIDs(b, 10)
			hit := 0
			for id := range ea {
				if eb[id] {
					hit++
				}
			}
			if len(ea) > 0 {
				overlap += float64(hit) / float64(len(ea))
			}
		}
		if avg := overlap / queries; avg < 0.85 {
			t.Fatalf("%s: top-10 overlap with exact search = %.3f, want >= 0.85", stage, avg)
		}
	}

	for _, e := range []*Engine{exact, fast} {
		if err := e.AddBatch(ctx, docs[:500]); err != nil {
			t.Fatal(err)
		}
	}
	if err := fast.Save(path); err != nil {
		t.Fatal(err)
	}
	check("after first flush", fast)

	for _, e := range []*Engine{exact, fast} {
		if err := e.AddBatch(ctx, docs[500:]); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 60; i++ { // deletes of segment-resident documents
			if err := e.Remove(ctx, docs[i*3].ID); err != nil {
				t.Fatal(err)
			}
		}
	}
	check("segment + delta", fast)
	if err := fast.Save(path); err != nil {
		t.Fatal(err)
	}
	check("after second flush", fast)

	if err := fast.Compact(); err != nil {
		t.Fatal(err)
	}
	check("after compaction (old mappings unmapped)", fast)

	re := newBagEngine(100)
	if err := re.Load(path); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { re.Close() })
	check("after reload (graph rebuilt from mapped vectors)", re)
}

// Load via one spelling of the path and Save via another must still be the same
// file: an incremental flush, not an "export" that rewrites mapped segments.
// The second spelling differs as a string and after Abs+Clean; only comparing
// file identity resolves it (case on Windows, a symlinked directory elsewhere).
func TestSave_SamePathDifferentSpelling(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	first := filepath.Join(dir, "same.db")
	var second string
	if runtime.GOOS == "windows" {
		second = strings.ToUpper(first) // NTFS is case-insensitive: same file, different string
	} else {
		link := dir + "-link"
		if err := os.Symlink(dir, link); err != nil {
			t.Skipf("cannot create a symlink here: %v", err)
		}
		t.Cleanup(func() { os.Remove(link) })
		second = filepath.Join(link, "same.db")
	}
	if first == second {
		t.Skip("could not construct a distinct spelling of the path")
	}

	e := diffEngine()
	t.Cleanup(func() { e.Close() })
	if err := e.Add(ctx, "a", "first document about kubernetes"); err != nil {
		t.Fatal(err)
	}
	if err := e.Save(first); err != nil {
		t.Fatal(err)
	}
	if err := e.Add(ctx, "b", "second document about docker"); err != nil {
		t.Fatal(err)
	}
	if err := e.Save(second); err != nil {
		t.Fatal(err)
	}
	if n := e.SegmentCount(); n != 2 {
		t.Fatalf("segments = %d, want 2: the second Save was treated as an export to a different file", n)
	}
	assertSame(t, "reload after respelled Save", e, mustLoad(t, first))
}
