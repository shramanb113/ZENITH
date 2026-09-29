package index

import (
	"context"
	"fmt"
	"hash/fnv"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/shramanb113/ZENITH/internal/ann"
)

func annEngine(threshold int) *Engine {
	e := diffEngine()
	e.SetANNThreshold(threshold)
	return e
}

func internalID(orig string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(orig))
	return h.Sum64()
}

func addDocs(t *testing.T, e *Engine, lo, hi int, seed int64) {
	t.Helper()
	r := rand.New(rand.NewSource(seed))
	for i := lo; i < hi; i++ {
		if err := e.Add(context.Background(), fmt.Sprintf("a%d", i), diffText(r)+fmt.Sprintf("uniq%d", i)); err != nil {
			t.Fatal(err)
		}
	}
}

// selfRecall: fraction of documents that the graph finds by their own vector.
func selfRecall(e *Engine, ids []string) float64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	found := 0
	for _, id := range ids {
		v := e.vecOf(internalID(id))
		if v == nil {
			continue
		}
		for _, h := range e.ann.Search(Float16ToFloats(v), 5, 100, nil) {
			if h.ID == internalID(id) {
				found++
				break
			}
		}
	}
	return float64(found) / float64(len(ids))
}

func names(lo, hi int) []string {
	out := make([]string, 0, hi-lo)
	for i := lo; i < hi; i++ {
		out = append(out, fmt.Sprintf("a%d", i))
	}
	return out
}

// The point of persisting the graph: the next open restores it instead of
// rebuilding, and search behaves the same.
func TestANNPersist_RestoredNotRebuilt(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ann.db")
	e := annEngine(100)
	defer e.Close()
	addDocs(t, e, 0, 600, 1)
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	if !e.ANNActive() {
		t.Fatal("ANN is not active above its threshold")
	}
	if err := e.SaveANN(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(annFilePath(path)); err != nil {
		t.Fatalf("no graph file written: %v", err)
	}

	e2 := annEngine(100)
	defer e2.Close()
	if err := e2.Load(path); err != nil {
		t.Fatal(err)
	}
	if !e2.ANNActive() || !e2.ANNLoadedFromDisk() {
		t.Fatalf("graph was rebuilt instead of restored (active=%v fromDisk=%v)", e2.ANNActive(), e2.ANNLoadedFromDisk())
	}
	assertSame(t, "engine restored from a persisted graph", e, e2)
	if r := selfRecall(e2, names(0, 600)); r < 0.97 {
		t.Fatalf("restored graph finds only %.3f of documents by their own vector", r)
	}
}

// A graph older than the segments is reconciled: added documents are inserted,
// replaced ones re-inserted, deleted ones never returned.
func TestANNPersist_StaleFileIsReconciledWithSegments(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ann.db")
	e := annEngine(100)
	defer e.Close()
	addDocs(t, e, 0, 500, 1)
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	if err := e.SaveANN(); err != nil {
		t.Fatal(err)
	}

	// After the graph file was written: new documents, replacements, removals.
	addDocs(t, e, 500, 560, 2)
	addDocs(t, e, 10, 20, 3) // replaces a10..a19 with new text (new vectors)
	removed := names(100, 130)
	for _, id := range removed {
		if err := e.Remove(context.Background(), id); err != nil {
			t.Fatal(err)
		}
	}
	if err := e.Save(path); err != nil { // flush only; the graph file stays old
		t.Fatal(err)
	}

	e2 := annEngine(100)
	defer e2.Close()
	if err := e2.Load(path); err != nil {
		t.Fatal(err)
	}
	if !e2.ANNLoadedFromDisk() {
		t.Fatal("a stale-but-mostly-valid graph file should be reused")
	}
	// Exactly what the file could not vouch for was reconciled: 60 new documents
	// and 10 replaced ones were inserted; 30 removed and 10 replaced (old) nodes
	// were tombstoned.
	e2.mu.RLock()
	reconciled := e2.annChanges
	e2.mu.RUnlock()
	if reconciled != 60+10+30+10 {
		t.Fatalf("reconciled %d graph changes, want 110 (60 new + 10 replaced inserted, 40 tombstoned)", reconciled)
	}
	live := append(names(0, 100), names(130, 560)...)
	if r := selfRecall(e2, live); r < 0.97 {
		t.Fatalf("after reconcile only %.3f of live documents are found by their own vector", r)
	}
	e2.mu.RLock()
	for _, id := range removed {
		for _, h := range e2.ann.Search(Float16ToFloats(e.vecOfSafe(id)), 10, 100, nil) {
			if h.ID == internalID(id) {
				e2.mu.RUnlock()
				t.Fatalf("removed document %s is returned by the restored graph", id)
			}
		}
	}
	e2.mu.RUnlock()
	// Replaced documents: the graph must hold their NEW vectors, not the old
	// links built for the old ones (found by the new vector = correct binding).
	if r := selfRecall(e2, names(10, 20)); r < 1.0 {
		t.Fatalf("replaced documents found by their new vector: %.2f, want 1.0", r)
	}
}

// vecOfSafe returns the stored vector of a (possibly removed) document as the
// reference engine e last knew it — for removed ids that is nil, so build a
// query from the document text instead.
func (e *Engine) vecOfSafe(orig string) []uint16 {
	v, _ := e.embedder.Embed(context.Background(), "kubernetes cluster "+orig)
	return FloatsToFloat16(normalizeVector(v))
}

func TestANNPersist_DamagedOrForeignFileFallsBackToRebuild(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mangle func(path string)
	}{
		{"corrupt", func(p string) {
			b, _ := os.ReadFile(annFilePath(p))
			b[len(b)/2] ^= 0xFF
			os.WriteFile(annFilePath(p), b, 0o644)
		}},
		{"truncated", func(p string) {
			b, _ := os.ReadFile(annFilePath(p))
			os.WriteFile(annFilePath(p), b[:len(b)/3], 0o644)
		}},
		{"garbage", func(p string) { os.WriteFile(annFilePath(p), []byte("not a graph"), 0o644) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "ann.db")
			e := annEngine(100)
			defer e.Close()
			addDocs(t, e, 0, 300, 1)
			e.Save(path)
			e.SaveANN()
			tc.mangle(path)
			e2 := annEngine(100)
			defer e2.Close()
			if err := e2.Load(path); err != nil {
				t.Fatalf("a damaged graph file must not fail Load: %v", err)
			}
			if !e2.ANNActive() || e2.ANNLoadedFromDisk() {
				t.Fatalf("expected a rebuild (active=%v fromDisk=%v)", e2.ANNActive(), e2.ANNLoadedFromDisk())
			}
			if r := selfRecall(e2, names(0, 300)); r < 0.97 {
				t.Fatalf("rebuilt graph self-recall %.3f", r)
			}
		})
	}
}

// After a compaction the file's segment tags are obsolete; the engine rewrites
// it, and the next open restores without inserting anything by hand.
func TestANNPersist_CompactionInvalidatesAndRewrites(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ann.db")
	e := annEngine(100)
	defer e.Close()
	for round := 0; round < 3; round++ {
		addDocs(t, e, round*200, (round+1)*200, int64(round))
		if err := e.Save(path); err != nil {
			t.Fatal(err)
		}
	}
	if err := e.Compact(); err != nil {
		t.Fatal(err)
	}
	if err := e.SaveANN(); err != nil { // waits its turn behind the background save, then is a no-op or a rewrite
		t.Fatal(err)
	}
	e2 := annEngine(100)
	defer e2.Close()
	if err := e2.Load(path); err != nil {
		t.Fatal(err)
	}
	if !e2.ANNLoadedFromDisk() {
		t.Fatal("graph was rebuilt after compaction + save")
	}
	e2.mu.RLock()
	stale := e2.annChanges
	e2.mu.RUnlock()
	if stale != 0 {
		t.Fatalf("%d documents had to be reconciled by hand after a compaction rewrite", stale)
	}
	if r := selfRecall(e2, names(0, 600)); r < 0.97 {
		t.Fatalf("self-recall %.3f", r)
	}
}

// A file that describes mostly other documents is not worth reconciling.
func TestANNPersist_MostlyStaleFileIsRebuilt(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ann.db")
	e := annEngine(100)
	defer e.Close()
	addDocs(t, e, 0, 200, 1)
	e.Save(path)
	e.SaveANN()
	addDocs(t, e, 200, 900, 2) // 78% of the documents are unknown to the file
	e.Save(path)
	e2 := annEngine(100)
	defer e2.Close()
	if err := e2.Load(path); err != nil {
		t.Fatal(err)
	}
	if !e2.ANNActive() || e2.ANNLoadedFromDisk() {
		t.Fatalf("expected a rebuild of a mostly-stale graph (active=%v fromDisk=%v)", e2.ANNActive(), e2.ANNLoadedFromDisk())
	}
}

// The graph file belongs to one embedder and one parameter set: a file that
// claims otherwise is ignored, however valid its checksum.
func TestANNPersist_ForeignIdentityIsIgnored(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ann.db")
	e := annEngine(100)
	defer e.Close()
	addDocs(t, e, 0, 300, 1)
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	// A structurally perfect file written for another embedder.
	f, err := os.Create(annFilePath(path))
	if err != nil {
		t.Fatal(err)
	}
	e.mu.RLock()
	_, err = e.ann.WriteTo(f, ann.WriteOptions{
		Tag:      func(id uint64) (uint64, bool) { li, _, ok := e.locate(id); return e.segs[li].gen, ok },
		Identity: []byte("zenith-ann/1 embedder=some-other-model dims=384 m=16 efc=100"),
	})
	e.mu.RUnlock()
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	e2 := annEngine(100)
	defer e2.Close()
	if err := e2.Load(path); err != nil {
		t.Fatal(err)
	}
	if !e2.ANNActive() || e2.ANNLoadedFromDisk() {
		t.Fatalf("a graph file for another embedder was used (active=%v fromDisk=%v)", e2.ANNActive(), e2.ANNLoadedFromDisk())
	}
}
