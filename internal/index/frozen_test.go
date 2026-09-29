package index

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// The frozen layer (a delta being flushed) must be invisible: the engine has to
// answer exactly like a reference engine that never froze anything, at every
// step — while writes, replacements and removals of frozen and older documents
// land during the flush, when the flush fails, and after reload.
func TestFrozen_DifferentialWritesDuringFlush(t *testing.T) {
	ctx := context.Background()
	r := rand.New(rand.NewSource(7))
	path := filepath.Join(t.TempDir(), "frozen.db")

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
			both(func(e *Engine) error { return e.AddWithVectorAttrs(ctx, id, text, vec, attrs) })
			live[id] = true
		}
	}
	liveIDs := func() []string {
		var ids []string
		for id := range live {
			ids = append(ids, id)
		}
		// map order is random; sort for a reproducible run
		for i := 1; i < len(ids); i++ {
			for j := i; j > 0 && ids[j] < ids[j-1]; j-- {
				ids[j], ids[j-1] = ids[j-1], ids[j]
			}
		}
		return ids
	}
	mutate := func(removes, replaces int) {
		for i := 0; i < removes; i++ {
			ids := liveIDs()
			if len(ids) == 0 {
				return
			}
			id := ids[r.Intn(len(ids))]
			both(func(e *Engine) error { return e.Remove(ctx, id) })
			delete(live, id)
		}
		for i := 0; i < replaces; i++ {
			ids := liveIDs()
			if len(ids) == 0 {
				return
			}
			id := ids[r.Intn(len(ids))]
			text := diffText(r)
			attrs := Attrs{"tenant": {Kind: AttrString, S: string(rune('a' + r.Intn(3)))}}
			vec := ref.EmbedText(ctx, text)
			both(func(e *Engine) error { return e.AddWithVectorAttrs(ctx, id, text, vec, attrs) })
		}
	}
	freeze := func(stage string) {
		t.Helper()
		froze, _, err := eng.BeginCheckpoint(path)
		if err != nil {
			t.Fatal(err)
		}
		if !froze {
			t.Fatalf("%s: nothing was frozen", stage)
		}
		if eng.frozen == nil {
			t.Fatalf("%s: engine reports no frozen layer", stage)
		}
		assertSame(t, stage+": just frozen", ref, eng)
	}
	finish := func(stage string) {
		t.Helper()
		if err := eng.FinishCheckpoint(path); err != nil {
			t.Fatal(err)
		}
		if eng.frozen != nil {
			t.Fatalf("%s: frozen layer survived the flush", stage)
		}
		assertSame(t, stage+": after flush", ref, eng)
	}

	// Round 1: freeze a delta, then hit it with every kind of write.
	addN(120)
	freeze("round 1")
	addN(30)       // new docs into the fresh delta
	mutate(15, 15) // removes/replaces that mostly land on frozen documents
	assertSame(t, "round 1: writes during freeze", ref, eng)
	finish("round 1")
	// A reload here would miss the removals made after the freeze (the caller's
	// journal covers them until the next flush), so flush again first.
	if err := eng.Save(path); err != nil {
		t.Fatal(err)
	}
	assertSame(t, "round 1: reload after the next flush", ref, mustLoad(t, path))

	// Round 2: the same again on top of two segments, then compaction.
	addN(50)
	mutate(10, 10)
	freeze("round 2")
	mutate(20, 20)
	addN(20)
	assertSame(t, "round 2: writes during freeze", ref, eng)
	finish("round 2")
	if err := eng.Compact(); err != nil {
		t.Fatal(err)
	}
	assertSame(t, "round 2: after compaction", ref, eng)
	if err := eng.Save(path); err != nil {
		t.Fatal(err)
	}
	assertSame(t, "round 2: reload", ref, mustLoad(t, path))
}

// Reload right after FinishCheckpoint sees the frozen documents but not the
// deletions made after the freeze (those are journalled by the caller until the
// next flush); after the next flush the reload must match exactly.
func TestFrozen_DeletionsDuringFlushPersistWithNextFlush(t *testing.T) {
	ctx := context.Background()
	r := rand.New(rand.NewSource(11))
	path := filepath.Join(t.TempDir(), "frozen2.db")
	ref, eng := diffEngine(), diffEngine()
	t.Cleanup(func() { eng.Close() })
	both := func(f func(e *Engine) error) {
		t.Helper()
		if err := f(ref); err != nil {
			t.Fatal(err)
		}
		if err := f(eng); err != nil {
			t.Fatal(err)
		}
	}
	add := func(id string) {
		text := diffText(r)
		vec := ref.EmbedText(ctx, text)
		attrs := Attrs{"tenant": {Kind: AttrString, S: "a"}}
		both(func(e *Engine) error { return e.AddWithVectorAttrs(ctx, id, text, vec, attrs) })
	}
	for i := 0; i < 80; i++ {
		add(fmt.Sprintf("d%d", i))
	}
	if _, _, err := eng.BeginCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ { // remove frozen docs
		id := fmt.Sprintf("d%d", i)
		both(func(e *Engine) error { return e.Remove(ctx, id) })
	}
	for i := 20; i < 30; i++ { // replace frozen docs
		add(fmt.Sprintf("d%d", i))
	}
	if err := eng.FinishCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	assertSame(t, "engine after flush", ref, eng)

	// Second flush carries the deletions and replacements.
	if err := eng.Save(path); err != nil {
		t.Fatal(err)
	}
	assertSame(t, "engine after second flush", ref, eng)
	assertSame(t, "reload after second flush", ref, mustLoad(t, path))

	// And a compaction over the result agrees too.
	if err := eng.Compact(); err != nil {
		t.Fatal(err)
	}
	assertSame(t, "after compaction", ref, eng)
	assertSame(t, "reload after compaction", ref, mustLoad(t, path))
}

// A flush that fails leaves the frozen layer in place, searchable and
// writable; the next Save writes it and nothing is lost or duplicated.
func TestFrozen_FailedFlushKeepsLayerAndRetries(t *testing.T) {
	ctx := context.Background()
	r := rand.New(rand.NewSource(3))
	path := filepath.Join(t.TempDir(), "frozen3.db")
	ref, eng := diffEngine(), diffEngine()
	t.Cleanup(func() { eng.Close() })
	both := func(f func(e *Engine) error) {
		t.Helper()
		if err := f(ref); err != nil {
			t.Fatal(err)
		}
		if err := f(eng); err != nil {
			t.Fatal(err)
		}
	}
	add := func(id string) {
		text := diffText(r)
		vec := ref.EmbedText(ctx, text)
		both(func(e *Engine) error { return e.AddWithVector(ctx, id, text, vec) })
	}
	for i := 0; i < 60; i++ {
		add(fmt.Sprintf("a%d", i))
	}
	if err := eng.Save(path); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 40; i++ {
		add(fmt.Sprintf("b%d", i))
	}

	// Block the next segment's file name with a directory, so the write fails.
	eng.mu.RLock()
	blocked := segmentFileName(eng.dbPath, eng.nextGen)
	eng.mu.RUnlock()
	if err := os.Mkdir(blocked, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := eng.Save(path); err == nil {
		t.Fatal("Save succeeded although the segment file could not be created")
	}
	if eng.frozen == nil {
		t.Fatal("the frozen layer was dropped by a failed flush")
	}
	assertSame(t, "after failed flush", ref, eng)
	add("c-during-failure")
	both(func(e *Engine) error { return e.Remove(ctx, "a3") })
	both(func(e *Engine) error { return e.Remove(ctx, "b3") }) // a frozen doc
	assertSame(t, "writes while a frozen layer is stuck", ref, eng)

	os.Remove(blocked)
	if err := eng.Save(path); err != nil {
		t.Fatalf("retry: %v", err)
	}
	if eng.frozen != nil {
		t.Fatal("frozen layer remained after a successful Save")
	}
	assertSame(t, "after retry", ref, eng)
	assertSame(t, "reload after retry", ref, mustLoad(t, path))
}

// Export while a frozen layer is pending must include it (minus its dead docs).
func TestFrozen_ExportIncludesFrozenLayer(t *testing.T) {
	ctx := context.Background()
	r := rand.New(rand.NewSource(5))
	dir := t.TempDir()
	path := filepath.Join(dir, "live.db")
	ref, eng := diffEngine(), diffEngine()
	t.Cleanup(func() { eng.Close() })
	both := func(f func(e *Engine) error) {
		t.Helper()
		if err := f(ref); err != nil {
			t.Fatal(err)
		}
		if err := f(eng); err != nil {
			t.Fatal(err)
		}
	}
	add := func(id string) {
		text := diffText(r)
		vec := ref.EmbedText(ctx, text)
		both(func(e *Engine) error { return e.AddWithVector(ctx, id, text, vec) })
	}
	for i := 0; i < 50; i++ {
		add(fmt.Sprintf("a%d", i))
	}
	if err := eng.Save(path); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 30; i++ {
		add(fmt.Sprintf("b%d", i))
	}
	if _, _, err := eng.BeginCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	both(func(e *Engine) error { return e.Remove(ctx, "b1") })
	add("b2") // replace a frozen doc
	add("c1")
	export := filepath.Join(dir, "export.db")
	if err := eng.Save(export); err != nil {
		t.Fatal(err)
	}
	assertSame(t, "export with a frozen layer pending", ref, mustLoad(t, export))
}

// The point of the design: searches keep completing while a flush runs, and the
// engine lock is held for a duration independent of the delta size.
func TestFrozen_SearchesProceedDuringFlush(t *testing.T) {
	if testing.Short() {
		t.Skip("timing test")
	}
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "conc.db")
	eng := diffEngine()
	t.Cleanup(func() { eng.Close() })
	r := rand.New(rand.NewSource(9))
	docs := make([]BatchDoc, 20000)
	for i := range docs {
		docs[i] = BatchDoc{ID: fmt.Sprintf("d%d", i), Text: diffText(r) + diffText(r) + diffText(r)}
	}
	if err := eng.AddBatch(ctx, docs); err != nil {
		t.Fatal(err)
	}

	var stop atomic.Bool
	var wg sync.WaitGroup
	var searches, maxNs atomic.Int64
	wg.Add(1)
	go func() {
		defer wg.Done()
		for !stop.Load() {
			t0 := time.Now()
			if _, err := eng.Search(ctx, "kubernetes cluster"); err != nil {
				t.Error(err)
				return
			}
			searches.Add(1)
			if d := int64(time.Since(t0)); d > maxNs.Load() {
				maxNs.Store(d)
			}
		}
	}()
	time.Sleep(100 * time.Millisecond)
	maxNs.Store(0)

	if _, _, err := eng.BeginCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	before := searches.Load()
	if err := eng.FinishCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	flush := time.Since(t0)
	during := searches.Load() - before
	stop.Store(true)
	wg.Wait()
	t.Logf("flush of 20k docs took %s; %d searches completed during it; longest search %s",
		flush.Round(time.Millisecond), during, time.Duration(maxNs.Load()).Round(time.Millisecond))
	if flush > 50*time.Millisecond && during < 5 {
		t.Fatalf("only %d searches finished during a %s flush: searches are being blocked", during, flush)
	}
}
