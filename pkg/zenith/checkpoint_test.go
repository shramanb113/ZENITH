package zenith

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// copyDir copies every regular file of src into a new directory, as a crash
// would leave the disk: everything the process had synced is there.
func copyDir(t *testing.T, src string) string {
	t.Helper()
	dst := t.TempDir()
	ents, err := os.ReadDir(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		if e.IsDir() || strings.HasSuffix(e.Name(), ".lock") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, e.Name()), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dst
}

func docText(i int) string { return fmt.Sprintf("checkpoint document number%d shared payload", i) }

// expectDocSet opens the index in dir and asserts it holds exactly want.
func expectDocSet(t *testing.T, what, dir string, want map[string]string) {
	t.Helper()
	db, err := Open(filepath.Join(dir, "ckpt.db"), WithBM25Only())
	if err != nil {
		t.Fatalf("%s: Open: %v", what, err)
	}
	defer db.Close()
	if db.engine.Count() != len(want) {
		t.Fatalf("%s: %d documents, want %d", what, db.engine.Count(), len(want))
	}
	for id, text := range want {
		got, ok, _ := db.Get(id)
		if !ok || got != text {
			t.Fatalf("%s: document %s = %q (present=%v), want %q", what, id, got, ok, text)
		}
	}
}

// A crash at each stage of a checkpoint must recover every acknowledged
// document and must not resurrect an acknowledged delete. The directory is
// snapshotted from inside the checkpoint, exactly where a kill would land.
func TestCheckpoint_CrashImagesAtEachStage(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	path := filepath.Join(dir, "ckpt.db")
	db, err := Open(path, WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			db.Close()
		}
	}()

	want := map[string]string{}
	add := func(i int, prefix string) {
		id := fmt.Sprintf("%s%d", prefix, i)
		want[id] = docText(i)
		if err := db.Add(ctx, id, docText(i)); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 100; i++ {
		add(i, "d")
	}

	var afterRotate, afterFlush string
	wantAtRotate := map[string]string{}
	for k, v := range want {
		wantAtRotate[k] = v
	}
	ckptTestHook = func(stage string) {
		switch stage {
		case "after-rotate":
			afterRotate = copyDir(t, dir)
		case "after-flush":
			afterFlush = copyDir(t, dir)
		}
	}
	defer func() { ckptTestHook = nil }()

	db.mu.Lock()
	job, err := db.beginCheckpointLocked()
	db.mu.Unlock()
	if err != nil || job == nil {
		t.Fatalf("begin: job=%v err=%v", job, err)
	}
	if len(job.archives) != 1 {
		t.Fatalf("expected one WAL archive after the cut, got %v", job.archives)
	}
	if db.docWAL.Size() != 0 {
		t.Fatalf("live WAL has %d bytes after the cut, want 0", db.docWAL.Size())
	}

	// While the checkpoint is "in flight": more documents and a delete of a
	// document that is in the frozen layer.
	for i := 0; i < 20; i++ {
		add(i, "e")
	}
	if err := db.Delete(ctx, "d5"); err != nil {
		t.Fatal(err)
	}
	delete(want, "d5")

	if err := db.finishCheckpoint(job); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(job.archives[0]); !os.IsNotExist(err) {
		t.Fatalf("archive %s still exists after the flush committed", job.archives[0])
	}

	expectDocSet(t, "crash after the WAL cut, before the flush", afterRotate, wantAtRotate)
	expectDocSet(t, "crash after the flush committed, before the archive was removed", afterFlush, want)

	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	closed = true
	expectDocSet(t, "clean close and reopen", dir, want)
}

// A background checkpoint racing with continuous writers and searchers must
// lose nothing, and Close must wait for it.
func TestCheckpoint_ConcurrentWritersSearchersAndSizeTriggeredCheckpoints(t *testing.T) {
	old := defaultWALCheckpointThreshold
	defaultWALCheckpointThreshold = 24 * 1024 // a checkpoint every ~150 documents
	defer func() { defaultWALCheckpointThreshold = old }()

	ctx := context.Background()
	dir := t.TempDir()
	path := filepath.Join(dir, "ckpt.db")
	db, err := Open(path, WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}

	const total = 1500
	var stop atomic.Bool
	var wg sync.WaitGroup
	var searches atomic.Int64
	for g := 0; g < 3; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for !stop.Load() {
				if _, err := db.Search(ctx, "checkpoint document payload"); err != nil {
					t.Errorf("search: %v", err)
					return
				}
				searches.Add(1)
			}
		}()
	}

	want := map[string]string{}
	for i := 0; i < total; i++ {
		id := fmt.Sprintf("w%d", i)
		want[id] = docText(i)
		if err := db.Add(ctx, id, docText(i)); err != nil {
			t.Fatal(err)
		}
		if i%10 == 3 { // deletes and replacements land on frozen and flushed documents
			victim := fmt.Sprintf("w%d", i-3)
			if err := db.Delete(ctx, victim); err != nil {
				t.Fatal(err)
			}
			delete(want, victim)
		}
		if i%25 == 7 {
			target := fmt.Sprintf("w%d", i-5)
			if _, ok := want[target]; ok {
				want[target] = docText(i) + " replaced"
				if err := db.Add(ctx, target, want[target]); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	stop.Store(true)
	wg.Wait()
	if searches.Load() == 0 {
		t.Fatal("no search completed")
	}
	segs := db.engine.SegmentCount()
	if segs == 0 {
		t.Fatal("no size-triggered checkpoint ran: the threshold was not reached")
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if archives, _ := walArchives(path + ".wal"); len(archives) != 0 {
		t.Fatalf("WAL archives left after a clean close: %v", archives)
	}
	expectDocSet(t, "after concurrent checkpoints and a clean close", dir, want)
}

// Archives a crashed process left behind are replayed before the live journal,
// and the next checkpoint deletes them once their documents are in a segment.
func TestCheckpoint_ReopenReplaysArchivesThenRemovesThem(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	path := filepath.Join(dir, "ckpt.db")
	db, err := Open(path, WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{}
	for i := 0; i < 30; i++ {
		id := fmt.Sprintf("d%d", i)
		want[id] = docText(i)
		if err := db.Add(ctx, id, docText(i)); err != nil {
			t.Fatal(err)
		}
	}
	var image string
	ckptTestHook = func(stage string) {
		if stage == "after-rotate" {
			image = copyDir(t, dir)
		}
	}
	defer func() { ckptTestHook = nil }()
	db.mu.Lock()
	_, err = db.beginCheckpointLocked()
	db.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	ckptTestHook = nil // the image is taken; later checkpoints must not overwrite it
	db.Close()

	// Reopen the crash image: archive + empty live journal + no segment yet.
	re, err := Open(filepath.Join(image, "ckpt.db"), WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}
	if archives, _ := walArchives(filepath.Join(image, "ckpt.db.wal")); len(archives) != 1 {
		t.Fatalf("recovered engine should still track the archive, files=%v", archives)
	}
	if err := re.Delete(ctx, "d1"); err != nil {
		t.Fatal(err)
	}
	delete(want, "d1")
	if err := re.Close(); err != nil { // its final checkpoint writes the segment and removes the archive
		t.Fatal(err)
	}
	if archives, _ := walArchives(filepath.Join(image, "ckpt.db.wal")); len(archives) != 0 {
		t.Fatalf("archives left after the recovered DB checkpointed: %v", archives)
	}
	expectDocSet(t, "reopen after recovery", image, want)
}
