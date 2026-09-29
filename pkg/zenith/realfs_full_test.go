package zenith

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

// TestRealFS_DiskFull fills a real, size-limited filesystem (not the fsx model)
// and checks the same promises the simulated disk-full tests make: a write that
// hits ENOSPC is refused and not acknowledged, the index never shows a refused
// document, a checkpoint that cannot write keeps its WAL archive, everything
// acknowledged survives a crash at the full-disk moment, and the DB keeps
// working once space returns.
//
// It needs a tiny filesystem, which Go cannot create portably, so it is opt-in:
//
//	# Linux / WSL, as root
//	mkdir -p /mnt/zsmall && mount -t tmpfs -o size=6m tmpfs /mnt/zsmall
//	ZENITH_SMALLFS=/mnt/zsmall ./zenith.test -test.run TestRealFS_DiskFull -test.v
func TestRealFS_DiskFull(t *testing.T) {
	small := os.Getenv("ZENITH_SMALLFS")
	if small == "" {
		t.Skip("set ZENITH_SMALLFS to a directory on a small real filesystem (see the comment above)")
	}
	ctx := context.Background()
	dir, err := os.MkdirTemp(small, "realfull-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	// A ballast file we can delete later to give space back, so the DB itself is
	// what runs the disk out and the recovery step is real too.
	ballast := filepath.Join(dir, "ballast")
	bf, err := os.Create(ballast)
	if err != nil {
		t.Fatal(err)
	}
	chunk := make([]byte, 64<<10)
	for i := range chunk {
		chunk[i] = byte(i)
	}
	for i := 0; i < 16; i++ { // 1 MiB of headroom to give back later
		if _, err := bf.Write(chunk); err != nil {
			t.Fatalf("ballast: %v", err)
		}
	}
	bf.Sync()
	bf.Close()

	path := filepath.Join(dir, plName)
	db, err := Open(path, WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}
	big := strings.Repeat("filler words to make the write ahead log grow ", 40) // ~1.8 KB
	text := func(i int) string { return fmt.Sprintf("realfull doc%d %s", i, big) }

	want := dbState{}
	var universe []string
	first := -1
	// Fill the disk through Add until the first refusal. tmpfs has no reserve, so
	// this ends in a genuine ENOSPC out of write(2) or fsync(2).
	for i := 0; i < 20000; i++ {
		id := fmt.Sprintf("d%d", i)
		universe = append(universe, id)
		if err := db.Add(ctx, id, text(i)); err != nil {
			if !errors.Is(err, syscall.ENOSPC) {
				t.Fatalf("Add #%d failed with %v, want a no-space error", i, err)
			}
			first = i
			break
		}
		want[id] = text(i)
	}
	if first < 0 {
		t.Fatal("the disk never filled: use a smaller filesystem (size=6m)")
	}
	t.Logf("first refused Add after %d acknowledged documents", len(want))

	// Keep pushing: each refusal must leave no visible trace and no acknowledgement.
	for j := 0; j < 5; j++ {
		id := fmt.Sprintf("refused%d", j)
		universe = append(universe, id)
		if err := db.Add(ctx, id, text(j)); err == nil {
			t.Fatalf("Add succeeded on a full disk (%s)", id)
		}
		if _, ok, _ := db.Get(id); ok {
			t.Fatalf("refused document %s is visible", id)
		}
	}
	if err := db.Delete(ctx, "d0"); err == nil {
		t.Log("Delete of d0 succeeded on the full disk (the record fit); modelling it as acknowledged")
		delete(want, "d0")
	} else if _, ok, _ := db.Get("d0"); !ok {
		t.Fatal("a refused Delete removed the document")
	}
	if n := db.engine.Count(); n != len(want) {
		t.Fatalf("index holds %d documents, %d acknowledged", n, len(want))
	}

	// A checkpoint on the full disk fails but loses nothing.
	db.mu.Lock()
	job, err := db.beginCheckpointLocked()
	db.mu.Unlock()
	if err != nil {
		t.Logf("checkpoint could not even begin on the full disk: %v", err)
	} else if job != nil {
		if err := db.finishCheckpoint(job); err == nil {
			t.Log("checkpoint fit in the remaining space")
		} else if !errors.Is(err, syscall.ENOSPC) {
			t.Fatalf("finishCheckpoint failed with %v, want a no-space error", err)
		} else if archives, _ := walArchives(path + ".wal"); len(archives) == 0 {
			t.Fatal("the WAL archive was deleted although the checkpoint failed")
		}
	}
	for id := range want {
		if _, ok, _ := db.Get(id); !ok {
			t.Fatalf("acknowledged document %s vanished after the failed checkpoint", id)
		}
	}

	// Crash right now: the files as they stand on the full disk, recovered on a
	// filesystem with room.
	image := copyDir(t, dir)
	os.Remove(filepath.Join(image, "ballast"))
	recoverAndCompare(t, "crash image taken while the disk was full", image, universe, want)

	// Give space back and carry on.
	if err := os.Remove(ballast); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("after%d", i)
		universe = append(universe, id)
		if err := db.Add(ctx, id, text(i)); err != nil {
			t.Fatalf("Add after space returned: %v", err)
		}
		want[id] = text(i)
	}
	// The ballast is only 1 MiB; Close writes a segment holding every document,
	// which may still not fit. That is allowed to fail, but only with ENOSPC, and
	// nothing acknowledged may be lost either way: the WAL still holds it.
	if err := db.Close(); err != nil {
		if !errors.Is(err, syscall.ENOSPC) {
			t.Fatalf("Close failed with %v, want success or a no-space error", err)
		}
		t.Logf("Close could not write its final segment in the space left: %v", err)
	}
	recoverAndCompare(t, "reopen after the disk-full episode (Close may have failed)", dir, universe, want)
	t.Logf("recovered %d documents after the real disk-full episode", len(want))
}
