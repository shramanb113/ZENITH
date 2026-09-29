package zenith

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/shramanb113/ZENITH/internal/fsx"
	"github.com/shramanb113/ZENITH/internal/storage/wal"
)

// Checkpoints do not stall searches or writes.
//
// A checkpoint has two halves. The first, under db.mu, is O(1): the engine
// freezes the documents added so far (see internal/index/frozen.go) and the WAL
// is cut at the same instant — the journal up to the cut becomes an archive
// `<db>.wal.old-NNNNNN`, and a fresh empty journal takes over. The second half
// runs with no lock: the engine writes and syncs the frozen documents as a
// segment and commits it, and only then are the archives deleted. Anything
// added meanwhile is in the fresh journal and the engine's new delta, so a crash
// at any point recovers the segment plus the archives plus the journal —
// replaying a document that is already in the segment is harmless, because
// adding is idempotent and a later delete in the journal still wins.
//
// Recovery replays archives oldest-first, then the live journal.

const walArchiveInfix = ".old-"

// ckptTestHook, when set by a test, is called at the named stages of a
// checkpoint ("after-rotate", "after-flush") so a test can snapshot the
// directory exactly as a crash at that moment would leave it.
var ckptTestHook func(stage string)

func ckptStage(stage string) {
	if ckptTestHook != nil {
		ckptTestHook(stage)
	}
}

// checkpointJob is the part of a checkpoint that runs after db.mu is released.
type checkpointJob struct {
	// archives are journal files whose records the frozen layer covers; they may
	// be deleted once that layer is committed. A journal cut that failed leaves
	// this empty: the records then stay in the live journal until a later cut.
	archives []string
}

// walArchives lists the archived journals of the WAL at walPath, oldest first,
// and the highest archive number in use.
func walArchives(walPath string) (files []string, maxSeq int) {
	matches, _ := filepath.Glob(walPath + walArchiveInfix + "*")
	for _, m := range matches {
		n, err := strconv.Atoi(strings.TrimPrefix(m, walPath+walArchiveInfix))
		if err != nil {
			continue // not one of ours
		}
		files = append(files, m)
		maxSeq = max(maxSeq, n)
	}
	// Archive numbers are zero-padded, so lexical order is chronological.
	return files, maxSeq
}

// beginCheckpointLocked freezes the engine's delta and cuts the WAL. db.mu must
// be held for writing. A nil job means there is nothing to write.
func (db *DB) beginCheckpointLocked() (*checkpointJob, error) {
	if db.path == "" || db.docWAL == nil {
		return nil, nil
	}
	froze, pending, err := db.engine.BeginCheckpoint(db.path)
	if err != nil {
		return nil, fmt.Errorf("zenith: %w", err)
	}
	if !pending {
		return nil, nil
	}
	if froze {
		db.walSeq++
		archive := fmt.Sprintf("%s%s%06d", db.walPath, walArchiveInfix, db.walSeq)
		if err := db.docWAL.Rotate(archive); err != nil {
			// The delta is frozen but the journal was not cut. Nothing is lost: the
			// records stay in the live journal, and a later checkpoint cuts it.
			slog.Warn("zenith: could not rotate the WAL; its records stay until the next checkpoint", "error", err)
		} else {
			db.walMu.Lock()
			db.walArchives = append(db.walArchives, archive)
			db.walMu.Unlock()
		}
		ckptStage("after-rotate")
	}
	db.walMu.Lock()
	defer db.walMu.Unlock()
	return &checkpointJob{archives: append([]string(nil), db.walArchives...)}, nil
}

// finishCheckpoint writes the frozen documents and, once they are committed,
// deletes the journal archives they made redundant. It needs no lock on db.mu.
func (db *DB) finishCheckpoint(job *checkpointJob) error {
	if err := db.engine.FinishCheckpoint(db.path); err != nil {
		return fmt.Errorf("zenith: %w", err)
	}
	ckptStage("after-flush")
	var removed []string
	for _, a := range job.archives {
		if err := fsx.Remove(a); err != nil && !os.IsNotExist(err) {
			slog.Warn("zenith: could not remove an archived WAL; it will be retried", "file", a, "error", err)
			continue
		}
		removed = append(removed, a)
	}
	db.walMu.Lock()
	kept := db.walArchives[:0]
	for _, a := range db.walArchives {
		gone := false
		for _, r := range removed {
			if a == r {
				gone = true
				break
			}
		}
		if !gone {
			kept = append(kept, a)
		}
	}
	db.walArchives = kept
	db.walMu.Unlock()
	return nil
}

// startCheckpointAsyncLocked begins a checkpoint under db.mu and finishes it in
// the background. At most one checkpoint runs at a time; a request while one is
// running is dropped (the journal keeps growing and the next request retries).
func (db *DB) startCheckpointAsyncLocked() {
	if !db.ckptBusy.CompareAndSwap(false, true) {
		return
	}
	job, err := db.beginCheckpointLocked()
	if err != nil || job == nil {
		db.ckptBusy.Store(false)
		if err != nil {
			slog.Warn("zenith: checkpoint failed", "error", err)
		}
		return
	}
	db.ckptWG.Add(1)
	go func() {
		defer db.ckptWG.Done()
		defer db.ckptBusy.Store(false)
		if err := db.finishCheckpoint(job); err != nil {
			slog.Warn("zenith: background checkpoint failed", "error", err)
		}
	}()
}

// checkpointSyncLocked makes everything added so far durable before returning.
// db.mu must be held for writing (Close is the caller: nothing else runs then).
func (db *DB) checkpointSyncLocked() error {
	for range 3 { // a layer left by a failed flush needs one more round for the delta
		job, err := db.beginCheckpointLocked()
		if err != nil {
			return err
		}
		if job == nil {
			return nil
		}
		if err := db.finishCheckpoint(job); err != nil {
			return err
		}
	}
	return nil
}

// walOpenArchives is used by Open: it reads the archives a crash left behind.
func walOpenArchives(walPath string) (files []string, records []wal.Record, maxSeq int, err error) {
	files, maxSeq = walArchives(walPath)
	for _, f := range files {
		recs, rerr := wal.ReadFile(f)
		if rerr != nil {
			return nil, nil, 0, fmt.Errorf("zenith: read archived wal %s: %w", filepath.Base(f), rerr)
		}
		records = append(records, recs...)
	}
	return files, records, maxSeq, nil
}
