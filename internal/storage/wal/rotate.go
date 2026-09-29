package wal

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/shramanb113/ZENITH/internal/fsx"
)

// Rotate ends the current log: everything appended so far is flushed and synced,
// the file is renamed to archivePath, and a fresh empty log takes over at the
// original path. Appends made after Rotate returns land in the new file.
//
// A checkpoint uses it to cut the journal at the exact point its snapshot
// covers, so the archive can be deleted once the snapshot is durable without
// touching records appended while the snapshot was being written. Recovery must
// replay archives (oldest first) before the live log — see ReadFile.
//
// Rotate refuses to overwrite an existing archive. If it fails before the rename
// the log keeps appending to the original file.
func (w *WAL) Rotate(archivePath string) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed.Load() {
		return errors.New("wal is closed")
	}
	if _, err := os.Stat(archivePath); err == nil {
		return fmt.Errorf("wal: archive %s already exists", archivePath)
	}
	if err := w.buf.Flush(); err != nil {
		return err
	}
	if err := w.file.Sync(); err != nil {
		return err
	}
	if err := w.file.Close(); err != nil {
		return err
	}
	if err := fsx.Rename(w.path, archivePath); err != nil {
		w.reopenLocked()
		return err
	}
	nf, err := fsx.OpenFile(w.path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		// The archive holds every record; the live log could not be recreated, so
		// stop accepting appends rather than silently dropping them.
		w.closed.Store(true)
		return err
	}
	w.file = nf
	w.buf.Reset(nf)
	w.byteWritten = 0
	syncDir(filepath.Dir(w.path))
	return nil
}

// reopenLocked resumes appending to the original file after a failed rotation.
func (w *WAL) reopenLocked() {
	f, err := fsx.OpenFile(w.path, os.O_RDWR, 0o644)
	if err != nil {
		w.closed.Store(true)
		return
	}
	if _, err := f.Seek(0, io.SeekEnd); err != nil {
		f.Close()
		w.closed.Store(true)
		return
	}
	w.file = f
	w.buf.Reset(f)
}

// ReadFile returns the valid records of a log file without opening it for
// writing (a torn tail is ignored, as in recovery). It is how archives left by
// Rotate are replayed.
func ReadFile(path string) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	recs, _, err := Recover(f)
	return recs, err
}

// syncDir makes a rename or create in dir durable. Directories cannot be opened
// for syncing on Windows; there the error is ignored (NTFS journals metadata).
func syncDir(dir string) { fsx.SyncDir(dir) }
