package storage

import (
	"encoding/json"
	"fmt"
	"os"
	"log/slog"
	"path/filepath"
	"regexp"
	"strconv"

	"github.com/shramanb113/ZENITH/internal/storage/compaction"
	"github.com/shramanb113/ZENITH/internal/storage/sstable"
)

// flushFileRe matches the plain "%010d.sst" name assigned to a freshly
// flushed L0 SSTable (as opposed to "compacted_%010d.sst" outputs). The
// captured number is also used as that file's recency Seq.
var flushFileRe = regexp.MustCompile(`(?:^|[\\/])(\d{10})\.sst$`)

func manifestPath(sstDir string) string {
	return filepath.Join(sstDir, "MANIFEST")
}

// writeManifest atomically persists the given SSTable set so a restart can
// rediscover it (fixes C6: without this, sstCounter and the compactor's
// level state both start empty on every restart, and the very next flush
// silently overwrites the previous run's first SSTable file).
func writeManifest(sstDir string, entries []compaction.ManifestEntry) error {
	data, err := json.Marshal(entries)
	if err != nil {
		return fmt.Errorf("storage: marshal manifest: %w", err)
	}

	// A uniquely-named temp file, not a fixed "MANIFEST.tmp": persistManifest
	// runs from every flush/compaction goroutine, so concurrent writers can
	// easily overlap. A shared fixed name lets two writers race on the same
	// file and rename, which on Windows fails outright ("file in use") and
	// on any OS can interleave two writes into one corrupt temp file.
	f, err := os.CreateTemp(sstDir, "MANIFEST.tmp.*")
	if err != nil {
		return fmt.Errorf("storage: create manifest temp file: %w", err)
	}
	tmp := f.Name()
	if _, err := f.Write(data); err != nil {
		f.Close()
		return fmt.Errorf("storage: write manifest: %w", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("storage: fsync manifest: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("storage: close manifest: %w", err)
	}
	if err := os.Rename(tmp, manifestPath(sstDir)); err != nil {
		return fmt.Errorf("storage: rename manifest into place: %w", err)
	}
	if dir, err := os.Open(sstDir); err == nil {
		_ = dir.Sync()
		_ = dir.Close()
	}
	return nil
}

// readManifest returns (nil, nil) if no manifest file exists yet (first ever
// run) — that is not an error.
func readManifest(sstDir string) ([]compaction.ManifestEntry, error) {
	data, err := os.ReadFile(manifestPath(sstDir))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var entries []compaction.ManifestEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, fmt.Errorf("storage: parse manifest: %w", err)
	}
	return entries, nil
}

// persistManifest is registered as the compactor's onChange callback so the
// manifest stays in sync with every flush registration and compaction pass.
func (e *Engine) persistManifest() {
	e.manifestMu.Lock()
	defer e.manifestMu.Unlock()
	if err := writeManifest(e.cfg.SSTDir, e.compactor.Snapshot()); err != nil {
		slog.Warn("storage: failed to persist SSTable manifest", "error", err)
	}
}

// rediscoverSSTables runs once at Open, before the compactor starts and
// before the WAL is replayed. It:
//  1. Reads the manifest (if present) to recover accurate per-file levels.
//  2. Also scans SSTDir directly and folds in any *.sst file the manifest
//     doesn't mention, defaulting it to level 0 — this way a missing,
//     stale, or corrupt manifest degrades to "treat unknown files as L0
//     candidates for re-compaction" rather than silently losing or
//     overwriting them.
//  3. Sets sstCounter above the highest flush-file number found on disk, so
//     the next flush can never reuse (and truncate) an existing file name.
func (e *Engine) rediscoverSSTables() error {
	entries, err := readManifest(e.cfg.SSTDir)
	if err != nil {
		slog.Warn("storage: manifest unreadable, falling back to a plain directory scan", "error", err)
		entries = nil
	}

	known := make(map[string]bool, len(entries))
	for _, me := range entries {
		known[me.Path] = true
	}

	files, err := filepath.Glob(filepath.Join(e.cfg.SSTDir, "*.sst"))
	if err != nil {
		return fmt.Errorf("storage: scan sstable dir: %w", err)
	}
	for _, f := range files {
		if known[f] {
			continue
		}
		entries = append(entries, compaction.ManifestEntry{Path: f, Level: 0})
	}

	var maxSeq int64
	for i := range entries {
		if m := flushFileRe.FindStringSubmatch(entries[i].Path); m != nil {
			if n, err := strconv.ParseInt(m[1], 10, 64); err == nil {
				if entries[i].Seq == 0 {
					entries[i].Seq = n
				}
				if n > maxSeq {
					maxSeq = n
				}
			}
		}
	}
	if maxSeq > 0 {
		e.sstCounter.Store(uint64(maxSeq))
	}

	if len(entries) == 0 {
		return nil
	}

	e.compactor.LoadSnapshot(entries, func(path string) ([]byte, []byte, int64, error) {
		r, err := sstable.OpenReader(path)
		if err != nil {
			return nil, nil, 0, err
		}
		defer r.Close()
		minKey, err := r.MinKey()
		if err != nil {
			return nil, nil, 0, err
		}
		maxKey := r.MaxKey()
		var size int64
		if info, statErr := os.Stat(path); statErr == nil {
			size = info.Size()
		}
		return minKey, maxKey, size, nil
	})

	slog.Info("storage: rediscovered SSTables from a previous run", "count", len(entries))
	return nil
}
