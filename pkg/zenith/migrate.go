package zenith

import (
	"errors"
	"fmt"
	"path/filepath"

	"github.com/shramanb113/ZENITH/internal/index"
)

// MigrateResult describes a completed Migrate.
type MigrateResult struct {
	FromVersion int    // on-disk format version of the original file
	Docs        int    // documents in the migrated index
	Embedder    string // embedding model recorded in the new file ("unknown" if the source predates the header)
	Backup      string // untouched copy of the original file
}

// Migrate converts an index written by an older release (gob format, versions
// 4 and 5) to the current segment format, in place.
//
// No embedding model is needed — vectors are copied, not recomputed. The
// original is copied to "<path>.v<N>.bak" first and kept, the new index is
// written beside it and swapped in atomically, and the result is re-opened and
// checksummed before Migrate returns, so a failure or crash at any point leaves
// the original usable. The database must not be open.
//
// embedder names the embedding model for version-4 files, which did not record
// one (pass "" to leave it unrecorded; the model check is then skipped on open).
// It is ignored for version-5 files.
//
// A WAL file next to the index (<path>.wal) is left alone: it replays on top of
// the migrated index at the next Open, exactly as it would have on the old one.
func Migrate(path, embedder string) (MigrateResult, error) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return MigrateResult{}, fmt.Errorf("zenith: %w", err)
	}
	fl, err := acquireLock(absPath)
	if err != nil {
		return MigrateResult{}, err
	}
	defer fl.release()

	res, err := index.Migrate(absPath, embedder)
	if err != nil {
		if errors.Is(err, index.ErrAlreadyCurrent) {
			return MigrateResult{}, fmt.Errorf("zenith: %s is already in the current format", path)
		}
		return MigrateResult{}, fmt.Errorf("zenith: migrate: %w", err)
	}
	return MigrateResult{FromVersion: int(res.FromVersion), Docs: res.Docs, Embedder: res.Embedder, Backup: res.Backup}, nil
}
