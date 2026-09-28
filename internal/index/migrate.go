package index

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	pathutil "path/filepath"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
	"github.com/shramanb113/ZENITH/internal/segment"
)

// identityEmbedder stands in for the real embedder while migrating: the vectors
// already exist in the file, so nothing is ever embedded — it only carries the
// recorded identity so the new file's header matches the old one.
type identityEmbedder struct {
	name string
	dims int
}

func (e identityEmbedder) Name() string    { return e.name }
func (e identityEmbedder) Dimensions() int { return e.dims }
func (e identityEmbedder) Embed(context.Context, string) ([]float32, error) {
	return nil, errors.New("index: migration never embeds")
}
func (e identityEmbedder) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return nil, errors.New("index: migration never embeds")
}

// MigrateResult describes a finished migration.
type MigrateResult struct {
	FromVersion uint16
	Docs        int
	Embedder    string
	Dims        int
	Backup      string // copy of the original file
}

// ErrAlreadyCurrent is returned by Migrate for a file that is already in the
// current (segment) format.
var ErrAlreadyCurrent = errors.New("index: file is already in the current format")

// Migrate converts a gob-format (v4/v5) index at path to the segment format, in
// place. No embedding model is needed: vectors are copied, not recomputed.
//
// The original is copied to "<path>.v<N>.bak" first and stays there. The new
// index is written as a segment next to path and only then does the manifest
// atomically replace the old file, so a crash at any point leaves either the
// old index or the new one, never a mixture. The result is re-opened and
// checked (document count, segment checksums) before Migrate returns.
//
// embedderOverride names the embedding model for v4 files, which recorded none;
// it is ignored for v5 files. Empty leaves it "unknown" (never checked).
func Migrate(path, embedderOverride string) (MigrateResult, error) {
	path = pathutil.Clean(path)
	info, err := Inspect(path)
	if err != nil {
		return MigrateResult{}, err
	}
	if info.Version == manifestVersion {
		return MigrateResult{}, ErrAlreadyCurrent
	}
	li, err := readLegacy(path)
	if err != nil {
		return MigrateResult{}, err
	}

	name, dims := li.Embedder, li.Dims
	if li.Version < 5 {
		name = embedderOverride
		dims = 0
		for _, v := range li.Vectors {
			dims = len(v.Vector)
			break
		}
		if dims == 0 {
			for _, v := range li.WordVectors {
				dims = len(v.Vector)
				break
			}
		}
	}
	if name == "" {
		name = "unknown"
	}

	backup, err := backupFile(path, fmt.Sprintf(".v%d.bak", li.Version))
	if err != nil {
		return MigrateResult{}, fmt.Errorf("index: backing up %s: %w", path, err)
	}

	stub := identityEmbedder{name: name, dims: dims}
	e := NewEngine(config.DefaultConfig(), stub, ranking.NewRRFRanker(0, 0), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	e.SetANNThreshold(0)
	e.replaceLegacy = true // the original is already backed up
	if err := e.LoadLegacy(path); err != nil {
		return MigrateResult{}, err
	}
	docs := e.Count()
	if err := e.Save(path); err != nil { // adopts path: segment first, then atomic manifest rename
		e.Close()
		return MigrateResult{}, fmt.Errorf("index: writing migrated index: %w", err)
	}
	e.Close()

	// Verify by reading it back through the normal path.
	chk := NewEngine(config.DefaultConfig(), stub, ranking.NewRRFRanker(0, 0), analysis.NewStandardAnalyzer())
	chk.SetANNThreshold(0)
	defer chk.Close()
	if err := chk.Load(path); err != nil {
		return MigrateResult{}, fmt.Errorf("index: migrated file failed to reopen (original kept at %s): %w", backup, err)
	}
	if got := chk.Count(); got != docs {
		return MigrateResult{}, fmt.Errorf("index: migrated file holds %d documents, expected %d (original kept at %s)", got, docs, backup)
	}
	if err := Verify(path); err != nil {
		return MigrateResult{}, fmt.Errorf("index: migrated file failed verification (original kept at %s): %w", backup, err)
	}
	return MigrateResult{FromVersion: li.Version, Docs: docs, Embedder: name, Dims: dims, Backup: backup}, nil
}

// backupFile copies path to path+suffix (adding a numeric tail if that exists)
// and fsyncs the copy. It returns the backup's path.
func backupFile(path, suffix string) (string, error) {
	dst := path + suffix
	for i := 1; ; i++ {
		if _, err := os.Stat(dst); os.IsNotExist(err) {
			break
		}
		dst = fmt.Sprintf("%s%s.%d", path, suffix, i)
	}
	in, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		os.Remove(dst)
		return "", err
	}
	if err := out.Sync(); err != nil {
		out.Close()
		os.Remove(dst)
		return "", err
	}
	return dst, out.Close()
}

// Verify checks a segment-format index without loading it into an engine: the
// manifest must be intact, every listed segment must exist, open, and pass its
// per-section checksums. It reads every byte of the index.
func Verify(path string) error {
	path = pathutil.Clean(path)
	_, _, m, err := readManifest(path)
	if err != nil {
		return err
	}
	for _, ms := range m.Segments {
		file := pathutil.Join(pathutil.Dir(path), ms.File)
		seg, err := segment.Open(file)
		if err != nil {
			return fmt.Errorf("segment %s: %w", ms.File, err)
		}
		verr := seg.Verify()
		seg.Close()
		if verr != nil {
			return fmt.Errorf("segment %s: %w", ms.File, verr)
		}
	}
	return nil
}

// CompactFile merges every segment of the index at path into one, dropping
// deleted documents and vocabulary. It needs no embedding model (nothing is
// embedded; the file's recorded identity is reused) and reports the segment
// count before and after.
func CompactFile(path string) (before, after int, err error) {
	path = pathutil.Clean(path)
	info, err := Inspect(path)
	if err != nil {
		return 0, 0, err
	}
	if info.Version != manifestVersion {
		return 0, 0, &LegacyFormatError{Version: info.Version}
	}
	name, dims := info.Embedder, info.Dims
	e := NewEngine(config.DefaultConfig(), identityEmbedder{name: name, dims: dims}, ranking.NewRRFRanker(0, 0), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	e.SetANNThreshold(0)
	defer e.Close()
	if err := e.Load(path); err != nil {
		return 0, 0, err
	}
	before = e.SegmentCount()
	if err := e.Compact(); err != nil {
		return before, before, err
	}
	return before, e.SegmentCount(), nil
}
