package index

import (
	"bytes"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/shramanb113/ZENITH/internal/ann"
	"github.com/shramanb113/ZENITH/internal/fsx"
)

// The HNSW graph is persisted next to the manifest as "<manifest>.ann" so a
// large index does not rebuild it on every open. The file is a hint (see
// internal/ann/persist.go): segments remain the truth, and Load reconciles the
// graph with them. It is written after a compaction, on Close (via SaveANN),
// and in the background once enough documents changed since the last write;
// never on every flush, whose cost must stay proportional to the delta.

const (
	annFileSuffix = ".ann"
	// annSaveEvery is how many graph mutations (inserts and deletes) may pile up
	// before a flush triggers a background rewrite of the file. It bounds the
	// number of documents a later open has to insert one by one.
	annSaveEvery = 50_000
)

func annFilePath(dbPath string) string { return dbPath + annFileSuffix }

// annIdentity binds the file to what it was built for: the embedder (a graph of
// another model's vectors is meaningless) and the graph parameters.
func (e *Engine) annIdentity() []byte {
	name, dims := e.embedderIdentity()
	return []byte(fmt.Sprintf("zenith-ann/1 embedder=%s dims=%d m=%d efc=%d", name, dims, annM, annEFConstruction))
}

// ANNLoadedFromDisk reports whether the current graph was restored from its
// sidecar file (as opposed to built from the vectors).
func (e *Engine) ANNLoadedFromDisk() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.annFromDisk
}

// SaveANN writes the graph sidecar if there is a graph and it changed since the
// last write. It does not stop searches or writes: the graph is copied in small
// chunks under the read lock. Failure is reported but harmless — the next open
// just rebuilds.
func (e *Engine) SaveANN() error {
	e.annSaveMu.Lock()
	defer e.annSaveMu.Unlock()

	e.mu.RLock()
	g, path, changes, saved := e.ann, e.dbPath, e.annChanges, e.annSaved
	ident := e.annIdentity()
	e.mu.RUnlock()
	if g == nil || path == "" {
		return nil
	}
	if changes == 0 && saved {
		return nil
	}

	start := time.Now()
	final := annFilePath(path)
	tmp := final + ".tmp"
	f, err := fsx.Create(tmp)
	if err != nil {
		return err
	}
	fail := func(err error) error { f.Close(); fsx.Remove(tmp); return err }
	n, err := g.WriteTo(f, ann.WriteOptions{
		Lock: func() func() { e.mu.RLock(); return e.mu.RUnlock },
		Tag: func(id uint64) (uint64, bool) {
			li, _, ok := e.locate(id)
			if !ok {
				return 0, false // still in the delta: not in a segment yet
			}
			return e.segs[li].gen, true
		},
		Identity: ident,
	})
	if err != nil {
		return fail(err)
	}
	if err := f.Sync(); err != nil {
		return fail(err)
	}
	if err := f.Close(); err != nil {
		fsx.Remove(tmp)
		return err
	}
	if err := fsx.Rename(tmp, final); err != nil {
		fsx.Remove(tmp)
		return err
	}
	syncDir(filepath.Dir(final))
	e.mu.Lock()
	if e.annChanges >= changes {
		e.annChanges -= changes
	} else {
		e.annChanges = 0
	}
	e.annSaved = true
	e.mu.Unlock()
	slog.Info("ANN graph saved", "path", final, "nodes", n, "duration", time.Since(start))
	return nil
}

// maybeSaveANNAsyncLocked starts a background SaveANN when many graph mutations
// piled up since the last one (force: whenever any did). Engine.mu held.
func (e *Engine) maybeSaveANNAsyncLocked(force bool) {
	if e.ann == nil || e.dbPath == "" {
		return
	}
	if e.annChanges == 0 || (!force && e.annChanges < annSaveEvery) {
		return
	}
	if !e.annSaving.CompareAndSwap(false, true) {
		return
	}
	e.annWG.Add(1)
	go func() {
		defer e.annWG.Done()
		defer e.annSaving.Store(false)
		if err := e.SaveANN(); err != nil {
			slog.Warn("index: could not save the ANN graph", "error", err)
		}
	}()
}

// tryLoadANNLocked restores the graph from its sidecar and reconciles it with
// the segments. It reports false — and the caller rebuilds — when there is no
// usable file. Engine.mu held for writing.
func (e *Engine) tryLoadANNLocked() bool {
	data, err := os.ReadFile(annFilePath(e.dbPath))
	if err != nil {
		return false
	}
	g, ident, err := ann.Read(data)
	if err != nil {
		slog.Info("index: ANN graph file unusable, rebuilding", "error", err)
		return false
	}
	if !bytes.Equal(ident, e.annIdentity()) {
		slog.Info("index: ANN graph file was built for something else, rebuilding")
		return false
	}

	type pending struct {
		id uint64
		v  []uint16
	}
	var todo []pending
	total := 0
	e.eachVectorGen(func(id uint64, v []uint16, gen uint64) {
		total++
		if tag, ok := g.NodeTag(id); ok && tag == gen {
			g.SetVec(id, v) // the node's links were built for exactly this vector
			return
		}
		todo = append(todo, pending{id, v}) // new since the file, or replaced
	})
	if total == 0 || len(todo)*2 > total {
		slog.Info("index: ANN graph file is too stale to reuse, rebuilding", "stale", len(todo), "vectors", total)
		return false
	}
	gone := g.Finish()
	for _, p := range todo {
		g.Insert(p.id, Float16ToFloats(p.v), p.v)
	}
	e.ann = g
	e.annFromDisk = true
	e.annSaved = true
	e.annChanges = int64(len(todo) + gone)
	slog.Info("index: ANN graph restored", "vectors", total, "inserted", len(todo), "removed", gone)
	return true
}
