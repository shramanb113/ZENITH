package index

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	pathutil "path/filepath"
	"sort"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/fsx"
	"github.com/shramanb113/ZENITH/internal/segment"
)

// checkEmbedder refuses to combine vectors from a different embedding space.
func (e *Engine) checkEmbedder(savedName string, savedDims int) error {
	curName, curDims := e.embedderIdentity()
	if curName != "unknown" && savedName != "unknown" && (curName != savedName || curDims != savedDims) {
		return fmt.Errorf("%w: file has %q (%d-dim), engine has %q (%d-dim)",
			ErrEmbedderMismatch, savedName, savedDims, curName, curDims)
	}
	return nil
}

// Save makes the engine's state durable at path.
//
// Called with the path the engine was loaded from (or the first path it is
// saved to), it flushes only what changed since the last Save: the documents
// added, as one new immutable segment, plus the IDs deleted. Cost is
// proportional to that delta, not to the index. Called with a different path
// it writes a complete, compacted copy there ("export") and leaves the engine
// bound to its own file.
//
// A flush does not stall searches or writes: the delta is frozen in O(1) under
// the engine lock (see frozen.go), the segment is written and synced with no
// engine lock held, and the engine lock is taken again only to swap the new
// segment in and commit the manifest. Only an export to another path (a merged
// copy) holds the lock for its whole duration.
func (e *Engine) Save(path string) error {
	e.saveMu.Lock()
	defer e.saveMu.Unlock()

	path = pathutil.Clean(path)
	start := time.Now()
	e.mu.RLock()
	bound := e.dbPath
	e.mu.RUnlock()
	var err error
	if bound == "" || samePath(path, bound) {
		err = e.flushAll(path)
	} else {
		e.mu.Lock()
		err = e.exportLocked(path)
		e.mu.Unlock()
	}
	if err != nil {
		return err
	}
	e.mu.RLock()
	nsegs := len(e.segs)
	e.mu.RUnlock()
	slog.Info("Index saved", "path", path, "segments", nsegs, "duration", time.Since(start))
	return nil
}

// BeginCheckpoint freezes the documents added so far so FinishCheckpoint can
// write them without holding any engine lock. It is O(1) plus a copy of the
// frozen documents' attributes. froze is true when a new frozen layer was
// created; pending is true when a frozen layer exists on return — the new one,
// or one left by an earlier failed flush, which FinishCheckpoint will write.
//
// A caller that keeps its own journal (pkg/zenith's WAL) must rotate it in the
// same critical section as BeginCheckpoint, so that exactly the journalled
// mutations before the freeze are covered by what FinishCheckpoint writes.
func (e *Engine) BeginCheckpoint(path string) (froze, pending bool, err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if _, err := e.bindLocked(pathutil.Clean(path)); err != nil {
		return false, false, err
	}
	if e.frozen != nil {
		return false, true, nil
	}
	if e.memEmptyLocked() {
		return false, false, nil
	}
	e.freezeLocked()
	return true, true, nil
}

// FinishCheckpoint writes the frozen layer (if any) as a segment and commits it.
// Searches and writes proceed meanwhile. It does nothing when nothing is frozen.
func (e *Engine) FinishCheckpoint(path string) error {
	e.saveMu.Lock()
	defer e.saveMu.Unlock()
	start := time.Now()
	if err := e.flushFrozen(pathutil.Clean(path)); err != nil {
		return err
	}
	slog.Info("Checkpoint written", "path", path, "duration", time.Since(start))
	return nil
}

// bindLocked binds the engine to path on the first flush, refusing to replace a
// readable old-format index. adopting reports whether this call bound it.
func (e *Engine) bindLocked(path string) (adopting bool, err error) {
	if e.dbPath != "" {
		return false, nil
	}
	// Never silently replace a readable old-format index with a new one that
	// holds different documents: that is the one way an upgrade could lose data.
	if _, _, _, err := readManifest(path); err != nil {
		var lfe *LegacyFormatError
		if errors.As(err, &lfe) && !e.replaceLegacy {
			return false, fmt.Errorf("index: refusing to overwrite %s: %w", path, err)
		}
	}
	e.dbPath = path
	e.needGC = true
	return true, nil
}

// nextGen allocates a segment file number.
func (e *Engine) nextGenLocked() uint64 {
	g := e.nextGen
	e.nextGen++
	return g
}

func (e *Engine) manifestLocked() manifestBody {
	m := manifestBody{Generation: e.manifestGen + 1, NextGen: e.nextGen}
	for _, l := range e.segs {
		m.Segments = append(m.Segments, manifestSegment{
			File: pathutil.Base(l.file), Gen: l.gen, Docs: l.seg.NumDocs(), Bytes: int64(l.seg.Size()),
		})
	}
	return m
}

func (e *Engine) commitManifestLocked() error {
	name, dims := e.embedderIdentity()
	m := e.manifestLocked()
	if err := writeManifest(e.dbPath, name, dims, m); err != nil {
		return err
	}
	e.manifestGen = m.Generation
	return nil
}

// memEmptyLocked reports whether the active delta holds nothing that needs flushing.
func (e *Engine) memEmptyLocked() bool {
	return len(e.idMapping) == 0 && len(e.vectors.GetWordVectors()) == 0 && len(e.pendingDels) == 0
}

// flushAll makes everything added before the call durable at path. saveMu held.
func (e *Engine) flushAll(path string) error {
	e.mu.Lock()
	adopting, err := e.bindLocked(path)
	if err != nil {
		e.mu.Unlock()
		return err
	}
	if e.frozen == nil && e.memEmptyLocked() {
		// Nothing new. Still make sure a manifest exists so the path is a valid
		// (possibly empty) index file.
		defer e.mu.Unlock()
		if _, err := os.Stat(path); err != nil || adopting {
			if err := e.commitManifestLocked(); err != nil {
				return err
			}
			if e.needGC {
				gcSegments(path, e.manifestLocked())
				e.needGC = false
			}
		}
		return nil
	}
	leftover := e.frozen != nil // a layer whose flush failed earlier
	e.mu.Unlock()

	for pass := 0; pass < 2; pass++ {
		e.mu.Lock()
		if e.frozen == nil {
			if e.memEmptyLocked() {
				e.mu.Unlock()
				return nil
			}
			e.freezeLocked()
		}
		e.mu.Unlock()
		if err := e.flushFrozen(path); err != nil {
			return err
		}
		if !leftover {
			break // documents added during the flush belong to the next checkpoint
		}
	}
	return nil
}

// flushFrozen writes the frozen layer as a new segment and swaps it in. The
// engine lock is not held while the segment is written, synced or mapped.
// saveMu held. On failure the frozen layer stays for the next attempt.
func (e *Engine) flushFrozen(path string) error {
	e.mu.Lock()
	f := e.frozen
	if f == nil {
		e.mu.Unlock()
		return nil
	}
	gen := e.nextGenLocked()
	e.mu.Unlock()

	file := segmentFileName(path, gen)
	if err := e.writeMem(f, file); err != nil {
		fsx.Remove(file)
		return fmt.Errorf("index: write segment: %w", err)
	}
	failpoint("flush-after-segment")

	seg, err := segment.Open(file)
	if err != nil {
		fsx.Remove(file)
		return fmt.Errorf("index: reopen segment: %w", err)
	}
	nl := newSegLayer(seg, file, gen)

	e.mu.Lock()
	defer e.mu.Unlock()
	e.segs = append(e.segs, nl)
	if err := e.commitManifestLocked(); err != nil {
		// The segment is not committed: drop it again so memory and disk agree.
		e.segs = e.segs[:len(e.segs)-1]
		seg.Close()
		fsx.Remove(file)
		return err
	}
	failpoint("flush-after-manifest")
	if e.needGC {
		gcSegments(path, e.manifestLocked())
		e.needGC = false
	}

	// The frozen documents now live in the new segment. Those deleted or replaced
	// since the freeze were written into it; mark them dead there.
	var moved []uint64
	if e.ann != nil {
		for id := range f.vectors.GetVectors() {
			if _, gone := f.dead[id]; !gone {
				moved = append(moved, id)
			}
		}
	}
	for id := range f.dead {
		if row, ok := seg.FindDoc(id); ok {
			nl.kill(row)
		}
	}
	e.frozen = nil
	e.rebindANNSomeLocked(moved) // O(delta): earlier segments did not move
	e.maybeCompactLocked()
	e.maybeSaveANNAsyncLocked(false)
	return nil
}

// resetMemLocked empties the mutable delta and drops any frozen layer
// (Load and legacy import replace the whole state).
func (e *Engine) resetMemLocked() {
	e.frozen = nil
	e.inverted.Lock()
	e.vectors.Lock()
	e.phonetics.Lock()
	e.inverted.Reset()
	e.vectors.ReplaceAll(make(map[uint64]VectorEntry), make(map[string]VectorEntry))
	e.phonetics.ReplaceAll(make(map[string][]uint64))
	e.idMapping = make(map[uint64]string)
	e.docText = make(map[uint64]string)
	e.inverted.Unlock()
	e.vectors.Unlock()
	e.phonetics.Unlock()
	e.bm25.Reset()
}

// writeMem writes a memory layer to file as a segment, without touching engine
// state. The layer must not be mutated while this runs: a frozen layer is
// immutable by construction, the active delta is only written under Engine.mu.
func (e *Engine) writeMem(m *memLayer, file string) error {
	ids := make([]uint64, 0, len(m.idMapping))
	for id := range m.idMapping {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	rowOf := make(map[uint64]int, len(ids))
	for r, id := range ids {
		rowOf[id] = r
	}

	lengths, termFreqs, _, _, _ := m.bm25.State()
	vecs := m.vectors.GetVectors()

	_, dims := e.embedderIdentity()
	if dims == 0 { // lexical-only engine: dimension is irrelevant but must be consistent
		dims = 1
	}
	w, err := segment.Create(file, dims)
	if err != nil {
		return err
	}

	w.WriteDocs(len(ids), func(i int) segment.DocIn {
		id := ids[i]
		d := segment.DocIn{
			ID: id, Orig: m.idMapping[id], Text: m.docText[id],
			Attrs: encodeAttrs(m.attrs[id]), Len: lengths[id],
		}
		if ent, ok := vecs[id]; ok && len(ent.Vector) == dims {
			d.Vec = ent.Vector
		}
		return d
	})

	// BM25 vocabulary + per-term postings, built from the delta's term frequencies.
	type acc struct {
		cf    int
		posts []segment.Posting
	}
	terms := make(map[string]*acc)
	for row, id := range ids {
		for term, tf := range termFreqs[id] {
			a := terms[term]
			if a == nil {
				a = &acc{}
				terms[term] = a
			}
			a.cf += tf
			a.posts = append(a.posts, segment.Posting{Doc: row, TF: tf})
		}
	}
	termList := make([]string, 0, len(terms))
	for t := range terms {
		termList = append(termList, t)
	}
	sort.Strings(termList)
	termIdx := make(map[string]int, len(termList))
	for i, t := range termList {
		termIdx[t] = i
	}
	ti := 0
	w.WriteTerms(func() (segment.TermIn, bool) {
		if ti >= len(termList) {
			return segment.TermIn{}, false
		}
		t := termList[ti]
		ti++
		return segment.TermIn{Term: t, CF: terms[t].cf, Posts: terms[t].posts}, true
	})

	keyIter := func(m map[string][]uint64) func() (segment.KeyIn, bool) {
		keys := make([]string, 0, len(m))
		for k, list := range m {
			if len(list) > 0 {
				keys = append(keys, k)
			}
		}
		sort.Strings(keys)
		i := 0
		return func() (segment.KeyIn, bool) {
			if i >= len(keys) {
				return segment.KeyIn{}, false
			}
			k := keys[i]
			i++
			rows := make([]int, 0, len(m[k]))
			for _, id := range m[k] {
				if r, ok := rowOf[id]; ok {
					rows = append(rows, r)
				}
			}
			sort.Ints(rows)
			return segment.KeyIn{Key: k, Docs: rows}, true
		}
	}
	w.WriteFrags(keyIter(m.inverted.GetData()))
	w.WritePhon(keyIter(m.phonetics.GetData()))

	w.WriteForward(len(ids), func(i int) []segment.TermFreq {
		tf := termFreqs[ids[i]]
		out := make([]segment.TermFreq, 0, len(tf))
		for term, f := range tf {
			out = append(out, segment.TermFreq{Term: termIdx[term], TF: f})
		}
		sort.Slice(out, func(a, b int) bool { return out[a].Term < out[b].Term })
		return out
	})

	words := make([]string, 0, len(m.vectors.GetWordVectors()))
	for word, ent := range m.vectors.GetWordVectors() {
		if len(ent.Vector) == dims {
			words = append(words, word)
		}
	}
	sort.Strings(words)
	w.WriteWords(len(words), func(i int) (string, []uint16) {
		return words[i], m.vectors.GetWordVectors()[words[i]].Vector
	})

	dels := make([]uint64, 0, len(m.dels))
	for id := range m.dels {
		dels = append(dels, id)
	}
	sort.Slice(dels, func(i, j int) bool { return dels[i] < dels[j] })
	w.WriteDels(dels)

	if err := w.Err(); err != nil {
		w.Abort()
		return err
	}
	name, edims := e.embedderIdentity()
	return w.Finish([]byte(fmt.Sprintf(`{"embedder":%q,"dims":%d,"created":%q}`, name, edims, time.Now().UTC().Format(time.RFC3339))))
}

// Load replaces the engine's contents with the index at path and binds the
// engine to it, so later Saves flush incrementally. Segment files are opened
// by memory-mapping, so this is fast and adds almost nothing to the Go heap.
//
// Gob-format files from older versions return an error matching
// ErrIncompatibleVersion (a *LegacyFormatError); convert them with LoadLegacy
// / `zenith migrate`.
func (e *Engine) Load(path string) error {
	e.compactMu.Lock() // a running merge reads the mapped segments Load would unmap
	defer e.compactMu.Unlock()
	e.saveMu.Lock() // and so does a running flush
	defer e.saveMu.Unlock()
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.loadLocked(pathutil.Clean(path))
}

func (e *Engine) loadLocked(path string) error {
	start := time.Now()
	name, dims, m, err := readManifest(path)
	if err != nil {
		if os.IsNotExist(err) {
			slog.Warn("No persistence file found, starting fresh", "path", path)
		}
		return err
	}
	if err := e.checkEmbedder(name, dims); err != nil {
		return err
	}

	// Open every segment before touching engine state, so a failure leaves the
	// engine exactly as it was.
	dir := pathutil.Dir(path)
	layers := make([]*segLayer, 0, len(m.Segments))
	closeAll := func() {
		for _, l := range layers {
			l.seg.Close()
		}
	}
	for _, ms := range m.Segments {
		file := pathutil.Join(dir, ms.File)
		seg, err := segment.Open(file)
		if err != nil {
			closeAll()
			return fmt.Errorf("index: open segment %s: %w", ms.File, err)
		}
		if seg.Dims() != dims && seg.NumVecs()+seg.NumWords() > 0 {
			seg.Close()
			closeAll()
			return fmt.Errorf("index: segment %s has %d-dim vectors, manifest says %d", ms.File, seg.Dims(), dims)
		}
		layers = append(layers, newSegLayer(seg, file, ms.Gen))
	}

	e.closeLayersLocked()
	e.resetMemLocked()
	e.attrs = make(map[uint64]Attrs)
	e.segs = layers
	e.dbPath = path
	e.needGC = false
	e.nextGen = m.NextGen
	e.manifestGen = m.Generation
	e.pendingDels = nil
	e.bk = nil
	e.bm25.SetBacking(segBacking{e})

	// Apply each segment's recorded deletions to the older segments, in order.
	for k, l := range e.segs {
		for _, id := range l.seg.Dels() {
			for j := k - 1; j >= 0; j-- {
				o := e.segs[j]
				if row, ok := o.seg.FindDoc(id); ok && !o.isDead(row) {
					o.kill(row)
					break
				}
			}
		}
	}
	// Attributes of live documents stay on the heap: filtered search evaluates
	// predicates per candidate and they are typically small.
	for _, l := range e.segs {
		for k, n := 0, l.seg.NumAttrRows(); k < n; k++ {
			row := l.seg.AttrRow(k)
			if l.isDead(row) {
				continue
			}
			if a := decodeAttrs(l.seg.AttrsAt(k)); a != nil {
				e.attrs[l.seg.DocID(row)] = a
			}
		}
	}

	e.attrIdx = rebuildAttrIndex(e.attrs)

	gcSegments(path, m)
	e.rebuildANNAfterLoadLocked()

	if e.fstPath != "" && e.tryLoadFSTFromDiskLocked() {
		slog.Info("Index loaded", "docs", e.docCountLocked(), "segments", len(e.segs), "duration", time.Since(start))
		return nil
	}
	if err := e.rebuildFSTLocked(); err != nil {
		slog.Warn("index: FST rebuild after load failed", "error", err)
	}
	slog.Info("Index loaded", "docs", e.docCountLocked(), "segments", len(e.segs), "duration", time.Since(start))
	return nil
}

// tryLoadFSTFromDiskLocked reuses a persisted FST when it matches the loaded vocabulary.
func (e *Engine) tryLoadFSTFromDiskLocked() bool {
	if err := e.fst.OpenFromFile(e.fstPath); err != nil {
		slog.Info("index: FST file not found, rebuilding", "path", e.fstPath)
		return false
	}
	// A stale FST file (from another run, or a save that did not complete) must
	// not survive: a term-count mismatch is a cheap, effective check.
	if vocab := len(e.liveTerms()); e.fst.Size() != vocab {
		slog.Warn("index: FST on disk does not match loaded vocabulary, rebuilding",
			"fst_terms", e.fst.Size(), "vocab_terms", vocab)
		return false
	}
	e.fstSize = e.fst.Size()
	e.fstDirty = false
	if w, ok := e.analyzer.(analysis.FSTWirer); ok {
		w.SetFST(e.fst)
	}
	slog.Info("index: FST loaded from disk", "path", e.fstPath, "terms", e.fst.Size())
	return true
}

// closeLayersLocked unmaps every segment. Anything aliasing them (the ANN
// graph's vector slices) must already be dropped or about to be replaced.
func (e *Engine) closeLayersLocked() {
	e.ann = nil
	for _, l := range e.segs {
		if err := l.seg.Close(); err != nil {
			slog.Warn("index: closing segment", "file", l.file, "error", err)
		}
	}
	e.segs = nil
}

// Close releases the engine's memory maps. The engine must not be used
// afterwards. Data is not saved: call Save first if there are unsaved changes.
func (e *Engine) Close() error {
	// Wait for any in-flight background ANN save (internal/index/ann_persist.go)
	// before taking mu below: SaveANN itself needs mu.RLock while it runs, so
	// waiting here (before Close holds the write lock) avoids a deadlock while
	// still guaranteeing the goroutine's file writes finish before Close tears
	// down the engine's files out from under it.
	e.annWG.Wait()

	e.compactMu.Lock()
	defer e.compactMu.Unlock()
	e.saveMu.Lock()
	defer e.saveMu.Unlock()
	e.mu.Lock()
	defer e.mu.Unlock()
	e.closeLayersLocked()
	if e.cache != nil {
		// Releases the L2 Redis client's connection pool, if one is
		// configured — a no-op otherwise. See Tiered.Close's doc comment
		// for why this matters (internal/collections idle-closes and
		// reopens an Engine per collection).
		if err := e.cache.Close(); err != nil {
			slog.Warn("index: query-result cache L2 close failed", "error", err)
		}
	}
	return nil
}

// docCountLocked is the number of live documents. Engine.mu held.
func (e *Engine) docCountLocked() int {
	return len(e.idMapping) + e.liveSegDocs()
}

// DBPath is the manifest path the engine is bound to ("" if never saved/loaded).
func (e *Engine) DBPath() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.dbPath
}

// SegmentCount is the number of segment files currently backing the engine.
func (e *Engine) SegmentCount() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.segs)
}

// samePath reports whether a and b name the same manifest file, however they are
// spelled (relative vs absolute, different separators or case on Windows,
// symlinks). Comparing strings alone would treat "zenith.db" and its absolute
// path as different files, and a "Save to another path" would then rewrite
// segment files the engine still has mapped.
func samePath(a, b string) bool {
	if a == b {
		return true
	}
	if aa, err := pathutil.Abs(a); err == nil {
		if bb, err := pathutil.Abs(b); err == nil && aa == bb {
			return true
		}
	}
	fa, err := os.Stat(a)
	if err != nil {
		return false
	}
	fb, err := os.Stat(b)
	if err != nil {
		return false
	}
	return os.SameFile(fa, fb)
}
