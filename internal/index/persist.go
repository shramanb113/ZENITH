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
// Save holds the engine's write lock while it writes: searches and writes
// wait for the flush. Because the delta is bounded by the checkpoint policy,
// that is short; large merges (Compact) run without the lock.
func (e *Engine) Save(path string) error {
	e.saveMu.Lock()
	defer e.saveMu.Unlock()
	e.mu.Lock()
	defer e.mu.Unlock()

	path = pathutil.Clean(path)
	start := time.Now()
	var err error
	if e.dbPath == "" || samePath(path, e.dbPath) {
		err = e.flushLocked(path)
	} else {
		err = e.exportLocked(path)
	}
	if err != nil {
		return err
	}
	slog.Info("Index saved", "path", path, "segments", len(e.segs), "duration", time.Since(start))
	return nil
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

// memEmpty reports whether the delta holds nothing that needs flushing.
func (e *Engine) memEmptyLocked() bool {
	return len(e.idMapping) == 0 && len(e.vectors.GetWordVectors()) == 0 && len(e.pendingDels) == 0
}

// flushLocked writes the delta as a new segment and commits it. Engine.mu held for writing.
func (e *Engine) flushLocked(path string) error {
	adopting := e.dbPath == ""
	if adopting {
		// Never silently replace a readable old-format index with a new one that
		// holds different documents: that is the one way an upgrade could lose data.
		if _, _, _, err := readManifest(path); err != nil {
			var lfe *LegacyFormatError
			if errors.As(err, &lfe) && !e.replaceLegacy {
				return fmt.Errorf("index: refusing to overwrite %s: %w", path, err)
			}
		}
		e.dbPath = path
	}
	if e.memEmptyLocked() {
		// Nothing new. Still make sure a manifest exists so the path is a valid
		// (possibly empty) index file.
		if _, err := os.Stat(path); err != nil || adopting {
			if err := e.commitManifestLocked(); err != nil {
				return err
			}
			if adopting {
				gcSegments(path, e.manifestLocked())
			}
		}
		return nil
	}

	gen := e.nextGenLocked()
	file := segmentFileName(path, gen)
	if err := e.writeMemSegment(file); err != nil {
		os.Remove(file)
		return fmt.Errorf("index: write segment: %w", err)
	}
	failpoint("flush-after-segment")

	seg, err := segment.Open(file)
	if err != nil {
		os.Remove(file)
		return fmt.Errorf("index: reopen segment: %w", err)
	}
	e.segs = append(e.segs, newSegLayer(seg, file, gen))
	if err := e.commitManifestLocked(); err != nil {
		// The segment is not committed: drop it again so memory and disk agree.
		e.segs = e.segs[:len(e.segs)-1]
		seg.Close()
		os.Remove(file)
		return err
	}
	failpoint("flush-after-manifest")
	if adopting {
		gcSegments(path, e.manifestLocked())
	}

	// The documents now live in the new segment: empty the delta and point
	// everything that referenced the heap copies at the mapping.
	var flushedVecIDs []uint64
	if e.ann != nil {
		for id := range e.vectors.GetVectors() {
			flushedVecIDs = append(flushedVecIDs, id)
		}
	}
	e.resetMemLocked()
	e.pendingDels = nil
	e.rebindANNSomeLocked(flushedVecIDs) // O(delta): earlier segments did not move
	e.fstDirty = false
	e.maybeCompactLocked()
	return nil
}

// resetMemLocked empties the mutable delta (documents already persisted).
func (e *Engine) resetMemLocked() {
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

// writeMemSegment writes the delta to file as a segment, without touching engine state.
func (e *Engine) writeMemSegment(file string) error {
	ids := make([]uint64, 0, len(e.idMapping))
	for id := range e.idMapping {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	rowOf := make(map[uint64]int, len(ids))
	for r, id := range ids {
		rowOf[id] = r
	}

	lengths, termFreqs, _, _, _ := e.bm25.State()
	vecs := e.vectors.GetVectors()

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
			ID: id, Orig: e.idMapping[id], Text: e.docText[id],
			Attrs: encodeAttrs(e.attrs[id]), Len: lengths[id],
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
	w.WriteFrags(keyIter(e.inverted.GetData()))
	w.WritePhon(keyIter(e.phonetics.GetData()))

	w.WriteForward(len(ids), func(i int) []segment.TermFreq {
		tf := termFreqs[ids[i]]
		out := make([]segment.TermFreq, 0, len(tf))
		for term, f := range tf {
			out = append(out, segment.TermFreq{Term: termIdx[term], TF: f})
		}
		sort.Slice(out, func(a, b int) bool { return out[a].Term < out[b].Term })
		return out
	})

	words := make([]string, 0, len(e.vectors.GetWordVectors()))
	for word, ent := range e.vectors.GetWordVectors() {
		if len(ent.Vector) == dims {
			words = append(words, word)
		}
	}
	sort.Strings(words)
	w.WriteWords(len(words), func(i int) (string, []uint16) {
		return words[i], e.vectors.GetWordVectors()[words[i]].Vector
	})

	dels := make([]uint64, 0, len(e.pendingDels))
	for id := range e.pendingDels {
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
	e.compactMu.Lock()
	defer e.compactMu.Unlock()
	e.saveMu.Lock()
	defer e.saveMu.Unlock()
	e.mu.Lock()
	defer e.mu.Unlock()
	e.closeLayersLocked()
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
