package index

import (
	"bytes"
	"fmt"
	"log/slog"
	"os"
	pathutil "path/filepath"
	"sort"
	"time"
	"unsafe"

	"github.com/shramanb113/ZENITH/internal/fsx"
	"github.com/shramanb113/ZENITH/internal/segment"
)

// mergeInput is one segment plus the set of its rows that must not be copied.
type mergeInput struct {
	seg  *segment.Segment
	dead []uint64 // bitset over rows; nil = none dead
}

func (m mergeInput) isDead(row int) bool {
	return m.dead != nil && m.dead[row>>6]&(1<<uint(row&63)) != 0
}

func rawString(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	return unsafe.String(&b[0], len(b))
}

// mergeSegments writes the live documents of inputs (oldest first) into one
// new segment at file and returns how many documents it holds. It reads the
// inputs' memory maps only, so it needs no engine lock, and streams postings
// through k-way merges so memory stays proportional to a single posting list
// plus a few integers per document and term, not to the index.
//
// If the same document ID is live in more than one input the newest input wins.
func mergeSegments(file string, dims int, inputs []mergeInput, embName string, embDims int) (int, error) {
	// --- document order: live rows sorted by ID, newest input winning ties.
	type ref struct{ in, row int32 }
	var refs []ref
	for i, in := range inputs {
		for r, n := 0, in.seg.NumDocs(); r < n; r++ {
			if !in.isDead(r) {
				refs = append(refs, ref{int32(i), int32(r)})
			}
		}
	}
	idOf := func(r ref) uint64 { return inputs[r.in].seg.DocID(int(r.row)) }
	sort.Slice(refs, func(a, b int) bool {
		ia, ib := idOf(refs[a]), idOf(refs[b])
		if ia != ib {
			return ia < ib
		}
		return refs[a].in > refs[b].in // newest first, so it is the one kept below
	})
	order := make([]ref, 0, len(refs))
	for i, r := range refs {
		if i > 0 && idOf(refs[i-1]) == idOf(r) {
			continue
		}
		order = append(order, r)
	}
	remap := make([][]int32, len(inputs))
	for i, in := range inputs {
		remap[i] = make([]int32, in.seg.NumDocs())
		for r := range remap[i] {
			remap[i][r] = -1
		}
	}
	for newRow, r := range order {
		remap[r.in][r.row] = int32(newRow)
	}

	w, err := segment.Create(file, dims)
	if err != nil {
		return 0, err
	}

	w.WriteDocs(len(order), func(i int) segment.DocIn {
		r := order[i]
		s := inputs[r.in].seg
		orig, text := s.DocRaw(int(r.row))
		return segment.DocIn{
			ID: s.DocID(int(r.row)), Orig: rawString(orig), Text: rawString(text),
			Attrs: s.AttrsRaw(int(r.row)), Len: s.DocLen(int(r.row)), Vec: s.Vec(int(r.row)),
		}
	})

	// --- terms: merged dictionary, with a per-input old-index -> new-index map.
	tmap := make([][]int32, len(inputs))
	for i, in := range inputs {
		tmap[i] = make([]int32, in.seg.NumTerms())
		for t := range tmap[i] {
			tmap[i][t] = -1
		}
	}
	cur := make([]int, len(inputs))
	nextTerm := int32(0)
	w.WriteTerms(func() (segment.TermIn, bool) {
		for {
			var min []byte
			for i, in := range inputs {
				if cur[i] < in.seg.NumTerms() {
					if k := in.seg.TermKey(cur[i]); min == nil || bytes.Compare(k, min) < 0 {
						min = k
					}
				}
			}
			if min == nil {
				return segment.TermIn{}, false
			}
			key := string(min)
			var posts []segment.Posting
			cf := 0
			var members, memberTerm []int
			for i, in := range inputs {
				if cur[i] < in.seg.NumTerms() && bytes.Equal(in.seg.TermKey(cur[i]), min) {
					t := cur[i]
					cur[i]++
					members = append(members, i)
					memberTerm = append(memberTerm, t)
					in.seg.TermPostings(t, func(row, tf int) bool {
						if nr := remap[i][row]; nr >= 0 {
							posts = append(posts, segment.Posting{Doc: int(nr), TF: tf})
							cf += tf
						}
						return true
					})
				}
			}
			if len(posts) == 0 {
				continue // every document containing the term is gone
			}
			sort.Slice(posts, func(a, b int) bool { return posts[a].Doc < posts[b].Doc })
			for j, i := range members {
				tmap[i][memberTerm[j]] = nextTerm
			}
			nextTerm++
			return segment.TermIn{Term: key, CF: cf, Posts: posts}, true
		}
	})

	// --- fragments and phonetic codes: k-way merge of row lists.
	mergeKeys := func(n func(*segment.Segment) int, key func(*segment.Segment, int) []byte,
		posts func(*segment.Segment, int, func(int) bool)) func() (segment.KeyIn, bool) {
		cur := make([]int, len(inputs))
		return func() (segment.KeyIn, bool) {
			for {
				var min []byte
				for i, in := range inputs {
					if cur[i] < n(in.seg) {
						if k := key(in.seg, cur[i]); min == nil || bytes.Compare(k, min) < 0 {
							min = k
						}
					}
				}
				if min == nil {
					return segment.KeyIn{}, false
				}
				name := string(min)
				var rows []int
				for i, in := range inputs {
					if cur[i] < n(in.seg) && bytes.Equal(key(in.seg, cur[i]), min) {
						posts(in.seg, cur[i], func(row int) bool {
							if nr := remap[i][row]; nr >= 0 {
								rows = append(rows, int(nr))
							}
							return true
						})
						cur[i]++
					}
				}
				if len(rows) == 0 {
					continue
				}
				sort.Ints(rows)
				return segment.KeyIn{Key: name, Docs: rows}, true
			}
		}
	}
	w.WriteFrags(mergeKeys((*segment.Segment).NumFrags, (*segment.Segment).FragKey, (*segment.Segment).FragPostings))
	w.WritePhon(mergeKeys((*segment.Segment).NumPhon, (*segment.Segment).PhonKey, (*segment.Segment).PhonPostings))

	// --- forward index, remapped to the merged term dictionary.
	w.WriteForward(len(order), func(i int) []segment.TermFreq {
		r := order[i]
		var out []segment.TermFreq
		inputs[r.in].seg.Forward(int(r.row), func(term, tf int) {
			if nt := tmap[r.in][term]; nt >= 0 {
				out = append(out, segment.TermFreq{Term: int(nt), TF: tf})
			}
		})
		return out
	})

	// --- word vectors: union; identical words carry identical vectors.
	type wref struct{ in, idx int }
	var words []wref
	wcur := make([]int, len(inputs))
	for {
		var min []byte
		for i, in := range inputs {
			if wcur[i] < in.seg.NumWords() {
				if k := in.seg.WordKey(wcur[i]); min == nil || bytes.Compare(k, min) < 0 {
					min = k
				}
			}
		}
		if min == nil {
			break
		}
		var pick wref
		for i, in := range inputs {
			if wcur[i] < in.seg.NumWords() && bytes.Equal(in.seg.WordKey(wcur[i]), min) {
				pick = wref{i, wcur[i]} // later inputs overwrite: newest wins
				wcur[i]++
			}
		}
		words = append(words, pick)
	}
	w.WriteWords(len(words), func(i int) (string, []uint16) {
		s := inputs[words[i].in].seg
		return rawString(s.WordKey(words[i].idx)), s.WordVec(words[i].idx)
	})
	w.WriteDels(nil)

	if err := w.Err(); err != nil {
		w.Abort()
		return 0, err
	}
	meta := fmt.Sprintf(`{"embedder":%q,"dims":%d,"created":%q,"merged":%d}`, embName, embDims, time.Now().UTC().Format(time.RFC3339), len(inputs))
	if err := w.Finish([]byte(meta)); err != nil {
		return 0, err
	}
	return len(order), nil
}

// exportLocked writes a complete, compacted copy of the index at path. The
// engine stays bound to its own file. Engine.mu held for writing.
func (e *Engine) exportLocked(path string) error {
	// Belt and braces beyond samePath: never write a file this engine has mapped.
	// Rewriting a live mapping corrupts it (or, on Windows, fails mid-export).
	for _, l := range e.segs {
		if samePath(segmentFileName(path, 1), l.file) || samePath(path, l.file) {
			return fmt.Errorf("index: refusing to export over %s, which is part of the open index", l.file)
		}
	}
	if err := os.MkdirAll(pathutil.Dir(path), 0o755); err != nil {
		return err
	}
	inputs := make([]mergeInput, 0, len(e.segs)+1)
	for _, l := range e.segs {
		inputs = append(inputs, mergeInput{seg: l.seg, dead: l.dead})
	}
	// The frozen layer (a flush that failed earlier) and the active delta go in
	// as extra inputs, oldest first.
	for i, m := range []*memLayer{e.frozen, e.activeMemLocked()} {
		if m == nil || (len(m.idMapping) == 0 && len(m.vectors.GetWordVectors()) == 0) {
			continue
		}
		tmp := fmt.Sprintf("%s.export-tmp%d", path, i)
		if err := e.writeMem(m, tmp); err != nil {
			fsx.Remove(tmp)
			return fmt.Errorf("index: export: %w", err)
		}
		defer fsx.Remove(tmp)
		seg, err := segment.Open(tmp)
		if err != nil {
			return fmt.Errorf("index: export: %w", err)
		}
		defer seg.Close()
		in := mergeInput{seg: seg}
		if len(m.dead) > 0 {
			in.dead = make([]uint64, (seg.NumDocs()+63)/64)
			for id := range m.dead {
				if row, ok := seg.FindDoc(id); ok {
					in.dead[row>>6] |= 1 << uint(row&63)
				}
			}
		}
		inputs = append(inputs, in)
	}

	name, dims := e.embedderIdentity()
	segDims := max(dims, 1)
	file := segmentFileName(path, 1)
	docs, err := mergeSegments(file, segDims, inputs, name, dims)
	if err != nil {
		fsx.Remove(file)
		return fmt.Errorf("index: export: %w", err)
	}
	var size int64
	if fi, err := os.Stat(file); err == nil {
		size = fi.Size()
	}
	m := manifestBody{Generation: 1, NextGen: 2, Segments: []manifestSegment{{File: pathutil.Base(file), Gen: 1, Docs: docs, Bytes: size}}}
	if err := writeManifest(path, name, dims, m); err != nil {
		fsx.Remove(file)
		return err
	}
	gcSegments(path, m)
	return nil
}

// Compact merges every segment into one, dropping deleted documents and
// deleted terms, so reads touch fewer files and the disk shrinks. It is safe to
// call while the engine serves searches and accepts writes: the merge reads
// only immutable segment files, and the engine lock is held just to take the
// snapshot and to swap the result in. It does nothing when there is at most one
// segment and nothing in it is deleted.
func (e *Engine) Compact() error {
	e.compactMu.Lock()
	defer e.compactMu.Unlock()

	// 1. Snapshot.
	e.mu.Lock()
	if e.dbPath == "" || len(e.segs) == 0 || (len(e.segs) == 1 && e.segs[0].nDead == 0) {
		e.mu.Unlock()
		return nil
	}
	snap := append([]*segLayer(nil), e.segs...)
	snapDead := make([][]uint64, len(snap))
	inputs := make([]mergeInput, len(snap))
	for i, l := range snap {
		snapDead[i] = l.cloneDead()
		inputs[i] = mergeInput{seg: l.seg, dead: snapDead[i]}
	}
	path := e.dbPath
	gen := e.nextGenLocked()
	name, dims := e.embedderIdentity()
	e.mu.Unlock()

	start := time.Now()
	file := segmentFileName(path, gen)

	// 2. Merge without the engine lock.
	docs, err := mergeSegments(file, max(dims, 1), inputs, name, dims)
	if err != nil {
		fsx.Remove(file)
		return fmt.Errorf("index: compact: %w", err)
	}
	failpoint("compact-after-segment")
	merged, err := segment.Open(file)
	if err != nil {
		fsx.Remove(file)
		return fmt.Errorf("index: compact: reopen merged segment: %w", err)
	}

	// 3. Swap in.
	e.mu.Lock()
	defer e.mu.Unlock()
	same := len(e.segs) >= len(snap)
	for i := 0; same && i < len(snap); i++ {
		same = e.segs[i] == snap[i]
	}
	if !same {
		merged.Close()
		fsx.Remove(file)
		return fmt.Errorf("index: compact: engine changed underneath the merge; retry")
	}
	nl := newSegLayer(merged, file, gen)
	// Documents deleted or replaced while the merge ran were copied into the
	// merged segment; mark them dead there so they stay invisible.
	for i, l := range snap {
		for row, n := 0, l.seg.NumDocs(); row < n; row++ {
			if l.isDead(row) && snapDead[i][row>>6]&(1<<uint(row&63)) == 0 {
				if nr, ok := merged.FindDoc(l.seg.DocID(row)); ok {
					nl.kill(nr)
				}
			}
		}
	}
	old := e.segs
	e.segs = append([]*segLayer{nl}, old[len(snap):]...)
	if err := e.commitManifestLocked(); err != nil {
		e.segs = old
		merged.Close()
		fsx.Remove(file)
		return err
	}
	failpoint("compact-after-manifest")
	// Rebind anything aliasing the old mappings before they are unmapped.
	e.rebindANNLocked()
	// The graph file's tags name the segments that no longer exist: rewrite it.
	if e.ann != nil {
		e.annSaved = false
		e.annChanges = max(e.annChanges, 1)
		e.maybeSaveANNAsyncLocked(true)
	}
	for _, l := range snap {
		if err := l.seg.Close(); err != nil {
			slog.Warn("index: closing compacted segment", "file", l.file, "error", err)
		}
		fsx.Remove(l.file)
	}
	slog.Info("Index compacted", "segments_in", len(snap), "docs", docs, "duration", time.Since(start))
	return nil
}

// maxSegments is how many segments accumulate before a flush triggers a
// background compaction; every read consults every segment, so the count is
// kept small.
const maxSegments = 8

// maybeCompactLocked starts a background compaction when the segment count or
// the fraction of deleted documents warrants it. Engine.mu held.
func (e *Engine) maybeCompactLocked() {
	if !e.autoCompact || e.compacting.Load() {
		return
	}
	deadRows, rows := 0, 0
	for _, l := range e.segs {
		deadRows += l.nDead
		rows += l.seg.NumDocs()
	}
	if len(e.segs) <= maxSegments && !(rows >= 10_000 && deadRows*4 > rows) {
		return
	}
	if !e.compacting.CompareAndSwap(false, true) {
		return
	}
	go func() {
		defer e.compacting.Store(false)
		if err := e.Compact(); err != nil {
			slog.Warn("index: background compaction failed", "error", err)
		}
	}()
}

// SetAutoCompact enables or disables background compaction after flushes
// (default enabled). Tests that need a deterministic segment count disable it.
func (e *Engine) SetAutoCompact(on bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.autoCompact = on
}
