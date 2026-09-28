package index

import (
	"encoding/json"
	"sort"

	"github.com/shramanb113/ZENITH/internal/segment"
)

// The engine's documents live in two places:
//
//   - segs: immutable, memory-mapped segment files (oldest first), opened at
//     Load or produced by a flush/compaction. They hold most of a large index's
//     bytes, so those bytes are page cache, not Go heap.
//   - the in-memory maps (inverted, phonetics, vectors, idMapping, docText,
//     bm25 …): the mutable "delta" holding documents added since the last flush.
//
// A document is live in exactly one place. Replacing or removing a document
// that lives in a segment marks its row dead (a bit in segLayer.dead) instead
// of touching the segment; the death is made durable by the next flush, which
// records the ID in the new segment's deletes.
//
// Everything below requires Engine.mu (read or write, as the caller's method
// dictates); layers are only mutated under the write lock.

// segLayer is one opened segment plus this process's view of which of its rows
// are still alive.
type segLayer struct {
	seg      *segment.Segment
	file     string // path of the segment file
	gen      uint64
	totalLen int // Σ DocLen over every row

	dead       []uint64 // bitset over rows
	nDead      int
	deadLen    int         // Σ DocLen of dead rows
	deadTermDF map[int]int // term index -> dead docs containing the term
	deadTermCF map[int]int // term index -> dead occurrences of the term
}

func newSegLayer(seg *segment.Segment, file string, gen uint64) *segLayer {
	l := &segLayer{seg: seg, file: file, gen: gen, dead: make([]uint64, (seg.NumDocs()+63)/64)}
	for i := 0; i < seg.NumDocs(); i++ {
		l.totalLen += seg.DocLen(i)
	}
	return l
}

func (l *segLayer) isDead(row int) bool { return l.dead[row>>6]&(1<<uint(row&63)) != 0 }

func (l *segLayer) liveDocs() int { return l.seg.NumDocs() - l.nDead }

// kill marks row dead and records the statistics adjustments BM25 and the
// vocabulary need. It reports the terms whose document count dropped, so the
// caller can tell whether a term vanished from the index entirely.
func (l *segLayer) kill(row int) {
	if l.isDead(row) {
		return
	}
	l.dead[row>>6] |= 1 << uint(row&63)
	l.nDead++
	l.deadLen += l.seg.DocLen(row)
	if l.deadTermDF == nil {
		l.deadTermDF = make(map[int]int)
		l.deadTermCF = make(map[int]int)
	}
	l.seg.Forward(row, func(term, tf int) {
		l.deadTermDF[term]++
		l.deadTermCF[term] += tf
	})
}

func (l *segLayer) cloneDead() []uint64 { return append([]uint64(nil), l.dead...) }

// ---- attribute (de)serialisation ----

type wireAttr struct {
	K AttrKind `json:"k"`
	S string   `json:"s,omitempty"`
	N float64  `json:"n,omitempty"`
}

func encodeAttrs(a Attrs) []byte {
	if len(a) == 0 {
		return nil
	}
	m := make(map[string]wireAttr, len(a))
	for k, v := range a {
		m[k] = wireAttr{K: v.Kind, S: v.S, N: v.N}
	}
	b, _ := json.Marshal(m) // string keys and plain scalars cannot fail to marshal
	return b
}

func decodeAttrs(b []byte) Attrs {
	if len(b) == 0 {
		return nil
	}
	var m map[string]wireAttr
	if err := json.Unmarshal(b, &m); err != nil {
		return nil
	}
	out := make(Attrs, len(m))
	for k, v := range m {
		out[k] = AttrValue{Kind: v.K, S: v.S, N: v.N}
	}
	return out
}

// ---- lookups across layers (Engine.mu held) ----

// locate finds the live segment row holding document id, newest segment first.
func (e *Engine) locate(id uint64) (li, row int, ok bool) {
	for i := len(e.segs) - 1; i >= 0; i-- {
		l := e.segs[i]
		if r, found := l.seg.FindDoc(id); found && !l.isDead(r) {
			return i, r, true
		}
	}
	return 0, 0, false
}

// hasDoc reports whether a live document with this internal ID exists anywhere.
func (e *Engine) hasDoc(id uint64) bool {
	if _, ok := e.idMapping[id]; ok {
		return true
	}
	_, _, ok := e.locate(id)
	return ok
}

// origID resolves an internal ID to the caller-supplied document ID ("" if unknown).
func (e *Engine) origID(id uint64) string {
	if s, ok := e.idMapping[id]; ok {
		return s
	}
	if li, row, ok := e.locate(id); ok {
		return e.segs[li].seg.Orig(row)
	}
	return ""
}

// textOf returns a live document's original text.
func (e *Engine) textOf(id uint64) (string, bool) {
	if t, ok := e.docText[id]; ok {
		return t, true
	}
	if li, row, ok := e.locate(id); ok {
		return e.segs[li].seg.Text(row), true
	}
	return "", false
}

// liveSegDocs is the number of live documents held in segments.
func (e *Engine) liveSegDocs() int {
	n := 0
	for _, l := range e.segs {
		n += l.liveDocs()
	}
	return n
}

// killBase marks the live segment copy of id (if any) dead and reports whether
// there was one. The ID is remembered so the next flush persists the deletion.
func (e *Engine) killBase(id uint64) bool {
	li, row, ok := e.locate(id)
	if !ok {
		return false
	}
	l := e.segs[li]
	// Terms whose only occurrences were in this document leave the vocabulary.
	var gone []int
	l.seg.Forward(row, func(term, tf int) { gone = append(gone, term) })
	l.kill(row)
	for _, t := range gone {
		if e.termRefs(string(l.seg.TermKey(t))) == 0 {
			e.fstDirty = true
			break
		}
	}
	if e.pendingDels == nil {
		e.pendingDels = make(map[uint64]struct{})
	}
	e.pendingDels[id] = struct{}{}
	delete(e.attrs, id)
	e.annDeleteLocked(id)
	return true
}

// termRefs is the live occurrence count of term across the mutable delta and
// every segment (0 = the term is not in the index).
func (e *Engine) termRefs(term string) int {
	n := e.inverted.GetGlobalSeen()[term]
	for _, l := range e.segs {
		if t := l.seg.FindTerm(term); t >= 0 {
			n += l.seg.TermCF(t) - l.deadTermCF[t]
		}
	}
	return n
}

// eachFragDoc calls fn with the internal ID of every live document indexed
// under the edge-n-gram fragment frag.
func (e *Engine) eachFragDoc(frag string, fn func(id uint64)) {
	for _, id := range e.inverted.GetData()[frag] {
		fn(id)
	}
	for _, l := range e.segs {
		if fi := l.seg.FindFrag(frag); fi >= 0 {
			l.seg.FragPostings(fi, func(row int) bool {
				if !l.isDead(row) {
					fn(l.seg.DocID(row))
				}
				return true
			})
		}
	}
}

// eachPhonDoc is eachFragDoc for phonetic codes.
func (e *Engine) eachPhonDoc(code string, fn func(id uint64)) {
	for _, id := range e.phonetics.GetData()[code] {
		fn(id)
	}
	for _, l := range e.segs {
		if pi := l.seg.FindPhon(code); pi >= 0 {
			l.seg.PhonPostings(pi, func(row int) bool {
				if !l.isDead(row) {
					fn(l.seg.DocID(row))
				}
				return true
			})
		}
	}
}

// eachVector calls fn for every live document that has a vector.
func (e *Engine) eachVector(fn func(id uint64, v []uint16)) {
	for id, ent := range e.vectors.GetVectors() {
		fn(id, ent.Vector)
	}
	for _, l := range e.segs {
		for row, n := 0, l.seg.NumDocs(); row < n; row++ {
			if l.isDead(row) {
				continue
			}
			if v := l.seg.Vec(row); v != nil {
				fn(l.seg.DocID(row), v)
			}
		}
	}
}

// vectorCount is the number of live documents that have a vector.
func (e *Engine) vectorCount() int {
	n := len(e.vectors.GetVectors())
	for _, l := range e.segs {
		if l.nDead == 0 {
			n += l.seg.NumVecs()
			continue
		}
		for row, m := 0, l.seg.NumDocs(); row < m; row++ {
			if !l.isDead(row) && l.seg.Vec(row) != nil {
				n++
			}
		}
	}
	return n
}

// vecOf returns the stored float16 vector of a live document (nil if none).
func (e *Engine) vecOf(id uint64) []uint16 {
	if ent, ok := e.vectors.GetVectors()[id]; ok {
		return ent.Vector
	}
	if li, row, ok := e.locate(id); ok {
		return e.segs[li].seg.Vec(row)
	}
	return nil
}

// hasWordVector reports whether a word vector exists in the delta or any segment.
func (e *Engine) hasWordVector(word string) bool {
	if e.vectors.HasWordVector(word) {
		return true
	}
	for _, l := range e.segs {
		if l.seg.FindWord(word) >= 0 {
			return true
		}
	}
	return false
}

// wordVec returns a word's float16 vector from the delta or a segment.
func (e *Engine) wordVec(word string) ([]uint16, bool) {
	if ent, ok := e.vectors.GetWordVectors()[word]; ok {
		return ent.Vector, true
	}
	for _, l := range e.segs {
		if i := l.seg.FindWord(word); i >= 0 {
			return l.seg.WordVec(i), true
		}
	}
	return nil, false
}

// eachWordVector calls fn for every stored word vector.
func (e *Engine) eachWordVector(fn func(word string, v []uint16)) {
	for w, ent := range e.vectors.GetWordVectors() {
		fn(w, ent.Vector)
	}
	for _, l := range e.segs {
		for i, n := 0, l.seg.NumWords(); i < n; i++ {
			fn(l.seg.Word(i), l.seg.WordVec(i))
		}
	}
}

// liveTerms returns the sorted set of terms with at least one live occurrence.
func (e *Engine) liveTerms() []string {
	if len(e.segs) == 0 {
		glob := e.inverted.GetGlobalSeen()
		terms := make([]string, 0, len(glob))
		for t := range glob {
			terms = append(terms, t)
		}
		return terms
	}
	refs := make(map[string]int, len(e.inverted.GetGlobalSeen()))
	for t, n := range e.inverted.GetGlobalSeen() {
		refs[t] = n
	}
	for _, l := range e.segs {
		for t, n := 0, l.seg.NumTerms(); t < n; t++ {
			if c := l.seg.TermCF(t) - l.deadTermCF[t]; c != 0 {
				refs[string(l.seg.TermKey(t))] += c
			}
		}
	}
	terms := make([]string, 0, len(refs))
	for t, n := range refs {
		if n > 0 {
			terms = append(terms, t)
		}
	}
	sort.Strings(terms)
	return terms
}

// eachDocTerms calls fn for every live document with a membership test over
// the document's distinct terms (used by Explain).
func (e *Engine) eachDocTerms(fn func(id uint64, has func(term string) bool)) {
	for id, toks := range e.inverted.GetDocTokens() {
		set := make(map[string]struct{}, len(toks))
		for _, t := range toks {
			set[t] = struct{}{}
		}
		fn(id, func(t string) bool { _, ok := set[t]; return ok })
	}
	for _, l := range e.segs {
		for row, n := 0, l.seg.NumDocs(); row < n; row++ {
			if l.isDead(row) {
				continue
			}
			set := make(map[string]struct{})
			l.seg.Forward(row, func(term, _ int) { set[string(l.seg.TermKey(term))] = struct{}{} })
			fn(l.seg.DocID(row), func(t string) bool { _, ok := set[t]; return ok })
		}
	}
}

// sampleVectorIDs returns up to n live document IDs that have vectors, spread
// evenly over each segment so a filter's selectivity estimate is not biased
// by insertion order.
func (e *Engine) sampleVectorIDs(n int) []uint64 {
	ids := make([]uint64, 0, n)
	for id := range e.vectors.GetVectors() {
		if len(ids) >= n/2 {
			break
		}
		ids = append(ids, id)
	}
	for _, l := range e.segs {
		rows := l.seg.NumDocs()
		if rows == 0 {
			continue
		}
		want := max(1, (n-len(ids))/max(1, len(e.segs)))
		step := max(1, rows/want)
		for row := 0; row < rows && len(ids) < n; row += step {
			if !l.isDead(row) && l.seg.Vec(row) != nil {
				ids = append(ids, l.seg.DocID(row))
			}
		}
	}
	return ids
}

// ---- BM25 backing ----

// segBacking exposes the segments to the BM25 scorer.
type segBacking struct{ e *Engine }

func (b segBacking) Totals() (docs, totalLen int) {
	for _, l := range b.e.segs {
		docs += l.liveDocs()
		totalLen += l.totalLen - l.deadLen
	}
	return docs, totalLen
}

func (b segBacking) DocFreq(term string) int {
	df := 0
	for _, l := range b.e.segs {
		if t := l.seg.FindTerm(term); t >= 0 {
			df += l.seg.TermDF(t) - l.deadTermDF[t]
		}
	}
	return df
}

func (b segBacking) EachPosting(term string, fn func(docID uint64, tf, docLen int)) {
	for _, l := range b.e.segs {
		t := l.seg.FindTerm(term)
		if t < 0 {
			continue
		}
		l.seg.TermPostings(t, func(row, tf int) bool {
			if !l.isDead(row) {
				fn(l.seg.DocID(row), tf, l.seg.DocLen(row))
			}
			return true
		})
	}
}

func (b segBacking) TermFreq(docID uint64, term string) (tf, docLen int, ok bool) {
	li, row, found := b.e.locate(docID)
	if !found {
		return 0, 0, false
	}
	l := b.e.segs[li]
	t := l.seg.FindTerm(term)
	if t < 0 {
		return 0, l.seg.DocLen(row), true
	}
	l.seg.Forward(row, func(term2, f int) {
		if term2 == t {
			tf = f
		}
	})
	return tf, l.seg.DocLen(row), true
}

// fragCount is an upper bound on how many documents are indexed under the
// fragment: exact for the delta, and including not-yet-compacted deleted rows
// for segments. It is only used to decide whether a fragment is too common to
// generate candidates from, where an upper bound is what is wanted.
func (e *Engine) fragCount(frag string) int {
	n := len(e.inverted.GetData()[frag])
	for _, l := range e.segs {
		if fi := l.seg.FindFrag(frag); fi >= 0 {
			n += l.seg.FragCount(fi)
		}
	}
	return n
}

// prefixCap resolves Config.PrefixFragmentCap: the largest posting list a
// short prefix fragment may have and still generate candidates. <= 0 means no cap.
func (e *Engine) prefixCap() int {
	c := e.config.PrefixFragmentCap
	if c < 0 {
		return 0
	}
	if c > 0 {
		return c
	}
	// Automatic: a prefix shared by more than 2% of the corpus carries almost no
	// information; keep a floor so small corpora are searched exhaustively.
	return max(2000, e.docCountLocked()/50)
}
