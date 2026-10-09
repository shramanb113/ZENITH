package index

import (
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// A checkpoint must not stall searches. The mutable delta is therefore retired
// in two steps instead of being written out under the engine lock:
//
//  1. freeze (O(1) under Engine.mu): the delta's sub-indexes become a read-only
//     memLayer (Engine.frozen) that searches keep reading, and fresh empty
//     sub-indexes take over writes;
//  2. flush (no engine lock): the frozen layer is written as a segment, opened,
//     and swapped in for the frozen layer under Engine.mu in O(1) plus the
//     manifest commit.
//
// A frozen layer is never mutated except for its dead set: replacing or
// removing one of its documents marks it dead (and remembers the ID in the
// active layer's pendingDels, so the next flush persists the deletion), exactly
// as for a segment row. When the frozen layer becomes a segment, the rows whose
// IDs are dead are killed in the new segLayer, so nothing observable changes at
// the swap. If writing fails the frozen layer simply stays and the next Save
// writes it again.

// memLayer is an in-memory layer of documents: the same data the engine's own
// sub-indexes hold for the delta.
type memLayer struct {
	inverted  *InvertedIndex
	vectors   *VectorStore
	phonetics *PhoneticIndex
	bm25      *ranking.BM25Scorer
	idMapping map[uint64]string
	docText   map[uint64]string
	attrs     map[uint64]Attrs    // attributes to write (a snapshot when frozen)
	dels      map[uint64]struct{} // documents of older segments this layer deleted

	// Frozen layers only.
	dead    map[uint64]struct{} // documents deleted or replaced since the freeze
	deadLen int
	deadDF  map[string]int
	deadCF  map[string]int
}

// activeMemLocked wraps the engine's mutable delta as a memLayer (no copying).
func (e *Engine) activeMemLocked() *memLayer {
	return &memLayer{
		inverted: e.inverted, vectors: e.vectors, phonetics: e.phonetics, bm25: e.bm25,
		idMapping: e.idMapping, docText: e.docText, attrs: e.attrs, dels: e.pendingDels,
	}
}

func (m *memLayer) live(id uint64) bool {
	if _, ok := m.idMapping[id]; !ok {
		return false
	}
	_, gone := m.dead[id]
	return !gone
}

func (m *memLayer) liveDocs() int { return len(m.idMapping) - len(m.dead) }

// kill marks a live document dead and returns the terms it contained.
func (m *memLayer) kill(id uint64) []string {
	if m.dead == nil {
		m.dead = make(map[uint64]struct{})
		m.deadDF = make(map[string]int)
		m.deadCF = make(map[string]int)
	}
	m.dead[id] = struct{}{}
	tf, dl, _ := m.bm25.LocalTermFreqs(id)
	m.deadLen += dl
	terms := make([]string, 0, len(tf))
	for t, f := range tf {
		m.deadDF[t]++
		m.deadCF[t] += f
		terms = append(terms, t)
	}
	return terms
}

// freezeLocked retires the delta into e.frozen and installs empty sub-indexes.
// Engine.mu held for writing; e.frozen must be nil.
func (e *Engine) freezeLocked() {
	f := &memLayer{
		inverted: e.inverted, vectors: e.vectors, phonetics: e.phonetics, bm25: e.bm25,
		idMapping: e.idMapping, docText: e.docText, dels: e.pendingDels,
		attrs: make(map[uint64]Attrs),
	}
	for id := range f.idMapping {
		if a, ok := e.attrs[id]; ok {
			f.attrs[id] = a // Attrs values are replaced, never edited in place
		}
	}
	e.frozen = f

	e.inverted = NewInvertedIndex()
	e.vectors = NewVectorStore()
	e.phonetics = NewPhoneticIndex()
	e.bm25 = ranking.NewBM25Scorer(ranking.BM25Params{})
	e.bm25.SetBacking(segBacking{e})
	e.idMapping = make(map[uint64]string)
	e.docText = make(map[uint64]string)
	e.pendingDels = nil
}
