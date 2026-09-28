package index

import (
	"sort"

	"github.com/shramanb113/ZENITH/internal/ann"
)

const (
	// defaultANNMinDocs is the corpus size at which vector search switches
	// from an exact scan to the HNSW graph. Below it the scan is exact and
	// costs well under a millisecond, so the graph's build time and recall
	// loss would buy nothing.
	defaultANNMinDocs = 20_000

	annM              = 16
	annEFConstruction = 100
	// annK is how many nearest documents the graph returns to rank fusion.
	// RRF gives rank r a weight of about 1/(k+r), so documents beyond the top
	// few hundred contribute almost nothing; scanning the whole corpus for
	// them was the cost being removed.
	annK  = 150
	annEF = 200
	// annMinSelectivity: below this fraction of allowed documents a filtered
	// graph traversal has to wander too far to find matches, so an exact scan
	// over the (few) allowed documents is both faster and exact.
	annMinSelectivity = 0.10
)

// SetANNThreshold sets the corpus size at which vector search uses the ANN
// graph; n <= 0 disables ANN entirely (always exact). Builds or drops the
// graph immediately if the current corpus is on the other side of n.
func (e *Engine) SetANNThreshold(n int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.vectors.Lock()
	defer e.vectors.Unlock()
	e.annMinDocs = n
	switch {
	case n <= 0:
		e.ann = nil
	case e.vectorCount() >= n:
		e.rebuildANNLocked()
	default:
		e.ann = nil
	}
}

// ANNActive reports whether vector search is currently served by the graph.
func (e *Engine) ANNActive() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.ann != nil
}

// The *Locked helpers require Engine.mu and vectors.mu held for writing.

// rebuildANNLocked builds the graph from every live vector (delta and
// segments), in ID order so the graph is deterministic.
func (e *Engine) rebuildANNLocked() {
	type item struct {
		id uint64
		v  []uint16
	}
	items := make([]item, 0, e.vectorCount())
	e.eachVector(func(id uint64, v []uint16) { items = append(items, item{id, v}) })
	sort.Slice(items, func(i, j int) bool { return items[i].id < items[j].id })
	g := ann.New(annM, annEFConstruction)
	for _, it := range items {
		g.Insert(it.id, Float16ToFloats(it.v), it.v)
	}
	e.ann = g
}

// rebuildANNAfterLoadLocked builds the graph after a snapshot load if the
// corpus is large enough; the graph itself is never persisted.
func (e *Engine) rebuildANNAfterLoadLocked() {
	e.ann = nil
	if e.annMinDocs > 0 && e.vectorCount() >= e.annMinDocs {
		e.rebuildANNLocked()
	}
}

// rebindANNLocked repoints the graph at the vectors' new homes after a flush
// or compaction moved them. If any vector is missing the graph is rebuilt.
func (e *Engine) rebindANNLocked() {
	if e.ann == nil {
		return
	}
	if err := e.ann.Rebind(e.vecOf); err != nil {
		e.rebuildANNLocked()
	}
}

// rebindANNSomeLocked is rebindANNLocked for a flush: only the flushed
// documents' vectors moved, so only those nodes are re-pointed. Falls back to a
// rebuild if any is missing.
func (e *Engine) rebindANNSomeLocked(ids []uint64) {
	if e.ann == nil || len(ids) == 0 {
		return
	}
	if err := e.ann.RebindSome(ids, e.vecOf); err != nil {
		e.rebuildANNLocked()
	}
}

// annInsertLocked adds a document to the graph. vec is the decoded float32
// vector and bits the stored float16 form (retained by the graph).
func (e *Engine) annInsertLocked(id uint64, vec []float32, bits []uint16) {
	if e.ann == nil {
		if e.annMinDocs > 0 && e.vectorCount() >= e.annMinDocs {
			e.rebuildANNLocked() // already includes id: it is in the store
		}
		return
	}
	e.ann.Insert(id, vec, bits)
	if e.ann.NeedsRebuild() {
		e.rebuildANNLocked()
	}
}

func (e *Engine) annDeleteLocked(id uint64) {
	if e.ann != nil {
		e.ann.Delete(id)
	}
}

// annSearch returns the graph's nearest documents, or ok=false when an exact
// scan should be used instead (very selective filter).
func (e *Engine) annSearch(q []float32, pred Predicate) ([]ann.Hit, bool) {
	ef := annEF
	var allow func(uint64) bool
	if pred != nil {
		sample := e.sampleVectorIDs(256)
		ok := 0
		for _, id := range sample {
			if pred(e.attrs[id]) {
				ok++
			}
		}
		if len(sample) == 0 {
			return nil, false
		}
		frac := float64(ok) / float64(len(sample))
		if frac < annMinSelectivity {
			return nil, false
		}
		ef = min(max(annEF, int(float64(annEF)/frac)), 2000)
		allow = func(id uint64) bool { return pred(e.attrs[id]) }
	}
	return e.ann.Search(q, annK, ef, allow), true
}
