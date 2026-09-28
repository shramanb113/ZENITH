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
	case len(e.vectors.vectors) >= n:
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

func (e *Engine) annVec(id uint64) []uint16 { return e.vectors.vectors[id].Vector }

func (e *Engine) rebuildANNLocked() {
	ids := make([]uint64, 0, len(e.vectors.vectors))
	for id := range e.vectors.vectors {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] }) // deterministic graph
	g := ann.New(annM, annEFConstruction, e.annVec)
	for _, id := range ids {
		g.Insert(id, Float16ToFloats(e.vectors.vectors[id].Vector))
	}
	e.ann = g
}

func (e *Engine) annInsertLocked(id uint64, vec []float32) {
	if e.ann == nil {
		if e.annMinDocs > 0 && len(e.vectors.vectors) >= e.annMinDocs {
			e.rebuildANNLocked() // already includes id: it is in the store
		}
		return
	}
	e.ann.Insert(id, vec)
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
func (e *Engine) annSearch(q []float32, pred Predicate, vecs map[uint64]VectorEntry) ([]ann.Hit, bool) {
	ef := annEF
	var allow func(uint64) bool
	if pred != nil {
		const sample = 256
		seen, ok := 0, 0
		for id := range vecs {
			if pred(e.attrs[id]) {
				ok++
			}
			if seen++; seen >= sample {
				break
			}
		}
		frac := float64(ok) / float64(seen)
		if frac < annMinSelectivity {
			return nil, false
		}
		ef = min(max(annEF, int(float64(annEF)/frac)), 2000)
		allow = func(id uint64) bool { return pred(e.attrs[id]) }
	}
	return e.ann.Search(q, annK, ef, allow), true
}
