// Package ann implements an approximate nearest-neighbour index (HNSW,
// Malkov & Yashunin 2018) over L2-normalised document vectors, so vector
// search stays sub-linear as the corpus grows instead of scanning every
// document per query.
//
// The index does not own vector data: each node keeps only a slice header
// pointing at the float16 vector the engine already stores (a heap slice, or a
// view into a memory-mapped segment), so ANN adds only the graph (links) on
// top of the existing memory footprint. Slices handed to Insert must stay
// valid and unmodified for as long as the index is used — drop the index
// before unmapping a segment. Similarity is the dot product, which equals
// cosine similarity for unit vectors.
//
// An Index is NOT safe for concurrent mutation. The engine serialises
// Insert/Delete under its write lock and runs Search under its read lock;
// Search allocates its own scratch so concurrent searches are safe.
package ann

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/x448/float16"
)

var (
	lutOnce sync.Once
	lut     [65536]float32
)

func initLUT() {
	for i := range lut {
		lut[i] = float16.Frombits(uint16(i)).Float32()
	}
}

// DotF32F16 returns dot(q, v) where v is stored as float16 bits, without
// allocating a float32 copy of v. The arithmetic (float64 products, same
// 4-way grouping) deliberately mirrors ranking.DotProduct so scores are
// bit-identical to decoding v first and calling that.
func DotF32F16(q []float32, v []uint16) float64 {
	lutOnce.Do(initLUT)
	if len(q) != len(v) || len(q) == 0 {
		return 0
	}
	var dot float64
	limit := len(q) - len(q)%4
	for i := 0; i < limit; i += 4 {
		dot += float64(q[i])*float64(lut[v[i]]) +
			float64(q[i+1])*float64(lut[v[i+1]]) +
			float64(q[i+2])*float64(lut[v[i+2]]) +
			float64(q[i+3])*float64(lut[v[i+3]])
	}
	for i := limit; i < len(q); i++ {
		dot += float64(q[i]) * float64(lut[v[i]])
	}
	return dot
}

func dotF16F16(a, b []uint16) float64 {
	lutOnce.Do(initLUT)
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var s float32
	for i := range a {
		s += lut[a[i]] * lut[b[i]]
	}
	return float64(s)
}

// Hit is one search result: a document ID and its similarity to the query.
type Hit struct {
	ID    uint64
	Score float64
}

// Index is an HNSW graph.
type Index struct {
	m, mmax0, efConstruction int
	ml                       float64
	rng                      *rand.Rand

	ids      []uint64
	vecs     [][]uint16 // vecs[node] is the stored vector; not owned
	idx      map[uint64]int32
	links    [][][]int32 // links[node][layer] -> neighbour node indexes
	dead     []bool
	live     int
	entry    int32
	maxLevel int
}

// New creates an empty index. m is the max links per node above layer 0
// (2*m at layer 0); efConstruction trades build time for graph quality.
func New(m, efConstruction int) *Index {
	if m < 4 {
		m = 16
	}
	if efConstruction < m {
		efConstruction = 100
	}
	return &Index{
		m: m, mmax0: 2 * m, efConstruction: efConstruction,
		ml:    1 / math.Log(float64(m)),
		rng:   rand.New(rand.NewSource(1)), // fixed seed: deterministic graphs for a given insert order
		idx:   make(map[uint64]int32),
		entry: -1,
	}
}

// Len is the number of live (non-deleted) vectors.
func (x *Index) Len() int { return x.live }

// NeedsRebuild reports whether tombstones have grown large enough that the
// graph should be rebuilt from the live vectors.
func (x *Index) NeedsRebuild() bool {
	return len(x.dead)-x.live > 1000 && (len(x.dead)-x.live)*3 > x.live
}

func (x *Index) randomLevel() int {
	return int(-math.Log(1-x.rng.Float64()) * x.ml)
}

// Insert adds a document. q is v decoded to float32; v is retained (not
// copied). Re-inserting an existing ID replaces it.
func (x *Index) Insert(id uint64, q []float32, v []uint16) {
	if _, ok := x.idx[id]; ok {
		x.Delete(id)
	}
	lvl := x.randomLevel()
	n := int32(len(x.ids))
	x.ids = append(x.ids, id)
	x.vecs = append(x.vecs, v)
	x.dead = append(x.dead, false)
	x.links = append(x.links, make([][]int32, lvl+1))
	x.idx[id] = n
	x.live++

	if x.entry < 0 {
		x.entry, x.maxLevel = n, lvl
		return
	}

	ep := x.entry
	for l := x.maxLevel; l > lvl; l-- {
		ep = x.greedy(q, ep, l)
	}
	eps := []int32{ep}
	for l := min(lvl, x.maxLevel); l >= 0; l-- {
		cands := x.searchLayer(q, eps, x.efConstruction, l, nil)
		maxLinks := x.m
		if l == 0 {
			maxLinks = x.mmax0
		}
		chosen := x.selectNeighbours(cands, x.m)
		x.links[n][l] = chosen
		for _, nb := range chosen {
			x.links[nb][l] = append(x.links[nb][l], n)
			if len(x.links[nb][l]) > maxLinks {
				x.links[nb][l] = x.shrink(nb, l, maxLinks)
			}
		}
		eps = eps[:0]
		for _, c := range cands {
			eps = append(eps, c.node)
		}
	}
	if lvl > x.maxLevel {
		x.entry, x.maxLevel = n, lvl
	}
}

// Delete tombstones a document: it stays in the graph as a routing node but
// is never returned.
func (x *Index) Delete(id uint64) {
	n, ok := x.idx[id]
	if !ok {
		return
	}
	delete(x.idx, id)
	x.dead[n] = true
	x.live--
	// The node stays as a routing point, but the vector it points at may live
	// in a memory-mapped segment that is about to be unmapped; own a copy.
	x.vecs[n] = append([]uint16(nil), x.vecs[n]...)
}

// RebindSome is Rebind for just the given IDs — for a flush, which moves only
// the flushed documents' vectors (every other node still points into an
// unchanged mapping), so the cost is O(delta) rather than O(index). IDs not in
// the graph are ignored.
func (x *Index) RebindSome(ids []uint64, f func(id uint64) []uint16) error {
	fresh := make(map[int32][]uint16, len(ids))
	for _, id := range ids {
		n, ok := x.idx[id]
		if !ok {
			continue
		}
		v := f(id)
		if v == nil {
			return fmt.Errorf("ann: rebind: no vector for id %d", id)
		}
		fresh[n] = v
	}
	for n, v := range fresh {
		x.vecs[n] = v
	}
	return nil
}

// Rebind repoints every live node at the vector f returns for its ID, after
// the engine moved vectors (flush into a new segment, compaction). It returns
// an error, leaving the index unchanged for nodes already visited, if f has no
// vector for a live ID; the caller should rebuild in that case.
func (x *Index) Rebind(f func(id uint64) []uint16) error {
	fresh := make(map[int32][]uint16, len(x.idx))
	for id, n := range x.idx {
		v := f(id)
		if v == nil {
			return fmt.Errorf("ann: rebind: no vector for id %d", id)
		}
		fresh[n] = v
	}
	for n, v := range fresh {
		x.vecs[n] = v
	}
	return nil
}

func (x *Index) nodeVec(n int32) []uint16 { return x.vecs[n] }

func (x *Index) simQ(q []float32, n int32) float64 {
	v := x.nodeVec(n)
	if v == nil {
		return math.Inf(-1)
	}
	return DotF32F16(q, v)
}

// greedy descends one layer by hill-climbing toward q.
func (x *Index) greedy(q []float32, ep int32, layer int) int32 {
	best, bestS := ep, x.simQ(q, ep)
	for improved := true; improved; {
		improved = false
		for _, nb := range x.links[best][layer] {
			if s := x.simQ(q, nb); s > bestS {
				best, bestS, improved = nb, s, true
			}
		}
	}
	return best
}

type cand struct {
	node int32
	sim  float64
}

// maxHeap pops the most similar first; minHeap pops the least similar first.
type maxHeap []cand

func (h maxHeap) Len() int           { return len(h) }
func (h maxHeap) Less(i, j int) bool { return h[i].sim > h[j].sim }
func (h maxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)        { *h = append(*h, x.(cand)) }
func (h *maxHeap) Pop() any {
	old := *h
	n := len(old)
	v := old[n-1]
	*h = old[:n-1]
	return v
}

type minHeap []cand

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].sim < h[j].sim }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(cand)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	v := old[n-1]
	*h = old[:n-1]
	return v
}

// searchLayer is the standard best-first beam search. Tombstoned nodes are
// traversed but never enter the result set; nodes rejected by allow are
// traversed but not returned. Construction therefore never links to deleted
// nodes either.
func (x *Index) searchLayer(q []float32, eps []int32, ef, layer int, allow func(uint64) bool) []cand {
	visited := make([]uint64, (len(x.ids)+63)/64)
	mark := func(n int32) bool {
		w, b := n>>6, uint(n)&63
		if visited[w]&(1<<b) != 0 {
			return false
		}
		visited[w] |= 1 << b
		return true
	}
	accept := func(n int32) bool {
		if x.dead[n] {
			return false
		}
		return allow == nil || allow(x.ids[n])
	}

	var cands maxHeap
	var results minHeap
	for _, e := range eps {
		if !mark(e) {
			continue
		}
		c := cand{e, x.simQ(q, e)}
		heap.Push(&cands, c)
		if accept(e) {
			heap.Push(&results, c)
		}
	}
	for cands.Len() > 0 {
		c := heap.Pop(&cands).(cand)
		if results.Len() >= ef && c.sim < results[0].sim {
			break
		}
		if layer >= len(x.links[c.node]) {
			continue
		}
		for _, nb := range x.links[c.node][layer] {
			if !mark(nb) {
				continue
			}
			s := x.simQ(q, nb)
			if results.Len() < ef || s > results[0].sim {
				heap.Push(&cands, cand{nb, s})
				if accept(nb) {
					heap.Push(&results, cand{nb, s})
					if results.Len() > ef {
						heap.Pop(&results)
					}
				}
			}
		}
	}
	out := make([]cand, len(results))
	copy(out, results)
	sort.Slice(out, func(i, j int) bool { return out[i].sim > out[j].sim })
	return out
}

// selectNeighbours applies the HNSW diversity heuristic: keep a candidate
// only if it is closer to the base point than to any already-kept neighbour,
// which preserves links across clusters instead of packing them into one.
// cands must be sorted best-first.
func (x *Index) selectNeighbours(cands []cand, m int) []int32 {
	out := make([]int32, 0, m)
	var kept [][]uint16
	for _, c := range cands {
		if len(out) >= m {
			break
		}
		cv := x.nodeVec(c.node)
		if cv == nil {
			continue
		}
		good := true
		for _, kv := range kept {
			if dotF16F16(cv, kv) > c.sim {
				good = false
				break
			}
		}
		if good {
			out = append(out, c.node)
			kept = append(kept, cv)
		}
	}
	// Top up with the best remaining so nodes never end up under-connected.
	if len(out) < m {
		have := make(map[int32]bool, len(out))
		for _, n := range out {
			have[n] = true
		}
		for _, c := range cands {
			if len(out) >= m {
				break
			}
			if !have[c.node] && x.nodeVec(c.node) != nil {
				out = append(out, c.node)
			}
		}
	}
	return out
}

// shrink re-selects node n's links at layer down to maxLinks.
func (x *Index) shrink(n int32, layer, maxLinks int) []int32 {
	base := x.nodeVec(n)
	if base == nil {
		return x.links[n][layer][:maxLinks]
	}
	cs := make([]cand, 0, len(x.links[n][layer]))
	for _, nb := range x.links[n][layer] {
		if v := x.nodeVec(nb); v != nil {
			cs = append(cs, cand{nb, dotF16F16(base, v)})
		}
	}
	sort.Slice(cs, func(i, j int) bool { return cs[i].sim > cs[j].sim })
	return x.selectNeighbours(cs, maxLinks)
}

// Search returns up to k live documents most similar to q, best first.
// allow (may be nil) restricts results; ef (>= k) is the beam width, the
// recall/latency knob.
func (x *Index) Search(q []float32, k, ef int, allow func(uint64) bool) []Hit {
	if x.entry < 0 || x.live == 0 || k <= 0 {
		return nil
	}
	if ef < k {
		ef = k
	}
	ep := x.entry
	for l := x.maxLevel; l > 0; l-- {
		ep = x.greedy(q, ep, l)
	}
	cs := x.searchLayer(q, []int32{ep}, ef, 0, allow)
	if len(cs) > k {
		cs = cs[:k]
	}
	out := make([]Hit, len(cs))
	for i, c := range cs {
		out[i] = Hit{ID: x.ids[c.node], Score: c.sim}
	}
	return out
}
