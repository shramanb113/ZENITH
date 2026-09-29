package index

import (
	"math"
	"sort"
)

// attrIndex is an inverted index over document attributes: for every
// (field, value) the documents carrying it. A filter such as tenant = "acme"
// then finds its documents directly instead of testing every document's
// attributes — which is what makes a selective filter cheap on a large index
// (the alternatives are a graph search that must wander far to find matches, or
// an exact scan over everything).
//
// Documents get dense ordinals in the order their attributes were set, so each
// posting list is an ascending []uint32 built by appending. Replacing or
// removing a document only clears its ordinal's live bit; the dead entries stay
// in the postings until the index is rebuilt from the engine's attribute map
// (when the dead outnumber the live). Only documents that have attributes are
// indexed: a filter that a document without attributes can satisfy (Not) is
// answered by the ordinary predicate scan.
//
// Every method needs Engine.mu (read for queries, write for updates).
type attrIndex struct {
	ord  map[uint64]uint32 // live doc -> ordinal
	ids  []uint64          // ordinal -> doc
	live []uint64          // bitset over ordinals
	dead int               // ordinals cleared but still present in postings

	eq  map[string]map[attrKey][]uint32
	all map[string][]uint32 // field -> documents that have it
}

type attrKey struct {
	kind AttrKind
	s    string
	n    float64
}

func keyOf(v AttrValue) attrKey { return attrKey{v.Kind, v.S, v.N} }

func newAttrIndex() *attrIndex {
	return &attrIndex{
		ord: make(map[uint64]uint32),
		eq:  make(map[string]map[attrKey][]uint32),
		all: make(map[string][]uint32),
	}
}

func (x *attrIndex) isLive(o uint32) bool { return x.live[o>>6]&(1<<(o&63)) != 0 }

// set indexes id's attributes, replacing any previous ones.
func (x *attrIndex) set(id uint64, a Attrs) {
	x.drop(id)
	if len(a) == 0 {
		return
	}
	o := uint32(len(x.ids))
	x.ids = append(x.ids, id)
	if int(o>>6) >= len(x.live) {
		x.live = append(x.live, 0)
	}
	x.live[o>>6] |= 1 << (o & 63)
	x.ord[id] = o
	for f, v := range a {
		m := x.eq[f]
		if m == nil {
			m = make(map[attrKey][]uint32)
			x.eq[f] = m
		}
		k := keyOf(v)
		m[k] = append(m[k], o)
		x.all[f] = append(x.all[f], o)
	}
}

// drop removes id from the index.
func (x *attrIndex) drop(id uint64) {
	o, ok := x.ord[id]
	if !ok {
		return
	}
	delete(x.ord, id)
	x.live[o>>6] &^= 1 << (o & 63)
	x.dead++
}

// needsRebuild reports whether dead postings have grown past the live ones.
func (x *attrIndex) needsRebuild() bool { return x.dead > 10_000 && x.dead > len(x.ord) }

// rebuildAttrIndex builds a fresh index from the engine's attribute map (deterministic order).
func rebuildAttrIndex(attrs map[uint64]Attrs) *attrIndex {
	ids := make([]uint64, 0, len(attrs))
	for id := range attrs {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	x := newAttrIndex()
	for _, id := range ids {
		x.set(id, attrs[id])
	}
	return x
}

// rangeScanMax bounds how many distinct values of one field a range condition
// may scan; beyond it (a timestamp-like field) the index declines and the
// predicate scan is used.
const rangeScanMax = 8192

// estimate is a cheap upper bound on how many documents satisfy s, or ok=false
// when the index cannot say (a condition it does not cover). It counts dead
// entries, so it can only overestimate.
func (x *attrIndex) estimate(s *FilterSpec) (n int, ok bool) {
	switch s.Op {
	case "eq":
		return len(x.eq[s.Field][keyOf(s.Value.AttrValue)]), true
	case "in":
		for _, v := range s.Values {
			n += len(x.eq[s.Field][keyOf(v.AttrValue)])
		}
		return n, true
	case "exists":
		return len(x.all[s.Field]), true
	case "range":
		m := x.eq[s.Field]
		if len(m) > rangeScanMax {
			return 0, false
		}
		lo, hi := bounds(s)
		for k, list := range m {
			if k.kind == AttrNumber && k.n >= lo && k.n <= hi {
				n += len(list)
			}
		}
		return n, true
	case "none":
		return 0, true
	case "and":
		best, any := 0, false
		for i := range s.Args {
			if c, ok := x.estimate(&s.Args[i]); ok && (!any || c < best) {
				best, any = c, true
			}
		}
		if !any && len(s.Args) > 0 {
			return 0, false
		}
		return best, any
	case "or":
		for i := range s.Args {
			c, ok := x.estimate(&s.Args[i])
			if !ok {
				return 0, false
			}
			n += c
		}
		return n, true
	}
	return 0, false // not
}

func bounds(s *FilterSpec) (lo, hi float64) {
	lo, hi = math.Inf(-1), math.Inf(1)
	if s.Min != nil {
		lo = *s.Min
	}
	if s.Max != nil {
		hi = *s.Max
	}
	return lo, hi
}

// candidates returns the ordinals (ascending, live only) that may satisfy s.
// exact reports that every one of them does; when false the caller must still
// apply the predicate (an `and` of a covered and an uncovered condition). ok is
// false when the index cannot narrow s at all.
func (x *attrIndex) candidates(s *FilterSpec) (ords []uint32, exact, ok bool) {
	liveOnly := func(list []uint32) []uint32 {
		out := make([]uint32, 0, len(list))
		for _, o := range list {
			if x.isLive(o) {
				out = append(out, o)
			}
		}
		return out
	}
	switch s.Op {
	case "eq":
		return liveOnly(x.eq[s.Field][keyOf(s.Value.AttrValue)]), true, true
	case "in":
		var acc []uint32
		for _, v := range s.Values {
			acc = union(acc, liveOnly(x.eq[s.Field][keyOf(v.AttrValue)]))
		}
		return acc, true, true
	case "exists":
		return liveOnly(x.all[s.Field]), true, true
	case "range":
		m := x.eq[s.Field]
		if len(m) > rangeScanMax {
			return nil, false, false
		}
		lo, hi := bounds(s)
		var acc []uint32
		for k, list := range m {
			if k.kind == AttrNumber && k.n >= lo && k.n <= hi {
				acc = union(acc, liveOnly(list))
			}
		}
		return acc, true, true
	case "none":
		return nil, true, true
	case "and":
		var acc []uint32
		have, allExact := false, true
		for i := range s.Args {
			c, ex, ok := x.candidates(&s.Args[i])
			if !ok {
				allExact = false
				continue
			}
			if !ex {
				allExact = false
			}
			if !have {
				acc, have = c, true
			} else {
				acc = intersect(acc, c)
			}
			if have && len(acc) == 0 {
				break
			}
		}
		if !have {
			return nil, false, false
		}
		return acc, allExact, true
	case "or":
		var acc []uint32
		allExact := true
		for i := range s.Args {
			c, ex, ok := x.candidates(&s.Args[i])
			if !ok {
				return nil, false, false
			}
			if !ex {
				allExact = false
			}
			acc = union(acc, c)
		}
		return acc, allExact, true
	}
	return nil, false, false // not
}

func union(a, b []uint32) []uint32 {
	if len(a) == 0 {
		return b
	}
	if len(b) == 0 {
		return a
	}
	out := make([]uint32, 0, len(a)+len(b))
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		switch {
		case a[i] < b[j]:
			out = append(out, a[i])
			i++
		case a[i] > b[j]:
			out = append(out, b[j])
			j++
		default:
			out = append(out, a[i])
			i++
			j++
		}
	}
	out = append(out, a[i:]...)
	return append(out, b[j:]...)
}

func intersect(a, b []uint32) []uint32 {
	out := make([]uint32, 0, min(len(a), len(b)))
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		switch {
		case a[i] < b[j]:
			i++
		case a[i] > b[j]:
			j++
		default:
			out = append(out, a[i])
			i++
			j++
		}
	}
	return out
}

// docIDs maps ordinals back to document IDs.
func (x *attrIndex) docIDs(ords []uint32) []uint64 {
	out := make([]uint64, len(ords))
	for i, o := range ords {
		out[i] = x.ids[o]
	}
	return out
}

// setAttrsLocked records a document's attributes (nil clears them) in the map
// and the index. Engine.mu held for writing.
func (e *Engine) setAttrsLocked(id uint64, a Attrs) {
	if a == nil {
		delete(e.attrs, id)
	} else {
		e.attrs[id] = a
	}
	e.attrIdx.set(id, a)
	e.rebuildAttrIdxIfNeededLocked()
}

// dropAttrsLocked forgets a document's attributes.
func (e *Engine) dropAttrsLocked(id uint64) {
	delete(e.attrs, id)
	e.attrIdx.drop(id)
	e.rebuildAttrIdxIfNeededLocked()
}

func (e *Engine) rebuildAttrIdxIfNeededLocked() {
	if e.attrIdx.needsRebuild() {
		e.attrIdx = rebuildAttrIndex(e.attrs)
	}
}
