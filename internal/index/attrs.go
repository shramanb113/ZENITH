package index

// AttrKind discriminates the type held by an AttrValue.
type AttrKind uint8

const (
	AttrString AttrKind = iota + 1
	AttrNumber
	AttrBool
	// AttrArray holds multiple values in Arr — never nested (an element of
	// Arr is never itself AttrArray). eq/in/range/prefix/contains on an
	// array-valued field mean "any element matches" (see anyMatch).
	AttrArray
)

// AttrValue is a single typed metadata value attached to a document. It is a
// concrete struct (not `any`) so it round-trips through encoding/gob without
// type registration. Numbers are stored as float64; bools as N=1/0.
//
// Adding Arr makes AttrValue incomparable (Go structs containing a slice
// can't use == or be a map key) — every place that used to compare two
// AttrValues directly now goes through keyOf (internal/index/attrindex.go),
// which already existed as a separate comparable key for the attribute
// index, or anyMatch (internal/index/filterspec.go) for predicate matching.
type AttrValue struct {
	Kind AttrKind
	S    string
	N    float64
	Arr  []AttrValue // only populated when Kind == AttrArray
}

// Attrs is the metadata attached to one document.
type Attrs map[string]AttrValue

// Predicate decides whether a document with the given attributes (nil if it
// has none) may appear in results.
type Predicate func(Attrs) bool

func copyAttrs(a Attrs) Attrs {
	if len(a) == 0 {
		return nil
	}
	out := make(Attrs, len(a))
	for k, v := range a {
		out[k] = v
	}
	return out
}

// filterCandidates drops every candidate whose attributes fail pred. Applied
// to the lexical and vector score maps *before* fusion so RRF ranks are
// computed over the filtered set only, instead of fetching a fixed window
// and discarding non-matches afterwards (which can return fewer than the
// requested limit even when enough matches exist).
func (e *Engine) filterCandidates(pred Predicate, scores map[uint64]float64) {
	if pred == nil {
		return
	}
	for id := range scores {
		if !pred(e.attrs[id]) {
			delete(scores, id)
		}
	}
}
