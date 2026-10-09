package index

import (
	"errors"
	"fmt"
	"math"
	"reflect"
)

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

// AttrValueFromAny converts a loosely-typed Go value — what encoding/json
// produces, or what a library caller passes — into an AttrValue: a string, a
// bool, any integer or float type (stored as float64), or a slice/array of
// those (never nested). nil, maps, non-finite numbers and nested slices are
// errors rather than silently dropped values.
func AttrValueFromAny(v any) (AttrValue, error) {
	switch x := v.(type) {
	case string:
		return AttrValue{Kind: AttrString, S: x}, nil
	case bool:
		n := 0.0
		if x {
			n = 1
		}
		return AttrValue{Kind: AttrBool, N: n}, nil
	case nil:
		return AttrValue{}, errors.New("null is not a supported attribute value")
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return AttrValue{Kind: AttrNumber, N: float64(rv.Int())}, nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return AttrValue{Kind: AttrNumber, N: float64(rv.Uint())}, nil
	case reflect.Float32, reflect.Float64:
		f := rv.Float()
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return AttrValue{}, errors.New("non-finite number")
		}
		return AttrValue{Kind: AttrNumber, N: f}, nil
	case reflect.Slice, reflect.Array:
		n := rv.Len()
		arr := make([]AttrValue, n)
		for i := 0; i < n; i++ {
			elem, err := AttrValueFromAny(rv.Index(i).Interface())
			if err != nil {
				return AttrValue{}, fmt.Errorf("[%d]: %w", i, err)
			}
			if elem.Kind == AttrArray {
				return AttrValue{}, fmt.Errorf("[%d]: nested arrays are not supported", i)
			}
			arr[i] = elem
		}
		return AttrValue{Kind: AttrArray, Arr: arr}, nil
	}
	return AttrValue{}, fmt.Errorf("unsupported attribute value type %T", v)
}

// Any is the inverse of AttrValueFromAny: a string, float64, bool, or []any
// of those (nil for a zero AttrValue). The result marshals to natural JSON.
func (v AttrValue) Any() any {
	switch v.Kind {
	case AttrString:
		return v.S
	case AttrNumber:
		return v.N
	case AttrBool:
		return v.N != 0
	case AttrArray:
		out := make([]any, len(v.Arr))
		for i, e := range v.Arr {
			out[i] = e.Any()
		}
		return out
	}
	return nil
}

// AttrsToAny converts a, value by value, with AttrValue.Any (nil for none).
func AttrsToAny(a Attrs) map[string]any {
	if len(a) == 0 {
		return nil
	}
	out := make(map[string]any, len(a))
	for k, v := range a {
		out[k] = v.Any()
	}
	return out
}

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
