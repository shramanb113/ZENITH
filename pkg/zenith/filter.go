package zenith

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"

	"github.com/shramanb113/ZENITH/internal/index"
)

// Attrs is the metadata attached to a document via AddWithAttrs. Values must
// be a string, bool, or any integer/float type (stored as float64).
type Attrs map[string]any

// ErrInvalidAttrs is returned for an empty attribute key or a value of an
// unsupported type.
var ErrInvalidAttrs = errors.New("zenith: invalid attributes: keys must be non-empty and values a string, bool or number")

func toAttrValue(v any) (index.AttrValue, error) {
	switch x := v.(type) {
	case string:
		return index.AttrValue{Kind: index.AttrString, S: x}, nil
	case bool:
		n := 0.0
		if x {
			n = 1
		}
		return index.AttrValue{Kind: index.AttrBool, N: n}, nil
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return index.AttrValue{Kind: index.AttrNumber, N: float64(rv.Int())}, nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return index.AttrValue{Kind: index.AttrNumber, N: float64(rv.Uint())}, nil
	case reflect.Float32, reflect.Float64:
		f := rv.Float()
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return index.AttrValue{}, fmt.Errorf("%w: non-finite number", ErrInvalidAttrs)
		}
		return index.AttrValue{Kind: index.AttrNumber, N: f}, nil
	}
	return index.AttrValue{}, fmt.Errorf("%w: unsupported type %T", ErrInvalidAttrs, v)
}

func toIndexAttrs(a Attrs) (index.Attrs, error) {
	if len(a) == 0 {
		return nil, nil
	}
	out := make(index.Attrs, len(a))
	for k, v := range a {
		if k == "" {
			return nil, ErrInvalidAttrs
		}
		av, err := toAttrValue(v)
		if err != nil {
			return nil, err
		}
		out[k] = av
	}
	return out, nil
}

// Filter restricts Search to documents whose attributes match. Build one with
// Eq, In, Range, Exists and combine with And, Or, Not — or decode one with
// FilterFromJSON. A document with no value for a key never matches a condition
// on that key (except under Not).
//
// A Filter is data as well as a function: it can be encoded with JSON (the
// format is documented on index.FilterSpec) and, on a large index, the engine
// answers selective conditions from an attribute index instead of testing every
// document.
type Filter struct {
	pred index.Predicate
	spec *index.FilterSpec
}

func (f Filter) indexFilter() *index.Filter {
	if f.pred == nil {
		return nil
	}
	return &index.Filter{Pred: f.pred, Spec: f.spec}
}

// FilterFromJSON decodes a filter such as
//
//	{"op":"and","args":[
//	   {"op":"eq","field":"tenant","value":"acme"},
//	   {"op":"range","field":"year","min":2020}]}
//
// Operators: eq, in, range, exists, and, or, not — the same meaning as the
// functions of the same names. Malformed or oversized input is rejected.
func FilterFromJSON(data []byte) (Filter, error) {
	spec, err := index.ParseFilterSpec(data)
	if err != nil {
		return Filter{}, fmt.Errorf("zenith: %w", err)
	}
	f, err := spec.Compile()
	if err != nil {
		return Filter{}, fmt.Errorf("zenith: %w", err)
	}
	return Filter{pred: f.Pred, spec: f.Spec}, nil
}

// JSON encodes the filter (see FilterFromJSON). The zero Filter, which matches
// everything, encodes as {"op":"and"}.
func (f Filter) JSON() ([]byte, error) {
	if f.spec == nil {
		return json.Marshal(index.FilterSpec{Op: "and"})
	}
	return json.Marshal(f.spec)
}

func noneSpec() *index.FilterSpec { return &index.FilterSpec{Op: "none"} }

func specValue(v any) (index.SpecValue, bool) {
	av, err := toAttrValue(v)
	if err != nil {
		return index.SpecValue{}, false
	}
	return index.SpecValue{AttrValue: av}, true
}

// Eq matches documents whose attribute key equals value. Strings compare
// exactly; numbers compare as float64; bools as bools. A type mismatch
// (e.g. Eq("year", "2024") against a numeric attribute) does not match.
func Eq(key string, value any) Filter {
	sv, ok := specValue(value)
	if !ok {
		return Filter{pred: func(index.Attrs) bool { return false }, spec: noneSpec()}
	}
	want := sv.AttrValue
	return Filter{
		pred: func(a index.Attrs) bool {
			got, ok := a[key]
			return ok && got == want
		},
		spec: &index.FilterSpec{Op: "eq", Field: key, Value: &sv},
	}
}

// In matches documents whose attribute key equals any of values.
func In(key string, values ...any) Filter {
	spec := &index.FilterSpec{Op: "in", Field: key}
	fs := make([]Filter, 0, len(values))
	for _, v := range values {
		if sv, ok := specValue(v); ok {
			spec.Values = append(spec.Values, sv)
		}
		fs = append(fs, Eq(key, v))
	}
	f := Or(fs...)
	f.spec = spec
	return f
}

// Range matches documents whose numeric attribute key lies in [min, max]
// (inclusive). Use math.Inf(-1) / math.Inf(1) for an open end. Non-numeric
// attributes never match.
func Range(key string, min, max float64) Filter {
	pred := func(a index.Attrs) bool {
		got, ok := a[key]
		return ok && got.Kind == index.AttrNumber && got.N >= min && got.N <= max
	}
	if math.IsNaN(min) || math.IsNaN(max) {
		return Filter{pred: pred, spec: noneSpec()}
	}
	spec := &index.FilterSpec{Op: "range", Field: key}
	if !math.IsInf(min, -1) {
		spec.Min = &min
	}
	if !math.IsInf(max, 1) {
		spec.Max = &max
	}
	return Filter{pred: pred, spec: spec}
}

// Exists matches documents that have any value for key.
func Exists(key string) Filter {
	return Filter{
		pred: func(a index.Attrs) bool { _, ok := a[key]; return ok },
		spec: &index.FilterSpec{Op: "exists", Field: key},
	}
}

// And matches when every filter matches (no filters: matches everything).
func And(fs ...Filter) Filter {
	spec := &index.FilterSpec{Op: "and"}
	for _, f := range fs {
		if f.pred != nil {
			spec.Args = append(spec.Args, *f.spec)
		}
	}
	return Filter{
		pred: func(a index.Attrs) bool {
			for _, f := range fs {
				if f.pred != nil && !f.pred(a) {
					return false
				}
			}
			return true
		},
		spec: spec,
	}
}

// Or matches when any filter matches (no filters: matches nothing).
func Or(fs ...Filter) Filter {
	spec := &index.FilterSpec{Op: "or"}
	for _, f := range fs {
		if f.pred != nil {
			spec.Args = append(spec.Args, *f.spec)
		}
	}
	return Filter{
		pred: func(a index.Attrs) bool {
			for _, f := range fs {
				if f.pred != nil && f.pred(a) {
					return true
				}
			}
			return false
		},
		spec: spec,
	}
}

// Not inverts a filter. A document lacking the attribute matches Not(Eq(...)).
func Not(f Filter) Filter {
	if f.pred == nil {
		return Filter{pred: func(index.Attrs) bool { return true }, spec: &index.FilterSpec{Op: "and"}}
	}
	return Filter{
		pred: func(a index.Attrs) bool { return !f.pred(a) },
		spec: &index.FilterSpec{Op: "not", Args: []index.FilterSpec{*f.spec}},
	}
}

// WithFilter restricts a Search to documents matching f. The filter is
// applied to the lexical and vector candidate sets before rank fusion, so
// results are the top matches among the filtered documents rather than a
// filtered view of the global top results. It may be combined with Explain.
func WithFilter(f Filter) SearchOption {
	return func(o *searchOptions) { o.filter = &f }
}

func (o *searchOptions) predicate() index.Predicate {
	if o.filter == nil {
		return nil
	}
	return o.filter.pred
}

func (o *searchOptions) indexFilter() *index.Filter {
	if o.filter == nil {
		return nil
	}
	return o.filter.indexFilter()
}
