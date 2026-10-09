package zenith

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/shramanb113/ZENITH/internal/index"
)

// Attrs is the metadata attached to a document via AddWithAttrs. Values must
// be a string, bool, or any integer/float type (stored as float64).
type Attrs map[string]any

// ErrInvalidAttrs is returned for an empty attribute key or a value of an
// unsupported type.
var ErrInvalidAttrs = errors.New("zenith: invalid attributes: keys must be non-empty and values a string, bool or number")

// toAttrValue is index.AttrValueFromAny with the error wrapped in
// ErrInvalidAttrs (the conversion itself lives in internal/index so that the
// CLI's JSONL input and this package share one implementation).
func toAttrValue(v any) (index.AttrValue, error) {
	av, err := index.AttrValueFromAny(v)
	if err != nil {
		return index.AttrValue{}, fmt.Errorf("%w: %v", ErrInvalidAttrs, err)
	}
	return av, nil
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

// anyMatchZ mirrors internal/index/filterspec.go's anyMatch: this package
// builds its own predicates directly (not by calling index.FilterSpec.
// predicate()), so it needs its own copy of the same "any element matches"
// rule for an array-valued attribute. attrEqual replaces a direct == on
// index.AttrValue, which became incomparable once it gained an Arr field.
func anyMatchZ(v index.AttrValue, cmp func(index.AttrValue) bool) bool {
	if v.Kind == index.AttrArray {
		for _, e := range v.Arr {
			if cmp(e) {
				return true
			}
		}
		return false
	}
	return cmp(v)
}

func attrEqual(a, b index.AttrValue) bool { return a.Kind == b.Kind && a.S == b.S && a.N == b.N }

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
			return ok && anyMatchZ(got, func(e index.AttrValue) bool { return attrEqual(e, want) })
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
	cmp := func(e index.AttrValue) bool { return e.Kind == index.AttrNumber && e.N >= min && e.N <= max }
	pred := func(a index.Attrs) bool {
		got, ok := a[key]
		return ok && anyMatchZ(got, cmp)
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

// Prefix matches documents whose string attribute key starts with prefix.
// Non-string attributes never match.
func Prefix(key, prefix string) Filter {
	cmp := func(e index.AttrValue) bool { return e.Kind == index.AttrString && strings.HasPrefix(e.S, prefix) }
	return Filter{
		pred: func(a index.Attrs) bool { got, ok := a[key]; return ok && anyMatchZ(got, cmp) },
		spec: &index.FilterSpec{Op: "prefix", Field: key, Value: &index.SpecValue{AttrValue: index.AttrValue{Kind: index.AttrString, S: prefix}}},
	}
}

// Contains matches documents whose string attribute key contains substr.
// Non-string attributes never match. Unlike Prefix, this can't be answered
// from the attribute index and always falls back to a predicate scan.
func Contains(key, substr string) Filter {
	cmp := func(e index.AttrValue) bool { return e.Kind == index.AttrString && strings.Contains(e.S, substr) }
	return Filter{
		pred: func(a index.Attrs) bool { got, ok := a[key]; return ok && anyMatchZ(got, cmp) },
		spec: &index.FilterSpec{Op: "contains", Field: key, Value: &index.SpecValue{AttrValue: index.AttrValue{Kind: index.AttrString, S: substr}}},
	}
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
