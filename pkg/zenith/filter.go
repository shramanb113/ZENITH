package zenith

import (
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
// Eq, In, Range, Exists and combine with And, Or, Not. A document with no
// value for a key never matches a condition on that key (except under Not).
type Filter struct {
	pred index.Predicate
}

// Eq matches documents whose attribute key equals value. Strings compare
// exactly; numbers compare as float64; bools as bools. A type mismatch
// (e.g. Eq("year", "2024") against a numeric attribute) does not match.
func Eq(key string, value any) Filter {
	want, err := toAttrValue(value)
	if err != nil {
		return Filter{pred: func(index.Attrs) bool { return false }}
	}
	return Filter{pred: func(a index.Attrs) bool {
		got, ok := a[key]
		return ok && got == want
	}}
}

// In matches documents whose attribute key equals any of values.
func In(key string, values ...any) Filter {
	fs := make([]Filter, len(values))
	for i, v := range values {
		fs[i] = Eq(key, v)
	}
	return Or(fs...)
}

// Range matches documents whose numeric attribute key lies in [min, max]
// (inclusive). Use math.Inf(-1) / math.Inf(1) for an open end. Non-numeric
// attributes never match.
func Range(key string, min, max float64) Filter {
	return Filter{pred: func(a index.Attrs) bool {
		got, ok := a[key]
		return ok && got.Kind == index.AttrNumber && got.N >= min && got.N <= max
	}}
}

// Exists matches documents that have any value for key.
func Exists(key string) Filter {
	return Filter{pred: func(a index.Attrs) bool {
		_, ok := a[key]
		return ok
	}}
}

// And matches when every filter matches (no filters: matches everything).
func And(fs ...Filter) Filter {
	return Filter{pred: func(a index.Attrs) bool {
		for _, f := range fs {
			if f.pred != nil && !f.pred(a) {
				return false
			}
		}
		return true
	}}
}

// Or matches when any filter matches (no filters: matches nothing).
func Or(fs ...Filter) Filter {
	return Filter{pred: func(a index.Attrs) bool {
		for _, f := range fs {
			if f.pred != nil && f.pred(a) {
				return true
			}
		}
		return false
	}}
}

// Not inverts a filter. A document lacking the attribute matches Not(Eq(...)).
func Not(f Filter) Filter {
	return Filter{pred: func(a index.Attrs) bool {
		if f.pred == nil {
			return true
		}
		return !f.pred(a)
	}}
}

// WithFilter restricts a Search to documents matching f. The filter is
// applied to the lexical and vector candidate sets before rank fusion, so
// results are the top matches among the filtered documents rather than a
// filtered view of the global top results. Cannot be combined with Explain.
func WithFilter(f Filter) SearchOption {
	return func(o *searchOptions) { o.filter = &f }
}

func (o *searchOptions) predicate() index.Predicate {
	if o.filter == nil {
		return nil
	}
	return o.filter.pred
}
