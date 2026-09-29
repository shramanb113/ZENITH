package index

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
)

// A Filter is a predicate over a document's attributes, optionally with the
// structured description (Spec) it was built from. Search only needs the
// predicate; the spec lets the engine narrow the candidates with the attribute
// index instead of testing every document, and lets a filter travel over gRPC,
// HTTP and the command line.
type Filter struct {
	Pred Predicate
	Spec *FilterSpec
}

// FilterSpec is a serialisable filter expression:
//
//	{"op":"eq",     "field":"tenant", "value":"acme"}
//	{"op":"in",     "field":"lang",   "values":["en","hi"]}
//	{"op":"range",  "field":"year",   "min":2020, "max":2024}   (either bound optional)
//	{"op":"exists", "field":"tenant"}
//	{"op":"and",    "args":[ ... ]}   {"op":"or", "args":[ ... ]}
//	{"op":"not",    "args":[ { ... } ]}
//
// The meaning of each operator is that of the same-named function in package
// zenith (Eq, In, Range, Exists, And, Or, Not): values are strings, numbers
// (float64) or bools, a type mismatch never matches, a document without the
// field never matches a condition on it, and Not matches documents that lack it.
type FilterSpec struct {
	Op     string       `json:"op"`
	Field  string       `json:"field,omitempty"`
	Value  *SpecValue   `json:"value,omitempty"`
	Values []SpecValue  `json:"values,omitempty"`
	Min    *float64     `json:"min,omitempty"`
	Max    *float64     `json:"max,omitempty"`
	Args   []FilterSpec `json:"args,omitempty"`
}

// SpecValue is a typed attribute value in a spec: a JSON string, number or bool.
type SpecValue struct{ AttrValue }

func (v SpecValue) MarshalJSON() ([]byte, error) {
	switch v.Kind {
	case AttrString:
		return json.Marshal(v.S)
	case AttrBool:
		return json.Marshal(v.N != 0)
	case AttrNumber:
		return json.Marshal(v.N)
	}
	return nil, errors.New("filter: value has no type")
}

func (v *SpecValue) UnmarshalJSON(b []byte) error {
	var x any
	if err := json.Unmarshal(b, &x); err != nil {
		return err
	}
	switch t := x.(type) {
	case string:
		v.AttrValue = AttrValue{Kind: AttrString, S: t}
	case bool:
		n := 0.0
		if t {
			n = 1
		}
		v.AttrValue = AttrValue{Kind: AttrBool, N: n}
	case float64:
		if math.IsNaN(t) || math.IsInf(t, 0) {
			return errors.New("filter: non-finite number")
		}
		v.AttrValue = AttrValue{Kind: AttrNumber, N: t}
	default:
		return fmt.Errorf("filter: value must be a string, number or bool, not %T", x)
	}
	return nil
}

// Limits keep a filter that arrives over the network from being an attack: a
// deeply nested or enormous expression is rejected before it is compiled.
const (
	MaxFilterDepth = 16
	MaxFilterNodes = 512
)

// ErrInvalidFilter is returned for a malformed filter expression.
var ErrInvalidFilter = errors.New("index: invalid filter")

// ParseFilterSpec decodes and validates a JSON filter expression.
func ParseFilterSpec(data []byte) (*FilterSpec, error) {
	var s FilterSpec
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidFilter, err)
	}
	if err := s.Validate(); err != nil {
		return nil, err
	}
	return &s, nil
}

// Validate checks structure and limits.
func (s *FilterSpec) Validate() error {
	nodes := 0
	return s.validate(0, &nodes)
}

func (s *FilterSpec) validate(depth int, nodes *int) error {
	*nodes++
	if depth > MaxFilterDepth {
		return fmt.Errorf("%w: nested deeper than %d", ErrInvalidFilter, MaxFilterDepth)
	}
	if *nodes > MaxFilterNodes {
		return fmt.Errorf("%w: more than %d nodes", ErrInvalidFilter, MaxFilterNodes)
	}
	needField := func() error {
		if s.Field == "" {
			return fmt.Errorf("%w: %q needs a field", ErrInvalidFilter, s.Op)
		}
		return nil
	}
	switch s.Op {
	case "eq":
		if err := needField(); err != nil {
			return err
		}
		if s.Value == nil || s.Value.Kind == 0 {
			return fmt.Errorf("%w: eq on %q needs a value", ErrInvalidFilter, s.Field)
		}
	case "in":
		if err := needField(); err != nil {
			return err
		}
		for _, v := range s.Values {
			if v.Kind == 0 {
				return fmt.Errorf("%w: in on %q has an untyped value", ErrInvalidFilter, s.Field)
			}
		}
	case "range":
		if err := needField(); err != nil {
			return err
		}
		if s.Min != nil && s.Max != nil && *s.Min > *s.Max {
			return fmt.Errorf("%w: range on %q has min > max", ErrInvalidFilter, s.Field)
		}
	case "exists":
		if err := needField(); err != nil {
			return err
		}
	case "none":
		// matches nothing (what a condition with an unusable value degrades to)
	case "and", "or":
		for i := range s.Args {
			if err := s.Args[i].validate(depth+1, nodes); err != nil {
				return err
			}
		}
	case "not":
		if len(s.Args) != 1 {
			return fmt.Errorf("%w: not takes exactly one argument", ErrInvalidFilter)
		}
		return s.Args[0].validate(depth+1, nodes)
	default:
		return fmt.Errorf("%w: unknown op %q", ErrInvalidFilter, s.Op)
	}
	return nil
}

// Compile turns a validated spec into a Filter (predicate plus the spec).
func (s *FilterSpec) Compile() (*Filter, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	return &Filter{Pred: s.predicate(), Spec: s}, nil
}

func (s *FilterSpec) predicate() Predicate {
	switch s.Op {
	case "eq":
		field, want := s.Field, s.Value.AttrValue
		return func(a Attrs) bool { got, ok := a[field]; return ok && got == want }
	case "in":
		field := s.Field
		set := make(map[AttrValue]struct{}, len(s.Values))
		for _, v := range s.Values {
			set[v.AttrValue] = struct{}{}
		}
		return func(a Attrs) bool {
			got, ok := a[field]
			if !ok {
				return false
			}
			_, in := set[got]
			return in
		}
	case "range":
		field := s.Field
		lo, hi := math.Inf(-1), math.Inf(1)
		if s.Min != nil {
			lo = *s.Min
		}
		if s.Max != nil {
			hi = *s.Max
		}
		return func(a Attrs) bool {
			got, ok := a[field]
			return ok && got.Kind == AttrNumber && got.N >= lo && got.N <= hi
		}
	case "exists":
		field := s.Field
		return func(a Attrs) bool { _, ok := a[field]; return ok }
	case "none":
		return func(Attrs) bool { return false }
	case "and":
		ps := make([]Predicate, len(s.Args))
		for i := range s.Args {
			ps[i] = s.Args[i].predicate()
		}
		return func(a Attrs) bool {
			for _, p := range ps {
				if !p(a) {
					return false
				}
			}
			return true
		}
	case "or":
		ps := make([]Predicate, len(s.Args))
		for i := range s.Args {
			ps[i] = s.Args[i].predicate()
		}
		return func(a Attrs) bool {
			for _, p := range ps {
				if p(a) {
					return true
				}
			}
			return false
		}
	case "not":
		p := s.Args[0].predicate()
		return func(a Attrs) bool { return !p(a) }
	}
	return func(Attrs) bool { return false }
}

func (f *Filter) pred() Predicate {
	if f == nil {
		return nil
	}
	return f.Pred
}

func (f *Filter) spec() *FilterSpec {
	if f == nil {
		return nil
	}
	return f.Spec
}
