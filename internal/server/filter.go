package server

import (
	"errors"
	"fmt"
	"math"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
	"github.com/shramanb113/ZENITH/internal/index"
)

// attrsFromProto converts request metadata into engine attributes, rejecting
// what the engine cannot store (empty keys, non-finite numbers, untyped values).
func attrsFromProto(m map[string]*zenithproto.AttrValue) (index.Attrs, error) {
	if len(m) == 0 {
		return nil, nil
	}
	out := make(index.Attrs, len(m))
	for k, v := range m {
		if k == "" {
			return nil, errors.New("attribute keys must not be empty")
		}
		av, err := attrValueFromProto(v)
		if err != nil {
			return nil, fmt.Errorf("attribute %q: %w", k, err)
		}
		out[k] = av
	}
	return out, nil
}

func attrValueFromProto(v *zenithproto.AttrValue) (index.AttrValue, error) {
	switch x := v.GetKind().(type) {
	case *zenithproto.AttrValue_StringValue:
		return index.AttrValue{Kind: index.AttrString, S: x.StringValue}, nil
	case *zenithproto.AttrValue_BoolValue:
		n := 0.0
		if x.BoolValue {
			n = 1
		}
		return index.AttrValue{Kind: index.AttrBool, N: n}, nil
	case *zenithproto.AttrValue_NumberValue:
		if math.IsNaN(x.NumberValue) || math.IsInf(x.NumberValue, 0) {
			return index.AttrValue{}, errors.New("numbers must be finite")
		}
		return index.AttrValue{Kind: index.AttrNumber, N: x.NumberValue}, nil
	case *zenithproto.AttrValue_ArrayValue:
		vals := x.ArrayValue.GetValues()
		arr := make([]index.AttrValue, len(vals))
		for i, elem := range vals {
			av, err := attrValueFromProto(elem)
			if err != nil {
				return index.AttrValue{}, err
			}
			if av.Kind == index.AttrArray {
				return index.AttrValue{}, errors.New("nested arrays are not supported")
			}
			arr[i] = av
		}
		return index.AttrValue{Kind: index.AttrArray, Arr: arr}, nil
	}
	return index.AttrValue{}, errors.New("value must be a string, number, bool or array")
}

func attrsToProto(a index.Attrs) map[string]*zenithproto.AttrValue {
	if len(a) == 0 {
		return nil
	}
	out := make(map[string]*zenithproto.AttrValue, len(a))
	for k, v := range a {
		if pv := attrValueToProto(v); pv != nil {
			out[k] = pv
		}
	}
	return out
}

func attrValueToProto(v index.AttrValue) *zenithproto.AttrValue {
	switch v.Kind {
	case index.AttrString:
		return &zenithproto.AttrValue{Kind: &zenithproto.AttrValue_StringValue{StringValue: v.S}}
	case index.AttrBool:
		return &zenithproto.AttrValue{Kind: &zenithproto.AttrValue_BoolValue{BoolValue: v.N != 0}}
	case index.AttrNumber:
		return &zenithproto.AttrValue{Kind: &zenithproto.AttrValue_NumberValue{NumberValue: v.N}}
	case index.AttrArray:
		elems := make([]*zenithproto.AttrValue, 0, len(v.Arr))
		for _, e := range v.Arr {
			if pv := attrValueToProto(e); pv != nil {
				elems = append(elems, pv)
			}
		}
		return &zenithproto.AttrValue{Kind: &zenithproto.AttrValue_ArrayValue{ArrayValue: &zenithproto.AttrValueArray{Values: elems}}}
	}
	return nil
}

// filterFromProto converts a request's filter tree into an engine filter. The
// tree arrives from the network, so depth and size are bounded before anything
// is compiled.
func filterFromProto(n *zenithproto.FilterNode) (*index.Filter, error) {
	nodes := 0
	spec, err := specFromProto(n, 0, &nodes)
	if err != nil {
		return nil, err
	}
	return spec.Compile()
}

func specFromProto(n *zenithproto.FilterNode, depth int, nodes *int) (*index.FilterSpec, error) {
	*nodes++
	if depth > index.MaxFilterDepth {
		return nil, fmt.Errorf("filter nested deeper than %d", index.MaxFilterDepth)
	}
	if *nodes > index.MaxFilterNodes {
		return nil, fmt.Errorf("filter has more than %d nodes", index.MaxFilterNodes)
	}
	group := func(op string, g *zenithproto.FilterGroup) (*index.FilterSpec, error) {
		s := &index.FilterSpec{Op: op}
		for _, c := range g.GetNodes() {
			cs, err := specFromProto(c, depth+1, nodes)
			if err != nil {
				return nil, err
			}
			s.Args = append(s.Args, *cs)
		}
		return s, nil
	}
	switch x := n.GetNode().(type) {
	case *zenithproto.FilterNode_And:
		return group("and", x.And)
	case *zenithproto.FilterNode_Or:
		return group("or", x.Or)
	case *zenithproto.FilterNode_Not:
		c, err := specFromProto(x.Not, depth+1, nodes)
		if err != nil {
			return nil, err
		}
		return &index.FilterSpec{Op: "not", Args: []index.FilterSpec{*c}}, nil
	case *zenithproto.FilterNode_Condition:
		c := x.Condition
		vals := make([]index.SpecValue, 0, len(c.GetValues()))
		for _, v := range c.GetValues() {
			av, err := attrValueFromProto(v)
			if err != nil {
				return nil, fmt.Errorf("filter on %q: %w", c.GetField(), err)
			}
			vals = append(vals, index.SpecValue{AttrValue: av})
		}
		s := &index.FilterSpec{Field: c.GetField()}
		switch c.GetOp() {
		case zenithproto.FilterCondition_EQ:
			if len(vals) != 1 {
				return nil, fmt.Errorf("filter EQ on %q needs exactly one value", c.GetField())
			}
			s.Op, s.Value = "eq", &vals[0]
		case zenithproto.FilterCondition_IN:
			s.Op, s.Values = "in", vals
		case zenithproto.FilterCondition_RANGE:
			s.Op = "range"
			if c.Min != nil {
				v := c.GetMin()
				s.Min = &v
			}
			if c.Max != nil {
				v := c.GetMax()
				s.Max = &v
			}
		case zenithproto.FilterCondition_EXISTS:
			s.Op = "exists"
		case zenithproto.FilterCondition_PREFIX:
			if len(vals) != 1 {
				return nil, fmt.Errorf("filter PREFIX on %q needs exactly one value", c.GetField())
			}
			s.Op, s.Value = "prefix", &vals[0]
		case zenithproto.FilterCondition_CONTAINS:
			if len(vals) != 1 {
				return nil, fmt.Errorf("filter CONTAINS on %q needs exactly one value", c.GetField())
			}
			s.Op, s.Value = "contains", &vals[0]
		default:
			return nil, fmt.Errorf("filter on %q has an unknown operator", c.GetField())
		}
		return s, nil
	}
	return nil, errors.New("filter node is empty")
}
