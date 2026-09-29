package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/shramanb113/ZENITH/internal/index"
)

// Attribute and filter flags shared by `zenith index --attr` and
// `zenith search --where / --filter`.
//
//	--attr  tenant=acme --attr year=2024 --attr public=true
//	--where tenant=acme --where year>=2020 --where lang!=fr
//	--filter '{"op":"or","args":[...]}'      (full expression; see index.FilterSpec)
//
// A value that reads as true/false is a bool, one that parses as a number is a
// number, anything else is a string. Quote a value ("2024") to force a string.

func parseScalar(s string) index.AttrValue {
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		if u, err := strconv.Unquote(s); err == nil {
			return index.AttrValue{Kind: index.AttrString, S: u}
		}
		return index.AttrValue{Kind: index.AttrString, S: s[1 : len(s)-1]}
	}
	switch s {
	case "true":
		return index.AttrValue{Kind: index.AttrBool, N: 1}
	case "false":
		return index.AttrValue{Kind: index.AttrBool}
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil && !math.IsNaN(f) && !math.IsInf(f, 0) {
		return index.AttrValue{Kind: index.AttrNumber, N: f}
	}
	return index.AttrValue{Kind: index.AttrString, S: s}
}

// parseAttrs turns repeated key=value flags into document attributes.
func parseAttrs(flags []string) (index.Attrs, error) {
	if len(flags) == 0 {
		return nil, nil
	}
	out := make(index.Attrs, len(flags))
	for _, f := range flags {
		k, v, ok := strings.Cut(f, "=")
		k = strings.TrimSpace(k)
		if !ok || k == "" {
			return nil, fmt.Errorf("--attr %q: want key=value", f)
		}
		out[k] = parseScalar(strings.TrimSpace(v))
	}
	return out, nil
}

// whereSpec parses one --where condition: key=value, key!=value, key>=n, key<=n.
func whereSpec(w string) (index.FilterSpec, error) {
	for _, op := range []string{"!=", ">=", "<=", "="} {
		k, v, ok := strings.Cut(w, op)
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		if k == "" || v == "" {
			break
		}
		val := parseScalar(v)
		switch op {
		case "=":
			return index.FilterSpec{Op: "eq", Field: k, Value: &index.SpecValue{AttrValue: val}}, nil
		case "!=":
			return index.FilterSpec{Op: "not", Args: []index.FilterSpec{
				{Op: "eq", Field: k, Value: &index.SpecValue{AttrValue: val}}}}, nil
		default:
			if val.Kind != index.AttrNumber {
				return index.FilterSpec{}, fmt.Errorf("--where %q: %s needs a number", w, op)
			}
			n := val.N
			if op == ">=" {
				return index.FilterSpec{Op: "range", Field: k, Min: &n}, nil
			}
			return index.FilterSpec{Op: "range", Field: k, Max: &n}, nil
		}
	}
	return index.FilterSpec{}, fmt.Errorf("--where %q: want key=value, key!=value, key>=n or key<=n", w)
}

// buildFilter combines --where conditions and an optional --filter JSON
// expression (all of them must hold). nil means no restriction.
func buildFilter(where []string, filterJSON string) (*index.Filter, error) {
	var parts []index.FilterSpec
	for _, w := range where {
		s, err := whereSpec(w)
		if err != nil {
			return nil, err
		}
		parts = append(parts, s)
	}
	if strings.TrimSpace(filterJSON) != "" {
		s, err := index.ParseFilterSpec([]byte(filterJSON))
		if err != nil {
			return nil, fmt.Errorf("--filter: %w", err)
		}
		parts = append(parts, *s)
	}
	switch len(parts) {
	case 0:
		return nil, nil
	case 1:
		return parts[0].Compile()
	}
	all := index.FilterSpec{Op: "and", Args: parts}
	return all.Compile()
}
