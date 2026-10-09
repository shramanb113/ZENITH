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

// parseAttrs turns repeated key=value flags into document attributes. A value
// written as [a,b,c] is an array attribute (AttrArray, the same thing
// `zenith txn add` accepts as a JSON array): each element is typed like a
// scalar value, a double-quoted element may contain commas ("a,b"), and []
// is an empty array. Quote the whole value ("[a]") to force a string.
func parseAttrs(flags []string) (index.Attrs, error) {
	if len(flags) == 0 {
		return nil, nil
	}
	out := make(index.Attrs, len(flags))
	for _, f := range flags {
		k, v, ok := strings.Cut(f, "=")
		k = strings.TrimSpace(k)
		if !ok || k == "" {
			return nil, fmt.Errorf("--attr %q: want key=value or key=[a,b,c]", f)
		}
		v = strings.TrimSpace(v)
		if len(v) >= 2 && v[0] == '[' && v[len(v)-1] == ']' {
			elems, err := splitArrayElems(v[1 : len(v)-1])
			if err != nil {
				return nil, fmt.Errorf("--attr %q: %w", f, err)
			}
			arr := make([]index.AttrValue, len(elems))
			for i, e := range elems {
				arr[i] = parseScalar(e)
			}
			out[k] = index.AttrValue{Kind: index.AttrArray, Arr: arr}
			continue
		}
		out[k] = parseScalar(v)
	}
	return out, nil
}

// splitArrayElems splits the inside of [a, "b,c", d] on commas outside double
// quotes, trimming each element (quotes are kept for parseScalar to strip).
// An empty or all-space body is an empty array; an empty element (a,,b) is an
// error rather than a silently dropped value.
func splitArrayElems(body string) ([]string, error) {
	if strings.TrimSpace(body) == "" {
		return []string{}, nil
	}
	var elems []string
	start, inQuote := 0, false
	for i := 0; i < len(body); i++ {
		switch body[i] {
		case '\\':
			if inQuote {
				i++ // skip the escaped character
			}
		case '"':
			inQuote = !inQuote
		case ',':
			if !inQuote {
				elems = append(elems, strings.TrimSpace(body[start:i]))
				start = i + 1
			}
		}
	}
	if inQuote {
		return nil, fmt.Errorf("unterminated quote in array value")
	}
	elems = append(elems, strings.TrimSpace(body[start:]))
	for _, e := range elems {
		if e == "" {
			return nil, fmt.Errorf("empty array element")
		}
	}
	return elems, nil
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
