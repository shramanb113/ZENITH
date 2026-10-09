package main

import (
	"reflect"
	"testing"

	"github.com/shramanb113/ZENITH/internal/index"
)

func TestParseAttrs_ArraySyntax(t *testing.T) {
	got, err := parseAttrs([]string{
		`tags=[go, search ,"a,b", 3, true]`,
		"empty=[]",
		`quoted="[x]"`,
		"year=2024",
	})
	if err != nil {
		t.Fatal(err)
	}
	s := func(v string) index.AttrValue { return index.AttrValue{Kind: index.AttrString, S: v} }
	want := index.Attrs{
		"tags": {Kind: index.AttrArray, Arr: []index.AttrValue{
			s("go"), s("search"), s("a,b"),
			{Kind: index.AttrNumber, N: 3}, {Kind: index.AttrBool, N: 1},
		}},
		"empty":  {Kind: index.AttrArray, Arr: []index.AttrValue{}},
		"quoted": s("[x]"),
		"year":   {Kind: index.AttrNumber, N: 2024},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("parseAttrs = %#v\nwant %#v", got, want)
	}
	for _, bad := range []string{"k=[a,,b]", `k=["a]`, "k=[a,]"} {
		if _, err := parseAttrs([]string{bad}); err == nil {
			t.Fatalf("%s: want error", bad)
		}
	}
}

func TestPageOf(t *testing.T) {
	rs := []index.SearchResponse{{ID: "a"}, {ID: "b"}, {ID: "c"}}
	ids := func(r []index.SearchResponse) []string {
		var out []string
		for _, x := range r {
			out = append(out, x.ID)
		}
		return out
	}
	for _, tc := range []struct {
		offset, max int
		want        []string
	}{
		{0, 2, []string{"a", "b"}},
		{1, 10, []string{"b", "c"}},
		{2, 0, []string{"c"}},
		{3, 5, nil},
	} {
		if got := ids(pageOf(rs, tc.offset, tc.max)); !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("pageOf(offset=%d, max=%d) = %v, want %v", tc.offset, tc.max, got, tc.want)
		}
	}
}

func TestParseFacetFields(t *testing.T) {
	got, err := parseFacetFields([]string{"lang, tags", "lang", "year"})
	if err != nil || !reflect.DeepEqual(got, []string{"lang", "tags", "year"}) {
		t.Fatalf("parseFacetFields = %v, %v", got, err)
	}
	if _, err := parseFacetFields([]string{"a,,b"}); err == nil {
		t.Fatal("empty field name: want error")
	}
}
