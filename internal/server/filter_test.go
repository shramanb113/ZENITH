package server_test

import (
	"context"
	"math"
	"testing"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/ranking"
	"github.com/shramanb113/ZENITH/internal/server"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func str(s string) *zenithproto.AttrValue {
	return &zenithproto.AttrValue{Kind: &zenithproto.AttrValue_StringValue{StringValue: s}}
}
func num(n float64) *zenithproto.AttrValue {
	return &zenithproto.AttrValue{Kind: &zenithproto.AttrValue_NumberValue{NumberValue: n}}
}
func boolean(b bool) *zenithproto.AttrValue {
	return &zenithproto.AttrValue{Kind: &zenithproto.AttrValue_BoolValue{BoolValue: b}}
}

func arr(vals ...*zenithproto.AttrValue) *zenithproto.AttrValue {
	return &zenithproto.AttrValue{Kind: &zenithproto.AttrValue_ArrayValue{ArrayValue: &zenithproto.AttrValueArray{Values: vals}}}
}

func cond(field string, op zenithproto.FilterCondition_Op, vals ...*zenithproto.AttrValue) *zenithproto.FilterNode {
	return &zenithproto.FilterNode{Node: &zenithproto.FilterNode_Condition{Condition: &zenithproto.FilterCondition{
		Field: field, Op: op, Values: vals}}}
}
func and(ns ...*zenithproto.FilterNode) *zenithproto.FilterNode {
	return &zenithproto.FilterNode{Node: &zenithproto.FilterNode_And{And: &zenithproto.FilterGroup{Nodes: ns}}}
}
func not(n *zenithproto.FilterNode) *zenithproto.FilterNode {
	return &zenithproto.FilterNode{Node: &zenithproto.FilterNode_Not{Not: n}}
}

func newSrv(t *testing.T) *server.ZenithServer {
	t.Helper()
	cfg := config.DefaultConfig()
	eng := index.NewEngine(cfg, embedding.NewDeterministicEmbedder(32), ranking.NewRRFRanker(0, 0), analysis.NewStandardAnalyzer())
	t.Cleanup(func() { eng.Close() })
	return &server.ZenithServer{Engine: eng}
}

func seed(t *testing.T, s *server.ZenithServer) {
	t.Helper()
	for _, d := range []struct {
		id    string
		attrs map[string]*zenithproto.AttrValue
	}{
		{"en2020", map[string]*zenithproto.AttrValue{"lang": str("en"), "year": num(2020), "public": boolean(true)}},
		{"en2024", map[string]*zenithproto.AttrValue{"lang": str("en"), "year": num(2024), "public": boolean(false)}},
		{"fr2022", map[string]*zenithproto.AttrValue{"lang": str("fr"), "year": num(2022), "public": boolean(true)}},
		{"plain", nil},
	} {
		if _, err := s.IndexDocuments(context.Background(), &zenithproto.IndexRequest{
			Id: d.id, Data: "kubernetes cluster networking guide", Attrs: d.attrs}); err != nil {
			t.Fatalf("IndexDocuments(%s): %v", d.id, err)
		}
	}
}

func idsOf(r *zenithproto.SearchResponse) map[string]bool {
	m := map[string]bool{}
	for _, x := range r.GetResults() {
		m[x.GetId()] = true
	}
	return m
}

func TestGRPC_SearchFilterAndAttrsRoundTrip(t *testing.T) {
	s := newSrv(t)
	seed(t, s)
	search := func(f *zenithproto.FilterNode) *zenithproto.SearchResponse {
		t.Helper()
		r, err := s.Search(context.Background(), &zenithproto.SearchRequest{Query: "kubernetes networking", Limit: 50, Filter: f})
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	if got := idsOf(search(nil)); len(got) != 4 {
		t.Fatalf("no filter: %v", got)
	}
	got := idsOf(search(cond("lang", zenithproto.FilterCondition_EQ, str("en"))))
	if len(got) != 2 || !got["en2020"] || !got["en2024"] {
		t.Fatalf("lang=en: %v", got)
	}
	got = idsOf(search(and(
		cond("lang", zenithproto.FilterCondition_IN, str("en"), str("fr")),
		cond("public", zenithproto.FilterCondition_EQ, boolean(true)))))
	if len(got) != 2 || !got["en2020"] || !got["fr2022"] {
		t.Fatalf("lang in (en,fr) and public: %v", got)
	}
	min := 2021.0
	rng := &zenithproto.FilterNode{Node: &zenithproto.FilterNode_Condition{Condition: &zenithproto.FilterCondition{
		Field: "year", Op: zenithproto.FilterCondition_RANGE, Min: &min}}}
	got = idsOf(search(rng))
	if len(got) != 2 || !got["en2024"] || !got["fr2022"] {
		t.Fatalf("year>=2021: %v", got)
	}
	got = idsOf(search(not(cond("lang", zenithproto.FilterCondition_EXISTS))))
	if len(got) != 1 || !got["plain"] {
		t.Fatalf("not exists(lang): %v", got)
	}

	// Attributes come back with the results.
	r := search(cond("lang", zenithproto.FilterCondition_EQ, str("fr")))
	if len(r.Results) != 1 {
		t.Fatalf("fr: %v", r.Results)
	}
	a := r.Results[0].GetAttrs()
	if a["lang"].GetStringValue() != "fr" || a["year"].GetNumberValue() != 2022 || !a["public"].GetBoolValue() {
		t.Fatalf("attrs did not round-trip: %v", a)
	}
}

// Array-valued attrs round-trip through the proto (AttrValue_ArrayValue) and
// eq/in/prefix/contains conditions match "any element" of the array, exactly
// as pkg/zenith's Filter does for the in-process path.
func TestGRPC_ArrayAttrsAndPrefixContainsRoundTrip(t *testing.T) {
	s := newSrv(t)
	for _, d := range []struct {
		id    string
		attrs map[string]*zenithproto.AttrValue
	}{
		{"a", map[string]*zenithproto.AttrValue{"tags": arr(str("go"), str("infra")), "path": str("/docs/guide")}},
		{"b", map[string]*zenithproto.AttrValue{"tags": arr(str("python"), str("infra")), "path": str("/docs/api")}},
		{"c", map[string]*zenithproto.AttrValue{"tags": arr(str("rust")), "path": str("/blog/release-notes")}},
	} {
		if _, err := s.IndexDocuments(context.Background(), &zenithproto.IndexRequest{
			Id: d.id, Data: "kubernetes cluster networking guide", Attrs: d.attrs}); err != nil {
			t.Fatalf("IndexDocuments(%s): %v", d.id, err)
		}
	}
	search := func(f *zenithproto.FilterNode) *zenithproto.SearchResponse {
		t.Helper()
		r, err := s.Search(context.Background(), &zenithproto.SearchRequest{Query: "kubernetes networking", Limit: 50, Filter: f})
		if err != nil {
			t.Fatal(err)
		}
		return r
	}

	got := idsOf(search(cond("tags", zenithproto.FilterCondition_EQ, str("infra"))))
	if len(got) != 2 || !got["a"] || !got["b"] {
		t.Fatalf("tags eq infra: %v", got)
	}
	got = idsOf(search(cond("path", zenithproto.FilterCondition_PREFIX, str("/docs/"))))
	if len(got) != 2 || !got["a"] || !got["b"] {
		t.Fatalf("path prefix /docs/: %v", got)
	}
	got = idsOf(search(cond("path", zenithproto.FilterCondition_CONTAINS, str("release"))))
	if len(got) != 1 || !got["c"] {
		t.Fatalf("path contains release: %v", got)
	}

	// Array attrs round-trip back out on the result too.
	r := search(cond("tags", zenithproto.FilterCondition_EQ, str("rust")))
	if len(r.Results) != 1 {
		t.Fatalf("rust: %v", r.Results)
	}
	gotTags := r.Results[0].GetAttrs()["tags"].GetArrayValue().GetValues()
	if len(gotTags) != 1 || gotTags[0].GetStringValue() != "rust" {
		t.Fatalf("tags did not round-trip: %v", gotTags)
	}
}

func TestGRPC_RejectsBadFiltersAndAttrs(t *testing.T) {
	s := newSrv(t)
	seed(t, s)
	deep := cond("lang", zenithproto.FilterCondition_EXISTS)
	for i := 0; i < 40; i++ {
		deep = not(deep)
	}
	for name, f := range map[string]*zenithproto.FilterNode{
		"empty node":       {},
		"eq without value": cond("lang", zenithproto.FilterCondition_EQ),
		"eq with two":      cond("lang", zenithproto.FilterCondition_EQ, str("a"), str("b")),
		"untyped value":    cond("lang", zenithproto.FilterCondition_EQ, &zenithproto.AttrValue{}),
		"too deep":         deep,
	} {
		_, err := s.Search(context.Background(), &zenithproto.SearchRequest{Query: "kubernetes", Filter: f})
		if status.Code(err) != codes.InvalidArgument {
			t.Errorf("%s: code %v (err=%v), want InvalidArgument", name, status.Code(err), err)
		}
	}
	for name, attrs := range map[string]map[string]*zenithproto.AttrValue{
		"empty key":     {"": str("x")},
		"untyped value": {"k": {}},
		"nan":           {"k": num(math.NaN())},
		"nested array":  {"k": arr(arr(str("a")))},
	} {
		_, err := s.IndexDocuments(context.Background(), &zenithproto.IndexRequest{Id: "x", Data: "text", Attrs: attrs})
		if status.Code(err) != codes.InvalidArgument {
			t.Errorf("%s: code %v, want InvalidArgument", name, status.Code(err))
		}
	}
}
