package server_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
	"github.com/shramanb113/ZENITH/internal/index"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func codeOf(err error) codes.Code { return status.Code(err) }

func f64(v float64) *float64 { return &v }

func TestGRPC_SearchWeights(t *testing.T) {
	s := newSrv(t)
	seed(t, s)
	ctx := context.Background()

	if _, err := s.Search(ctx, &zenithproto.SearchRequest{
		Query: "kubernetes", VectorWeight: f64(3), PhoneticWeight: f64(0.5), RrfK: f64(10),
	}); err != nil {
		t.Fatalf("valid weights: %v", err)
	}
	for name, req := range map[string]*zenithproto.SearchRequest{
		"NaN vector":        {Query: "kubernetes", VectorWeight: f64(math.NaN())},
		"Inf phonetic":      {Query: "kubernetes", PhoneticWeight: f64(math.Inf(1))},
		"negative rrf_k":    {Query: "kubernetes", RrfK: f64(-1)},
		"negative vector":   {Query: "kubernetes", VectorWeight: f64(-0.1)},
		"-Inf rrf_k":        {Query: "kubernetes", RrfK: f64(math.Inf(-1))},
		"negative phonetic": {Query: "kubernetes", PhoneticWeight: f64(-3)},
	} {
		if _, err := s.Search(ctx, req); codeOf(err) != codes.InvalidArgument {
			t.Fatalf("%s: code = %v (%v), want InvalidArgument", name, codeOf(err), err)
		}
	}
}

func facetsByField(r *zenithproto.SearchResponse) map[string]map[string]int64 {
	out := map[string]map[string]int64{}
	for _, f := range r.GetFacets() {
		m := out[f.GetField()]
		if m == nil {
			m = map[string]int64{}
			out[f.GetField()] = m
		}
		var k string
		switch v := f.GetValue().GetKind().(type) {
		case *zenithproto.AttrValue_StringValue:
			k = v.StringValue
		case *zenithproto.AttrValue_BoolValue:
			k = map[bool]string{true: "true", false: "false"}[v.BoolValue]
		case *zenithproto.AttrValue_NumberValue:
			k = "num"
		}
		m[k] += f.GetCount()
	}
	return out
}

func TestGRPC_SearchFacets(t *testing.T) {
	s := newSrv(t)
	seed(t, s)
	ctx := context.Background()

	// Facets cover every match even though limit is 1.
	resp, err := s.Search(ctx, &zenithproto.SearchRequest{
		Query: "kubernetes networking", Limit: 1, FacetFields: []string{"lang", "public"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.GetResults()) != 1 {
		t.Fatalf("limit 1 returned %d results", len(resp.GetResults()))
	}
	got := facetsByField(resp)
	if got["lang"]["en"] != 2 || got["lang"]["fr"] != 1 || got["public"]["true"] != 2 || got["public"]["false"] != 1 {
		t.Fatalf("facets = %v", got)
	}
	// Grouped by field in request order, highest count first.
	if f := resp.GetFacets(); f[0].GetField() != "lang" || f[0].GetValue().GetStringValue() != "en" {
		t.Fatalf("first facet = %v, want lang=en", f[0])
	}

	// Empty query + facet_fields: corpus-wide counts, no results.
	corpus, err := s.Search(ctx, &zenithproto.SearchRequest{FacetFields: []string{"lang"}, FacetTopK: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(corpus.GetResults()) != 0 || len(corpus.GetFacets()) != 1 || corpus.GetFacets()[0].GetCount() != 2 {
		t.Fatalf("corpus facets = %v / results %v", corpus.GetFacets(), corpus.GetResults())
	}

	for name, req := range map[string]*zenithproto.SearchRequest{
		"empty query, no facets":    {},
		"empty facet field":         {Query: "kubernetes", FacetFields: []string{""}},
		"negative top_k":            {Query: "kubernetes", FacetFields: []string{"lang"}, FacetTopK: -1},
		"corpus facets with filter": {FacetFields: []string{"lang"}, Filter: &zenithproto.FilterNode{}},
	} {
		if _, err := s.Search(ctx, req); codeOf(err) != codes.InvalidArgument {
			t.Fatalf("%s: code = %v (%v), want InvalidArgument", name, codeOf(err), err)
		}
	}
}

func TestGRPC_GetDocumentReturnsAttrs(t *testing.T) {
	s := newSrv(t)
	seed(t, s)
	resp, err := s.GetDocument(context.Background(), &zenithproto.GetDocumentRequest{Id: "fr2022"})
	if err != nil || !resp.GetFound() {
		t.Fatalf("GetDocument: %v %v", resp, err)
	}
	a := resp.GetAttrs()
	if a["lang"].GetStringValue() != "fr" || a["year"].GetNumberValue() != 2022 || !a["public"].GetBoolValue() {
		t.Fatalf("attrs = %v", a)
	}
	plain, _ := s.GetDocument(context.Background(), &zenithproto.GetDocumentRequest{Id: "plain"})
	if len(plain.GetAttrs()) != 0 {
		t.Fatalf("plain attrs = %v, want none", plain.GetAttrs())
	}
}

func TestGRPC_IndexPDF_NilIndexerIsUnimplemented(t *testing.T) {
	s := newSrv(t)
	_, err := s.IndexPDF(context.Background(), &zenithproto.IndexPDFRequest{DocumentId: "d", FilePath: "x.pdf"})
	if codeOf(err) != codes.Unimplemented {
		t.Fatalf("code = %v (%v), want Unimplemented", codeOf(err), err)
	}
}

func TestGRPC_IndexPDF_PassesAttrs(t *testing.T) {
	mock := &mockPDFIndexer{returnCount: 1}
	s := newSrv(t)
	s.PDFIndexer = mock
	if _, err := s.IndexPDF(context.Background(), &zenithproto.IndexPDFRequest{
		DocumentId: "d", FilePath: "x.pdf",
		Attrs: map[string]*zenithproto.AttrValue{"lang": str("en")},
	}); err != nil {
		t.Fatal(err)
	}
	if mock.lastAttrs["lang"].S != "en" {
		t.Fatalf("attrs passed = %v", mock.lastAttrs)
	}
	if _, err := s.IndexPDF(context.Background(), &zenithproto.IndexPDFRequest{
		DocumentId: "d", FilePath: "x.pdf",
		Attrs: map[string]*zenithproto.AttrValue{"bad": num(math.NaN())},
	}); codeOf(err) != codes.InvalidArgument {
		t.Fatalf("NaN attr: code = %v, want InvalidArgument", codeOf(err))
	}
}

// memTxn is an in-memory index.Txn: staged writes become visible in a
// committed map only on Commit, so a test can see all-or-nothing behavior.
type memTxn struct {
	store     map[string][]byte
	staged    map[string][]byte
	deleted   map[string]bool
	commitErr error
}

func (m *memTxn) Put(k, v []byte) error { m.staged[string(k)] = v; return nil }
func (m *memTxn) Delete(k []byte) error { m.deleted[string(k)] = true; return nil }
func (m *memTxn) Discard() error        { return nil }
func (m *memTxn) Commit(context.Context) error {
	if m.commitErr != nil {
		return m.commitErr
	}
	for k, v := range m.staged {
		m.store[k] = v
	}
	for k := range m.deleted {
		delete(m.store, k)
	}
	return nil
}

func TestGRPC_IndexBatchAndDeleteBatch(t *testing.T) {
	s := newSrv(t)
	ctx := context.Background()

	// No storage engine wired: Unimplemented, cleanly.
	if _, err := s.IndexBatch(ctx, &zenithproto.IndexBatchRequest{Docs: []*zenithproto.IndexRequest{{Id: "a", Data: "x"}}}); codeOf(err) != codes.Unimplemented {
		t.Fatalf("IndexBatch without NewTxn: code = %v", codeOf(err))
	}
	if _, err := s.DeleteBatch(ctx, &zenithproto.DeleteBatchRequest{Ids: []string{"a"}}); codeOf(err) != codes.Unimplemented {
		t.Fatalf("DeleteBatch without NewTxn: code = %v", codeOf(err))
	}

	store := map[string][]byte{}
	var commitErr error
	s.NewTxn = func() index.Txn {
		return &memTxn{store: store, staged: map[string][]byte{}, deleted: map[string]bool{}, commitErr: commitErr}
	}

	resp, err := s.IndexBatch(ctx, &zenithproto.IndexBatchRequest{Docs: []*zenithproto.IndexRequest{
		{Id: "a", Data: "kubernetes operators", Attrs: map[string]*zenithproto.AttrValue{"lang": str("en")}},
		{Id: "b", Data: "kubernetes ingress"},
	}})
	if err != nil || !resp.GetStatus() || resp.GetIndexed() != 2 {
		t.Fatalf("IndexBatch: %v %v", resp, err)
	}
	if len(store) != 2 {
		t.Fatalf("committed keys = %d, want 2", len(store))
	}
	if got, _ := s.GetDocument(ctx, &zenithproto.GetDocumentRequest{Id: "a"}); !got.GetFound() || got.GetAttrs()["lang"].GetStringValue() != "en" {
		t.Fatalf("doc a after batch: %v", got)
	}

	// One invalid document rejects the whole batch before anything is staged.
	for name, docs := range map[string][]*zenithproto.IndexRequest{
		"empty batch": nil,
		"empty id":    {{Id: "c", Data: "ok"}, {Id: "", Data: "x"}},
		"empty data":  {{Id: "c", Data: "ok"}, {Id: "d", Data: ""}},
		"bad attr":    {{Id: "c", Data: "ok", Attrs: map[string]*zenithproto.AttrValue{"": str("x")}}},
	} {
		if _, err := s.IndexBatch(ctx, &zenithproto.IndexBatchRequest{Docs: docs}); codeOf(err) != codes.InvalidArgument {
			t.Fatalf("%s: code = %v, want InvalidArgument", name, codeOf(err))
		}
	}
	if found, _ := s.GetDocument(ctx, &zenithproto.GetDocumentRequest{Id: "c"}); found.GetFound() {
		t.Fatal("doc c indexed although its batch was rejected")
	}

	// A failed commit applies nothing.
	commitErr = errors.New("disk on fire")
	if _, err := s.IndexBatch(ctx, &zenithproto.IndexBatchRequest{Docs: []*zenithproto.IndexRequest{{Id: "c", Data: "ok"}}}); codeOf(err) != codes.Internal {
		t.Fatalf("failed commit: code = %v, want Internal", codeOf(err))
	}
	if found, _ := s.GetDocument(ctx, &zenithproto.GetDocumentRequest{Id: "c"}); found.GetFound() {
		t.Fatal("doc c visible after a failed commit")
	}
	commitErr = nil

	if _, err := s.DeleteBatch(ctx, &zenithproto.DeleteBatchRequest{Ids: []string{"a", ""}}); codeOf(err) != codes.InvalidArgument {
		t.Fatalf("DeleteBatch with empty id: code = %v", codeOf(err))
	}
	del, err := s.DeleteBatch(ctx, &zenithproto.DeleteBatchRequest{Ids: []string{"a", "b", "never-existed"}})
	if err != nil || del.GetDeleted() != 3 {
		t.Fatalf("DeleteBatch: %v %v", del, err)
	}
	if len(store) != 0 {
		t.Fatalf("store after delete = %v, want empty", store)
	}
	for _, id := range []string{"a", "b"} {
		if got, _ := s.GetDocument(ctx, &zenithproto.GetDocumentRequest{Id: id}); got.GetFound() {
			t.Fatalf("%s still found after DeleteBatch", id)
		}
	}
}

func TestGRPC_Suggest(t *testing.T) {
	s := newSrv(t)
	seed(t, s)
	ctx := context.Background()
	resp, err := s.Suggest(ctx, &zenithproto.SuggestRequest{Prefix: "Kub"})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.GetTerms()) != 1 || resp.GetTerms()[0] != "kubernet" {
		t.Fatalf("Suggest(Kub) = %v, want [kubernet]", resp.GetTerms())
	}
	if _, err := s.Suggest(ctx, &zenithproto.SuggestRequest{}); codeOf(err) != codes.InvalidArgument {
		t.Fatalf("empty prefix: code = %v", codeOf(err))
	}
	if _, err := s.Suggest(ctx, &zenithproto.SuggestRequest{Prefix: "k", Limit: -1}); codeOf(err) != codes.InvalidArgument {
		t.Fatalf("negative limit: code = %v", codeOf(err))
	}
}
