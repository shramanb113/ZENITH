package zenith_test

// This file is an external test package (zenith_test, not zenith) on
// purpose: internal/sidecar imports pkg/zenith, so a white-box test that
// imports internal/sidecar from inside package zenith would be an import
// cycle. zenith_test is a distinct package from zenith, so no cycle exists.

import (
	"context"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/collections"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/sidecar"
	"github.com/shramanb113/ZENITH/pkg/zenith"
)

// newTestServer spins up the real sidecar.Server (with persistent
// collections backed by a temp dir and a deterministic, non-ONNX embedder
// so the test has no CGo/model dependency) behind an httptest.Server, and
// returns an admin-keyed zenith.Client pointed at it.
func newTestServer(t *testing.T) (*zenith.Client, *httptest.Server, func()) {
	t.Helper()
	dir := t.TempDir()
	mgr, err := collections.New(collections.Config{Root: dir, Embedder: embedding.NewDeterministicEmbedder(384)})
	if err != nil {
		t.Fatalf("collections.New: %v", err)
	}
	srv := sidecar.New(sidecar.Config{Key: "admin-key", Collections: mgr})
	ts := httptest.NewServer(srv.Handler())
	cleanup := func() {
		ts.Close()
		_ = mgr.CloseAll()
	}
	return zenith.NewClient(ts.URL, "admin-key"), ts, cleanup
}

func TestClient_CollectionLifecycle(t *testing.T) {
	c, _, cleanup := newTestServer(t)
	defer cleanup()
	ctx := context.Background()

	info, key, err := c.CreateCollection(ctx, "acme", zenith.CreateCollectionOptions{})
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if info.ID != "acme" || key == "" {
		t.Fatalf("CreateCollection: got info=%+v key=%q", info, key)
	}

	list, err := c.ListCollections(ctx)
	if err != nil {
		t.Fatalf("ListCollections: %v", err)
	}
	if len(list) != 1 || list[0].ID != "acme" {
		t.Fatalf("ListCollections: want [acme], got %+v", list)
	}

	stat, err := c.StatCollection(ctx, "acme")
	if err != nil {
		t.Fatalf("StatCollection: %v", err)
	}
	if stat.ID != "acme" {
		t.Fatalf("StatCollection: got %+v", stat)
	}

	newKey, err := c.RotateKey(ctx, "acme")
	if err != nil {
		t.Fatalf("RotateKey: %v", err)
	}
	if newKey == "" || newKey == key {
		t.Fatalf("RotateKey: want a fresh key, got %q (old %q)", newKey, key)
	}

	if err := c.DeleteCollection(ctx, "acme"); err != nil {
		t.Fatalf("DeleteCollection: %v", err)
	}
	list, err = c.ListCollections(ctx)
	if err != nil {
		t.Fatalf("ListCollections after delete: %v", err)
	}
	if len(list) != 0 {
		t.Fatalf("ListCollections after delete: want empty, got %+v", list)
	}
}

func TestClient_DocumentLifecycleAndSearch(t *testing.T) {
	c, _, cleanup := newTestServer(t)
	defer cleanup()
	ctx := context.Background()

	if _, _, err := c.CreateCollection(ctx, "docs", zenith.CreateCollectionOptions{}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	res, err := c.UpsertDocuments(ctx, "docs", []zenith.Document{
		{ID: "d1", Text: "the quick brown fox jumps over the lazy dog", Attrs: zenith.Attrs{"lang": "en"}},
		{ID: "d2", Text: "a completely unrelated sentence about oceans"},
	})
	if err != nil {
		t.Fatalf("UpsertDocuments: %v", err)
	}
	if res.New != 2 || res.DocCount != 2 {
		t.Fatalf("UpsertDocuments: got %+v", res)
	}

	text, err := c.GetDocument(ctx, "docs", "d1")
	if err != nil {
		t.Fatalf("GetDocument: %v", err)
	}
	if text != "the quick brown fox jumps over the lazy dog" {
		t.Fatalf("GetDocument: got %q", text)
	}

	results, err := c.Search(ctx, "docs", []zenith.SearchQuery{{ID: "q1", Text: "quick fox"}},
		zenith.WithSearchLimit(5), zenith.WithSearchFilter(zenith.Eq("lang", "en")))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	qr, ok := results["q1"]
	if !ok || len(qr.Hits) != 1 || qr.Hits[0].ID != "d1" {
		t.Fatalf("Search: want 1 hit d1 (filtered to lang=en), got %+v", results)
	}

	if err := c.DeleteDocument(ctx, "docs", "d1"); err != nil {
		t.Fatalf("DeleteDocument: %v", err)
	}
	if _, err := c.GetDocument(ctx, "docs", "d1"); err == nil {
		t.Fatalf("GetDocument after delete: want error, got nil")
	} else {
		var apiErr *zenith.APIError
		if !asAPIError(err, &apiErr) || apiErr.Code != "doc_not_found" {
			t.Fatalf("GetDocument after delete: want APIError doc_not_found, got %v", err)
		}
	}
}

func TestClient_NotFoundIsAPIError(t *testing.T) {
	c, _, cleanup := newTestServer(t)
	defer cleanup()
	ctx := context.Background()

	_, err := c.StatCollection(ctx, "ghost")
	var apiErr *zenith.APIError
	if !asAPIError(err, &apiErr) {
		t.Fatalf("StatCollection on unknown id: want *zenith.APIError, got %v (%T)", err, err)
	}
	if apiErr.StatusCode != 404 || apiErr.Code != "not_found" {
		t.Fatalf("StatCollection on unknown id: got %+v", apiErr)
	}
}

func TestClient_WrongKeyIsUnauthorized(t *testing.T) {
	_, ts, cleanup := newTestServer(t)
	defer cleanup()
	bad := zenith.NewClient(ts.URL, "not-the-key")
	_, err := bad.ListCollections(context.Background())
	var apiErr *zenith.APIError
	if !asAPIError(err, &apiErr) || apiErr.StatusCode != 401 {
		t.Fatalf("ListCollections with wrong key: want 401 APIError, got %v", err)
	}
}

// asAPIError is errors.As without importing "errors" twice for a single use.
func asAPIError(err error, target **zenith.APIError) bool {
	ae, ok := err.(*zenith.APIError)
	if ok {
		*target = ae
	}
	return ok
}

func TestClient_SuggestAndFacets(t *testing.T) {
	c, _, cleanup := newTestServer(t)
	defer cleanup()
	ctx := context.Background()

	if _, _, err := c.CreateCollection(ctx, "docs", zenith.CreateCollectionOptions{}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	_, err := c.UpsertDocuments(ctx, "docs", []zenith.Document{
		{ID: "d1", Text: "kubernetes networking guide", Attrs: zenith.Attrs{"lang": "en"}},
		{ID: "d2", Text: "kubernetes storage guide", Attrs: zenith.Attrs{"lang": "en"}},
		{ID: "d3", Text: "guide de reseau kubernetes", Attrs: zenith.Attrs{"lang": "fr"}},
	})
	if err != nil {
		t.Fatalf("UpsertDocuments: %v", err)
	}

	terms, err := c.Suggest(ctx, "docs", "netw", 10)
	if err != nil {
		t.Fatalf("Suggest: %v", err)
	}
	if len(terms) != 1 || terms[0] != "network" {
		t.Fatalf("Suggest(netw): got %v, want [network] (stemmed)", terms)
	}

	facets, err := c.Facets(ctx, "docs", []string{"lang"}, 10)
	if err != nil {
		t.Fatalf("Facets: %v", err)
	}
	want := []zenith.FacetCount{{Value: "en", Count: 2}, {Value: "fr", Count: 1}}
	if !reflect.DeepEqual(facets["lang"], want) {
		t.Fatalf("Facets[lang]: got %#v, want %#v", facets["lang"], want)
	}

	if _, err := c.Facets(ctx, "docs", nil, 10); err == nil {
		t.Fatal("Facets with no fields: want error")
	}
}

func TestClient_SearchOptions_OffsetSortFacetsWeights(t *testing.T) {
	c, _, cleanup := newTestServer(t)
	defer cleanup()
	ctx := context.Background()

	if _, _, err := c.CreateCollection(ctx, "docs", zenith.CreateCollectionOptions{}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	_, err := c.UpsertDocuments(ctx, "docs", []zenith.Document{
		{ID: "a", Text: "kubernetes networking guide", Attrs: zenith.Attrs{"year": float64(2020), "lang": "en"}},
		{ID: "b", Text: "kubernetes networking tutorial", Attrs: zenith.Attrs{"year": float64(2024), "lang": "en"}},
		{ID: "c", Text: "kubernetes networking basics", Attrs: zenith.Attrs{"year": float64(2022), "lang": "fr"}},
	})
	if err != nil {
		t.Fatalf("UpsertDocuments: %v", err)
	}
	queries := []zenith.SearchQuery{{ID: "q1", Text: "kubernetes networking"}}

	// Weights and offset are accepted and round-trip without error; the
	// collection is too small to assert a specific reordering from the
	// weight override, so this just proves the wire path works end to end.
	withWeights, err := c.Search(ctx, "docs", queries, zenith.WithSearchWeights(3, 0.5, 10), zenith.WithSearchOffset(1))
	if err != nil {
		t.Fatalf("Search with weights+offset: %v", err)
	}
	if got := len(withWeights["q1"].Hits); got != 2 {
		t.Fatalf("Search with offset=1 of 3 matches: got %d hits, want 2", got)
	}

	// SortBy replaces score ordering entirely.
	sorted, err := c.Search(ctx, "docs", queries, zenith.WithSearchSort("year", false))
	if err != nil {
		t.Fatalf("Search with sort: %v", err)
	}
	hits := sorted["q1"].Hits
	if len(hits) != 3 || hits[0].ID != "a" || hits[1].ID != "c" || hits[2].ID != "b" {
		ids := make([]string, len(hits))
		for i, h := range hits {
			ids[i] = h.ID
		}
		t.Fatalf("Search sorted by year ascending: got %v, want [a c b]", ids)
	}

	// Facets on the search itself, scoped to this query's matches.
	withFacets, err := c.Search(ctx, "docs", queries, zenith.WithSearchFacets([]string{"lang"}, 10))
	if err != nil {
		t.Fatalf("Search with facets: %v", err)
	}
	want := []zenith.FacetCount{{Value: "en", Count: 2}, {Value: "fr", Count: 1}}
	if !reflect.DeepEqual(withFacets["q1"].Facets["lang"], want) {
		t.Fatalf("Search facets[lang]: got %#v, want %#v", withFacets["q1"].Facets["lang"], want)
	}
}

func TestClient_IngestFile_PDF(t *testing.T) {
	c, _, cleanup := newTestServer(t)
	defer cleanup()
	ctx := context.Background()

	if _, _, err := c.CreateCollection(ctx, "docs", zenith.CreateCollectionOptions{}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	res, err := c.IngestFile(ctx, "docs", "report1", filepath.Join("..", "..", "internal", "pdf", "testdata", "multipage.pdf"),
		zenith.Attrs{"tenant": "acme"})
	if err != nil {
		t.Fatalf("IngestFile: %v", err)
	}
	// multipage.pdf: page 1 has 301 words (2 overlapping chunks), page 2 has
	// 46 (1 more) — same fixture internal/pdf's own test uses.
	if res.New < 3 || res.DocCount < 3 {
		t.Fatalf("IngestFile: got %+v, want >= 3 new chunks", res)
	}

	// "word0000" is the first word of page 1 and appears nowhere else in the
	// fixture, so it uniquely identifies chunk p1/c0 (internal/pdf's own
	// TestIndex_RealPDF_ChunkBBoxesArePlausible relies on the same word).
	results, err := c.Search(ctx, "docs", []zenith.SearchQuery{{ID: "q1", Text: "word0000"}})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	hits := results["q1"].Hits
	if len(hits) == 0 || hits[0].ID != "report1#p1#c0" {
		t.Fatalf("Search(word0000): got %+v, want top hit report1#p1#c0", hits)
	}

	text, err := c.GetDocument(ctx, "docs", hits[0].ID)
	if err != nil {
		t.Fatalf("GetDocument(%s): %v", hits[0].ID, err)
	}
	if !strings.Contains(text, "word0000") {
		t.Fatalf("GetDocument(%s): got %q, want it to contain word0000", hits[0].ID, text)
	}
}

func TestClient_IngestFile_RejectsUnsupportedExtension(t *testing.T) {
	c, _, cleanup := newTestServer(t)
	defer cleanup()
	ctx := context.Background()

	if _, _, err := c.CreateCollection(ctx, "docs", zenith.CreateCollectionOptions{}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	tmp := filepath.Join(t.TempDir(), "notes.txt")
	if err := os.WriteFile(tmp, []byte("plain text, not a supported upload type"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := c.IngestFile(ctx, "docs", "notes", tmp, nil)
	var apiErr *zenith.APIError
	if !asAPIError(err, &apiErr) || apiErr.StatusCode != 400 {
		t.Fatalf("IngestFile(.txt): want 400 APIError, got %v", err)
	}
}
