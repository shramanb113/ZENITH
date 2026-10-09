package zenith_test

// This file is an external test package (zenith_test, not zenith) on
// purpose: internal/sidecar imports pkg/zenith, so a white-box test that
// imports internal/sidecar from inside package zenith would be an import
// cycle. zenith_test is a distinct package from zenith, so no cycle exists.

import (
	"context"
	"net/http/httptest"
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
