package sidecar

import (
	"bytes"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/collections"
	"github.com/shramanb113/ZENITH/internal/embedding"
)

func newColTest(t *testing.T, mut func(*Config)) (*Server, *httptest.Server, *collections.Manager, *bytes.Buffer) {
	t.Helper()
	dir := t.TempDir()
	mgr, err := collections.New(collections.Config{Root: dir, Embedder: embedding.NewDeterministicEmbedder(384)})
	if err != nil {
		t.Fatalf("collections.New: %v", err)
	}
	t.Cleanup(func() { _ = mgr.CloseAll() })
	s, ts, _, logs := newTest(t, func(c *Config) {
		c.Collections = mgr
		if mut != nil {
			mut(c)
		}
	})
	return s, ts, mgr, logs
}

func TestCollections_CreateRequiresAdminKey(t *testing.T) {
	_, ts, _, _ := newColTest(t, func(c *Config) { c.Key = "admin-key" })
	resp, body := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "acme"}, nil)
	if resp.StatusCode != 401 {
		t.Fatalf("want 401, got %d %v", resp.StatusCode, body)
	}
	if body["code"] != "unauthorized" {
		t.Fatalf("want code=unauthorized, got %v", body)
	}
}

func TestCollections_Create(t *testing.T) {
	_, ts, _, _ := newColTest(t, func(c *Config) { c.Key = "admin-key" })
	resp, body := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "acme"}, map[string]string{"X-Zenith-Key": "admin-key"})
	if resp.StatusCode != 201 {
		t.Fatalf("want 201, got %d %v", resp.StatusCode, body)
	}
	key, _ := body["key"].(string)
	if !strings.HasPrefix(key, "zk_") {
		t.Fatalf("want key with zk_ prefix, got %v", body)
	}
}

func TestCollections_DataPlaneKeyIsolation(t *testing.T) {
	_, ts, _, _ := newColTest(t, func(c *Config) { c.Key = "admin-key" })
	admin := map[string]string{"X-Zenith-Key": "admin-key"}
	if resp, body := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "a"}, admin); resp.StatusCode != 201 {
		t.Fatalf("create a: %d %v", resp.StatusCode, body)
	}
	_, bodyB := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "b"}, admin)
	keyB := bodyB["key"].(string)

	put := map[string]any{"docs": []map[string]any{{"id": "d1", "text": "hello"}}}
	for _, hdr := range []map[string]string{nil, {"X-Zenith-Key": "wrong"}, {"X-Zenith-Key": keyB}} {
		resp, body := do(t, ts, "PUT", "/v1/collections/a/docs", put, hdr)
		if resp.StatusCode != 401 {
			t.Fatalf("hdr=%v: want 401, got %d %v", hdr, resp.StatusCode, body)
		}
	}
}

func TestCollections_PutDocsWithOwnOrAdminKey(t *testing.T) {
	_, ts, _, _ := newColTest(t, func(c *Config) { c.Key = "admin-key" })
	_, bodyA := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "a"}, map[string]string{"X-Zenith-Key": "admin-key"})
	keyA := bodyA["key"].(string)

	put1 := map[string]any{"docs": []map[string]any{{"id": "d1", "text": "hello"}}}
	if resp, body := do(t, ts, "PUT", "/v1/collections/a/docs", put1, map[string]string{"X-Zenith-Key": keyA}); resp.StatusCode != 200 {
		t.Fatalf("own key: want 200, got %d %v", resp.StatusCode, body)
	}
	put2 := map[string]any{"docs": []map[string]any{{"id": "d2", "text": "world"}}}
	if resp, body := do(t, ts, "PUT", "/v1/collections/a/docs", put2, map[string]string{"X-Zenith-Key": "admin-key"}); resp.StatusCode != 200 {
		t.Fatalf("admin key: want 200, got %d %v", resp.StatusCode, body)
	}
}

func TestCollections_SearchSpansPointAtOriginalText(t *testing.T) {
	s, ts, _, _ := newColTest(t, nil)
	_, bodyA := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "a"}, nil)
	key := bodyA["key"].(string)
	text := "Basement mein paani bhar jaata hai"
	put := map[string]any{"docs": []map[string]any{{"id": "d1", "text": text}}}
	if resp, body := do(t, ts, "PUT", "/v1/collections/a/docs", put, map[string]string{"X-Zenith-Key": key}); resp.StatusCode != 200 {
		t.Fatalf("put: %d %v", resp.StatusCode, body)
	}
	sreq := map[string]any{"queries": []map[string]string{{"id": "q", "text": "paani bhar"}}}
	resp, body := do(t, ts, "POST", "/v1/collections/a/search", sreq, map[string]string{"X-Zenith-Key": key})
	if resp.StatusCode != 200 {
		t.Fatalf("search: %d %v", resp.StatusCode, body)
	}
	hs := hits(body, "q")
	if len(hs) != 1 {
		t.Fatalf("want 1 hit, got %v", hs)
	}
	h := hs[0].(map[string]any)
	runes := []rune(text)
	terms := h["terms"].([]any)
	if len(terms) == 0 {
		t.Fatalf("want at least one matched term, got %v", h)
	}
	for _, raw := range terms {
		tm := raw.(map[string]any)
		st, en := int(tm["start"].(float64)), int(tm["end"].(float64))
		surface := string(runes[st:en])
		if got := s.ana.TokenizeExact(surface); len(got) != 1 || got[0] != tm["matched"] {
			t.Fatalf("span %d:%d = %q analyses to %v, want %v", st, en, surface, got, tm["matched"])
		}
	}
}

func TestCollections_BodyTooLarge(t *testing.T) {
	_, ts, _, _ := newColTest(t, nil)
	_, bodyA := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "a", "max_body_bytes": 1024}, nil)
	key := bodyA["key"].(string)
	big := strings.Repeat("flood ", 1000)
	put := map[string]any{"docs": []map[string]any{{"id": "d1", "text": big}}}
	resp, body := do(t, ts, "PUT", "/v1/collections/a/docs", put, map[string]string{"X-Zenith-Key": key})
	if resp.StatusCode != 413 {
		t.Fatalf("want 413, got %d %v", resp.StatusCode, body)
	}
	if body["code"] != "body_too_large" {
		t.Fatalf("want code=body_too_large, got %v", body)
	}
}

func TestCollections_QuotaExceededHTTP(t *testing.T) {
	_, ts, _, _ := newColTest(t, nil)
	_, bodyA := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "a", "max_docs": 1}, nil)
	key := bodyA["key"].(string)
	put1 := map[string]any{"docs": []map[string]any{{"id": "d1", "text": "one"}}}
	if resp, body := do(t, ts, "PUT", "/v1/collections/a/docs", put1, map[string]string{"X-Zenith-Key": key}); resp.StatusCode != 200 {
		t.Fatalf("first put: %d %v", resp.StatusCode, body)
	}
	put2 := map[string]any{"docs": []map[string]any{{"id": "d2", "text": "two"}}}
	resp, body := do(t, ts, "PUT", "/v1/collections/a/docs", put2, map[string]string{"X-Zenith-Key": key})
	if resp.StatusCode != 403 || body["code"] != "quota_exceeded" {
		t.Fatalf("want 403 quota_exceeded, got %d %v", resp.StatusCode, body)
	}
}

func TestCollections_UnknownCollection(t *testing.T) {
	_, ts, _, _ := newColTest(t, func(c *Config) { c.Key = "admin-key" })
	resp, body := do(t, ts, "GET", "/v1/collections/ghost/stats", nil, map[string]string{"X-Zenith-Key": "whatever"})
	if resp.StatusCode != 401 {
		t.Fatalf("non-admin: want 401, got %d %v", resp.StatusCode, body)
	}
	resp, body = do(t, ts, "GET", "/v1/collections/ghost/stats", nil, map[string]string{"X-Zenith-Key": "admin-key"})
	if resp.StatusCode != 404 || body["code"] != "not_found" {
		t.Fatalf("admin: want 404 not_found, got %d %v", resp.StatusCode, body)
	}
}

func TestCollections_DocIDWithSlash(t *testing.T) {
	_, ts, _, _ := newColTest(t, nil)
	_, bodyA := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "a"}, nil)
	key := bodyA["key"].(string)
	docID := "folder/file.txt"
	put := map[string]any{"docs": []map[string]any{{"id": docID, "text": "contents"}}}
	if resp, body := do(t, ts, "PUT", "/v1/collections/a/docs", put, map[string]string{"X-Zenith-Key": key}); resp.StatusCode != 200 {
		t.Fatalf("put: %d %v", resp.StatusCode, body)
	}
	docPath := "/v1/collections/a/docs/" + url.PathEscape(docID)
	resp, body := do(t, ts, "GET", docPath, nil, map[string]string{"X-Zenith-Key": key})
	if resp.StatusCode != 200 || body["text"] != "contents" {
		t.Fatalf("get: want 200 contents, got %d %v", resp.StatusCode, body)
	}
	resp, body = do(t, ts, "DELETE", docPath, nil, map[string]string{"X-Zenith-Key": key})
	if resp.StatusCode != 204 {
		t.Fatalf("delete: want 204, got %d %v", resp.StatusCode, body)
	}
	resp, body = do(t, ts, "GET", docPath, nil, map[string]string{"X-Zenith-Key": key})
	if resp.StatusCode != 404 || body["code"] != "doc_not_found" {
		t.Fatalf("get after delete: want 404 doc_not_found, got %d %v", resp.StatusCode, body)
	}
}

func TestCollections_StatsWALBytes(t *testing.T) {
	_, ts, _, _ := newColTest(t, nil)
	_, bodyA := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "a"}, nil)
	key := bodyA["key"].(string)
	put := map[string]any{"docs": []map[string]any{{"id": "d1", "text": "hello"}}}
	if resp, body := do(t, ts, "PUT", "/v1/collections/a/docs", put, map[string]string{"X-Zenith-Key": key}); resp.StatusCode != 200 {
		t.Fatalf("put: %d %v", resp.StatusCode, body)
	}
	resp, body := do(t, ts, "GET", "/v1/collections/a/stats", nil, map[string]string{"X-Zenith-Key": key})
	if resp.StatusCode != 200 {
		t.Fatalf("stats: %d %v", resp.StatusCode, body)
	}
	wal, _ := body["wal_bytes"].(float64)
	if wal <= 0 {
		t.Fatalf("want wal_bytes > 0, got %v", body)
	}
}

func TestCollections_RotateKey(t *testing.T) {
	_, ts, _, _ := newColTest(t, func(c *Config) { c.Key = "admin-key" })
	admin := map[string]string{"X-Zenith-Key": "admin-key"}
	_, bodyA := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "a"}, admin)
	oldKey := bodyA["key"].(string)
	resp, body := do(t, ts, "POST", "/v1/collections/a/rotate-key", nil, admin)
	if resp.StatusCode != 200 {
		t.Fatalf("rotate: %d %v", resp.StatusCode, body)
	}
	newKey := body["key"].(string)
	if newKey == oldKey {
		t.Fatalf("rotate-key returned the same key")
	}
	stats := "/v1/collections/a/stats"
	if resp, _ := do(t, ts, "GET", stats, nil, map[string]string{"X-Zenith-Key": oldKey}); resp.StatusCode != 401 {
		t.Fatalf("old key: want 401, got %d", resp.StatusCode)
	}
	if resp, _ := do(t, ts, "GET", stats, nil, map[string]string{"X-Zenith-Key": newKey}); resp.StatusCode != 200 {
		t.Fatalf("new key: want 200, got %d", resp.StatusCode)
	}
}

func TestCollections_ListAndDelete(t *testing.T) {
	_, ts, _, _ := newColTest(t, func(c *Config) { c.Key = "admin-key" })
	hdr := map[string]string{"X-Zenith-Key": "admin-key"}
	for _, id := range []string{"c2", "c1"} {
		if resp, body := do(t, ts, "POST", "/v1/collections", map[string]any{"id": id}, hdr); resp.StatusCode != 201 {
			t.Fatalf("create %s: %d %v", id, resp.StatusCode, body)
		}
	}
	resp, body := do(t, ts, "GET", "/v1/collections", nil, hdr)
	if resp.StatusCode != 200 {
		t.Fatalf("list: %d %v", resp.StatusCode, body)
	}
	list := body["collections"].([]any)
	if len(list) != 2 {
		t.Fatalf("want 2 collections, got %v", list)
	}
	id0 := list[0].(map[string]any)["id"].(string)
	id1 := list[1].(map[string]any)["id"].(string)
	if id0 != "c1" || id1 != "c2" {
		t.Fatalf("want sorted [c1 c2], got [%s %s]", id0, id1)
	}

	resp, _ = do(t, ts, "DELETE", "/v1/collections/c1", nil, hdr)
	if resp.StatusCode != 204 {
		t.Fatalf("delete: want 204, got %d", resp.StatusCode)
	}
	resp, body = do(t, ts, "GET", "/v1/collections", nil, hdr)
	list = body["collections"].([]any)
	if len(list) != 1 || list[0].(map[string]any)["id"] != "c2" {
		t.Fatalf("after delete, want [c2], got %v", list)
	}
}

func TestCollections_LogsNeverContainText(t *testing.T) {
	_, ts, _, logs := newColTest(t, nil)
	_, bodyA := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "a"}, nil)
	key := bodyA["key"].(string)
	secret := "Lakeview basement floods zqxjk"
	put := map[string]any{"docs": []map[string]any{{"id": "d1", "text": secret}}}
	do(t, ts, "PUT", "/v1/collections/a/docs", put, map[string]string{"X-Zenith-Key": key})
	sreq := map[string]any{"queries": []map[string]string{{"id": "q", "text": "zqxjk basement"}}}
	do(t, ts, "POST", "/v1/collections/a/search", sreq, map[string]string{"X-Zenith-Key": key})
	if strings.Contains(logs.String(), "zqxjk") {
		t.Fatalf("snippet or query text leaked into logs: %s", logs.String())
	}
}

func TestCollections_NilManagerLeavesRoutesUnregistered(t *testing.T) {
	_, ts, _, _ := newTest(t, nil) // Config.Collections left nil
	resp, _ := do(t, ts, "GET", "/v1/collections", nil, nil)
	if resp.StatusCode != 404 {
		t.Fatalf("want 404 (route not registered), got %d", resp.StatusCode)
	}
	put(t, ts, "ephemeral", "still works")
	code, body := search(t, ts, "ephemeral", map[string]string{"q": "works"})
	if code != 200 || len(hits(body, "q")) == 0 {
		t.Fatalf("ephemeral routes affected: %d %v", code, body)
	}
}
