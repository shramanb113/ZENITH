package sidecar

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/shramanb113/ZENITH/internal/collections"
	"github.com/shramanb113/ZENITH/internal/embedding"
)

// newChromaTest mirrors newColTest (collections_http_test.go) but is kept
// local so this file reads standalone.
func newChromaTest(t *testing.T, mut func(*Config)) *httptest.Server {
	t.Helper()
	dir := t.TempDir()
	mgr, err := collections.New(collections.Config{Root: dir, Embedder: embedding.NewDeterministicEmbedder(384)})
	if err != nil {
		t.Fatalf("collections.New: %v", err)
	}
	t.Cleanup(func() { _ = mgr.CloseAll() })
	_, ts, _, _ := newTest(t, func(c *Config) {
		c.Collections = mgr
		if mut != nil {
			mut(c)
		}
	})
	return ts
}

// doAny is like the package's `do` helper but decodes into `any`, so it
// also handles a bare JSON array or scalar response (Chroma's wire shape
// for list/count/version), not just a JSON object.
func doAny(t *testing.T, ts *httptest.Server, method, path string, body any, hdr map[string]string) (*http.Response, any) {
	t.Helper()
	var rd *bytes.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			t.Fatalf("marshal body: %v", err)
		}
		rd = bytes.NewReader(b)
	} else {
		rd = bytes.NewReader(nil)
	}
	req, err := http.NewRequest(method, ts.URL+path, rd)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp, out
}

func TestChroma_Heartbeat(t *testing.T) {
	ts := newChromaTest(t, nil)
	resp, out := doAny(t, ts, "GET", "/api/v1/heartbeat", nil, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("heartbeat: want 200, got %d", resp.StatusCode)
	}
	m := out.(map[string]any)
	if _, ok := m["nanosecond heartbeat"]; !ok {
		t.Fatalf("heartbeat: missing \"nanosecond heartbeat\" key, got %v", out)
	}
}

func TestChroma_CreateGetListDeleteCollection(t *testing.T) {
	ts := newChromaTest(t, nil)

	resp, out := doAny(t, ts, "POST", "/api/v1/collections", map[string]any{"name": "acme"}, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("create: want 200, got %d %v", resp.StatusCode, out)
	}
	m := out.(map[string]any)
	if m["name"] != "acme" || m["id"] != "acme" {
		t.Fatalf("create: want name=id=acme, got %v", out)
	}

	resp, out = doAny(t, ts, "GET", "/api/v1/collections", nil, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("list: want 200, got %d", resp.StatusCode)
	}
	list := out.([]any)
	if len(list) != 1 || list[0].(map[string]any)["name"] != "acme" {
		t.Fatalf("list: want [acme], got %v", out)
	}

	resp, out = doAny(t, ts, "GET", "/api/v1/collections/acme", nil, nil)
	if resp.StatusCode != 200 || out.(map[string]any)["name"] != "acme" {
		t.Fatalf("get: want 200 acme, got %d %v", resp.StatusCode, out)
	}

	resp, _ = doAny(t, ts, "DELETE", "/api/v1/collections/acme", nil, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("delete: want 200, got %d", resp.StatusCode)
	}
	resp, out = doAny(t, ts, "GET", "/api/v1/collections", nil, nil)
	if list := out.([]any); len(list) != 0 {
		t.Fatalf("list after delete: want empty, got %v", list)
	}
	_ = resp
}

func TestChroma_CreateDuplicateFailsUnlessGetOrCreate(t *testing.T) {
	ts := newChromaTest(t, nil)
	doAny(t, ts, "POST", "/api/v1/collections", map[string]any{"name": "acme"}, nil)

	resp, out := doAny(t, ts, "POST", "/api/v1/collections", map[string]any{"name": "acme"}, nil)
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("duplicate create: want 409, got %d %v", resp.StatusCode, out)
	}

	resp, out = doAny(t, ts, "POST", "/api/v1/collections", map[string]any{"name": "acme", "get_or_create": true}, nil)
	if resp.StatusCode != 200 || out.(map[string]any)["name"] != "acme" {
		t.Fatalf("get_or_create: want 200 acme, got %d %v", resp.StatusCode, out)
	}
}

func TestChroma_InvalidCollectionNameRejected(t *testing.T) {
	ts := newChromaTest(t, nil)
	resp, _ := doAny(t, ts, "POST", "/api/v1/collections", map[string]any{"name": "Not_Valid_Chars!"}, nil)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("invalid name: want 400, got %d", resp.StatusCode)
	}
}

func TestChroma_AddRequiresDocuments(t *testing.T) {
	ts := newChromaTest(t, nil)
	doAny(t, ts, "POST", "/api/v1/collections", map[string]any{"name": "docs"}, nil)

	resp, out := doAny(t, ts, "POST", "/api/v1/collections/docs/add", map[string]any{
		"ids":        []string{"d1"},
		"embeddings": [][]float32{{0.1, 0.2, 0.3}},
	}, nil)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("add without documents: want 400, got %d %v", resp.StatusCode, out)
	}
}

func TestChroma_AddGetQueryDelete(t *testing.T) {
	ts := newChromaTest(t, nil)
	doAny(t, ts, "POST", "/api/v1/collections", map[string]any{"name": "docs"}, nil)

	resp, out := doAny(t, ts, "POST", "/api/v1/collections/docs/add", map[string]any{
		"ids":       []string{"d1", "d2"},
		"documents": []string{"the quick brown fox jumps over the lazy dog", "oceans are large bodies of salt water"},
		"metadatas": []map[string]any{{"lang": "en"}, {"lang": "fr"}},
	}, nil)
	if resp.StatusCode != 200 || out != true {
		t.Fatalf("add: want 200 true, got %d %v", resp.StatusCode, out)
	}

	resp, out = doAny(t, ts, "GET", "/api/v1/collections/docs/count", nil, nil)
	if resp.StatusCode != 200 || out.(float64) != 2 {
		t.Fatalf("count: want 200 2, got %d %v", resp.StatusCode, out)
	}

	resp, out = doAny(t, ts, "POST", "/api/v1/collections/docs/get", map[string]any{
		"ids": []string{"d1", "d2", "ghost"},
	}, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("get: want 200, got %d %v", resp.StatusCode, out)
	}
	getResp := out.(map[string]any)
	ids := getResp["ids"].([]any)
	if len(ids) != 2 || ids[0] != "d1" || ids[1] != "d2" {
		t.Fatalf("get: want [d1 d2] (ghost silently skipped), got %v", ids)
	}
	metas := getResp["metadatas"].([]any)
	if metas[0].(map[string]any)["lang"] != "en" {
		t.Fatalf("get: want metadatas[0].lang=en, got %v", metas)
	}

	resp, out = doAny(t, ts, "POST", "/api/v1/collections/docs/query", map[string]any{
		"query_texts": []string{"quick fox"},
		"n_results":   5,
	}, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("query: want 200, got %d %v", resp.StatusCode, out)
	}
	qResp := out.(map[string]any)
	qIDs := qResp["ids"].([]any)[0].([]any)
	if len(qIDs) != 1 || qIDs[0] != "d1" {
		t.Fatalf("query: want [[d1]], got %v", qResp["ids"])
	}

	resp, out = doAny(t, ts, "POST", "/api/v1/collections/docs/delete", map[string]any{
		"ids": []string{"d1"},
	}, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("delete: want 200, got %d %v", resp.StatusCode, out)
	}
	deleted := out.([]any)
	if len(deleted) != 1 || deleted[0] != "d1" {
		t.Fatalf("delete: want [d1], got %v", deleted)
	}

	resp, out = doAny(t, ts, "GET", "/api/v1/collections/docs/count", nil, nil)
	if out.(float64) != 1 {
		t.Fatalf("count after delete: want 1, got %v", out)
	}
}

func TestChroma_QueryRejectsEmbeddingsOnlyQuery(t *testing.T) {
	ts := newChromaTest(t, nil)
	doAny(t, ts, "POST", "/api/v1/collections", map[string]any{"name": "docs"}, nil)
	resp, _ := doAny(t, ts, "POST", "/api/v1/collections/docs/query", map[string]any{
		"query_embeddings": [][]float32{{0.1, 0.2}},
		"n_results":        5,
	}, nil)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("query_embeddings-only: want 400, got %d", resp.StatusCode)
	}
}

func TestChroma_WhereFilterNarrowsQueryAndGet(t *testing.T) {
	ts := newChromaTest(t, nil)
	doAny(t, ts, "POST", "/api/v1/collections", map[string]any{"name": "docs"}, nil)
	doAny(t, ts, "POST", "/api/v1/collections/docs/add", map[string]any{
		"ids":       []string{"d1", "d2"},
		"documents": []string{"shared words here", "shared words here"},
		"metadatas": []map[string]any{{"lang": "en"}, {"lang": "fr"}},
	}, nil)

	resp, out := doAny(t, ts, "POST", "/api/v1/collections/docs/query", map[string]any{
		"query_texts": []string{"shared words"},
		"n_results":   5,
		"where":       map[string]any{"lang": "fr"},
	}, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("query with where: want 200, got %d %v", resp.StatusCode, out)
	}
	qIDs := out.(map[string]any)["ids"].([]any)[0].([]any)
	if len(qIDs) != 1 || qIDs[0] != "d2" {
		t.Fatalf("query with where lang=fr: want [d2], got %v", qIDs)
	}

	resp, out = doAny(t, ts, "POST", "/api/v1/collections/docs/get", map[string]any{
		"ids":   []string{"d1", "d2"},
		"where": map[string]any{"lang": map[string]any{"$ne": "fr"}},
	}, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("get with where: want 200, got %d %v", resp.StatusCode, out)
	}
	ids := out.(map[string]any)["ids"].([]any)
	if len(ids) != 1 || ids[0] != "d1" {
		t.Fatalf("get with where lang!=fr: want [d1], got %v", ids)
	}
}

func TestChroma_RequiresSharedTokenWhenKeyConfigured(t *testing.T) {
	ts := newChromaTest(t, func(c *Config) { c.Key = "admin-key" })

	resp, _ := doAny(t, ts, "GET", "/api/v1/collections", nil, nil)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("no token: want 401, got %d", resp.StatusCode)
	}
	resp, _ = doAny(t, ts, "GET", "/api/v1/collections", nil, map[string]string{"X-Chroma-Token": "wrong"})
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("wrong token: want 401, got %d", resp.StatusCode)
	}
	resp, _ = doAny(t, ts, "GET", "/api/v1/collections", nil, map[string]string{"X-Chroma-Token": "admin-key"})
	if resp.StatusCode != 200 {
		t.Fatalf("right token: want 200, got %d", resp.StatusCode)
	}
	// Heartbeat is unauthenticated even with a key configured, matching
	// real Chroma (used for liveness probing before a client has a token).
	resp, _ = doAny(t, ts, "GET", "/api/v1/heartbeat", nil, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("heartbeat without token: want 200, got %d", resp.StatusCode)
	}
}

func TestChroma_NativeAndChromaRoutesShareOneCollection(t *testing.T) {
	ts := newChromaTest(t, nil)
	// Created through the native /v1/collections route...
	resp, body := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "shared"}, nil)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("native create: want 201, got %d %v", resp.StatusCode, body)
	}
	key, _ := body["key"].(string)

	// ...is visible and writable through the Chroma route, with no
	// separate provisioning step.
	resp, out := doAny(t, ts, "POST", "/api/v1/collections/shared/add", map[string]any{
		"ids":       []string{"d1"},
		"documents": []string{"hello from the chroma shim"},
	}, nil)
	if resp.StatusCode != 200 || out != true {
		t.Fatalf("chroma add into natively-created collection: want 200 true, got %d %v", resp.StatusCode, out)
	}

	// ...and the document is visible through the native route too.
	resp, body = do(t, ts, "GET", "/v1/collections/shared/docs/d1", nil, map[string]string{"X-Zenith-Key": key})
	if resp.StatusCode != 200 || body["text"] != "hello from the chroma shim" {
		t.Fatalf("native get after chroma add: want 200, got %d %v", resp.StatusCode, body)
	}
}
