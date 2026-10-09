package sidecar

import (
	"reflect"
	"testing"
)

var surfaceDocs = []map[string]any{
	{"id": "d1", "text": "kubernetes cluster networking", "attrs": map[string]any{"lang": "en", "year": 2020, "tags": []any{"k8s", "net"}}},
	{"id": "d2", "text": "kubernetes cluster networking", "attrs": map[string]any{"lang": "en", "year": 2024, "tags": []any{"k8s"}}},
	{"id": "d3", "text": "kubernetes cluster networking", "attrs": map[string]any{"lang": "fr", "year": 2022}},
	{"id": "d4", "text": "kubernetes cluster networking"},
}

func hitIDs(out map[string]any, qid string) []string {
	var ids []string
	for _, h := range hits(out, qid) {
		ids = append(ids, h.(map[string]any)["id"].(string))
	}
	return ids
}

func facetsOf(out map[string]any, qid string) map[string]any {
	f, _ := out["results"].(map[string]any)[qid].(map[string]any)["facets"].(map[string]any)
	return f
}

// checkSearchExtras runs the same sort/offset/facets/weights/attrs checks
// against a namespace or collection search path.
func checkSearchExtras(t *testing.T, search func(body map[string]any) (int, map[string]any)) {
	t.Helper()
	q := []map[string]string{{"id": "q", "text": "kubernetes networking"}}

	// sort replaces score order; offset skips from the top of that order.
	code, out := search(map[string]any{
		"queries": q, "limit": 2, "offset": 1,
		"sort":   map[string]any{"field": "year", "desc": true},
		"facets": map[string]any{"fields": []string{"lang", "tags"}},
	})
	if code != 200 {
		t.Fatalf("search: %d %v", code, out)
	}
	if got := hitIDs(out, "q"); !reflect.DeepEqual(got, []string{"d3", "d1"}) {
		t.Fatalf("sort year desc, offset 1, limit 2 = %v, want [d3 d1]", got)
	}
	// Hits carry attrs (numbers decode as float64 from JSON).
	h0 := hits(out, "q")[0].(map[string]any)
	if a, _ := h0["attrs"].(map[string]any); a["lang"] != "fr" || a["year"] != float64(2022) {
		t.Fatalf("hit attrs = %v", h0["attrs"])
	}
	// Facets cover every match, not just the 2-hit page.
	f := facetsOf(out, "q")
	wantLang := []any{map[string]any{"value": "en", "count": float64(2)}, map[string]any{"value": "fr", "count": float64(1)}}
	if !reflect.DeepEqual(f["lang"], wantLang) {
		t.Fatalf("lang facets = %v, want %v", f["lang"], wantLang)
	}
	wantTags := []any{map[string]any{"value": "k8s", "count": float64(2)}, map[string]any{"value": "net", "count": float64(1)}}
	if !reflect.DeepEqual(f["tags"], wantTags) {
		t.Fatalf("tags facets = %v, want %v", f["tags"], wantTags)
	}

	// No facets requested: no facets key.
	if _, out := search(map[string]any{"queries": q}); facetsOf(out, "q") != nil {
		t.Fatalf("unrequested facets present: %v", out)
	}
	// Valid weights are accepted.
	if code, out := search(map[string]any{"queries": q, "weights": map[string]any{"vector": 3, "phonetic": 0.5, "rrf_k": 10}}); code != 200 {
		t.Fatalf("weights: %d %v", code, out)
	}
	for name, body := range map[string]map[string]any{
		"negative weight":  {"queries": q, "weights": map[string]any{"rrf_k": -1}},
		"negative offset":  {"queries": q, "offset": -1},
		"huge offset":      {"queries": q, "offset": 1 << 40},
		"empty facet name": {"queries": q, "facets": map[string]any{"fields": []string{""}}},
		"negative top_k":   {"queries": q, "facets": map[string]any{"fields": []string{"lang"}, "top_k": -1}},
	} {
		if code, out := search(body); code != 400 {
			t.Fatalf("%s: want 400, got %d %v", name, code, out)
		}
	}
}

func TestNamespaceSearchExtrasSuggestFacets(t *testing.T) {
	_, ts, _, _ := newTest(t, nil)
	if resp, out := do(t, ts, "PUT", "/v1/ns/s/docs", map[string]any{"docs": surfaceDocs}, nil); resp.StatusCode != 200 {
		t.Fatalf("put: %d %v", resp.StatusCode, out)
	}
	checkSearchExtras(t, func(body map[string]any) (int, map[string]any) {
		resp, out := do(t, ts, "POST", "/v1/ns/s/search", body, nil)
		return resp.StatusCode, out
	})

	resp, out := do(t, ts, "GET", "/v1/ns/s/suggest?q=Kub&n=5", nil, nil)
	if resp.StatusCode != 200 || !reflect.DeepEqual(out["terms"], []any{"kubernet"}) {
		t.Fatalf("suggest: %d %v", resp.StatusCode, out)
	}
	if resp, _ := do(t, ts, "GET", "/v1/ns/s/suggest", nil, nil); resp.StatusCode != 400 {
		t.Fatalf("suggest without q: %d", resp.StatusCode)
	}
	if resp, _ := do(t, ts, "GET", "/v1/ns/missing/suggest?q=k", nil, nil); resp.StatusCode != 404 {
		t.Fatalf("suggest unknown ns: %d", resp.StatusCode)
	}

	resp, out = do(t, ts, "GET", "/v1/ns/s/facets?fields=lang,year&top_k=1", nil, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("facets: %d %v", resp.StatusCode, out)
	}
	f := out["facets"].(map[string]any)
	if !reflect.DeepEqual(f["lang"], []any{map[string]any{"value": "en", "count": float64(2)}}) {
		t.Fatalf("corpus lang facets = %v", f["lang"])
	}
	if resp, _ := do(t, ts, "GET", "/v1/ns/s/facets", nil, nil); resp.StatusCode != 400 {
		t.Fatalf("facets without fields: %d", resp.StatusCode)
	}
}

func TestCollectionSearchExtrasDocAttrsSuggestFacets(t *testing.T) {
	_, ts, _, _ := newColTest(t, func(c *Config) { c.Key = "admin" })
	admin := map[string]string{"X-Zenith-Key": "admin"}
	if resp, body := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "c"}, admin); resp.StatusCode != 201 {
		t.Fatalf("create: %d %v", resp.StatusCode, body)
	}
	if resp, body := do(t, ts, "PUT", "/v1/collections/c/docs", map[string]any{"docs": surfaceDocs}, admin); resp.StatusCode != 200 {
		t.Fatalf("put: %d %v", resp.StatusCode, body)
	}
	checkSearchExtras(t, func(body map[string]any) (int, map[string]any) {
		resp, out := do(t, ts, "POST", "/v1/collections/c/search", body, admin)
		return resp.StatusCode, out
	})

	resp, out := do(t, ts, "GET", "/v1/collections/c/docs/d1", nil, admin)
	if resp.StatusCode != 200 || out["text"] != "kubernetes cluster networking" {
		t.Fatalf("get doc: %d %v", resp.StatusCode, out)
	}
	wantAttrs := map[string]any{"lang": "en", "year": float64(2020), "tags": []any{"k8s", "net"}}
	if !reflect.DeepEqual(out["attrs"], wantAttrs) {
		t.Fatalf("doc attrs = %v, want %v", out["attrs"], wantAttrs)
	}
	if _, out := do(t, ts, "GET", "/v1/collections/c/docs/d4", nil, admin); out["attrs"] != nil {
		t.Fatalf("attr-less doc has attrs: %v", out)
	}
	if resp, out := do(t, ts, "GET", "/v1/collections/c/docs/nope", nil, admin); resp.StatusCode != 404 || out["code"] != "doc_not_found" {
		t.Fatalf("missing doc: %d %v", resp.StatusCode, out)
	}

	resp, out = do(t, ts, "GET", "/v1/collections/c/suggest?q=netw", nil, admin)
	if resp.StatusCode != 200 || !reflect.DeepEqual(out["terms"], []any{"network"}) {
		t.Fatalf("suggest: %d %v", resp.StatusCode, out)
	}
	if resp, _ := do(t, ts, "GET", "/v1/collections/c/suggest?q=k&n=-1", nil, admin); resp.StatusCode != 400 {
		t.Fatalf("suggest n=-1: %d", resp.StatusCode)
	}

	resp, out = do(t, ts, "GET", "/v1/collections/c/facets?fields=tags", nil, admin)
	if resp.StatusCode != 200 {
		t.Fatalf("facets: %d %v", resp.StatusCode, out)
	}
	wantTags := []any{map[string]any{"value": "k8s", "count": float64(2)}, map[string]any{"value": "net", "count": float64(1)}}
	if got := out["facets"].(map[string]any)["tags"]; !reflect.DeepEqual(got, wantTags) {
		t.Fatalf("corpus tags = %v, want %v", got, wantTags)
	}
}
