package sidecar

import (
	"testing"
)

// The sidecar namespaces are per-request document sets; documents may carry
// attrs, and a search may carry a filter (which composes with Explain).
func TestSearchFilterRestrictsToMatchingAttrs(t *testing.T) {
	_, ts, _, _ := newTest(t, nil)
	body := map[string]any{"docs": []map[string]any{
		{"id": "d1", "text": "kubernetes cluster networking", "attrs": map[string]any{"lang": "en", "year": 2020, "public": true}},
		{"id": "d2", "text": "kubernetes cluster networking", "attrs": map[string]any{"lang": "en", "year": 2024, "public": false}},
		{"id": "d3", "text": "kubernetes cluster networking", "attrs": map[string]any{"lang": "fr", "year": 2022, "public": true}},
		{"id": "d4", "text": "kubernetes cluster networking"},
	}}
	if resp, out := do(t, ts, "PUT", "/v1/ns/f/docs", body, nil); resp.StatusCode != 200 {
		t.Fatalf("put: %d %v", resp.StatusCode, out)
	}
	ids := func(filter any) []string {
		req := map[string]any{"queries": []map[string]string{{"id": "q", "text": "kubernetes networking"}}}
		if filter != nil {
			req["filter"] = filter
		}
		resp, out := do(t, ts, "POST", "/v1/ns/f/search", req, nil)
		if resp.StatusCode != 200 {
			t.Fatalf("search: status %d %v", resp.StatusCode, out)
		}
		var got []string
		for _, h := range hits(out, "q") {
			got = append(got, h.(map[string]any)["id"].(string))
		}
		return got
	}
	same := func(name string, got []string, want ...string) {
		t.Helper()
		seen := map[string]bool{}
		for _, g := range got {
			seen[g] = true
		}
		if len(got) != len(want) {
			t.Fatalf("%s: %v, want %v", name, got, want)
		}
		for _, w := range want {
			if !seen[w] {
				t.Fatalf("%s: %v, want %v", name, got, want)
			}
		}
	}

	same("no filter", ids(nil), "d1", "d2", "d3", "d4")
	same("eq", ids(map[string]any{"op": "eq", "field": "lang", "value": "en"}), "d1", "d2")
	same("and", ids(map[string]any{"op": "and", "args": []any{
		map[string]any{"op": "eq", "field": "lang", "value": "en"},
		map[string]any{"op": "eq", "field": "public", "value": true}}}), "d1")
	same("range", ids(map[string]any{"op": "range", "field": "year", "min": 2021}), "d2", "d3")
	same("not exists", ids(map[string]any{"op": "not", "args": []any{map[string]any{"op": "exists", "field": "lang"}}}), "d4")
}

func TestSearchRejectsABadFilterAndBadAttrs(t *testing.T) {
	_, ts, _, _ := newTest(t, nil)
	put(t, ts, "g", "some text here")
	for name, filter := range map[string]any{
		"unknown op":       map[string]any{"op": "regex", "field": "a", "value": "x"},
		"eq without value": map[string]any{"op": "eq", "field": "a"},
		"not a filter":     "tenant=acme",
	} {
		resp, _ := do(t, ts, "POST", "/v1/ns/g/search",
			map[string]any{"queries": []map[string]string{{"id": "q", "text": "text"}}, "filter": filter}, nil)
		if resp.StatusCode != 400 {
			t.Errorf("%s: status %d, want 400", name, resp.StatusCode)
		}
	}
	resp, _ := do(t, ts, "PUT", "/v1/ns/bad/docs", map[string]any{"docs": []map[string]any{
		{"id": "d", "text": "text", "attrs": map[string]any{"nested": map[string]any{"x": 1}}}}}, nil)
	if resp.StatusCode != 400 {
		t.Errorf("nested attr value: status %d, want 400", resp.StatusCode)
	}
}
