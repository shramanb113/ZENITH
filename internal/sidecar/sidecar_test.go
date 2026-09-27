// internal/sidecar/sidecar_test.go
package sidecar

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

type clock struct{ t time.Time }

func (c *clock) now() time.Time { return c.t }

func newTest(t *testing.T, mut func(*Config)) (*Server, *httptest.Server, *clock, *bytes.Buffer) {
	t.Helper()
	c := &clock{t: time.Date(2026, 9, 30, 10, 0, 0, 0, time.UTC)}
	logs := &bytes.Buffer{}
	cfg := Config{Version: "test", Model: "none", Now: c.now, Log: slog.New(slog.NewTextHandler(logs, nil))}
	if mut != nil {
		mut(&cfg)
	}
	s := New(cfg)
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() { ts.Close(); s.Close() })
	return s, ts, c, logs
}

func do(t *testing.T, ts *httptest.Server, method, path string, body any, hdr map[string]string) (*http.Response, map[string]any) {
	t.Helper()
	var rd *bytes.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	} else {
		rd = bytes.NewReader(nil)
	}
	req, _ := http.NewRequest(method, ts.URL+path, rd)
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp, out
}

func put(t *testing.T, ts *httptest.Server, ns string, texts ...string) {
	t.Helper()
	docs := make([]map[string]string, len(texts))
	for i, tx := range texts {
		docs[i] = map[string]string{"id": string(rune('0' + i)), "text": tx}
	}
	if resp, body := do(t, ts, "PUT", "/v1/ns/"+ns+"/docs", map[string]any{"docs": docs}, nil); resp.StatusCode != 200 {
		t.Fatalf("PUT %s: %d %v", ns, resp.StatusCode, body)
	}
}

func search(t *testing.T, ts *httptest.Server, ns string, queries map[string]string) (int, map[string]any) {
	t.Helper()
	qs := []map[string]string{}
	for id, q := range queries {
		qs = append(qs, map[string]string{"id": id, "text": q})
	}
	resp, body := do(t, ts, "POST", "/v1/ns/"+ns+"/search",
		map[string]any{"queries": qs, "explain": true, "limit": 50, "min_semantic": 0.3}, nil)
	return resp.StatusCode, body
}

func hits(body map[string]any, qid string) []any {
	return body["results"].(map[string]any)[qid].(map[string]any)["hits"].([]any)
}

func TestHealthReportsModel(t *testing.T) {
	_, ts, _, _ := newTest(t, func(c *Config) { c.Key = "k"; c.Model = "all-MiniLM-L6-v2"; c.Synonyms = "abcd1234" })
	resp, body := do(t, ts, "GET", "/healthz", nil, nil)
	if resp.StatusCode != 200 || body["model"] != "all-MiniLM-L6-v2" || body["version"] != "test" || body["synonyms"] != "abcd1234" {
		t.Fatalf("healthz = %d %v", resp.StatusCode, body)
	}
}

func TestKeyRequiredOnV1(t *testing.T) {
	_, ts, _, _ := newTest(t, func(c *Config) { c.Key = "secret" })
	body := map[string]any{"docs": []map[string]string{{"id": "0", "text": "x y"}}}
	if resp, _ := do(t, ts, "PUT", "/v1/ns/a/docs", body, nil); resp.StatusCode != 401 {
		t.Fatalf("missing key: %d", resp.StatusCode)
	}
	if resp, _ := do(t, ts, "PUT", "/v1/ns/a/docs", body, map[string]string{"X-Zenith-Key": "wrong"}); resp.StatusCode != 401 {
		t.Fatalf("wrong key: %d", resp.StatusCode)
	}
	if resp, _ := do(t, ts, "PUT", "/v1/ns/a/docs", body, map[string]string{"X-Zenith-Key": "secret"}); resp.StatusCode != 200 {
		t.Fatalf("right key: %d", resp.StatusCode)
	}
}

func TestSearchSpansPointAtOriginalText(t *testing.T) {
	s, ts, _, _ := newTest(t, nil)
	texts := []string{"Basement mein paani bhar jaata hai", "बेसमेंट में पानी भर जाता है"}
	put(t, ts, "m_1", texts...)
	code, body := search(t, ts, "m_1", map[string]string{"en": "paani bhar", "hi": "पानी भर"})
	if code != 200 {
		t.Fatalf("search: %d %v", code, body)
	}
	for qid, doc := range map[string]int{"en": 0, "hi": 1} {
		hs := hits(body, qid)
		if len(hs) != 1 {
			t.Fatalf("%s: want 1 hit, got %v", qid, hs)
		}
		h := hs[0].(map[string]any)
		runes := []rune(texts[doc])
		terms := h["terms"].([]any)
		if len(terms) != 2 {
			t.Fatalf("%s: want 2 terms, got %v", qid, terms)
		}
		for _, raw := range terms {
			tm := raw.(map[string]any)
			st, en := int(tm["start"].(float64)), int(tm["end"].(float64))
			surface := string(runes[st:en])
			// The surface form must analyse back to exactly the matched term.
			if got := s.ana.TokenizeExact(surface); len(got) != 1 || got[0] != tm["matched"] {
				t.Fatalf("%s: span %d:%d = %q analyses to %v, want %v", qid, st, en, surface, got, tm["matched"])
			}
		}
	}
}

func TestUnknownNamespaceIs404(t *testing.T) {
	_, ts, _, _ := newTest(t, nil)
	if code, _ := search(t, ts, "nope", map[string]string{"q": "x"}); code != 404 {
		t.Fatalf("want 404, got %d", code)
	}
}

func TestBadNamespaceNameIs400(t *testing.T) {
	_, ts, _, _ := newTest(t, nil)
	if resp, _ := do(t, ts, "DELETE", "/v1/ns/bad.name", nil, nil); resp.StatusCode != 400 {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
}

func TestBodyLimitIs413(t *testing.T) {
	_, ts, _, _ := newTest(t, func(c *Config) { c.MaxBody = 64 })
	body := map[string]any{"docs": []map[string]string{{"id": "0", "text": strings.Repeat("flood ", 50)}}}
	if resp, _ := do(t, ts, "PUT", "/v1/ns/a/docs", body, nil); resp.StatusCode != 413 {
		t.Fatalf("want 413, got %d", resp.StatusCode)
	}
}

func TestEmptyDocTextIs400(t *testing.T) {
	_, ts, _, _ := newTest(t, nil)
	body := map[string]any{"docs": []map[string]string{{"id": "0", "text": "   "}}}
	if resp, _ := do(t, ts, "PUT", "/v1/ns/a/docs", body, nil); resp.StatusCode != 400 {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
}

func TestPutReplacesDocuments(t *testing.T) {
	_, ts, _, _ := newTest(t, nil)
	put(t, ts, "a", "sewage overflow near gate")
	put(t, ts, "a", "lovely park")
	if _, body := search(t, ts, "a", map[string]string{"q": "sewage"}); len(hits(body, "q")) != 0 {
		t.Fatal("old documents must be gone after a second PUT")
	}
}

func TestLRUEvictsOldest(t *testing.T) {
	_, ts, c, _ := newTest(t, func(cfg *Config) { cfg.MaxNS = 2 })
	put(t, ts, "a", "flood")
	c.t = c.t.Add(time.Second)
	put(t, ts, "b", "flood")
	c.t = c.t.Add(time.Second)
	put(t, ts, "c", "flood")
	if code, _ := search(t, ts, "a", map[string]string{"q": "flood"}); code != 404 {
		t.Fatalf("a should be evicted, got %d", code)
	}
	if code, _ := search(t, ts, "c", map[string]string{"q": "flood"}); code != 200 {
		t.Fatalf("c should exist, got %d", code)
	}
}

func TestSweepExpiresIdleNamespaces(t *testing.T) {
	s, ts, c, _ := newTest(t, nil)
	put(t, ts, "a", "flood")
	c.t = c.t.Add(11 * time.Minute)
	s.Sweep()
	if code, _ := search(t, ts, "a", map[string]string{"q": "flood"}); code != 404 {
		t.Fatalf("a should have expired, got %d", code)
	}
}

func TestTooManyQueriesIs400(t *testing.T) {
	_, ts, _, _ := newTest(t, func(c *Config) { c.MaxQueries = 1 })
	put(t, ts, "a", "flood")
	if code, _ := search(t, ts, "a", map[string]string{"q1": "flood", "q2": "rain"}); code != 400 {
		t.Fatalf("want 400, got %d", code)
	}
}

func TestDeleteIsIdempotent(t *testing.T) {
	_, ts, _, _ := newTest(t, nil)
	put(t, ts, "a", "flood")
	for i := 0; i < 2; i++ {
		if resp, _ := do(t, ts, "DELETE", "/v1/ns/a", nil, nil); resp.StatusCode != 204 {
			t.Fatalf("delete %d: %d", i, resp.StatusCode)
		}
	}
}

func TestLogsNeverContainText(t *testing.T) {
	_, ts, _, logs := newTest(t, nil)
	secret := "Lakeview basement floods zqxjk"
	put(t, ts, "a", secret)
	search(t, ts, "a", map[string]string{"q": "zqxjk basement"})
	if strings.Contains(logs.String(), "zqxjk") {
		t.Fatalf("snippet or query text leaked into logs: %s", logs.String())
	}
}

func TestRunStopsOnContextCancel(t *testing.T) {
	s, _, _, _ := newTest(t, nil)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { s.Run(ctx, time.Millisecond); close(done) }()
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not return after cancel")
	}
}
