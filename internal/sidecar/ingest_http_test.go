package sidecar

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// doMultipart posts a multipart/form-data request built from fields plus an
// optional single file part, matching do's shape (parsed JSON response body)
// for tests that need a file upload rather than a JSON body.
func doMultipart(t *testing.T, ts *httptest.Server, path string, fields map[string]string, fileField, fileName string, fileContent []byte, hdr map[string]string) (*http.Response, map[string]any) {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for k, v := range fields {
		if err := mw.WriteField(k, v); err != nil {
			t.Fatalf("WriteField(%s): %v", k, err)
		}
	}
	if fileField != "" {
		fw, err := mw.CreateFormFile(fileField, fileName)
		if err != nil {
			t.Fatalf("CreateFormFile: %v", err)
		}
		if _, err := fw.Write(fileContent); err != nil {
			t.Fatalf("writing file content: %v", err)
		}
	}
	if err := mw.Close(); err != nil {
		t.Fatalf("mw.Close: %v", err)
	}
	req, err := http.NewRequest(http.MethodPost, ts.URL+path, &buf)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	var out map[string]any
	_ = json.Unmarshal(data, &out)
	return resp, out
}

func TestIngest_PDF_AttrsIncludePageAndBBox(t *testing.T) {
	_, ts, _, _ := newColTest(t, nil)
	_, bodyA := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "a"}, nil)
	key := bodyA["key"].(string)

	pdfBytes, err := os.ReadFile(filepath.Join("..", "pdf", "testdata", "multipage.pdf"))
	if err != nil {
		t.Fatalf("reading fixture: %v", err)
	}

	resp, body := doMultipart(t, ts, "/v1/collections/a/ingest",
		map[string]string{"id": "report1", "attrs": `{"tenant":"acme"}`},
		"file", "multipage.pdf", pdfBytes,
		map[string]string{"X-Zenith-Key": key})
	if resp.StatusCode != 200 {
		t.Fatalf("ingest: want 200, got %d %v", resp.StatusCode, body)
	}
	if int(body["upserted"].(float64)) < 3 {
		t.Fatalf("ingest: want >= 3 upserted, got %v", body)
	}

	// "#" (not pdf.Chunk.ID's "||") is this route's chunk-ID separator — see
	// ingest_http.go's addChunk — so the doc path must percent-escape it.
	resp2, doc := do(t, ts, "GET", "/v1/collections/a/docs/report1%23p1%23c0", nil, map[string]string{"X-Zenith-Key": key})
	if resp2.StatusCode != 200 {
		t.Fatalf("GetDocument(report1#p1#c0): want 200, got %d %v", resp2.StatusCode, doc)
	}
	attrs, ok := doc["attrs"].(map[string]any)
	if !ok {
		t.Fatalf("GetDocument: want attrs in response, got %v", doc)
	}
	if attrs["tenant"] != "acme" {
		t.Fatalf("attrs[tenant]: got %v, want acme", attrs["tenant"])
	}
	if _, ok := attrs["_page"]; !ok {
		t.Fatalf("attrs: want _page, got %v", attrs)
	}
	if _, ok := attrs["_bbox"]; !ok {
		t.Fatalf("attrs: want _bbox, got %v", attrs)
	}
}

func TestIngest_RejectsMissingFile(t *testing.T) {
	_, ts, _, _ := newColTest(t, nil)
	_, bodyA := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "a"}, nil)
	key := bodyA["key"].(string)

	resp, body := doMultipart(t, ts, "/v1/collections/a/ingest",
		map[string]string{"id": "doc1"}, "", "", nil,
		map[string]string{"X-Zenith-Key": key})
	if resp.StatusCode != 400 {
		t.Fatalf("want 400, got %d %v", resp.StatusCode, body)
	}
}

func TestIngest_RejectsUnsupportedExtension(t *testing.T) {
	_, ts, _, _ := newColTest(t, nil)
	_, bodyA := do(t, ts, "POST", "/v1/collections", map[string]any{"id": "a"}, nil)
	key := bodyA["key"].(string)

	resp, body := doMultipart(t, ts, "/v1/collections/a/ingest",
		map[string]string{"id": "doc1"}, "file", "notes.txt", []byte("plain text"),
		map[string]string{"X-Zenith-Key": key})
	if resp.StatusCode != 400 {
		t.Fatalf("want 400, got %d %v", resp.StatusCode, body)
	}
}
