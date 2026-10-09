package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/collections"
	"github.com/shramanb113/ZENITH/internal/embedding"
)

// newTestServer builds a server backed by a real collections.Manager
// (temp dir, deterministic embedder — no ONNX/CGo dependency) so these
// tests exercise the real tool handlers end to end, not a mock.
func newTestMCPServer(t *testing.T) *server {
	t.Helper()
	dir := t.TempDir()
	mgr, err := collections.New(collections.Config{Root: dir, Embedder: embedding.NewDeterministicEmbedder(384)})
	if err != nil {
		t.Fatalf("collections.New: %v", err)
	}
	t.Cleanup(func() { _ = mgr.CloseAll() })
	tools, handlers := buildTools(mgr)
	return &server{name: "test", version: "0.0.0", tools: tools, handlers: handlers}
}

func call(t *testing.T, s *server, method string, id any, params any) *rpcResponse {
	t.Helper()
	req := rpcRequest{JSONRPC: "2.0", ID: id, Method: method}
	if params != nil {
		b, err := json.Marshal(params)
		if err != nil {
			t.Fatalf("marshal params: %v", err)
		}
		req.Params = b
	}
	return s.dispatch(context.Background(), req)
}

func TestInitializeAndToolsList(t *testing.T) {
	s := newTestMCPServer(t)

	resp := call(t, s, "initialize", 1, map[string]any{"protocolVersion": "2024-11-05"})
	if resp == nil || resp.Error != nil {
		t.Fatalf("initialize: got %+v", resp)
	}
	result, ok := resp.Result.(map[string]any)
	if !ok || result["protocolVersion"] == nil {
		t.Fatalf("initialize: unexpected result %+v", resp.Result)
	}

	if r := call(t, s, "notifications/initialized", nil, nil); r != nil {
		t.Fatalf("notification: want no response, got %+v", r)
	}

	resp = call(t, s, "tools/list", 2, nil)
	if resp == nil || resp.Error != nil {
		t.Fatalf("tools/list: got %+v", resp)
	}
	result = resp.Result.(map[string]any)
	tools := result["tools"].([]toolDef)
	if len(tools) != 4 {
		t.Fatalf("tools/list: want 4 tools, got %d", len(tools))
	}
	names := map[string]bool{}
	for _, td := range tools {
		names[td.Name] = true
		if td.Description == "" {
			t.Fatalf("tool %s has no description", td.Name)
		}
		if td.InputSchema == nil {
			t.Fatalf("tool %s has no input schema", td.Name)
		}
	}
	for _, want := range []string{"list_collections", "search_collection", "upsert_document", "get_document"} {
		if !names[want] {
			t.Fatalf("tools/list: missing tool %q, got %v", want, names)
		}
	}
}

func toolCallResult(t *testing.T, resp *rpcResponse) (text string, isError bool) {
	t.Helper()
	if resp == nil || resp.Error != nil {
		t.Fatalf("tools/call: unexpected rpc error: %+v", resp)
	}
	result := resp.Result.(map[string]any)
	content := result["content"].([]map[string]any)
	if len(content) != 1 {
		t.Fatalf("tools/call: want 1 content block, got %v", content)
	}
	return content[0]["text"].(string), result["isError"].(bool)
}

func TestUpsertAutoCreatesCollectionAndGetRoundTrips(t *testing.T) {
	s := newTestMCPServer(t)

	resp := call(t, s, "tools/call", 1, map[string]any{
		"name":      "upsert_document",
		"arguments": map[string]any{"collection": "memory", "id": "note1", "text": "remember the sky is blue"},
	})
	text, isErr := toolCallResult(t, resp)
	if isErr {
		t.Fatalf("upsert_document: want success, got error: %s", text)
	}
	if !strings.Contains(text, "note1") {
		t.Fatalf("upsert_document: want confirmation mentioning note1, got %q", text)
	}

	resp = call(t, s, "tools/call", 2, map[string]any{
		"name":      "get_document",
		"arguments": map[string]any{"collection": "memory", "id": "note1"},
	})
	text, isErr = toolCallResult(t, resp)
	if isErr || text != "remember the sky is blue" {
		t.Fatalf("get_document: got text=%q isError=%v", text, isErr)
	}

	resp = call(t, s, "tools/call", 3, map[string]any{
		"name":      "list_collections",
		"arguments": map[string]any{},
	})
	text, isErr = toolCallResult(t, resp)
	if isErr || !strings.Contains(text, "memory") {
		t.Fatalf("list_collections: got text=%q isError=%v", text, isErr)
	}
}

func TestSearchCollectionFindsUpsertedDocument(t *testing.T) {
	s := newTestMCPServer(t)

	call(t, s, "tools/call", 1, map[string]any{
		"name":      "upsert_document",
		"arguments": map[string]any{"collection": "kb", "id": "d1", "text": "the quick brown fox jumps over the lazy dog"},
	})
	call(t, s, "tools/call", 2, map[string]any{
		"name":      "upsert_document",
		"arguments": map[string]any{"collection": "kb", "id": "d2", "text": "oceans are large bodies of salt water"},
	})

	resp := call(t, s, "tools/call", 3, map[string]any{
		"name":      "search_collection",
		"arguments": map[string]any{"collection": "kb", "query": "quick fox", "limit": 5},
	})
	text, isErr := toolCallResult(t, resp)
	if isErr {
		t.Fatalf("search_collection: want success, got error: %s", text)
	}
	if !strings.Contains(text, "d1") {
		t.Fatalf("search_collection: want d1 in results, got %q", text)
	}
}

func TestGetDocumentNotFoundIsToolError(t *testing.T) {
	s := newTestMCPServer(t)
	call(t, s, "tools/call", 1, map[string]any{
		"name":      "upsert_document",
		"arguments": map[string]any{"collection": "kb", "id": "d1", "text": "hello"},
	})
	resp := call(t, s, "tools/call", 2, map[string]any{
		"name":      "get_document",
		"arguments": map[string]any{"collection": "kb", "id": "missing"},
	})
	text, isErr := toolCallResult(t, resp)
	if !isErr {
		t.Fatalf("get_document on missing id: want isError=true, got text=%q", text)
	}
}

func TestUnknownToolIsInvalidParamsError(t *testing.T) {
	s := newTestMCPServer(t)
	resp := call(t, s, "tools/call", 1, map[string]any{"name": "does_not_exist", "arguments": map[string]any{}})
	if resp == nil || resp.Error == nil || resp.Error.Code != codeInvalidParams {
		t.Fatalf("want invalid-params rpc error, got %+v", resp)
	}
}

// TestFullStdioRoundTrip exercises server.run end to end over an in-memory
// pipe, the same transport real clients use (newline-delimited JSON-RPC on
// stdin/stdout), instead of calling dispatch directly.
func TestFullStdioRoundTrip(t *testing.T) {
	s := newTestMCPServer(t)
	in := strings.NewReader(
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}` + "\n" +
			`{"jsonrpc":"2.0","method":"notifications/initialized"}` + "\n" +
			`{"jsonrpc":"2.0","id":2,"method":"tools/list"}` + "\n",
	)
	var out bytes.Buffer
	if err := s.run(context.Background(), in, &out); err != nil {
		t.Fatalf("run: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 response lines (notification gets none), got %d: %q", len(lines), out.String())
	}
	var r1, r2 map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &r1); err != nil {
		t.Fatalf("decode line 1: %v", err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &r2); err != nil {
		t.Fatalf("decode line 2: %v", err)
	}
	if r1["id"] != float64(1) || r2["id"] != float64(2) {
		t.Fatalf("want ids 1,2 in order, got %v, %v", r1["id"], r2["id"])
	}
}
