package main

// A minimal MCP (Model Context Protocol) server over stdio: just enough of
// the JSON-RPC 2.0 envelope and the "tools" capability to let an MCP
// client (Claude Code, Claude Desktop, Cursor, ...) discover and call this
// binary's tools. There is no vendored MCP SDK in go.mod, and the stdio
// transport is small enough (newline-delimited JSON-RPC messages on
// stdin/stdout, no Content-Length framing like LSP) that implementing it
// directly is simpler than adding a dependency.
//
// Every log line and diagnostic in this program goes to stderr, never
// stdout — stdout is the protocol channel, and a single stray non-JSON
// line on it would break every client reading the stream.

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
)

const (
	codeParseError     = -32700
	codeMethodNotFound = -32601
	codeInvalidParams  = -32602
)

type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string    `json:"jsonrpc"`
	ID      any       `json:"id,omitempty"`
	Result  any       `json:"result,omitempty"`
	Error   *rpcError `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// toolDef is one entry in a tools/list response. InputSchema is a JSON
// Schema object describing the tool's arguments.
type toolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

// toolHandler executes one tool call. It never returns a transport-level
// error: a failure (bad arguments, not found, quota exceeded, ...) is
// reported as content with isError=true so the calling model can see and
// react to it, per the MCP convention that tool errors are part of the
// result, not the JSON-RPC envelope.
type toolHandler func(ctx context.Context, args json.RawMessage) (content string, isError bool)

// server holds everything dispatch needs: the advertised tool list (for
// tools/list) and the name -> handler table (for tools/call).
type server struct {
	name     string
	version  string
	tools    []toolDef
	handlers map[string]toolHandler
}

// run reads newline-delimited JSON-RPC requests from in and writes
// responses to out until in is exhausted or ctx is done. It returns nil on
// a clean EOF (the client closed stdin, the normal shutdown signal for an
// MCP stdio server).
func (s *server) run(ctx context.Context, in io.Reader, out io.Writer) error {
	reader := bufio.NewReaderSize(in, 1<<20)
	w := bufio.NewWriter(out)
	for {
		if ctx.Err() != nil {
			return nil
		}
		line, err := reader.ReadString('\n')
		trimmed := trimSpace(line)
		if trimmed != "" {
			resp := s.handleLine(ctx, trimmed)
			if resp != nil {
				if werr := writeResponse(w, *resp); werr != nil {
					return werr
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

func trimSpace(s string) string {
	start, end := 0, len(s)
	for start < end && isSpace(s[start]) {
		start++
	}
	for end > start && isSpace(s[end-1]) {
		end--
	}
	return s[start:end]
}

func isSpace(b byte) bool { return b == ' ' || b == '\t' || b == '\r' || b == '\n' }

func writeResponse(w *bufio.Writer, resp rpcResponse) error {
	b, err := json.Marshal(resp)
	if err != nil {
		return fmt.Errorf("zenith-mcp: marshal response: %w", err)
	}
	if _, err := w.Write(b); err != nil {
		return err
	}
	if err := w.WriteByte('\n'); err != nil {
		return err
	}
	return w.Flush()
}

// handleLine parses and dispatches one request line. It returns nil for a
// notification (no "id" field) — JSON-RPC notifications get no response,
// even an error one.
func (s *server) handleLine(ctx context.Context, line string) *rpcResponse {
	var req rpcRequest
	if err := json.Unmarshal([]byte(line), &req); err != nil {
		return &rpcResponse{JSONRPC: "2.0", Error: &rpcError{Code: codeParseError, Message: "parse error: " + err.Error()}}
	}
	return s.dispatch(ctx, req)
}

func errResp(id any, code int, msg string) *rpcResponse {
	return &rpcResponse{JSONRPC: "2.0", ID: id, Error: &rpcError{Code: code, Message: msg}}
}

func (s *server) dispatch(ctx context.Context, req rpcRequest) *rpcResponse {
	isNotification := req.ID == nil

	switch req.Method {
	case "initialize":
		result := map[string]any{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      map[string]any{"name": s.name, "version": s.version},
		}
		return &rpcResponse{JSONRPC: "2.0", ID: req.ID, Result: result}

	case "notifications/initialized", "notifications/cancelled":
		return nil // no response to any notification

	case "ping":
		if isNotification {
			return nil
		}
		return &rpcResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{}}

	case "tools/list":
		if isNotification {
			return nil
		}
		return &rpcResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{"tools": s.tools}}

	case "tools/call":
		if isNotification {
			return nil
		}
		var p struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return errResp(req.ID, codeInvalidParams, "invalid tools/call params: "+err.Error())
		}
		h, ok := s.handlers[p.Name]
		if !ok {
			return errResp(req.ID, codeInvalidParams, "unknown tool: "+p.Name)
		}
		content, isErr := h(ctx, p.Arguments)
		return &rpcResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{
			"content": []map[string]any{{"type": "text", "text": content}},
			"isError": isErr,
		}}

	default:
		if isNotification {
			return nil
		}
		return errResp(req.ID, codeMethodNotFound, "method not found: "+req.Method)
	}
}
