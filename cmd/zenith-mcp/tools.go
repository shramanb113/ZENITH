package main

// Tool surface exposed to MCP clients: four small, well-documented tools
// backed directly by internal/collections.Manager (no HTTP hop — this
// binary embeds the collection engine in-process, matching ZENITH's
// "zero-infra" pitch: there is no separate `zenith serve` to run first).

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/shramanb113/ZENITH/internal/collections"
	"github.com/shramanb113/ZENITH/pkg/zenith"
)

func buildTools(mgr *collections.Manager) ([]toolDef, map[string]toolHandler) {
	tools := []toolDef{
		{
			Name:        "list_collections",
			Description: "List every ZENITH collection managed by this server, with each collection's document count and creation time. Takes no arguments.",
			InputSchema: map[string]any{
				"type":                 "object",
				"properties":           map[string]any{},
				"additionalProperties": false,
			},
		},
		{
			Name: "search_collection",
			Description: "Run a hybrid lexical+semantic search over one collection's documents and return the best matches with their scores. " +
				"Use this to retrieve relevant memory/context before answering a question.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"collection": map[string]any{"type": "string", "description": "Collection ID to search."},
					"query":      map[string]any{"type": "string", "description": "Natural-language search query."},
					"limit":      map[string]any{"type": "integer", "description": "Maximum number of hits to return (default 10)."},
					"filter": map[string]any{
						"type":        "object",
						"description": "Optional ZENITH FilterSpec restricting results to documents whose attrs match, e.g. {\"op\":\"eq\",\"field\":\"tenant\",\"value\":\"acme\"}.",
					},
				},
				"required":             []string{"collection", "query"},
				"additionalProperties": false,
			},
		},
		{
			Name: "upsert_document",
			Description: "Add or overwrite a document in a collection, identified by an ID you choose. Creates the collection " +
				"automatically on first use — no separate setup step is required.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"collection": map[string]any{"type": "string", "description": "Collection ID (created automatically if it doesn't exist)."},
					"id":         map[string]any{"type": "string", "description": "Document ID; upserting an existing ID replaces its text."},
					"text":       map[string]any{"type": "string", "description": "Document text to index."},
					"attrs": map[string]any{
						"type":        "object",
						"description": "Optional flat metadata (string, number, or bool values only) usable later as a search filter.",
					},
				},
				"required":             []string{"collection", "id", "text"},
				"additionalProperties": false,
			},
		},
		{
			Name:        "get_document",
			Description: "Fetch a document's full text from a collection by its ID.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"collection": map[string]any{"type": "string", "description": "Collection ID."},
					"id":         map[string]any{"type": "string", "description": "Document ID."},
				},
				"required":             []string{"collection", "id"},
				"additionalProperties": false,
			},
		},
	}

	handlers := map[string]toolHandler{
		"list_collections":  handleListCollections(mgr),
		"search_collection": handleSearchCollection(mgr),
		"upsert_document":   handleUpsertDocument(mgr),
		"get_document":      handleGetDocument(mgr),
	}
	return tools, handlers
}

func handleListCollections(mgr *collections.Manager) toolHandler {
	return func(_ context.Context, _ json.RawMessage) (string, bool) {
		list := mgr.List()
		if len(list) == 0 {
			return "No collections exist yet. upsert_document creates one automatically on first use.", false
		}
		sort.Slice(list, func(i, j int) bool { return list[i].ID < list[j].ID })
		var b strings.Builder
		fmt.Fprintf(&b, "%d collection(s):\n", len(list))
		for _, c := range list {
			fmt.Fprintf(&b, "- %s: %d document(s), embedder=%s, created=%s\n",
				c.ID, c.DocCount, c.Embedder, c.CreatedAt.Format("2006-01-02T15:04:05Z07:00"))
		}
		return b.String(), false
	}
}

func handleSearchCollection(mgr *collections.Manager) toolHandler {
	return func(ctx context.Context, args json.RawMessage) (string, bool) {
		var req struct {
			Collection string          `json:"collection"`
			Query      string          `json:"query"`
			Limit      int             `json:"limit"`
			Filter     json.RawMessage `json:"filter"`
		}
		if err := json.Unmarshal(args, &req); err != nil {
			return "invalid arguments: " + err.Error(), true
		}
		if req.Collection == "" || strings.TrimSpace(req.Query) == "" {
			return "both \"collection\" and \"query\" are required", true
		}
		limit := req.Limit
		if limit <= 0 {
			limit = 10
		}
		opts := []zenith.SearchOption{zenith.Limit(limit)}
		if len(req.Filter) > 0 && string(req.Filter) != "null" {
			f, err := zenith.FilterFromJSON(req.Filter)
			if err != nil {
				return "invalid filter: " + err.Error(), true
			}
			opts = append(opts, zenith.WithFilter(f))
		}
		results, err := mgr.Search(ctx, req.Collection, req.Query, opts...)
		if err != nil {
			return describeErr(err), true
		}
		if len(results) == 0 {
			return "No matches.", false
		}
		var b strings.Builder
		fmt.Fprintf(&b, "%d result(s):\n", len(results))
		for i, r := range results {
			fmt.Fprintf(&b, "%d. %s (score=%.4f)\n", i+1, r.ID, r.Score)
		}
		return b.String(), false
	}
}

func handleUpsertDocument(mgr *collections.Manager) toolHandler {
	return func(ctx context.Context, args json.RawMessage) (string, bool) {
		var req struct {
			Collection string         `json:"collection"`
			ID         string         `json:"id"`
			Text       string         `json:"text"`
			Attrs      map[string]any `json:"attrs"`
		}
		if err := json.Unmarshal(args, &req); err != nil {
			return "invalid arguments: " + err.Error(), true
		}
		if req.Collection == "" || req.ID == "" || strings.TrimSpace(req.Text) == "" {
			return "\"collection\", \"id\", and non-empty \"text\" are required", true
		}
		docs := map[string]string{req.ID: req.Text}
		var attrs map[string]zenith.Attrs
		if len(req.Attrs) > 0 {
			attrs = map[string]zenith.Attrs{req.ID: zenith.Attrs(req.Attrs)}
		}

		res, err := mgr.Upsert(ctx, req.Collection, docs, attrs)
		if errors.Is(err, collections.ErrNotFound) {
			// Zero-infra: auto-provision the collection on first write
			// instead of requiring a separate create step.
			if _, _, cerr := mgr.Create(req.Collection, collections.CreateOptions{}); cerr != nil && !errors.Is(cerr, collections.ErrExists) {
				return "failed to auto-create collection " + req.Collection + ": " + cerr.Error(), true
			}
			res, err = mgr.Upsert(ctx, req.Collection, docs, attrs)
		}
		if err != nil {
			return describeErr(err), true
		}
		return fmt.Sprintf("upserted document %q into %q (collection now has %d document(s))", req.ID, req.Collection, res.DocCount), false
	}
}

func handleGetDocument(mgr *collections.Manager) toolHandler {
	return func(ctx context.Context, args json.RawMessage) (string, bool) {
		var req struct {
			Collection string `json:"collection"`
			ID         string `json:"id"`
		}
		if err := json.Unmarshal(args, &req); err != nil {
			return "invalid arguments: " + err.Error(), true
		}
		if req.Collection == "" || req.ID == "" {
			return "both \"collection\" and \"id\" are required", true
		}
		text, err := mgr.GetDoc(ctx, req.Collection, req.ID)
		if err != nil {
			return describeErr(err), true
		}
		return text, false
	}
}

// describeErr turns a collections/zenith package error into a short,
// model-readable message without leaking internal error wrapping.
func describeErr(err error) string {
	switch {
	case errors.Is(err, collections.ErrNotFound):
		return "collection not found"
	case errors.Is(err, collections.ErrDocNotFound):
		return "document not found"
	case errors.Is(err, collections.ErrQuota):
		return "collection document quota exceeded"
	case errors.Is(err, collections.ErrInvalidID):
		return "invalid collection id"
	case errors.Is(err, zenith.ErrInvalidID), errors.Is(err, zenith.ErrIDTooLong):
		return "invalid document id"
	case errors.Is(err, zenith.ErrEmptyDocument):
		return "document text is empty"
	case errors.Is(err, zenith.ErrInvalidAttrs):
		return "invalid attrs: keys must be non-empty and values string, bool, or number"
	default:
		return "error: " + err.Error()
	}
}
