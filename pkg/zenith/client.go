package zenith

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Client is a thin HTTP client for a running ZENITH sidecar's persistent
// collections API (internal/sidecar's /v1/collections/* routes, documented
// in CLAUDE.md under "internal/sidecar"). It is the supported way for an
// out-of-process application — in any language, but this is the Go
// convenience wrapper — to talk to ZENITH over the network instead of
// linking this package's embedded DB directly.
//
// A Client is safe for concurrent use by multiple goroutines; it holds no
// mutable state beyond its *http.Client.
type Client struct {
	baseURL string
	apiKey  string
	hc      *http.Client
}

// ClientOption configures optional Client behaviour.
type ClientOption func(*Client)

// WithHTTPClient overrides the *http.Client used for requests (default:
// http.DefaultClient). Use this to set timeouts, transports, or proxies.
func WithHTTPClient(hc *http.Client) ClientOption {
	return func(c *Client) {
		if hc != nil {
			c.hc = hc
		}
	}
}

// NewClient builds a Client for the sidecar at baseURL (e.g.
// "http://localhost:7700"). apiKey is sent as the X-Zenith-Key header on
// every request: pass the sidecar's admin key for management calls
// (create/list/delete collection, rotate-key), or a single collection's own
// key for data-plane calls scoped to that collection. Pass "" if the
// sidecar was started without --key.
func NewClient(baseURL, apiKey string, opts ...ClientOption) *Client {
	c := &Client{baseURL: strings.TrimRight(baseURL, "/"), apiKey: apiKey, hc: http.DefaultClient}
	for _, o := range opts {
		o(c)
	}
	return c
}

// APIError is returned for any non-2xx response from the sidecar. Code is
// the machine-readable error code from the JSON error body (see colErrBody
// in internal/sidecar/collections_http.go), e.g. "not_found",
// "quota_exceeded", "invalid_request", "unauthorized".
type APIError struct {
	StatusCode int
	Code       string
	Message    string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("zenith: sidecar returned %d (%s): %s", e.StatusCode, e.Code, e.Message)
}

// doJSON sends method+path with an optional JSON-encoded body and decodes a
// successful JSON response into out (which may be nil to discard the body).
func (c *Client) doJSON(ctx context.Context, method, path string, body, out any) error {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("zenith: encoding request body: %w", err)
		}
		rdr = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, rdr)
	if err != nil {
		return fmt.Errorf("zenith: building request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.apiKey != "" {
		req.Header.Set("X-Zenith-Key", c.apiKey)
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return fmt.Errorf("zenith: request failed: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("zenith: reading response: %w", err)
	}
	if resp.StatusCode >= 300 {
		var eb struct {
			Error string `json:"error"`
			Code  string `json:"code"`
		}
		_ = json.Unmarshal(data, &eb)
		return &APIError{StatusCode: resp.StatusCode, Code: eb.Code, Message: eb.Error}
	}
	if out == nil || len(data) == 0 {
		return nil
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("zenith: decoding response: %w", err)
	}
	return nil
}

// CollectionInfo is a point-in-time snapshot of one collection's state, the
// same shape as internal/collections.Info.
type CollectionInfo struct {
	ID            string    `json:"id"`
	Embedder      string    `json:"embedder"`
	CreatedAt     time.Time `json:"created_at"`
	LastUsed      time.Time `json:"last_used,omitempty"`
	DocCount      int       `json:"doc_count"`
	DocCountExact bool      `json:"doc_count_exact"`
	MaxDocs       int       `json:"max_docs"`
	MaxBodyBytes  int64     `json:"max_body_bytes"`
	Open          bool      `json:"open"`
	DiskBytes     int64     `json:"disk_bytes,omitempty"`
	WALBytes      int64     `json:"wal_bytes,omitempty"`
}

// CreateCollectionOptions configures CreateCollection. Zero values take the
// sidecar's configured defaults.
type CreateCollectionOptions struct {
	MaxDocs      int
	MaxBodyBytes int64
}

// CreateCollection creates a new persistent collection and returns its info
// plus a freshly generated API key (returned only once — the sidecar never
// stores the plaintext key). Requires the admin key.
func (c *Client) CreateCollection(ctx context.Context, id string, opts CreateCollectionOptions) (CollectionInfo, string, error) {
	body := map[string]any{"id": id}
	if opts.MaxDocs > 0 {
		body["max_docs"] = opts.MaxDocs
	}
	if opts.MaxBodyBytes > 0 {
		body["max_body_bytes"] = opts.MaxBodyBytes
	}
	var resp struct {
		CollectionInfo
		Key string `json:"key"`
	}
	if err := c.doJSON(ctx, http.MethodPost, "/v1/collections", body, &resp); err != nil {
		return CollectionInfo{}, "", err
	}
	return resp.CollectionInfo, resp.Key, nil
}

// ListCollections returns every collection's info, sorted by ID. Requires
// the admin key.
func (c *Client) ListCollections(ctx context.Context) ([]CollectionInfo, error) {
	var resp struct {
		Collections []CollectionInfo `json:"collections"`
	}
	if err := c.doJSON(ctx, http.MethodGet, "/v1/collections", nil, &resp); err != nil {
		return nil, err
	}
	return resp.Collections, nil
}

// DeleteCollection permanently deletes a collection and its documents.
// Requires the admin key.
func (c *Client) DeleteCollection(ctx context.Context, id string) error {
	return c.doJSON(ctx, http.MethodDelete, "/v1/collections/"+url.PathEscape(id), nil, nil)
}

// StatCollection returns one collection's current info, including
// disk/WAL size. Works with the admin key or the collection's own key.
func (c *Client) StatCollection(ctx context.Context, id string) (CollectionInfo, error) {
	var info CollectionInfo
	err := c.doJSON(ctx, http.MethodGet, "/v1/collections/"+url.PathEscape(id)+"/stats", nil, &info)
	return info, err
}

// RotateKey replaces a collection's API key and returns the new plaintext
// key. The old key stops working immediately. Requires the admin key.
func (c *Client) RotateKey(ctx context.Context, id string) (string, error) {
	var resp struct {
		Key string `json:"key"`
	}
	err := c.doJSON(ctx, http.MethodPost, "/v1/collections/"+url.PathEscape(id)+"/rotate-key", nil, &resp)
	return resp.Key, err
}

// Document is one document to upsert via UpsertDocuments.
type Document struct {
	ID    string `json:"id"`
	Text  string `json:"text"`
	Attrs Attrs  `json:"attrs,omitempty"`
}

// UpsertResult reports the outcome of UpsertDocuments.
type UpsertResult struct {
	Upserted int `json:"upserted"`
	New      int `json:"new"`
	DocCount int `json:"doc_count"`
}

// UpsertDocuments adds or replaces up to the sidecar's configured batch
// limit of documents in one call. Works with the admin key or the
// collection's own key.
func (c *Client) UpsertDocuments(ctx context.Context, collectionID string, docs []Document) (UpsertResult, error) {
	if len(docs) == 0 {
		return UpsertResult{}, errors.New("zenith: UpsertDocuments requires at least one document")
	}
	var res UpsertResult
	path := "/v1/collections/" + url.PathEscape(collectionID) + "/docs"
	err := c.doJSON(ctx, http.MethodPut, path, map[string]any{"docs": docs}, &res)
	return res, err
}

// GetDocument fetches one document's full text by ID.
func (c *Client) GetDocument(ctx context.Context, collectionID, docID string) (string, error) {
	var resp struct {
		Text string `json:"text"`
	}
	path := "/v1/collections/" + url.PathEscape(collectionID) + "/docs/" + url.PathEscape(docID)
	err := c.doJSON(ctx, http.MethodGet, path, nil, &resp)
	return resp.Text, err
}

// DeleteDocument deletes one document by ID. Deleting a nonexistent
// document returns a *APIError with Code "doc_not_found".
func (c *Client) DeleteDocument(ctx context.Context, collectionID, docID string) error {
	path := "/v1/collections/" + url.PathEscape(collectionID) + "/docs/" + url.PathEscape(docID)
	return c.doJSON(ctx, http.MethodDelete, path, nil, nil)
}

// SearchQuery is one named query in a Search call. Running several queries
// in one call lets the sidecar amortise per-document work (span resolution)
// across them.
type SearchQuery struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

// SearchTerm records how one analysed query term was matched in a result's
// document, including its character span in the original text.
type SearchTerm struct {
	Term    string `json:"term"`
	Matched string `json:"matched"`
	Dist    int    `json:"dist"`
	Synonym bool   `json:"synonym"`
	Start   int    `json:"start"`
	End     int    `json:"end"`
	Pos     int    `json:"pos"`
}

// SearchHit is one matched document.
type SearchHit struct {
	ID       string       `json:"id"`
	Score    float64      `json:"score"`
	Lexical  float64      `json:"lexical"`
	Semantic float64      `json:"semantic"`
	Terms    []SearchTerm `json:"terms"`
}

// SearchResult is the result of one SearchQuery.
type SearchResult struct {
	QueryTerms []string    `json:"query_terms"`
	Hits       []SearchHit `json:"hits"`
}

// collectionSearchRequest is the wire shape of POST .../search, matching
// the request struct decoded in internal/sidecar/collections_http.go's
// searchCollection. Only the fields that handler currently reads are
// included: limit, min_semantic, and filter. The sidecar does not yet
// accept per-query ranking weights, a sort field, or an offset on this
// endpoint (unlike pkg/zenith.Search's WithWeights/SortBy/WithFilter, which
// are library-only); CollectionSearchOption is written so adding one later
// (WithSearchWeights, WithSearchSort, WithSearchOffset, ...) is a pure
// addition here with no signature break.
type collectionSearchRequest struct {
	Queries     []SearchQuery   `json:"queries"`
	Limit       int             `json:"limit,omitempty"`
	MinSemantic float64         `json:"min_semantic,omitempty"`
	Filter      json.RawMessage `json:"filter,omitempty"`
}

// CollectionSearchOption configures a Search call.
type CollectionSearchOption func(*collectionSearchRequest)

// WithSearchLimit caps the number of hits returned per query (sidecar
// default: 10; sidecar-side max is configurable, default 100).
func WithSearchLimit(n int) CollectionSearchOption {
	return func(r *collectionSearchRequest) { r.Limit = n }
}

// WithMinSemantic drops hits with no matched lexical term and a semantic
// score below threshold.
func WithMinSemantic(threshold float64) CollectionSearchOption {
	return func(r *collectionSearchRequest) { r.MinSemantic = threshold }
}

// WithSearchFilter restricts results to documents whose attrs match f (see
// Eq, In, Range, Prefix, Contains, Exists, And, Or, Not, and FilterFromJSON).
func WithSearchFilter(f Filter) CollectionSearchOption {
	return func(r *collectionSearchRequest) {
		if b, err := f.JSON(); err == nil {
			r.Filter = b
		}
	}
}

// Search runs one or more named queries against a collection and returns a
// map keyed by each query's ID. Works with the admin key or the
// collection's own key.
func (c *Client) Search(ctx context.Context, collectionID string, queries []SearchQuery, opts ...CollectionSearchOption) (map[string]SearchResult, error) {
	if len(queries) == 0 {
		return nil, errors.New("zenith: Search requires at least one query")
	}
	req := collectionSearchRequest{Queries: queries}
	for _, o := range opts {
		o(&req)
	}
	var resp struct {
		Results map[string]SearchResult `json:"results"`
	}
	path := "/v1/collections/" + url.PathEscape(collectionID) + "/search"
	if err := c.doJSON(ctx, http.MethodPost, path, req, &resp); err != nil {
		return nil, err
	}
	return resp.Results, nil
}
