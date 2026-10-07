// Package sidecar serves short-lived, per-request ZENITH namespaces over HTTP/JSON. Each namespace
// is an in-memory index built from one PUT; searches return raw Explain signals with character
// spans. It never logs document or query text.
package sidecar

import (
	"container/list"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/collections"
	"github.com/shramanb113/ZENITH/pkg/zenith"
)

// Config holds sidecar limits and identity. Zero values take the defaults noted.
type Config struct {
	Key        string          // required in X-Zenith-Key on /v1/* when non-empty
	TTL        time.Duration   // idle namespace lifetime; default 10m
	MaxNS      int             // namespace cap with LRU eviction; default 200
	MaxBody    int64           // request body cap in bytes; default 1 MiB
	MaxDocs    int             // documents per namespace; default 2000
	MaxQueries int             // queries per search; default 300
	Version    string          // ZENITH version reported by /healthz
	Model      string          // embedding model id, "none" or "deterministic"
	Synonyms   string          // short hash of the loaded synonyms file; "" if none
	Embedder   zenith.Embedder // shared across namespaces; nil means BM25-only
	Now        func() time.Time
	Log        *slog.Logger

	// Collections serves persistent, multi-tenant collections at
	// /v1/collections/*. nil (the default) leaves those routes unregistered;
	// the ephemeral /v1/ns/* routes above are unaffected either way.
	Collections *collections.Manager
	MaxBatch    int // docs per PUT .../docs; default 1000
	MaxLimit    int // max search limit; default 100
}

type span struct{ start, end, pos int }

type namespace struct {
	db    *zenith.DB
	ndocs int
	spans map[string]map[string]span // doc id → analysed term → first span
	used  time.Time
	elem  *list.Element
}

// Server is an http.Handler factory plus the namespace table. Safe for concurrent use.
type Server struct {
	cfg Config
	ana *analysis.StandardAnalyzer
	mu  sync.Mutex
	ns  map[string]*namespace
	lru *list.List // front = most recently used; values are namespace names
}

var nsName = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

// New builds a Server, filling defaults.
func New(cfg Config) *Server {
	if cfg.TTL <= 0 {
		cfg.TTL = 10 * time.Minute
	}
	if cfg.MaxNS <= 0 {
		cfg.MaxNS = 200
	}
	if cfg.MaxBody <= 0 {
		cfg.MaxBody = 1 << 20
	}
	if cfg.MaxDocs <= 0 {
		cfg.MaxDocs = 2000
	}
	if cfg.MaxQueries <= 0 {
		cfg.MaxQueries = 300
	}
	if cfg.MaxBatch <= 0 {
		cfg.MaxBatch = 1000
	}
	if cfg.MaxLimit <= 0 {
		cfg.MaxLimit = 100
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.Log == nil {
		cfg.Log = slog.Default()
	}
	return &Server{cfg: cfg, ana: analysis.NewStandardAnalyzer(), ns: map[string]*namespace{}, lru: list.New()}
}

// Handler returns the HTTP routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	s.Register(mux)
	return mux
}

// Register attaches every route this Server serves to mux: the ephemeral
// /v1/ns/* namespace routes, and, when Config.Collections is set, the
// persistent /v1/collections/* routes (collections_http.go).
func (s *Server) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /healthz", s.health)
	mux.Handle("PUT /v1/ns/{ns}/docs", s.guard(s.putDocs))
	mux.Handle("POST /v1/ns/{ns}/search", s.guard(s.search))
	mux.Handle("DELETE /v1/ns/{ns}", s.guard(s.deleteNS))
	if s.cfg.Collections != nil {
		s.registerCollections(mux)
	}
}

// Run sweeps expired namespaces every interval until ctx is done.
func (s *Server) Run(ctx context.Context, interval time.Duration) {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.Sweep()
		}
	}
}

// Sweep closes namespaces idle for longer than TTL.
func (s *Server) Sweep() {
	s.mu.Lock()
	defer s.mu.Unlock()
	cutoff := s.cfg.Now().Add(-s.cfg.TTL)
	for name, n := range s.ns {
		if n.used.Before(cutoff) {
			s.dropLocked(name)
		}
	}
}

// Close drops every namespace.
func (s *Server) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for name := range s.ns {
		s.dropLocked(name)
	}
}

func (s *Server) dropLocked(name string) {
	n, ok := s.ns[name]
	if !ok {
		return
	}
	delete(s.ns, name)
	s.lru.Remove(n.elem)
	_ = n.db.Close()
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

func (s *Server) guard(h http.HandlerFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.Key != "" &&
			subtle.ConstantTimeCompare([]byte(r.Header.Get("X-Zenith-Key")), []byte(s.cfg.Key)) != 1 {
			writeErr(w, http.StatusUnauthorized, "unauthorized")
			return
		}
		if !nsName.MatchString(r.PathValue("ns")) {
			writeErr(w, http.StatusBadRequest, "invalid namespace name")
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, s.cfg.MaxBody)
		h(w, r)
	})
}

func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeErr(w, http.StatusRequestEntityTooLarge, "body too large")
		} else {
			writeErr(w, http.StatusBadRequest, "invalid JSON body")
		}
		return false
	}
	return true
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	s.mu.Lock()
	n := len(s.ns)
	s.mu.Unlock()
	out := map[string]any{"status": "ok", "version": s.cfg.Version, "model": s.cfg.Model,
		"synonyms": s.cfg.Synonyms, "namespaces": n}
	if s.cfg.Collections != nil {
		total, open := s.cfg.Collections.Counts()
		out["collections"] = total
		out["collections_open"] = open
	}
	writeJSON(w, http.StatusOK, out)
}

type docIn struct {
	ID   string `json:"id"`
	Text string `json:"text"`
	// Attrs is optional metadata (string, number or bool values) that a search's
	// "filter" can test.
	Attrs map[string]any `json:"attrs,omitempty"`
}

func (s *Server) putDocs(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("ns")
	var req struct {
		Docs []docIn `json:"docs"`
	}
	if !decode(w, r, &req) {
		return
	}
	if len(req.Docs) == 0 || len(req.Docs) > s.cfg.MaxDocs {
		writeErr(w, http.StatusBadRequest, "docs must contain 1..MaxDocs entries")
		return
	}
	docs := make(map[string]string, len(req.Docs))
	attrs := make(map[string]zenith.Attrs)
	for _, d := range req.Docs {
		if d.ID == "" || strings.TrimSpace(d.Text) == "" {
			writeErr(w, http.StatusBadRequest, "every doc needs a non-empty id and text")
			return
		}
		if _, dup := docs[d.ID]; dup {
			writeErr(w, http.StatusBadRequest, "duplicate doc id")
			return
		}
		docs[d.ID] = d.Text
		if len(d.Attrs) > 0 {
			attrs[d.ID] = zenith.Attrs(d.Attrs)
		}
	}

	opts := []zenith.Option{zenith.WithoutWordVectors(), zenith.WithLimit(s.cfg.MaxDocs)}
	if s.cfg.Embedder != nil {
		opts = append(opts, zenith.WithEmbedder(s.cfg.Embedder))
	} else {
		opts = append(opts, zenith.WithBM25Only())
	}
	start := time.Now()
	db, err := zenith.Open(":memory:", opts...)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "open failed")
		return
	}
	if err := db.AddBatchWithAttrs(r.Context(), docs, attrs); err != nil {
		_ = db.Close()
		if errors.Is(err, zenith.ErrInvalidAttrs) {
			writeErr(w, http.StatusBadRequest, "attrs must be non-empty keys with string, number or bool values")
			return
		}
		writeErr(w, http.StatusInternalServerError, "index failed")
		return
	}
	spans := make(map[string]map[string]span, len(docs))
	for id, text := range docs {
		spans[id] = firstSpans(s.ana, text)
	}

	s.mu.Lock()
	s.dropLocked(name)
	for len(s.ns) >= s.cfg.MaxNS {
		oldest := s.lru.Back()
		s.dropLocked(oldest.Value.(string))
	}
	n := &namespace{db: db, ndocs: len(docs), spans: spans, used: s.cfg.Now()}
	n.elem = s.lru.PushFront(name)
	s.ns[name] = n
	s.mu.Unlock()

	s.cfg.Log.Info("namespace indexed", "ns", name, "docs", len(docs), "ms", time.Since(start).Milliseconds())
	writeJSON(w, http.StatusOK, map[string]int{"indexed": len(docs)})
}

func (s *Server) touch(name string) *namespace {
	s.mu.Lock()
	defer s.mu.Unlock()
	n, ok := s.ns[name]
	if !ok {
		return nil
	}
	n.used = s.cfg.Now()
	s.lru.MoveToFront(n.elem)
	return n
}

type termOut struct {
	Term    string `json:"term"`
	Matched string `json:"matched"`
	Dist    int    `json:"dist"`
	Synonym bool   `json:"synonym"`
	Start   int    `json:"start"`
	End     int    `json:"end"`
	Pos     int    `json:"pos"`
}

type hitOut struct {
	ID       string    `json:"id"`
	Score    float64   `json:"score"`
	Lexical  float64   `json:"lexical"`
	Semantic float64   `json:"semantic"`
	Terms    []termOut `json:"terms"`
}

type queryOut struct {
	QueryTerms []string `json:"query_terms"`
	Hits       []hitOut `json:"hits"`
}

func (s *Server) search(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("ns")
	var req struct {
		Queries []struct {
			ID   string `json:"id"`
			Text string `json:"text"`
		} `json:"queries"`
		Limit       int     `json:"limit"`
		MinSemantic float64 `json:"min_semantic"`
		// Filter restricts the search to documents whose attrs match; see
		// zenith.FilterFromJSON for the format.
		Filter json.RawMessage `json:"filter"`
	}
	if !decode(w, r, &req) {
		return
	}
	var searchOpts []zenith.SearchOption
	if len(req.Filter) > 0 && string(req.Filter) != "null" {
		f, err := zenith.FilterFromJSON(req.Filter)
		if err != nil {
			writeErr(w, http.StatusBadRequest, "invalid filter: "+strings.TrimPrefix(err.Error(), "zenith: "))
			return
		}
		searchOpts = append(searchOpts, zenith.WithFilter(f))
	}
	if len(req.Queries) == 0 || len(req.Queries) > s.cfg.MaxQueries {
		writeErr(w, http.StatusBadRequest, "queries must contain 1..MaxQueries entries")
		return
	}
	seen := map[string]bool{}
	for _, q := range req.Queries {
		if q.ID == "" || strings.TrimSpace(q.Text) == "" || seen[q.ID] {
			writeErr(w, http.StatusBadRequest, "every query needs a unique id and non-empty text")
			return
		}
		seen[q.ID] = true
	}
	n := s.touch(name)
	if n == nil {
		writeErr(w, http.StatusNotFound, "unknown namespace")
		return
	}
	limit := req.Limit
	if limit <= 0 || limit > n.ndocs {
		limit = n.ndocs
	}

	start := time.Now()
	out := make(map[string]queryOut, len(req.Queries))
	spansFor := func(docID string) map[string]span { return n.spans[docID] }
	for _, q := range req.Queries {
		res, err := n.db.Search(r.Context(), q.Text, append([]zenith.SearchOption{zenith.Explain(), zenith.Limit(limit)}, searchOpts...)...)
		if err != nil {
			if errors.Is(err, zenith.ErrClosed) {
				writeErr(w, http.StatusNotFound, "namespace closed")
				return
			}
			writeErr(w, http.StatusInternalServerError, "search failed")
			return
		}
		out[q.ID] = buildQueryOut(res, spansFor, req.MinSemantic)
	}
	s.cfg.Log.Info("namespace searched", "ns", name, "queries", len(req.Queries), "ms", time.Since(start).Milliseconds())
	writeJSON(w, http.StatusOK, map[string]any{"results": out})
}

// buildQueryOut converts raw Explain results into the wire shape, resolving
// each matched term's character span via spansFor (called at most once per
// document ID actually referenced by res, since callers that memoise it do
// the expensive lookup — GetDoc + AnalyzeSpans for collections — only once
// per document instead of once per matched term).
func buildQueryOut(res []zenith.Result, spansFor func(docID string) map[string]span, minSemantic float64) queryOut {
	qo := queryOut{QueryTerms: []string{}, Hits: []hitOut{}}
	for _, res1 := range res {
		sig := res1.Signals
		if sig == nil {
			continue
		}
		if len(qo.QueryTerms) == 0 && len(sig.QueryTerms) > 0 {
			qo.QueryTerms = sig.QueryTerms
		}
		spans := spansFor(res1.ID)
		terms := make([]termOut, 0, len(sig.Terms))
		for _, tm := range sig.Terms {
			sp, ok := spans[tm.Matched]
			if !ok {
				continue
			}
			terms = append(terms, termOut{tm.Term, tm.Matched, tm.Dist, tm.Synonym, sp.start, sp.end, sp.pos})
		}
		if len(terms) == 0 && sig.Semantic < minSemantic {
			continue
		}
		qo.Hits = append(qo.Hits, hitOut{res1.ID, res1.Score, sig.Lexical, sig.Semantic, terms})
	}
	return qo
}

func (s *Server) deleteNS(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	s.dropLocked(r.PathValue("ns"))
	s.mu.Unlock()
	w.WriteHeader(http.StatusNoContent)
}
