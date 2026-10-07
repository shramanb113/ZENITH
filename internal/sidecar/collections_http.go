package sidecar

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/collections"
	"github.com/shramanb113/ZENITH/pkg/zenith"
)

const adminBodyCap = 64 << 10

// registerCollections attaches the persistent-collection routes to mux.
// Called from Register only when Config.Collections is set.
func (s *Server) registerCollections(mux *http.ServeMux) {
	mux.Handle("POST /v1/collections", s.adminGuard(s.createCollection))
	mux.Handle("GET /v1/collections", s.adminGuard(s.listCollections))
	mux.Handle("DELETE /v1/collections/{id}", s.adminGuard(s.deleteCollection))
	mux.Handle("POST /v1/collections/{id}/rotate-key", s.adminGuard(s.rotateKey))

	mux.Handle("PUT /v1/collections/{id}/docs", s.colGuard(s.putCollectionDocs))
	mux.Handle("GET /v1/collections/{id}/docs/{doc...}", s.colGuard(s.getCollectionDoc))
	mux.Handle("DELETE /v1/collections/{id}/docs/{doc...}", s.colGuard(s.deleteCollectionDoc))
	mux.Handle("POST /v1/collections/{id}/search", s.colGuard(s.searchCollection))
	mux.Handle("GET /v1/collections/{id}/stats", s.colGuard(s.statCollection))
}

type colErrBody struct {
	Error string `json:"error"`
	Code  string `json:"code"`
}

func writeColErr(w http.ResponseWriter, code int, errCode, msg string) {
	writeJSON(w, code, colErrBody{Error: msg, Code: errCode})
}

// colErr maps an error from the collections package or pkg/zenith to the
// wire error shape. The catch-all branch logs err (never request text, since
// err never carries doc or query bodies) before reporting a generic 500.
func (s *Server) colErr(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, collections.ErrInvalidID):
		writeColErr(w, http.StatusBadRequest, "invalid_id", err.Error())
	case errors.Is(err, collections.ErrNotFound):
		writeColErr(w, http.StatusNotFound, "not_found", err.Error())
	case errors.Is(err, collections.ErrDocNotFound):
		writeColErr(w, http.StatusNotFound, "doc_not_found", err.Error())
	case errors.Is(err, collections.ErrExists):
		writeColErr(w, http.StatusConflict, "exists", err.Error())
	case errors.Is(err, zenith.ErrEmbedderMismatch):
		writeColErr(w, http.StatusConflict, "embedder_mismatch", err.Error())
	case errors.Is(err, zenith.ErrIncompatibleVersion):
		writeColErr(w, http.StatusConflict, "incompatible_format", err.Error())
	case errors.Is(err, collections.ErrQuota):
		writeColErr(w, http.StatusForbidden, "quota_exceeded", err.Error())
	case errors.Is(err, collections.ErrTooManyCollections):
		writeColErr(w, http.StatusForbidden, "collection_limit", err.Error())
	case errors.Is(err, collections.ErrShuttingDown), errors.Is(err, zenith.ErrLocked), errors.Is(err, zenith.ErrClosed):
		writeColErr(w, http.StatusServiceUnavailable, "unavailable", err.Error())
	case errors.Is(err, zenith.ErrInvalidID), errors.Is(err, zenith.ErrIDTooLong),
		errors.Is(err, zenith.ErrEmptyDocument), errors.Is(err, zenith.ErrInvalidAttrs):
		writeColErr(w, http.StatusBadRequest, "invalid_request", err.Error())
	default:
		s.cfg.Log.Error("collections: request failed", "error", err)
		writeColErr(w, http.StatusInternalServerError, "internal", "internal error")
	}
}

func decodeCol(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeColErr(w, http.StatusRequestEntityTooLarge, "body_too_large", "body too large")
		} else {
			writeColErr(w, http.StatusBadRequest, "invalid_json", "invalid JSON body")
		}
		return false
	}
	return true
}

func (s *Server) adminKeyOK(r *http.Request) bool {
	return s.cfg.Key == "" || subtle.ConstantTimeCompare([]byte(r.Header.Get("X-Zenith-Key")), []byte(s.cfg.Key)) == 1
}

// adminGuard protects the management routes (create/list/delete/rotate-key).
// Open when Config.Key is empty, matching the ephemeral routes' behavior.
func (s *Server) adminGuard(h http.HandlerFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !s.adminKeyOK(r) {
			writeColErr(w, http.StatusUnauthorized, "unauthorized", "unauthorized")
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, adminBodyCap)
		h(w, r)
	})
}

// colGuard protects the data-plane routes. The admin key always works; a
// collection's own key works only for that collection. CheckKey returns
// false for a missing collection, so an unauthenticated or wrongly-keyed
// request to a nonexistent id gets the same 401 as a real one — only an
// admin caller can tell the difference (via the 404 that follows).
func (s *Server) colGuard(h http.HandlerFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("id")
		if !collections.ValidID(id) {
			writeColErr(w, http.StatusBadRequest, "invalid_id", "invalid collection id")
			return
		}
		isAdmin := s.cfg.Key != "" && subtle.ConstantTimeCompare([]byte(r.Header.Get("X-Zenith-Key")), []byte(s.cfg.Key)) == 1
		if !isAdmin && !s.cfg.Collections.CheckKey(id, r.Header.Get("X-Zenith-Key")) {
			writeColErr(w, http.StatusUnauthorized, "unauthorized", "unauthorized")
			return
		}
		maxBody, err := s.cfg.Collections.MaxBody(id)
		if err != nil {
			if isAdmin {
				s.colErr(w, err)
			} else {
				writeColErr(w, http.StatusUnauthorized, "unauthorized", "unauthorized")
			}
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, maxBody)
		h(w, r)
	})
}

func (s *Server) createCollection(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID           string `json:"id"`
		MaxDocs      int    `json:"max_docs"`
		MaxBodyBytes int64  `json:"max_body_bytes"`
	}
	if !decodeCol(w, r, &req) {
		return
	}
	if req.ID == "" {
		writeColErr(w, http.StatusBadRequest, "invalid_request", "id is required")
		return
	}
	info, key, err := s.cfg.Collections.Create(req.ID, collections.CreateOptions{
		MaxDocs: req.MaxDocs, MaxBodyBytes: req.MaxBodyBytes,
	})
	if err != nil {
		switch {
		case errors.Is(err, collections.ErrInvalidID):
			writeColErr(w, http.StatusBadRequest, "invalid_id", err.Error())
		case errors.Is(err, collections.ErrExists):
			writeColErr(w, http.StatusConflict, "exists", err.Error())
		case errors.Is(err, collections.ErrTooManyCollections):
			writeColErr(w, http.StatusForbidden, "collection_limit", err.Error())
		case errors.Is(err, collections.ErrShuttingDown):
			writeColErr(w, http.StatusServiceUnavailable, "unavailable", err.Error())
		default:
			writeColErr(w, http.StatusBadRequest, "invalid_request", err.Error())
		}
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{
		"id": info.ID, "key": key, "embedder": info.Embedder,
		"max_docs": info.MaxDocs, "max_body_bytes": info.MaxBodyBytes, "created_at": info.CreatedAt,
	})
}

func (s *Server) listCollections(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"collections": s.cfg.Collections.List()})
}

func (s *Server) deleteCollection(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !collections.ValidID(id) {
		writeColErr(w, http.StatusBadRequest, "invalid_id", "invalid collection id")
		return
	}
	if err := s.cfg.Collections.Delete(id); err != nil {
		s.colErr(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) rotateKey(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !collections.ValidID(id) {
		writeColErr(w, http.StatusBadRequest, "invalid_id", "invalid collection id")
		return
	}
	key, err := s.cfg.Collections.RotateKey(id)
	if err != nil {
		s.colErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"key": key})
}

type colDocIn struct {
	ID    string         `json:"id"`
	Text  string         `json:"text"`
	Attrs map[string]any `json:"attrs,omitempty"`
}

func (s *Server) putCollectionDocs(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var req struct {
		Docs []colDocIn `json:"docs"`
	}
	if !decodeCol(w, r, &req) {
		return
	}
	if len(req.Docs) == 0 || len(req.Docs) > s.cfg.MaxBatch {
		writeColErr(w, http.StatusBadRequest, "invalid_request", "docs must contain 1..MaxBatch entries")
		return
	}
	docs := make(map[string]string, len(req.Docs))
	attrs := make(map[string]zenith.Attrs)
	for _, d := range req.Docs {
		if d.ID == "" || strings.TrimSpace(d.Text) == "" {
			writeColErr(w, http.StatusBadRequest, "invalid_request", "every doc needs a non-empty id and text")
			return
		}
		if _, dup := docs[d.ID]; dup {
			writeColErr(w, http.StatusBadRequest, "invalid_request", "duplicate doc id")
			return
		}
		docs[d.ID] = d.Text
		if len(d.Attrs) > 0 {
			attrs[d.ID] = zenith.Attrs(d.Attrs)
		}
	}

	start := time.Now()
	res, err := s.cfg.Collections.Upsert(r.Context(), id, docs, attrs)
	if err != nil {
		s.colErr(w, err)
		return
	}
	s.cfg.Log.Info("collection upsert", "id", id, "docs", len(docs), "ms", time.Since(start).Milliseconds())
	writeJSON(w, http.StatusOK, map[string]int{"upserted": res.Upserted, "new": res.New, "doc_count": res.DocCount})
}

func (s *Server) getCollectionDoc(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	docID := r.PathValue("doc")
	text, err := s.cfg.Collections.GetDoc(r.Context(), id, docID)
	if err != nil {
		s.colErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"id": docID, "text": text})
}

func (s *Server) deleteCollectionDoc(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	docID := r.PathValue("doc")
	if err := s.cfg.Collections.DeleteDoc(r.Context(), id, docID); err != nil {
		s.colErr(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) searchCollection(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var req struct {
		Queries []struct {
			ID   string `json:"id"`
			Text string `json:"text"`
		} `json:"queries"`
		Limit       int             `json:"limit"`
		MinSemantic float64         `json:"min_semantic"`
		Filter      json.RawMessage `json:"filter"`
	}
	if !decodeCol(w, r, &req) {
		return
	}
	var searchOpts []zenith.SearchOption
	if len(req.Filter) > 0 && string(req.Filter) != "null" {
		f, err := zenith.FilterFromJSON(req.Filter)
		if err != nil {
			writeColErr(w, http.StatusBadRequest, "invalid_request", "invalid filter: "+strings.TrimPrefix(err.Error(), "zenith: "))
			return
		}
		searchOpts = append(searchOpts, zenith.WithFilter(f))
	}
	if len(req.Queries) == 0 || len(req.Queries) > s.cfg.MaxQueries {
		writeColErr(w, http.StatusBadRequest, "invalid_request", "queries must contain 1..MaxQueries entries")
		return
	}
	seen := make(map[string]bool, len(req.Queries))
	for _, q := range req.Queries {
		if q.ID == "" || strings.TrimSpace(q.Text) == "" || seen[q.ID] {
			writeColErr(w, http.StatusBadRequest, "invalid_request", "every query needs a unique id and non-empty text")
			return
		}
		seen[q.ID] = true
	}

	limit := req.Limit
	if limit <= 0 {
		limit = 10
	}
	if limit > s.cfg.MaxLimit {
		limit = s.cfg.MaxLimit
	}

	start := time.Now()
	out := make(map[string]queryOut, len(req.Queries))
	spansFor := memoSpans(r.Context(), s.cfg.Collections, id, s.ana)
	for _, q := range req.Queries {
		res, err := s.cfg.Collections.Search(r.Context(), id, q.Text,
			append([]zenith.SearchOption{zenith.Explain(), zenith.Limit(limit)}, searchOpts...)...)
		if err != nil {
			s.colErr(w, err)
			return
		}
		out[q.ID] = buildQueryOut(res, spansFor, req.MinSemantic)
	}
	s.cfg.Log.Info("collection searched", "id", id, "queries", len(req.Queries), "ms", time.Since(start).Milliseconds())
	writeJSON(w, http.StatusOK, map[string]any{"results": out})
}

func (s *Server) statCollection(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	info, err := s.cfg.Collections.Stat(id)
	if err != nil {
		s.colErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, info)
}

// memoSpans returns a spansFor function (see buildQueryOut) that resolves a
// collection document's term spans by fetching its text and analysing it at
// most once per document, no matter how many queries or matched terms in
// this request reference it.
func memoSpans(ctx context.Context, mgr *collections.Manager, colID string, ana *analysis.StandardAnalyzer) func(docID string) map[string]span {
	cache := make(map[string]map[string]span)
	return func(docID string) map[string]span {
		if m, ok := cache[docID]; ok {
			return m
		}
		text, err := mgr.GetDoc(ctx, colID, docID)
		var m map[string]span
		if err == nil {
			m = firstSpans(ana, text)
		}
		cache[docID] = m
		return m
	}
}

// firstSpans returns, for each analysed term in text, the character span of
// its first occurrence.
func firstSpans(ana *analysis.StandardAnalyzer, text string) map[string]span {
	m := map[string]span{}
	for _, sp := range ana.AnalyzeSpans(text) {
		if _, seen := m[sp.Term]; !seen {
			m[sp.Term] = span{sp.Start, sp.End, sp.Pos}
		}
	}
	return m
}
