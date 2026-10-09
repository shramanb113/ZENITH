package sidecar

// A Chroma-compatible REST shim over the same persistent collections engine
// registerCollections wires up (internal/collections.Manager): a Chroma
// collection and a ZENITH collection are literally the same object here —
// creating one through either route family makes it visible through both.
//
// This targets Chroma's legacy v1 wire protocol (POST/GET
// /api/v1/collections, .../add, .../upsert, .../get, .../query,
// .../delete, .../count, plus /api/v1/heartbeat and /api/v1/version for
// client-side connectivity checks), the shape named in the original task
// and still what a lot of deployed LangChain/LlamaIndex Chroma integrations
// speak. It is NOT the newer v2 protocol (tenant/database-scoped paths,
// /api/v2/auth/identity, /api/v2/pre-flight-checks) that chromadb's Python
// client defaults to from v0.5 onward — a client pinned to v2 needs a
// different base path or an older client version to reach these routes.
//
// What's implemented: collection create/get/list/delete (with
// get_or_create), add/upsert (always upsert semantics — see below), get
// (by ids only), query (nearest-neighbour search by query_texts, with
// `where` metadata filters translated to zenith.Filter), delete (by ids
// and/or where), count, heartbeat, version.
//
// What's deliberately left out, and why:
//   - query_embeddings / add-with-only-embeddings (no documents): ZENITH
//     embeds server-side from text through its own model; it has no public
//     API to index or search against a caller-supplied raw vector. A
//     request that omits `documents` on add, or omits `query_texts` on
//     query, gets a clear 400 explaining this rather than silently
//     misbehaving.
//   - get by `where` with no `ids`: there is no "list every document
//     matching a filter" call in internal/collections.Manager (only
//     "search by text, filtered"), so an id-less get is rejected with a
//     clear 400 rather than approximated.
//   - add vs. upsert: real Chroma's /add fails on a duplicate id; ZENITH's
//     Manager.Upsert always replaces. Both routes here just upsert — a
//     second /add with the same id succeeds instead of erroring.
//   - whereDocument (full-text filter on document content) and $contains/
//     $not_contains operators: not translated; only metadata `where`
//     clauses are.
//   - $gt/$lt are implemented as inclusive (same as $gte/$lte) because
//     zenith.Range has no exclusive-bound form.
//   - embeddings are never returned by get/query (ZENITH's public API has
//     no accessor for the stored vector, only for text and attrs); the
//     field is always null, even if explicitly included.
//   - multi-tenancy/auth: one shared token (X-Chroma-Token, checked against
//     Config.Key) across every collection, not Config.Collections' real
//     per-collection keys — Chroma's own wire protocol has no per-collection
//     credential to carry one.

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/shramanb113/ZENITH/internal/collections"
	"github.com/shramanb113/ZENITH/pkg/zenith"
)

const chromaBodyCap = 8 << 20 // 8 MiB, matching collections.Config.DefaultMaxBody

// registerChroma attaches the Chroma-compatible routes to mux. Called from
// Register only when Config.Collections is set — same gate as the native
// /v1/collections/* routes, since both share the same Manager.
func (s *Server) registerChroma(mux *http.ServeMux) {
	route(mux, "GET /api/v1/heartbeat", http.HandlerFunc(s.chromaHeartbeat))
	route(mux, "GET /api/v1/version", http.HandlerFunc(s.chromaVersion))

	route(mux, "POST /api/v1/collections", s.chromaGuard(s.chromaCreateCollection))
	route(mux, "GET /api/v1/collections", s.chromaGuard(s.chromaListCollections))
	route(mux, "GET /api/v1/collections/{name}", s.chromaGuard(s.chromaGetCollection))
	route(mux, "DELETE /api/v1/collections/{name}", s.chromaGuard(s.chromaDeleteCollection))
	route(mux, "GET /api/v1/collections/{name}/count", s.chromaGuard(s.chromaCount))

	route(mux, "POST /api/v1/collections/{name}/add", s.chromaGuard(s.chromaAdd))
	route(mux, "POST /api/v1/collections/{name}/upsert", s.chromaGuard(s.chromaAdd))
	route(mux, "POST /api/v1/collections/{name}/get", s.chromaGuard(s.chromaGet))
	route(mux, "POST /api/v1/collections/{name}/query", s.chromaGuard(s.chromaQuery))
	route(mux, "POST /api/v1/collections/{name}/delete", s.chromaGuard(s.chromaDelete))
}

func writeChromaErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

// chromaGuard checks the single shared token (if Config.Key is set) and
// caps the request body. Unlike colGuard, there is no per-collection key:
// Chroma's wire protocol carries no such credential.
func (s *Server) chromaGuard(h http.HandlerFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.Key != "" && r.Header.Get("X-Chroma-Token") != s.cfg.Key {
			writeChromaErr(w, http.StatusUnauthorized, "unauthorized")
			return
		}
		if name := r.PathValue("name"); name != "" && !collections.ValidID(name) {
			writeChromaErr(w, http.StatusBadRequest, "invalid collection name: must match ZENITH collection id rules (lowercase alphanumeric, '_', '-')")
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, chromaBodyCap)
		h(w, r)
	})
}

func (s *Server) chromaHeartbeat(w http.ResponseWriter, _ *http.Request) {
	// Matches real Chroma's exact response key; some clients check for its
	// presence as a liveness probe on construction.
	writeJSON(w, http.StatusOK, map[string]int64{"nanosecond heartbeat": time.Now().UnixNano()})
}

func (s *Server) chromaVersion(w http.ResponseWriter, _ *http.Request) {
	v := s.cfg.Version
	if v == "" {
		v = "0.0.0"
	}
	writeJSON(w, http.StatusOK, v+"-zenith-chroma-shim")
}

type chromaCollection struct {
	ID       string         `json:"id"`
	Name     string         `json:"name"`
	Metadata map[string]any `json:"metadata"`
}

func toChromaCollection(info collections.Info) chromaCollection {
	return chromaCollection{ID: info.ID, Name: info.ID, Metadata: nil}
}

func (s *Server) chromaCreateCollection(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name        string         `json:"name"`
		Metadata    map[string]any `json:"metadata"`
		GetOrCreate bool           `json:"get_or_create"`
	}
	if !decodeCol(w, r, &req) {
		return
	}
	if req.Name == "" {
		writeChromaErr(w, http.StatusBadRequest, "name is required")
		return
	}
	if !collections.ValidID(req.Name) {
		writeChromaErr(w, http.StatusBadRequest, "invalid collection name: must match ZENITH collection id rules (lowercase alphanumeric, '_', '-')")
		return
	}
	info, _, err := s.cfg.Collections.Create(req.Name, collections.CreateOptions{})
	if err != nil {
		if errors.Is(err, collections.ErrExists) && req.GetOrCreate {
			info, err = s.cfg.Collections.Stat(req.Name)
		}
		if err != nil {
			s.chromaErr(w, err)
			return
		}
	}
	writeJSON(w, http.StatusOK, toChromaCollection(info))
}

func (s *Server) chromaListCollections(w http.ResponseWriter, _ *http.Request) {
	list := s.cfg.Collections.List()
	out := make([]chromaCollection, len(list))
	for i, info := range list {
		out[i] = toChromaCollection(info)
	}
	writeJSON(w, http.StatusOK, out) // a bare array, matching Chroma's wire shape (not {"collections":[...]})
}

func (s *Server) chromaGetCollection(w http.ResponseWriter, r *http.Request) {
	info, err := s.cfg.Collections.Stat(r.PathValue("name"))
	if err != nil {
		s.chromaErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, toChromaCollection(info))
}

func (s *Server) chromaDeleteCollection(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if err := s.cfg.Collections.Delete(name); err != nil {
		s.chromaErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"name": name})
}

func (s *Server) chromaCount(w http.ResponseWriter, r *http.Request) {
	info, err := s.cfg.Collections.Stat(r.PathValue("name"))
	if err != nil {
		s.chromaErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, info.DocCount)
}

// chromaErr maps a collections/zenith error to a Chroma-shaped {"error":
// "..."} body. Unlike colErr (the native route's richer {"error","code"}
// shape), Chroma clients generally only look at the HTTP status and a
// human string, so there is no separate machine code here.
func (s *Server) chromaErr(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, collections.ErrInvalidID):
		writeChromaErr(w, http.StatusBadRequest, err.Error())
	case errors.Is(err, collections.ErrNotFound):
		writeChromaErr(w, http.StatusNotFound, "Collection not found: "+err.Error())
	case errors.Is(err, collections.ErrDocNotFound):
		writeChromaErr(w, http.StatusNotFound, err.Error())
	case errors.Is(err, collections.ErrExists):
		writeChromaErr(w, http.StatusConflict, err.Error())
	case errors.Is(err, collections.ErrQuota), errors.Is(err, collections.ErrTooManyCollections):
		writeChromaErr(w, http.StatusForbidden, err.Error())
	case errors.Is(err, collections.ErrShuttingDown), errors.Is(err, zenith.ErrLocked), errors.Is(err, zenith.ErrClosed):
		writeChromaErr(w, http.StatusServiceUnavailable, err.Error())
	case errors.Is(err, zenith.ErrInvalidID), errors.Is(err, zenith.ErrIDTooLong),
		errors.Is(err, zenith.ErrEmptyDocument), errors.Is(err, zenith.ErrInvalidAttrs):
		writeChromaErr(w, http.StatusBadRequest, err.Error())
	default:
		s.cfg.Log.Error("chroma shim: request failed", "error", err)
		writeChromaErr(w, http.StatusInternalServerError, "internal error")
	}
}

// chromaAddRequest is the wire shape POST .../add and .../upsert share.
type chromaAddRequest struct {
	IDs        []string         `json:"ids"`
	Embeddings [][]float32      `json:"embeddings"`
	Documents  []*string        `json:"documents"`
	Metadatas  []map[string]any `json:"metadatas"`
}

func (s *Server) chromaAdd(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req chromaAddRequest
	if !decodeCol(w, r, &req) {
		return
	}
	if len(req.IDs) == 0 {
		writeChromaErr(w, http.StatusBadRequest, "ids is required and must be non-empty")
		return
	}
	if len(req.Documents) == 0 {
		writeChromaErr(w, http.StatusBadRequest,
			"this shim requires \"documents\" (ZENITH embeds text server-side and has no API to index a caller-supplied raw vector); "+
				"add/upsert with embeddings only is not supported")
		return
	}
	if len(req.Documents) != len(req.IDs) {
		writeChromaErr(w, http.StatusBadRequest, "documents must be the same length as ids")
		return
	}
	if len(req.Metadatas) > 0 && len(req.Metadatas) != len(req.IDs) {
		writeChromaErr(w, http.StatusBadRequest, "metadatas must be the same length as ids")
		return
	}

	docs := make(map[string]string, len(req.IDs))
	attrs := make(map[string]zenith.Attrs)
	for i, id := range req.IDs {
		if id == "" {
			writeChromaErr(w, http.StatusBadRequest, "every id must be non-empty")
			return
		}
		if req.Documents[i] == nil || strings.TrimSpace(*req.Documents[i]) == "" {
			writeChromaErr(w, http.StatusBadRequest, fmt.Sprintf("document for id %q is empty; this shim cannot index embeddings-only records", id))
			return
		}
		docs[id] = *req.Documents[i]
		if len(req.Metadatas) > 0 && len(req.Metadatas[i]) > 0 {
			attrs[id] = zenith.Attrs(req.Metadatas[i])
		}
	}

	if _, err := s.cfg.Collections.Upsert(r.Context(), name, docs, attrs); err != nil {
		s.chromaErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, true)
}

type chromaGetRequest struct {
	IDs     []string       `json:"ids"`
	Where   map[string]any `json:"where"`
	Limit   int            `json:"limit"`
	Offset  int            `json:"offset"`
	Include []string       `json:"include"`
}

type chromaGetResponse struct {
	IDs        []string         `json:"ids"`
	Embeddings any              `json:"embeddings"`
	Documents  []*string        `json:"documents,omitempty"`
	Metadatas  []map[string]any `json:"metadatas,omitempty"`
}

func (s *Server) chromaGet(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req chromaGetRequest
	if !decodeCol(w, r, &req) {
		return
	}
	if len(req.IDs) == 0 {
		writeChromaErr(w, http.StatusBadRequest,
			"this shim requires \"ids\": getting every document matching \"where\" with no ids is not supported "+
				"(ZENITH's collections API has no id-less document scan — use query instead)")
		return
	}
	include := req.Include
	if len(include) == 0 {
		include = []string{"metadatas", "documents"}
	}
	wantDocs, wantMeta := chromaIncludes(include)

	var filter *zenith.Filter
	if len(req.Where) > 0 {
		f, err := chromaWhereToFilter(req.Where)
		if err != nil {
			writeChromaErr(w, http.StatusBadRequest, err.Error())
			return
		}
		filter = &f
	}

	resp := chromaGetResponse{Embeddings: nil}
	off := req.Offset
	for _, id := range req.IDs {
		if off > 0 {
			off--
			continue
		}
		if req.Limit > 0 && len(resp.IDs) >= req.Limit {
			break
		}
		text, err := s.cfg.Collections.GetDoc(r.Context(), name, id)
		if err != nil {
			if errors.Is(err, collections.ErrDocNotFound) {
				continue // Chroma's get silently skips missing ids
			}
			s.chromaErr(w, err)
			return
		}
		var attrs zenith.Attrs
		if filter != nil || wantMeta {
			attrs, err = s.cfg.Collections.GetDocAttrs(r.Context(), name, id)
			if err != nil {
				s.chromaErr(w, err)
				return
			}
		}
		if filter != nil && !filter.Matches(attrs) {
			continue
		}
		resp.IDs = append(resp.IDs, id)
		if wantDocs {
			t := text
			resp.Documents = append(resp.Documents, &t)
		}
		if wantMeta {
			resp.Metadatas = append(resp.Metadatas, map[string]any(attrs))
		}
	}
	writeJSON(w, http.StatusOK, resp)
}

type chromaQueryRequest struct {
	QueryEmbeddings [][]float32    `json:"query_embeddings"`
	QueryTexts      []string       `json:"query_texts"`
	NResults        int            `json:"n_results"`
	Where           map[string]any `json:"where"`
	Include         []string       `json:"include"`
}

type chromaQueryResponse struct {
	IDs        [][]string         `json:"ids"`
	Embeddings any                `json:"embeddings"`
	Documents  [][]*string        `json:"documents,omitempty"`
	Metadatas  [][]map[string]any `json:"metadatas,omitempty"`
	Distances  [][]float64        `json:"distances,omitempty"`
}

func (s *Server) chromaQuery(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req chromaQueryRequest
	if !decodeCol(w, r, &req) {
		return
	}
	if len(req.QueryTexts) == 0 {
		if len(req.QueryEmbeddings) > 0 {
			writeChromaErr(w, http.StatusBadRequest,
				"this shim requires \"query_texts\": ZENITH searches by text through its own embedder and has no API to "+
					"search against a caller-supplied raw query_embeddings vector")
			return
		}
		writeChromaErr(w, http.StatusBadRequest, "query_texts is required")
		return
	}
	n := req.NResults
	if n <= 0 {
		n = 10
	}
	include := req.Include
	if len(include) == 0 {
		include = []string{"metadatas", "documents", "distances"}
	}
	wantDocs, wantMeta := chromaIncludes(include)
	wantDist := chromaIncludesOne(include, "distances")

	var opts []zenith.SearchOption
	opts = append(opts, zenith.Limit(n))
	if len(req.Where) > 0 {
		f, err := chromaWhereToFilter(req.Where)
		if err != nil {
			writeChromaErr(w, http.StatusBadRequest, err.Error())
			return
		}
		opts = append(opts, zenith.WithFilter(f))
	}

	resp := chromaQueryResponse{Embeddings: nil}
	for _, qt := range req.QueryTexts {
		results, err := s.cfg.Collections.Search(r.Context(), name, qt, opts...)
		if err != nil {
			s.chromaErr(w, err)
			return
		}
		ids := make([]string, 0, len(results))
		var docs []*string
		var metas []map[string]any
		var dists []float64
		for _, res := range results {
			ids = append(ids, res.ID)
			if wantDist {
				dists = append(dists, 1-res.Score) // approximate: ZENITH's normalised [0,1] similarity, not a true metric distance
			}
			if wantDocs || wantMeta {
				text, attrs := s.fetchDocForQuery(r.Context(), name, res.ID)
				if wantDocs {
					docs = append(docs, &text)
				}
				if wantMeta {
					metas = append(metas, map[string]any(attrs))
				}
			}
		}
		resp.IDs = append(resp.IDs, ids)
		if wantDocs {
			resp.Documents = append(resp.Documents, docs)
		}
		if wantMeta {
			resp.Metadatas = append(resp.Metadatas, metas)
		}
		if wantDist {
			resp.Distances = append(resp.Distances, dists)
		}
	}
	writeJSON(w, http.StatusOK, resp)
}

// fetchDocForQuery best-efforts a doc's text/attrs for a query result; a
// failure (e.g. the document was deleted a moment ago) degrades to empty
// values instead of failing the whole query response.
func (s *Server) fetchDocForQuery(ctx context.Context, collectionID, docID string) (string, zenith.Attrs) {
	text, _ := s.cfg.Collections.GetDoc(ctx, collectionID, docID)
	attrs, _ := s.cfg.Collections.GetDocAttrs(ctx, collectionID, docID)
	return text, attrs
}

type chromaDeleteRequest struct {
	IDs   []string       `json:"ids"`
	Where map[string]any `json:"where"`
}

func (s *Server) chromaDelete(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req chromaDeleteRequest
	if !decodeCol(w, r, &req) {
		return
	}
	if len(req.IDs) == 0 {
		writeChromaErr(w, http.StatusBadRequest,
			"this shim requires \"ids\": deleting by \"where\" alone with no ids is not supported")
		return
	}
	var filter *zenith.Filter
	if len(req.Where) > 0 {
		f, err := chromaWhereToFilter(req.Where)
		if err != nil {
			writeChromaErr(w, http.StatusBadRequest, err.Error())
			return
		}
		filter = &f
	}

	deleted := make([]string, 0, len(req.IDs))
	for _, id := range req.IDs {
		if filter != nil {
			attrs, err := s.cfg.Collections.GetDocAttrs(r.Context(), name, id)
			if err != nil {
				if errors.Is(err, collections.ErrDocNotFound) {
					continue
				}
				s.chromaErr(w, err)
				return
			}
			if !filter.Matches(attrs) {
				continue
			}
		}
		if err := s.cfg.Collections.DeleteDoc(r.Context(), name, id); err != nil {
			if errors.Is(err, collections.ErrDocNotFound) {
				continue
			}
			s.chromaErr(w, err)
			return
		}
		deleted = append(deleted, id)
	}
	writeJSON(w, http.StatusOK, deleted)
}

func chromaIncludes(include []string) (docs, meta bool) {
	for _, s := range include {
		switch s {
		case "documents":
			docs = true
		case "metadatas":
			meta = true
		}
	}
	return
}

func chromaIncludesOne(include []string, want string) bool {
	for _, s := range include {
		if s == want {
			return true
		}
	}
	return false
}

// chromaWhereToFilter translates a Chroma `where` metadata filter object
// into a zenith.Filter. Supported: implicit equality ({"field": value}),
// {"field": {"$eq"|"$ne"|"$gt"|"$gte"|"$lt"|"$lte"|"$in"|"$nin": value}},
// and {"$and"|"$or": [...]} at any nesting level. See the file-level
// comment for what's deliberately not translated.
func chromaWhereToFilter(where map[string]any) (zenith.Filter, error) {
	var clauses []zenith.Filter
	for key, val := range where {
		switch key {
		case "$and", "$or":
			subs, ok := val.([]any)
			if !ok {
				return zenith.Filter{}, fmt.Errorf("chroma: %q must be a list", key)
			}
			fs := make([]zenith.Filter, 0, len(subs))
			for _, sub := range subs {
				sm, ok := sub.(map[string]any)
				if !ok {
					return zenith.Filter{}, fmt.Errorf("chroma: %q entries must be objects", key)
				}
				f, err := chromaWhereToFilter(sm)
				if err != nil {
					return zenith.Filter{}, err
				}
				fs = append(fs, f)
			}
			if key == "$and" {
				clauses = append(clauses, zenith.And(fs...))
			} else {
				clauses = append(clauses, zenith.Or(fs...))
			}
		default:
			f, err := chromaFieldFilter(key, val)
			if err != nil {
				return zenith.Filter{}, err
			}
			clauses = append(clauses, f)
		}
	}
	if len(clauses) == 1 {
		return clauses[0], nil
	}
	return zenith.And(clauses...), nil
}

func chromaFieldFilter(field string, val any) (zenith.Filter, error) {
	m, ok := val.(map[string]any)
	if !ok {
		return zenith.Eq(field, val), nil
	}
	if len(m) != 1 {
		return zenith.Filter{}, fmt.Errorf("chroma: operator object for %q must have exactly one operator", field)
	}
	for op, v := range m {
		switch op {
		case "$eq":
			return zenith.Eq(field, v), nil
		case "$ne":
			return zenith.Not(zenith.Eq(field, v)), nil
		case "$in":
			arr, ok := v.([]any)
			if !ok {
				return zenith.Filter{}, fmt.Errorf("chroma: $in value for %q must be a list", field)
			}
			return zenith.In(field, arr...), nil
		case "$nin":
			arr, ok := v.([]any)
			if !ok {
				return zenith.Filter{}, fmt.Errorf("chroma: $nin value for %q must be a list", field)
			}
			return zenith.Not(zenith.In(field, arr...)), nil
		case "$gt", "$gte":
			f, ok := chromaNum(v)
			if !ok {
				return zenith.Filter{}, fmt.Errorf("chroma: %s value for %q must be numeric", op, field)
			}
			return zenith.Range(field, f, math.Inf(1)), nil
		case "$lt", "$lte":
			f, ok := chromaNum(v)
			if !ok {
				return zenith.Filter{}, fmt.Errorf("chroma: %s value for %q must be numeric", op, field)
			}
			return zenith.Range(field, math.Inf(-1), f), nil
		default:
			return zenith.Filter{}, fmt.Errorf("chroma: unsupported operator %q", op)
		}
	}
	panic("unreachable: len(m) == 1 guaranteed a single iteration")
}

func chromaNum(v any) (float64, bool) {
	f, ok := v.(float64) // encoding/json decodes every JSON number as float64 into `any`
	return f, ok
}
