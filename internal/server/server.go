package server

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
	"github.com/shramanb113/ZENITH/internal/activitylog"
	"github.com/shramanb113/ZENITH/internal/index"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// PDFIndexer is satisfied by pdf.PDFIndexer.
// Defined here to avoid an import cycle (server → pdf → nerve).
type PDFIndexer interface {
	Index(ctx context.Context, docID, filePath string, attrs index.Attrs) (int, error)
}

// ZenithServer implements zenithproto.SearchServiceServer.
type ZenithServer struct {
	zenithproto.UnimplementedSearchServiceServer
	Engine *index.Engine
	// PDFIndexer serves IndexPDF; nil makes IndexPDF return UNIMPLEMENTED.
	PDFIndexer PDFIndexer
	// NewTxn starts an atomic multi-document transaction against the
	// durable storage engine (storage.Engine.NewTxn) for IndexBatch and
	// DeleteBatch; nil makes both return UNIMPLEMENTED.
	NewTxn func() index.Txn
	Logger *activitylog.Logger
}

func (s *ZenithServer) logger() *activitylog.Logger {
	if s.Logger == nil {
		return activitylog.Noop()
	}
	return s.Logger
}

func (s *ZenithServer) IndexDocuments(
	ctx context.Context,
	req *zenithproto.IndexRequest,
) (*zenithproto.IndexResponse, error) {
	if req.GetId() == "" {
		return &zenithproto.IndexResponse{
			Status:  false,
			Message: "document id must not be empty",
		}, status.Error(codes.InvalidArgument, "document id must not be empty")
	}
	if req.GetData() == "" {
		return &zenithproto.IndexResponse{
			Status:  false,
			Message: "document data must not be empty",
		}, status.Error(codes.InvalidArgument, "document data must not be empty")
	}

	attrs, err := attrsFromProto(req.GetAttrs())
	if err != nil {
		return &zenithproto.IndexResponse{Status: false, Message: err.Error()}, status.Error(codes.InvalidArgument, err.Error())
	}
	if len(attrs) > 0 {
		err = s.Engine.AddWithVectorAttrs(ctx, req.GetId(), req.GetData(), s.Engine.EmbedText(ctx, req.GetData()), attrs)
	} else {
		err = s.Engine.Add(ctx, req.GetId(), req.GetData())
	}
	if err != nil {
		slog.Error("Failed to index document", "id", req.GetId(), "error", err)
		msg := fmt.Sprintf("indexing failed: %v", err)
		return &zenithproto.IndexResponse{
			Status:  false,
			Message: msg,
		}, status.Error(codes.Internal, msg)
	}

	slog.Info("Document indexed", "id", req.GetId())
	return &zenithproto.IndexResponse{
		Status:  true,
		Message: fmt.Sprintf("document %s indexed successfully", req.GetId()),
	}, nil
}

func (s *ZenithServer) Search(
	ctx context.Context,
	req *zenithproto.SearchRequest,
) (*zenithproto.SearchResponse, error) {
	for _, f := range req.GetFacetFields() {
		if f == "" {
			return nil, status.Error(codes.InvalidArgument, "facet_fields must not contain an empty name")
		}
	}
	if req.GetFacetTopK() < 0 {
		return nil, status.Error(codes.InvalidArgument, "facet_top_k must not be negative")
	}
	facetTopK := int(req.GetFacetTopK())
	if facetTopK == 0 {
		facetTopK = defaultFacetTopK
	}

	if req.GetQuery() == "" {
		// An empty query is only meaningful as a corpus-wide facet request.
		if len(req.GetFacetFields()) == 0 {
			return nil, status.Error(codes.InvalidArgument, "query must not be empty")
		}
		if req.GetFilter() != nil {
			return nil, status.Error(codes.InvalidArgument, "filter needs a query: corpus-wide facets (empty query) count every live document")
		}
		facets, err := s.Engine.CorpusFacetCounts(req.GetFacetFields(), facetTopK)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "facets: %v", err)
		}
		return &zenithproto.SearchResponse{Facets: facetsToProto(req.GetFacetFields(), facets)}, nil
	}

	weights := index.Weights{
		Vector:   req.GetVectorWeight(),
		Phonetic: req.GetPhoneticWeight(),
		RRF:      req.GetRrfK(),
	}
	if err := weights.Validate(); err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid ranking weights: %v", err)
	}

	var filter *index.Filter
	if req.GetFilter() != nil {
		var ferr error
		if filter, ferr = filterFromProto(req.GetFilter()); ferr != nil {
			return nil, status.Errorf(codes.InvalidArgument, "invalid filter: %v", ferr)
		}
	}
	results, err := s.Engine.SearchFilteredWeighted(ctx, req.GetQuery(), filter, weights)
	if err != nil {
		slog.Error("Search failed", "query", req.GetQuery(), "error", err)
		return nil, status.Errorf(codes.Internal, "search failed: %v", err)
	}

	// Facets describe the whole fused candidate list, not just the page.
	var protoFacets []*zenithproto.FacetResult
	if len(req.GetFacetFields()) > 0 {
		ids := make([]string, len(results))
		for i, r := range results {
			ids[i] = r.ID
		}
		facets, err := s.Engine.FacetCounts(ids, req.GetFacetFields(), facetTopK)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "facets: %v", err)
		}
		protoFacets = facetsToProto(req.GetFacetFields(), facets)
	}

	// sort_field replaces score ordering entirely, so it must run on the
	// full candidate list before paginate truncates it.
	s.Engine.SortByAttribute(results, req.GetSortField(), req.GetSortDesc())

	results = paginate(results, int(req.GetOffset()), int(req.GetLimit()))

	protoResults := make([]*zenithproto.SearchResult, 0, len(results))
	for _, r := range results {
		text, _ := s.Engine.GetText(r.ID)
		protoResults = append(protoResults, &zenithproto.SearchResult{
			Id:           r.ID,
			Score:        r.Score,
			Fields:       parseChunkFields(r.ID),
			OriginalText: text,
			Attrs:        attrsToProto(s.Engine.GetAttrs(r.ID)),
		})
	}

	s.logger().Log("SEARCH", fmt.Sprintf("%q → %d results", req.GetQuery(), len(protoResults)))

	slog.Info("Search complete", "query", req.GetQuery(), "hits", len(protoResults))
	return &zenithproto.SearchResponse{Results: protoResults, Facets: protoFacets}, nil
}

// defaultFacetTopK is the per-field value cap when facet_top_k is unset.
const defaultFacetTopK = 10

// facetsToProto flattens facets into one FacetResult per (field, value),
// grouped by field in the order fields lists them (each field once).
func facetsToProto(fields []string, f index.Facets) []*zenithproto.FacetResult {
	var out []*zenithproto.FacetResult
	done := make(map[string]bool, len(fields))
	for _, field := range fields {
		if done[field] {
			continue
		}
		done[field] = true
		for _, c := range f[field] {
			out = append(out, &zenithproto.FacetResult{
				Field: field,
				Value: attrValueToProto(c.Value),
				Count: int64(c.Count),
			})
		}
	}
	return out
}

func (s *ZenithServer) IndexPDF(
	ctx context.Context,
	req *zenithproto.IndexPDFRequest,
) (*zenithproto.IndexPDFResponse, error) {
	if req.GetDocumentId() == "" {
		return &zenithproto.IndexPDFResponse{Status: false, Message: "document_id must not be empty"},
			status.Error(codes.InvalidArgument, "document_id must not be empty")
	}
	if req.GetFilePath() == "" {
		return &zenithproto.IndexPDFResponse{Status: false, Message: "file_path must not be empty"},
			status.Error(codes.InvalidArgument, "file_path must not be empty")
	}
	if s.PDFIndexer == nil {
		return &zenithproto.IndexPDFResponse{Status: false, Message: "PDF indexing is not configured on this server"},
			status.Error(codes.Unimplemented, "PDF indexing is not configured on this server")
	}
	attrs, err := attrsFromProto(req.GetAttrs())
	if err != nil {
		return &zenithproto.IndexPDFResponse{Status: false, Message: err.Error()}, status.Error(codes.InvalidArgument, err.Error())
	}

	count, err := s.PDFIndexer.Index(ctx, req.GetDocumentId(), req.GetFilePath(), attrs)
	if err != nil {
		slog.Error("PDF indexing failed", "id", req.GetDocumentId(), "path", req.GetFilePath(), "error", err)
		msg := fmt.Sprintf("pdf indexing failed: %v", err)
		return &zenithproto.IndexPDFResponse{Status: false, Message: msg},
			status.Error(codes.Internal, msg)
	}

	slog.Info("PDF indexed", "id", req.GetDocumentId(), "chunks", count)
	return &zenithproto.IndexPDFResponse{
		Status:        true,
		Message:       fmt.Sprintf("indexed %d chunks from %s", count, req.GetFilePath()),
		ChunksIndexed: int32(count),
	}, nil
}

func (s *ZenithServer) GetDocument(
	ctx context.Context,
	req *zenithproto.GetDocumentRequest,
) (*zenithproto.GetDocumentResponse, error) {
	if req.GetId() == "" {
		return nil, status.Error(codes.InvalidArgument, "id must not be empty")
	}

	text, found := s.Engine.GetText(req.GetId())
	if !found {
		return &zenithproto.GetDocumentResponse{Found: false, Id: req.GetId()}, nil
	}
	return &zenithproto.GetDocumentResponse{
		Found: true, Id: req.GetId(), Text: text,
		Attrs: attrsToProto(s.Engine.GetAttrs(req.GetId())),
	}, nil
}

func (s *ZenithServer) DeleteDocument(
	ctx context.Context,
	req *zenithproto.DeleteDocumentRequest,
) (*zenithproto.DeleteDocumentResponse, error) {
	if req.GetId() == "" {
		return &zenithproto.DeleteDocumentResponse{Status: false, Message: "id must not be empty"},
			status.Error(codes.InvalidArgument, "id must not be empty")
	}

	if err := s.Engine.Remove(ctx, req.GetId()); err != nil {
		msg := fmt.Sprintf("delete failed: %v", err)
		return &zenithproto.DeleteDocumentResponse{Status: false, Message: msg}, status.Error(codes.Internal, msg)
	}

	s.logger().Log("DELETE", req.GetId())

	slog.Info("Document deleted", "id", req.GetId())
	return &zenithproto.DeleteDocumentResponse{
		Status:  true,
		Message: fmt.Sprintf("document %s deleted", req.GetId()),
	}, nil
}

// errNoTxn is returned by the batch RPCs when no storage engine is wired.
var errNoTxn = status.Error(codes.Unimplemented, "atomic batch operations need a transactional storage engine, which this server was started without")

// IndexBatch indexes every document atomically: each one is validated
// exactly as IndexDocuments validates a single document, then all of them are
// staged into one storage transaction and committed together — either every
// document is durably indexed or none is.
func (s *ZenithServer) IndexBatch(
	ctx context.Context,
	req *zenithproto.IndexBatchRequest,
) (*zenithproto.IndexBatchResponse, error) {
	fail := func(code codes.Code, msg string) (*zenithproto.IndexBatchResponse, error) {
		return &zenithproto.IndexBatchResponse{Status: false, Message: msg}, status.Error(code, msg)
	}
	if s.NewTxn == nil {
		return &zenithproto.IndexBatchResponse{Status: false, Message: "atomic batches are not enabled on this server"}, errNoTxn
	}
	if len(req.GetDocs()) == 0 {
		return fail(codes.InvalidArgument, "docs must not be empty")
	}
	docs := make([]index.BatchDoc, 0, len(req.GetDocs()))
	for i, d := range req.GetDocs() {
		if d.GetId() == "" {
			return fail(codes.InvalidArgument, fmt.Sprintf("docs[%d]: document id must not be empty", i))
		}
		if d.GetData() == "" {
			return fail(codes.InvalidArgument, fmt.Sprintf("docs[%d] (%q): document data must not be empty", i, d.GetId()))
		}
		attrs, err := attrsFromProto(d.GetAttrs())
		if err != nil {
			return fail(codes.InvalidArgument, fmt.Sprintf("docs[%d] (%q): %v", i, d.GetId(), err))
		}
		docs = append(docs, index.BatchDoc{ID: d.GetId(), Text: d.GetData(), Attrs: attrs})
	}

	if err := s.Engine.AddTransaction(ctx, docs, s.NewTxn()); err != nil {
		slog.Error("Atomic batch index failed", "docs", len(docs), "error", err)
		return fail(codes.Internal, fmt.Sprintf("batch indexing failed: %v", err))
	}

	s.logger().Log("INDEX", fmt.Sprintf("%d documents (atomic batch)", len(docs)))
	slog.Info("Documents indexed atomically", "docs", len(docs))
	return &zenithproto.IndexBatchResponse{
		Status:  true,
		Message: fmt.Sprintf("%d documents indexed atomically", len(docs)),
		Indexed: int32(len(docs)),
	}, nil
}

// DeleteBatch removes every id atomically (one storage transaction). Like
// DeleteDocument, deleting an id that does not exist is not an error.
func (s *ZenithServer) DeleteBatch(
	ctx context.Context,
	req *zenithproto.DeleteBatchRequest,
) (*zenithproto.DeleteBatchResponse, error) {
	fail := func(code codes.Code, msg string) (*zenithproto.DeleteBatchResponse, error) {
		return &zenithproto.DeleteBatchResponse{Status: false, Message: msg}, status.Error(code, msg)
	}
	if s.NewTxn == nil {
		return &zenithproto.DeleteBatchResponse{Status: false, Message: "atomic batches are not enabled on this server"}, errNoTxn
	}
	if len(req.GetIds()) == 0 {
		return fail(codes.InvalidArgument, "ids must not be empty")
	}
	for i, id := range req.GetIds() {
		if id == "" {
			return fail(codes.InvalidArgument, fmt.Sprintf("ids[%d] must not be empty", i))
		}
	}

	if err := s.Engine.RemoveBatch(ctx, req.GetIds(), s.NewTxn()); err != nil {
		slog.Error("Atomic batch delete failed", "ids", len(req.GetIds()), "error", err)
		return fail(codes.Internal, fmt.Sprintf("batch delete failed: %v", err))
	}

	s.logger().Log("DELETE", fmt.Sprintf("%d documents (atomic batch)", len(req.GetIds())))
	slog.Info("Documents deleted atomically", "ids", len(req.GetIds()))
	return &zenithproto.DeleteBatchResponse{
		Status:  true,
		Message: fmt.Sprintf("%d documents deleted atomically", len(req.GetIds())),
		Deleted: int32(len(req.GetIds())),
	}, nil
}

// Suggest returns indexed terms starting with a prefix. See
// index.Engine.Suggest for what the terms are (analysed and stemmed, in
// lexicographic order — not frequency-ranked surface forms).
func (s *ZenithServer) Suggest(
	ctx context.Context,
	req *zenithproto.SuggestRequest,
) (*zenithproto.SuggestResponse, error) {
	if req.GetPrefix() == "" {
		return nil, status.Error(codes.InvalidArgument, "prefix must not be empty")
	}
	if req.GetLimit() < 0 {
		return nil, status.Error(codes.InvalidArgument, "limit must not be negative")
	}
	terms, err := s.Engine.Suggest(req.GetPrefix(), int(req.GetLimit()))
	if err != nil {
		return nil, status.Errorf(codes.Internal, "suggest failed: %v", err)
	}
	return &zenithproto.SuggestResponse{Terms: terms}, nil
}

// defaultSearchLimit mirrors pkg/zenith's default page size so the gRPC and
// embedded-library surfaces behave the same way when no limit is given.
const defaultSearchLimit = 10

// paginate applies offset then limit to a ranked result slice. A limit <= 0
// falls back to defaultSearchLimit; a negative offset is treated as 0.
func paginate(results []index.SearchResponse, offset, limit int) []index.SearchResponse {
	if offset < 0 {
		offset = 0
	}
	if limit <= 0 {
		limit = defaultSearchLimit
	}
	if offset >= len(results) {
		return nil
	}
	end := offset + limit
	if end > len(results) {
		end = len(results)
	}
	return results[offset:end]
}

// parseChunkFields parses the structured chunk ID format
// "{docID}||p{page}||c{chunk}||{type}||{bbox}" into a fields map.
// Returns nil for regular (non-PDF) document IDs.
func parseChunkFields(id string) map[string]string {
	parts := strings.SplitN(id, "||", 5)
	if len(parts) != 5 {
		return nil
	}
	return map[string]string{
		"pdf_source":  parts[0],
		"page":        strings.TrimPrefix(parts[1], "p"),
		"chunk":       strings.TrimPrefix(parts[2], "c"),
		"source_type": parts[3],
		"bbox":        parts[4],
	}
}
