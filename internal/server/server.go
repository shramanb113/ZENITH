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
	Index(ctx context.Context, docID, filePath string) (int, error)
}

// ZenithServer implements zenithproto.SearchServiceServer.
type ZenithServer struct {
	zenithproto.UnimplementedSearchServiceServer
	Engine     *index.Engine
	PDFIndexer PDFIndexer
	Logger     *activitylog.Logger
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
	if req.GetQuery() == "" {
		return nil, status.Error(codes.InvalidArgument, "query must not be empty")
	}

	var filter *index.Filter
	if req.GetFilter() != nil {
		var ferr error
		if filter, ferr = filterFromProto(req.GetFilter()); ferr != nil {
			return nil, status.Errorf(codes.InvalidArgument, "invalid filter: %v", ferr)
		}
	}
	results, err := s.Engine.SearchFiltered(ctx, req.GetQuery(), filter)
	if err != nil {
		slog.Error("Search failed", "query", req.GetQuery(), "error", err)
		return nil, status.Errorf(codes.Internal, "search failed: %v", err)
	}

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

	l := s.Logger
	if l == nil {
		l = activitylog.Noop()
	}
	l.Log("SEARCH", fmt.Sprintf("%q → %d results", req.GetQuery(), len(protoResults)))

	slog.Info("Search complete", "query", req.GetQuery(), "hits", len(protoResults))
	return &zenithproto.SearchResponse{Results: protoResults}, nil
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

	count, err := s.PDFIndexer.Index(ctx, req.GetDocumentId(), req.GetFilePath())
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
	return &zenithproto.GetDocumentResponse{Found: true, Id: req.GetId(), Text: text}, nil
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

	l := s.Logger
	if l == nil {
		l = activitylog.Noop()
	}
	l.Log("DELETE", req.GetId())

	slog.Info("Document deleted", "id", req.GetId())
	return &zenithproto.DeleteDocumentResponse{
		Status:  true,
		Message: fmt.Sprintf("document %s deleted", req.GetId()),
	}, nil
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

// ParseChunkFieldsForTest exports parseChunkFields for white-box testing.
var ParseChunkFieldsForTest = parseChunkFields
