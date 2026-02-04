package server

import (
	"context"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/index"
)

type ZenithServer struct {
	zenithproto.UnimplementedSearchServiceServer
	Index     *index.InMemoryIndex
	Tokenizer *analysis.StandardTokenizer
}

func (s *ZenithServer) IndexDocuments(ctx context.Context, req *zenithproto.IndexRequest) (*zenithproto.IndexResponse, error) {

	tokens := s.Tokenizer.Tokenize(req.Data)

	s.Index.Add(req.Id, tokens)

	return &zenithproto.IndexResponse{
		Status:  true,
		Message: "Document Indexed successfully",
	}, nil
}

func (s *ZenithServer) Search(ctx context.Context, req *zenithproto.SearchRequest) (*zenithproto.SearchResponse, error) {

	tokens := s.Tokenizer.Tokenize(req.Query)

	results := s.Index.Search(tokens)

	var protoResults []*zenithproto.SearchResult

	for _, res := range results {
		protoResults = append(protoResults, &zenithproto.SearchResult{
			Id:    res.ID,
			Score: res.Score,
		})
	}

	return &zenithproto.SearchResponse{
		Results: protoResults,
	}, nil

}
