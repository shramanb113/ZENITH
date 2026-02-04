package main

import (
	"context"
	"log"
	"time"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {

	conn, err := grpc.NewClient("localhost:8080", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Did not connect: %v", err)
	}
	defer conn.Close()

	client := zenithproto.NewSearchServiceClient(conn)

	docs := []struct {
		ID   string
		Text string
	}{
		{"DOC-1", "The gopher is a fast animal and great for search engines."},
		{"DOC-2", "Zenith provides blazingly fast engine performance."},
		{"DOC-3", "A king was cycling across the glass bridge."},
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
	defer cancel()

	log.Println("--- 📨 INDEXING PHASE ---")
	for _, d := range docs {
		res, err := client.IndexDocuments(ctx, &zenithproto.IndexRequest{
			Id:   d.ID,
			Data: d.Text,
		})
		if err != nil {
			log.Printf("Could not index %s: %v", d.ID, err)
		} else {
			log.Printf("Server Response: %s (Success: %v)", res.Message, res.Status)
		}
	}

	log.Println("\n--- 🔍 SEARCH PHASE ---")
	searchQuery := "fast engine"

	searchRes, err := client.Search(ctx, &zenithproto.SearchRequest{
		Query: searchQuery,
	})
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	log.Printf("Results for query '%s':", searchQuery)
	for _, r := range searchRes.Results {
		log.Printf(" -> [%s] Score: %.4f", r.Id, r.Score)
	}
}
