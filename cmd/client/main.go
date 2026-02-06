package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	mode := flag.String("mode", "index", "Mode to run: 'index' or 'search'")
	count := flag.Int("count", 1000, "Number of documents to index in stress test")
	flag.Parse()

	conn, err := grpc.Dial("localhost:8080", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	client := zenithproto.NewSearchServiceClient(conn)

	if *mode == "index" {
		runStressIndex(client, *count)
	} else {
		runSearch(client, "document")
	}
}

func runStressIndex(client zenithproto.SearchServiceClient, totalDocs int) {
	var wg sync.WaitGroup
	start := time.Now()

	fmt.Printf("🚀 Starting Stress Test: Indexing %d documents...\n", totalDocs)

	for i := 0; i < totalDocs; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()

			docID := fmt.Sprintf("STRESS-%d", id)
			text := fmt.Sprintf("This is document number %d which contains specific keywords for the search engine stress test", id)

			_, err := client.IndexDocuments(ctx, &zenithproto.IndexRequest{
				Id:   docID,
				Data: text,
			})
			if err != nil {
				log.Printf("Error indexing %s: %v", docID, err)
			}
		}(i)
	}

	wg.Wait()
	duration := time.Since(start)
	fmt.Printf("✅ Finished! Total Time: %v | Avg per doc: %v\n", duration, duration/time.Duration(totalDocs))
}

func runSearch(client zenithproto.SearchServiceClient, query string) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	start := time.Now()
	res, err := client.Search(ctx, &zenithproto.SearchRequest{Query: query})
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("🔍 Search Results for '%s':\n", query)
	for _, doc := range res.Results {
		fmt.Printf(" -> [%s] Score: %.4f\n", doc.Id, doc.Score)
	}
	fmt.Printf("\nFound %d results in %v\n", len(res.Results), time.Since(start))
}
