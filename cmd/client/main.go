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
	mode := flag.String("mode", "semantic", "Modes: 'index', 'search', 'semantic'")
	count := flag.Int("count", 1000, "Number of docs for stress test")
	query := flag.String("query", "fruit", "Search query")
	flag.Parse()

	// Connect to the Zenith Engine
	conn, err := grpc.Dial("localhost:8080", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("❌ Did not connect: %v", err)
	}
	defer conn.Close()
	client := zenithproto.NewSearchServiceClient(conn)

	switch *mode {
	case "index":
		runStressIndex(client, *count)
	case "search":
		runSearch(client, *query)
	case "semantic":
		runSemanticGauntlet(client)
	default:
		fmt.Println("Unknown mode. Use 'index', 'search', or 'semantic'")
	}
}

// runSemanticGauntlet tests the "Brain" (Challenge 17 Hybrid Search)
func runSemanticGauntlet(client zenithproto.SearchServiceClient) {
	fmt.Println("🧠 PHASE 2: Testing Neural Intelligence...")

	testDocs := []struct {
		id   string
		text string
	}{
		{"GO-1", "Golang is a statically typed, compiled programming language designed at Google."},
		{"PY-1", "Python is an interpreted, high-level, general-purpose programming language."},
		{"WEATHER-1", "The lightning flashed across the sky while the storm rumbled."},
		{"SYS-1", "High performance systems require optimized memory management and low latency."},
		{"FRUIT-1", "I love eating oranges and lemons in the summer."},
	}

	for _, d := range testDocs {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		_, err := client.IndexDocuments(ctx, &zenithproto.IndexRequest{Id: d.id, Data: d.text})
		cancel()
		if err != nil {
			log.Printf("Failed to index %s: %v", d.id, err)
		}
	}

	fmt.Println("✅ Semantic corpus indexed. Let's see if ZENITH can 'think'...")

	// Testing Semantic Overlap (The Vector Engine)
	// Query "fruit" doesn't exist in FRUIT-1, but the vector should find it.
	queries := []string{"fruit", "coding", "thunderstorm", "efficiency"}

	for _, q := range queries {
		runSearch(client, q)
		fmt.Println("--------------------------------------------------")
	}
}

// runStressIndex tests the "Skeleton" (Challenge 10 Concurrency)
func runStressIndex(client zenithproto.SearchServiceClient, totalDocs int) {
	start := time.Now()
	var wg sync.WaitGroup
	// Semaphore to prevent OS resource exhaustion
	semaphore := make(chan struct{}, 100)

	fmt.Printf("🚀 Starting Stress Test: Indexing %d documents...\n", totalDocs)

	for i := 0; i < totalDocs; i++ {
		wg.Add(1)
		semaphore <- struct{}{} // Acquire

		go func(id int) {
			defer wg.Done()
			defer func() { <-semaphore }() // Release

			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()

			docID := fmt.Sprintf("STRESS-%d", id)
			text := "Go is a great language for building distributed systems and search engines."

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
	fmt.Printf("📊 Result: %d docs in %v | Speed: %.2f docs/sec\n",
		totalDocs, duration, float64(totalDocs)/duration.Seconds())
}

func runSearch(client zenithproto.SearchServiceClient, query string) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	start := time.Now()
	res, err := client.Search(ctx, &zenithproto.SearchRequest{Query: query})
	if err != nil {
		log.Printf("❌ Search failed for '%s': %v", query, err)
		return
	}

	fmt.Printf("🔍 Results for '%s' (%v):\n", query, time.Since(start))
	if len(res.Results) == 0 {
		fmt.Println("   [No results found]")
		return
	}

	for _, doc := range res.Results {
		fmt.Printf(" -> [%-10s] Score: %.4f\n", doc.Id, doc.Score)
	}
}
