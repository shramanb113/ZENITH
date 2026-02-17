package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {

	conn, err := grpc.NewClient("localhost:8080", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("❌ gRPC Connection Failed: %v", err)
	}
	defer conn.Close()
	client := zenithproto.NewSearchServiceClient(conn)

	runNeuralGauntlet(client)
}

func runNeuralGauntlet(client zenithproto.SearchServiceClient) {
	fmt.Println("🚀 PHASE 4 TEST: The Linguistic Gauntlet")
	fmt.Println("--------------------------------------------------")

	challengingDocs := []struct {
		id   string
		text string
	}{
		{"TECH-01", "The PageRank algorithm uses backlink structures to determine the perceived importance of web pages."},
		{"DATA-08", "Modern ranking systems prioritize various ranks and ranked signals to ensure high-quality rankings."},
		{"LEGAL-03", "The relational database was revolutionary for its time, despite many relational anomalies."},
		{"AI-04", "Transformer ensembles often over-rely on lexical overlap instead of capturing deep semantic similarity."},
		{"ENV-06", "Global warming requires environmental solutions and atmospheric carbon capture."},
	}

	// 📦 CONCURRENT INDEXING
	fmt.Printf("📦 Indexing %d linguistic-heavy documents...\n", len(challengingDocs))
	var wg sync.WaitGroup
	start := time.Now()

	for _, d := range challengingDocs {
		wg.Add(1)
		go func(id, text string) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_, err := client.IndexDocuments(ctx, &zenithproto.IndexRequest{Id: id, Data: text})
			if err != nil {
				log.Printf("❌ Failed to index %s: %v", id, err)
			}
		}(d.id, d.text)
	}
	wg.Wait()
	fmt.Printf("✅ Indexing Complete in %v\n\n", time.Since(start))

	// 🧪 THE LINGUISTIC TRIALS
	linguisticTests := []struct {
		query    string
		expected string
		reason   string
	}{
		{
			query:    "PageRan",
			expected: "TECH-01",
			reason:   "Edge N-Gram: PageRank (stemmed to rank) -> 'ran'",
		},
		{
			query:    "rankings",
			expected: "DATA-08",
			reason:   "Stemming: rankings -> rank",
		},
		{
			query:    "relationship",
			expected: "LEGAL-03",
			reason:   "Stemming: relational -> relat",
		},
		{
			query:    "atmos",
			expected: "ENV-06",
			reason:   "Edge N-Gram: atmospheric (stemmed to atmospher) -> 'atmos'",
		},
	}

	fmt.Println("🧠 Analyzing Linguistic Accuracy...")
	fmt.Println("--------------------------------------------------")

	for _, test := range linguisticTests {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		res, err := client.Search(ctx, &zenithproto.SearchRequest{Query: test.query})
		cancel()

		if err != nil {
			log.Printf("❌ Search Error for '%s': %v", test.query, err)
			continue
		}

		// ... (after the search request) ...

		fmt.Printf("🔍 Query: [%s]\n", test.query)
		fmt.Printf("🎯 Goal:  Match %s (%s)\n", test.expected, test.reason)

		// Increase limit to see the whole small dataset
		displayLimit := 5
		if len(res.Results) < displayLimit {
			displayLimit = len(res.Results)
		}

		if len(res.Results) == 0 {
			fmt.Println("  🚫 NO RESULTS FOUND")
		}

		for i := 0; i < displayLimit; i++ {
			r := res.Results[i]
			status := "  "

			if r.Id == test.expected {
				if i == 0 {
					status = "✅" // Target is at the top!
				} else {
					status = "⚠️ " // Target found, but buried at Rank i+1
				}
			}

			fmt.Printf("   %s Rank %d: [%-8s] Score: %.6f\n", status, i+1, r.Id, r.Score)
		}
		fmt.Println("--------------------------------------------------")
	}
}
