package main

import (
	"context"
	"fmt"
	"log"
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
	fmt.Println("🚀 ZENITH NEURAL & LINGUISTIC STRESS TEST")
	fmt.Println("--------------------------------------------------")

	challengingDocs := []struct {
		id   string
		text string
	}{
		{"TECH-01", "The PageRank algorithm uses backlink structures to determine the perceived importance of web pages."},
		{"DATA-08", "Modern ranking systems prioritize various signals to ensure high-quality results."},
		{"LEGAL-03", "The relational database was revolutionary for its time, despite many relational anomalies."},
		{"AI-04", "Transformer ensembles often over-rely on lexical overlap instead of capturing deep semantic similarity."},
		{"ENV-06", "Global warming requires environmental solutions and atmospheric carbon capture."},
	}

	// 1. 📦 DETERMINISTIC INDEXING (No Goroutines here!)
	// We need these to enter the engine in the exact same order every time
	// to ensure internal IDs and TF-IDF weights are stable.
	fmt.Printf("📦 Feeding %d documents into Zenith's brain...\n", len(challengingDocs))
	for _, d := range challengingDocs {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_, err := client.IndexDocuments(ctx, &zenithproto.IndexRequest{
			Id:   d.id,
			Data: d.text,
		})
		cancel()
		if err != nil {
			log.Fatalf("❌ Critical Indexing Failure for %s: %v", d.id, err)
		}
	}

	fmt.Println("✅ Indexing Complete. Memory brain is warm.")
	fmt.Println("--------------------------------------------------")

	// 2. THE TRIALS
	trials := []struct {
		query    string
		expected string
		category string
		reason   string
	}{

		{
			query:    "PageRanc",
			expected: "TECH-01",
			category: "PHONETIC",
			reason:   "Soundex match",
		},
		{
			query:    "machine learning",
			expected: "AI-04",
			category: "NEURAL",
			reason:   "Semantic match: 'machine learning' ≈ 'Transformer'",
		},
		{
			query:    "climate change",
			expected: "ENV-06",
			category: "NEURAL",
			reason:   "Semantic match: 'climate' ≈ 'environmental', 'change' ≈ 'warming'",
		},
		{
			query:    "Transfomer",
			expected: "AI-04",
			category: "FUZZY",
			reason:   "Levenshtein Distance 1",
		},
	}

	for _, test := range trials {
		// Use a fresh context per search
		ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
		start := time.Now()
		res, err := client.Search(ctx, &zenithproto.SearchRequest{Query: test.query})
		duration := time.Since(start)
		cancel()

		if err != nil {
			fmt.Printf("❌ Search Error for [%s]: %v\n", test.query, err)
			continue
		}

		fmt.Printf("🔍 Query: [%-15s] | Cat: %-8s | Latency: %v\n", test.query, test.category, duration)
		fmt.Printf("🎯 Goal:  Match %s (%s)\n", test.expected, test.reason)

		if len(res.Results) == 0 {
			fmt.Println("   🚫 NO RESULTS FOUND")
		} else {
			// Find the actual rank
			foundAt := -1
			for i, r := range res.Results {
				if r.Id == test.expected {
					foundAt = i + 1
					break
				}
			}

			if foundAt == 1 {
				fmt.Printf("   ✅ TOP MATCH: [%s] Score: %.4f\n", res.Results[0].Id, res.Results[0].Score)
			} else if foundAt > 0 {
				fmt.Printf("   ⚠️  FOUND AT RANK %d: [%s] (Top was [%s])\n", foundAt, test.expected, res.Results[0].Id)
			} else {
				fmt.Printf("   ❌ FAIL: Top result was [%s]\n", res.Results[0].Id)
			}
		}
		fmt.Println("--------------------------------------------------")
	}
}
