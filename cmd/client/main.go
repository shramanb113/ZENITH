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
	fmt.Println("🚀 ZENITH LINGUISTIC & PHONETIC GAUNTLET")
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
	fmt.Printf("📦 Indexing %d documents into the Inverted & Phonetic Index...\n", len(challengingDocs))
	var wg sync.WaitGroup
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
	fmt.Println("✅ Indexing Complete.")
	fmt.Println("--------------------------------------------------")

	// 🧪 THE TRIALS: Stemming, N-Grams, and Phonetics
	trials := []struct {
		query    string
		expected string
		category string
		reason   string
	}{
		{
			query:    "PageRan",
			expected: "TECH-01",
			category: "LEXICAL",
			reason:   "Edge N-Gram match (PageRank -> ran)",
		},
		{
			query:    "rankings",
			expected: "DATA-08",
			category: "STEMMING",
			reason:   "Porter Stemmer (rankings -> rank)",
		},
		{
			query:    "PageRanc",
			expected: "TECH-01",
			category: "PHONETIC",
			reason:   "Soundex: PageRanc (P265) matches PageRank (P265)",
		},
		{
			query:    "relayshun",
			expected: "LEGAL-03",
			category: "PHONETIC",
			reason:   "Soundex: relayshun (R425) sounds like relational (R435) - *Testing sound proximity*",
		},
		{
			query:    "Amospher",
			expected: "ENV-06",
			category: "PHONETIC",
			reason:   "Soundex: Amospher (A521) matches Atmospheric (A352) anchor 'A'",
		},
		{
			query:    "relat",
			expected: "LEGAL-03",
			category: "LEXICAL",
			reason:   "Stemming overlap (relational -> relat)",
		},
		{
			query:    "Transfomer", // Missing the 'r' (Levenshtein Distance 1)
			expected: "AI-04",
			category: "FUZZY",
			reason:   "Levenshtein: Transfomer -> Transformer (Distance 1)",
		},
		{
			query:    "envirmental", // Missing 'on' (Levenshtein Distance 2)
			expected: "ENV-06",
			category: "FUZZY",
			reason:   "Levenshtein: envirmental -> environmental (Distance 2)",
		},
	}

	for _, test := range trials {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		res, err := client.Search(ctx, &zenithproto.SearchRequest{Query: test.query})
		cancel()

		if err != nil {
			fmt.Printf("❌ Search Error for [%s]: %v\n", test.query, err)
			continue
		}

		fmt.Printf("🔍 Query: [%-10s] | Category: %-10s\n", test.query, test.category)
		fmt.Printf("🎯 Goal:  Match %s (%s)\n", test.expected, test.reason)

		if len(res.Results) == 0 {
			fmt.Println("   🚫 NO RESULTS FOUND")
		} else {
			foundAt := -1
			for i, r := range res.Results {
				if r.Id == test.expected {
					foundAt = i + 1
					break
				}
			}

			if foundAt == 1 {
				fmt.Printf("   ✅ TOP MATCH: [%s] Score: %.4f\n", res.Results[0].Id, res.Results[0].Score)
			} else if foundAt > 1 {
				fmt.Printf("   ⚠️  FOUND AT RANK %d: [%s] Score: %.4f\n", foundAt, test.expected, res.Results[foundAt-1].Score)
			} else {
				fmt.Printf("   ❌ FAIL: Target %s not in top results. Top was [%s]\n", test.expected, res.Results[0].Id)
			}
		}
		fmt.Println("--------------------------------------------------")
	}
}
