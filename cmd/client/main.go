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
	count := flag.Int("count", 10, "Number of heavy documents")
	flag.Parse()

	// Using NewClient as per latest gRPC standards
	conn, err := grpc.NewClient("localhost:8080", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("❌ gRPC Connection Failed: %v", err)
	}
	defer conn.Close()
	client := zenithproto.NewSearchServiceClient(conn)

	runNeuralGauntlet(client, *count)
}

func runNeuralGauntlet(client zenithproto.SearchServiceClient, count int) {
	fmt.Println("🚀 PHASE 2 STRESS TEST: The Neural Gauntlet (Top 3 View)")
	fmt.Println("--------------------------------------------------")

	challengingDocs := []struct {
		id   string
		text string
	}{
		{"TECH-01", "The PageRank algorithm uses backlink structures to determine the perceived importance of web pages within a fully automated crawling ecosystem."},
		{"MED-02", "Clinical pharmacology studies indicate that drug-drug interactions between H2 blockers and organics can lead to metabolic anomalies in water-based solvent environments."},
		{"LEGAL-03", "The robots exclusion standard (robots.txt) acts as a hint rather than a legal directive, yet it remains the primary mechanism for preventing crawler-based indexing of sensitive directories."},
		{"AI-04", "Transformer ensembles often over-rely on lexical overlap (surface features) instead of capturing deep semantic similarity, especially in domain-specific technical jargon."},
		{"SYS-05", "High-performance distributed systems achieve sub-linear time complexity using Approximate Nearest Neighbor (ANN) algorithms like HNSW or FAISS to navigate high-dimensional vector spaces."},
		{"ENV-06", "Carbon capture and sequestration technologies utilize underground storage to mitigate CO2 emissions, effectively decoupling industrial output from atmospheric pollution."},
		{"EDU-07", "Asymmetric semantic search involves retrieving long, informative paragraphs to answer short, intent-heavy queries where keywords might not explicitly overlap."},
		{"DATA-08", "Min-Max Normalization squashes diverse score distributions into a fixed 0.0 to 1.0 range, preventing one algorithm from drowning out others in hybrid ranking systems."},
		{"LOG-09", "Shift logs in the process industry document incidents, maintenance activities, and product quality using proprietary industry-specific syntax and non-standard acronyms."},
		{"META-10", "Search engine optimization (SEO) requires balancing verifiability, neutrality, and notability, ensuring that popularity does not trump accuracy in indexed results."},
	}

	// Indexing
	fmt.Printf("📦 Indexing %d documents concurrently...\n", len(challengingDocs))
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

	// Semantic Queries
	semanticTests := []string{
		"how to rank websites",
		"medicine mixing",
		"preventing web spiders",
		"neural network failure",
		"fast search math",
		"global warming solutions",
	}

	fmt.Println("🧠 Analyzing Top 3 Hybrid Results...")
	fmt.Println("--------------------------------------------------")

	for _, query := range semanticTests {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		res, err := client.Search(ctx, &zenithproto.SearchRequest{Query: query})
		cancel()

		if err != nil {
			log.Printf("❌ Search Error for '%s': %v", query, err)
			continue
		}

		fmt.Printf("🔍 Query: [%s]\n", query)

		// Logic to show up to Top 3
		displayLimit := 3
		if len(res.Results) < displayLimit {
			displayLimit = len(res.Results)
		}

		for i := 0; i < displayLimit; i++ {
			r := res.Results[i]
			medal := "  "
			switch i {
			case 0:
				medal = "🥇"
			case 1:
				medal = "🥈"
			case 2:
				medal = "🥉"
			}
			fmt.Printf("   %s Rank %d: [%-8s] Score: %.4f\n", medal, i+1, r.Id, r.Score)
		}
		fmt.Println("--------------------------------------------------")
	}
}
