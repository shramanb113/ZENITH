package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/index"
)

func main() {
	tokenizer := analysis.NewStandardTokenizer()
	engine := index.NewInMemoryIndex()

	words := []string{"gopher", "zenith", "search", "engine", "fast", "blazing", "go", "high", "performance", "system"}
	docCount := 1000

	fmt.Printf("--- ZENITH STRESS TEST: INDEXING %d DOCUMENTS ---\n", docCount)
	start := time.Now()

	for i := 0; i < docCount; i++ {
		id := fmt.Sprintf("DOC-%d", i)
		content := ""
		for j := 0; j < 20; j++ {
			content += words[rand.Intn(len(words))] + " "
		}

		tokens := tokenizer.Tokenize(content)
		engine.Add(id, tokens)
	}

	fmt.Printf("Indexing complete in: %v\n", time.Since(start))

	query := "fast gopher engine"
	iterations := 5000
	fmt.Printf("\n--- ZENITH STRESS TEST: %d SEARCHES FOR '%s' ---\n", iterations, query)

	searchStart := time.Now()
	queryTokens := tokenizer.Tokenize(query)

	for i := 0; i < iterations; i++ {
		_ = engine.Search(queryTokens)
	}

	duration := time.Since(searchStart)
	fmt.Printf("Total search time: %v\n", duration)
	fmt.Printf("Average search time: %v\n", duration/time.Duration(iterations))
}
