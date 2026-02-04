package main

import (
	"fmt"

	"github.com/shramanb113/ZENITH/internal/analysis"
)

func main() {
	tokenizer := analysis.NewStandardTokenizer()

	testInput := "The king was cycling across the glass bridge."

	fmt.Println("--- ZENITH DEBUGGER ---")
	fmt.Printf("Input: %s\n", testInput)

	tokens := tokenizer.Tokenize(testInput)

	fmt.Printf("Tokens: %v\n", tokens)
	fmt.Printf("Count: %d tokens extracted\n", len(tokens))

	for _, t := range tokens {
		if len(t) <= 1 {
			fmt.Printf("Warning: Short token found: %s\n", t)
		}
	}
}
