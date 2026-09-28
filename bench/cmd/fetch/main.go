// Command fetch downloads and caches the MS MARCO passage-retrieval dataset
// (collection, dev queries, qrels) so benchmarks and diagnostics can run
// without re-downloading. Files land in the cache directory (default .cache,
// which is gitignored); a completed download is never repeated.
//
// Usage: go run ./cmd/fetch [--cache=.cache] [--limit=100000]
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/shramanb113/ZENITH/bench/internal/corpus"
)

func main() {
	cache := flag.String("cache", ".cache", "directory to store the dataset")
	limit := flag.Int("limit", 100_000, "number of passages to load (verifies the cache)")
	flag.Parse()

	passages, queries, qrels, err := corpus.Load(*cache, *limit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fetch: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("ready: %d passages, %d queries, %d qrels in %s\n", len(passages), len(queries), len(qrels), *cache)
}
