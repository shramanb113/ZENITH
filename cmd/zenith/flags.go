package main

import (
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"
)

// addEngineFlags attaches the common engine-configuration flags to cmd.
// All commands that build an engine (index, search, watch, serve) call this.
func addEngineFlags(cmd *cobra.Command) {
	cmd.Flags().StringVar(&cliFlags.dbPath, "db", zenithDataPath("zenith.db"), "Index database file")
	cmd.Flags().StringVar(&cliFlags.fstPath, "fst", "", "On-disk FST path (default: <db>.fst, tied to --db so different --db paths never load each other's FST)")
	cmd.Flags().StringVar(&cliFlags.storageDir, "storage-dir", "", "Pebble-backed document-journal + embedding-cache directory (default: <db>.pebble, tied to --db so different --db paths never share one journal)")
	cmd.Flags().StringVar(&cliFlags.embedder, "embedder", "auto",
		`Embedding backend:
  auto          Embedded ONNX model (requires CGo build; see docs/DECISIONS.md)
  local         Alias for auto
  ollama        Local Ollama (needs: ollama serve + ollama pull nomic-embed-text)
  deterministic Hash-based, zero dependencies`)
	cmd.Flags().StringVar(&cliFlags.model, "model", "",
		"Embedding model id from `zenith models list` (default: the one bundled in this binary; others need `zenith models pull <id>`)")
	cmd.Flags().StringVar(&cliFlags.ollamaURL, "ollama-url", "http://localhost:11434", "Ollama server URL")
	cmd.Flags().StringVar(&cliFlags.ollamaModel, "ollama-model", "nomic-embed-text", "Ollama embedding model")
	cmd.Flags().IntVar(&cliFlags.queryCacheSize, "query-cache-size", 1000, "Query-result cache entry count (L1, in-process); 0 disables the query-result cache")
	cmd.Flags().DurationVar(&cliFlags.queryCacheTTL, "query-cache-ttl", 5*time.Minute, "L2 (Redis) query-result cache entry TTL; has no effect without --query-cache-redis-addr")
	cmd.Flags().StringVar(&cliFlags.queryCacheRedisAddr, "query-cache-redis-addr", "", "Optional L2 Redis address for the query-result cache (e.g. localhost:6379); empty keeps the cache in-process only")
	cmd.Flags().Float64Var(&cliFlags.queryCacheSemanticThreshold, "query-cache-semantic-threshold", 0, "Enable near-duplicate query-cache matching above this cosine similarity (0 disables it; 0.97 is a conservative starting point)")
	cmd.Flags().Float64Var(&cliFlags.annThresholdBandPct, "ann-threshold-band-pct", 0, "Make the ANN-vs-exact vector search choice near the ANN threshold follow measured latency within this fraction of the threshold (0 disables it, static threshold only)")
}

// zenithDataPath returns an absolute path inside the user's ~/.zenith/ directory.
// Falls back to the bare filename if the home directory cannot be determined.
func zenithDataPath(rel string) string {
	home, err := os.UserHomeDir()
	if err != nil {
		return rel
	}
	return filepath.Join(home, ".zenith", rel)
}
