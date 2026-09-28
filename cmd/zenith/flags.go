package main

import (
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

// addEngineFlags attaches the common engine-configuration flags to cmd.
// All commands that build an engine (index, search, watch, serve) call this.
func addEngineFlags(cmd *cobra.Command) {
	cmd.Flags().StringVar(&cliFlags.dbPath, "db", zenithDataPath("zenith.db"), "Index database file")
	cmd.Flags().StringVar(&cliFlags.fstPath, "fst", zenithDataPath("data/index.fst"), "On-disk FST path")
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
