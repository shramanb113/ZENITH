package main

// engine.go — shared engine construction used by index, search, watch, serve.

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/shramanb113/ZENITH/internal/activitylog"
	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/ranking"
	storage "github.com/shramanb113/ZENITH/internal/storage"
	"github.com/shramanb113/ZENITH/internal/storage/wal"
)

// cliFlags holds the flag values shared across all commands.
var cliFlags struct {
	dbPath      string
	fstPath     string
	embedder    string // "auto" | "local" | "ollama" | "deterministic"
	ollamaURL   string
	ollamaModel string
}

// defaultStorageConfig returns a storage config rooted at ~/.zenith/.
func defaultStorageConfig() storage.EngineConfig {
	cfg := storage.DefaultEngineConfig()
	home, err := os.UserHomeDir()
	if err != nil {
		return cfg
	}
	base := filepath.Join(home, ".zenith")
	cfg.WALPath = filepath.Join(base, "data", "wal", "zenith.wal")
	cfg.WALConfig = wal.WALConfig{
		SyncMode: wal.SyncAlways,
		Dir:      filepath.Join(base, "data", "wal"),
	}
	cfg.SSTDir = filepath.Join(base, "data", "sst")
	cfg.FSTPath = filepath.Join(base, "data", "terms.fst")
	return cfg
}

// buildEngine constructs and optionally loads a ready-to-use index.Engine.
func buildEngine(load bool) (*index.Engine, *activitylog.Logger, func(), error) {
	appConfig := config.DefaultConfig()

	alog := activitylog.Open()

	var (
		storageEng   *storage.Engine
		storageErr   error
		emb          embedding.Embedder
		embedderName string
	)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		storageEng, storageErr = storage.Open(defaultStorageConfig())
	}()
	go func() {
		defer wg.Done()
		emb, embedderName = resolveEmbedder(appConfig, alog)
	}()
	wg.Wait()

	if storageErr != nil {
		alog.Close()
		return nil, nil, nil, storageErr
	}

	tkz := analysis.NewStandardAnalyzer()
	scorer := ranking.NewWeightedRRFRanker(appConfig.RRFConstant, appConfig.MaxResults, 1.0, appConfig.VectorWeight)
	engine := index.NewEngine(appConfig, emb, scorer, tkz)
	engine.SetFSTPath(cliFlags.fstPath)
	engine.SetTermStore(storageEng)

	_ = embedderName

	if load {
		if err := engine.Load(cliFlags.dbPath); err != nil {
			slog.Info("No existing index, starting fresh.")
		} else {
			alog.Log("LOADED", cliFlags.dbPath)
		}
	}

	teardown := func() {
		if err := engine.Save(cliFlags.dbPath); err != nil {
			slog.Error("Failed to save index", "error", err)
		} else {
			alog.Log("SAVED", cliFlags.dbPath)
		}
		if err := storageEng.Close(); err != nil {
			slog.Error("Storage engine close failed", "error", err)
		}
		alog.Close()
	}

	return engine, alog, teardown, nil
}

// resolveEmbedder selects the embedder based on cliFlags.embedder.
func resolveEmbedder(_ *config.Config, alog *activitylog.Logger) (embedding.Embedder, string) {
	switch cliFlags.embedder {
	case "auto", "local":
		return localEmbedderOrFallback(alog)

	case "ollama":
		base := cliFlags.ollamaURL
		if base == "" {
			base = "http://localhost:11434"
		}
		model := cliFlags.ollamaModel
		if model == "" {
			model = "nomic-embed-text"
		}
		raw := embedding.NewOllamaEmbedder(base, model, 30*time.Second)
		cached, err := embedding.NewCachingEmbedder(raw, 10_000)
		if err != nil {
			return embedding.NewDeterministicEmbedder(384), "deterministic"
		}
		return cached, "ollama"

	default: // "deterministic"
		return embedding.NewDeterministicEmbedder(384), "deterministic"
	}
}

// localEmbedderOrFallback loads the embedded ONNX model.
// Falls back to deterministic embeddings if CGo is unavailable or initialisation fails.
func localEmbedderOrFallback(alog *activitylog.Logger) (embedding.Embedder, string) {
	emb, err := localembedder.New()
	if err != nil {
		alog.Log("EMBEDDER", fmt.Sprintf("local embedder unavailable: %v — using deterministic", err))
		return embedding.NewDeterministicEmbedder(384), "deterministic"
	}
	cached, err := embedding.NewCachingEmbedder(emb, 10_000)
	if err != nil {
		return emb, "local"
	}
	return cached, "local"
}

func setupLogger() {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelWarn,
	})))
}
