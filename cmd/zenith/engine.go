package main

// engine.go — shared engine construction used by index, search, watch, serve.

import (
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
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
	model       string // registered local model id; "" = the bundled one
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

	// The index decides which embedding model to use: an existing index records the
	// model that built its vectors, and opening it with any other would mix vector
	// spaces. So unless --model says otherwise, follow the index (which makes changing
	// the bundled default painless for existing users).
	if load && cliFlags.model == "" {
		cliFlags.model = modelFromIndex(cliFlags.dbPath)
	}

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
		err := engine.Load(cliFlags.dbPath)
		switch {
		case err == nil:
			alog.Log("LOADED", cliFlags.dbPath)
		case errors.Is(err, fs.ErrNotExist):
			slog.Info("No existing index, starting fresh.")
		default:
			// Never fall through to "fresh": teardown saves to this path, so an
			// index we merely failed to read (older format, different embedder,
			// damage) would be overwritten by an empty one.
			_ = storageEng.Close()
			alog.Close()
			return nil, nil, nil, fmt.Errorf("cannot open index %s: %w%s", cliFlags.dbPath, err, mismatchHint(err))
		}
	}

	teardown := func() {
		if err := engine.Save(cliFlags.dbPath); err != nil {
			slog.Error("Failed to save index", "error", err)
		} else {
			alog.Log("SAVED", cliFlags.dbPath)
		}
		if err := engine.SaveANN(); err != nil {
			slog.Warn("Could not save the ANN graph; the next open will rebuild it", "error", err)
		}
		if err := engine.Close(); err != nil {
			slog.Error("Failed to release index files", "error", err)
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
	emb, err := localembedder.NewByID(cliFlags.model, modelsDir())
	if err != nil && cliFlags.model != "" {
		// An explicitly requested model must not silently degrade to another one.
		alog.Log("EMBEDDER", fmt.Sprintf("model %q unavailable: %v", cliFlags.model, err))
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
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

// mismatchHint turns an embedder-mismatch error into the two ways forward.
// The model that built the index is read from its header (nothing is loaded).
func mismatchHint(err error) string {
	if !errors.Is(err, index.ErrEmbedderMismatch) {
		return ""
	}
	info, ierr := index.Inspect(cliFlags.dbPath)
	if ierr != nil || !strings.HasPrefix(info.Embedder, "onnx:") {
		return "\n  This index was built with a different embedding model. Re-index, or open it with the model that built it (--model <id>; see 'zenith models list')."
	}
	id := strings.TrimPrefix(info.Embedder, "onnx:")
	return fmt.Sprintf("\n  This index was built with the %s model. Either:\n"+
		"    - keep using it:  zenith models pull %s   (skip if it is the bundled model), then add  --model %s\n"+
		"    - switch models:  remove the index and re-index (your documents are untouched; vectors differ per model)", id, id, id)
}

// modelFromIndex returns the registry model recorded in the index at path, or ""
// when there is no index, it records no registry model, or that model is the one
// bundled in this binary (which needs no special handling).
func modelFromIndex(path string) string {
	info, err := index.Inspect(path)
	if err != nil || !strings.HasPrefix(info.Embedder, "onnx:") {
		return ""
	}
	id := strings.TrimPrefix(info.Embedder, "onnx:")
	if _, err := localembedder.Lookup(id); err != nil || strings.EqualFold(id, localembedder.BundledID()) {
		return ""
	}
	return id
}
