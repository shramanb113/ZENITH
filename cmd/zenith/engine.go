package main

// engine.go — shared engine construction used by index, search, watch, serve.

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/shramanb113/ZENITH/internal/activitylog"
	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/metrics"
	"github.com/shramanb113/ZENITH/internal/ranking"
	"github.com/shramanb113/ZENITH/internal/storage"
)

// cliFlags holds the flag values shared across all commands.
var cliFlags struct {
	dbPath      string
	fstPath     string
	storageDir  string // Pebble-backed document journal + persistent embedding cache
	embedder    string // "auto" | "local" | "ollama" | "deterministic"
	model       string // registered local model id; "" = the bundled one
	ollamaURL   string
	ollamaModel string

	queryCacheSize              int
	queryCacheTTL               time.Duration
	queryCacheRedisAddr         string
	queryCacheSemanticThreshold float64
	annThresholdBandPct         float64
}

// buildEngine constructs and optionally loads a ready-to-use index.Engine.
//
// withStorage controls whether a storage.Engine (document journal +
// persistent embedding cache) is opened at all. Pebble takes an exclusive
// lock on its directory for as long as it is open, so opening it
// unconditionally on every command — including a purely read-only one like
// `zenith search` — would make `search` fail whenever a long-running
// `zenith watch`/`zenith serve` (or another CLI invocation) already holds
// that lock, breaking a workflow that worked before storage was wired into
// cmd/zenith at all. Pass false for read-only commands that never mutate
// the index and don't need crash-safe journaling or the embedding cache;
// pass true for anything that adds, removes, or otherwise writes. The
// returned *storage.Engine is nil when withStorage is false.
//
// When non-nil, the returned *storage.Engine is the same instance
// buildEngine opened and wired as the document journal — callers that need
// NewTxn() (zenith txn, zenith storage) use it directly rather than opening
// a second storage.Engine on the same --storage-dir, which would hit the
// same exclusive-lock problem. teardown closes it; callers must not close
// it themselves.
func buildEngine(load bool, withStorage bool) (*index.Engine, *storage.Engine, *activitylog.Logger, func(), error) {
	appConfig := config.DefaultConfig()

	if cliFlags.queryCacheSize >= 0 {
		appConfig.QueryCacheSize = cliFlags.queryCacheSize
	}
	if cliFlags.queryCacheTTL > 0 {
		appConfig.QueryCacheTTL = cliFlags.queryCacheTTL
	}
	appConfig.QueryCacheRedisAddr = cliFlags.queryCacheRedisAddr
	appConfig.QueryCacheSemanticThreshold = cliFlags.queryCacheSemanticThreshold
	appConfig.ANNThresholdBandPct = cliFlags.annThresholdBandPct

	// The index decides which embedding model to use: an existing index records the
	// model that built its vectors, and opening it with any other would mix vector
	// spaces. So unless --model says otherwise, follow the index (which makes changing
	// the bundled default painless for existing users).
	if load && cliFlags.model == "" {
		cliFlags.model = modelFromIndex(cliFlags.dbPath)
	}

	alog := activitylog.Open()

	var storageEng *storage.Engine
	if withStorage {
		var err error
		storageEng, err = storage.Open(storage.EngineConfig{Dir: effectiveStorageDir()})
		if err != nil {
			alog.Close()
			return nil, nil, nil, nil, fmt.Errorf("cannot open storage engine at %s: %w", effectiveStorageDir(), err)
		}
	}

	emb, embedderName := resolveEmbedder(appConfig, alog)
	if storageEng != nil {
		if pc, ok := emb.(embedding.PersistentCacheSetter); ok {
			pc.SetPersistentCache(storageEng)
		}
	}

	tkz := analysis.NewStandardAnalyzer()
	scorer := ranking.NewWeightedRRFRanker(appConfig.RRFConstant, appConfig.MaxResults, 1.0, appConfig.VectorWeight)
	engine := index.NewEngine(appConfig, emb, scorer, tkz)
	engine.SetFSTPath(effectiveFSTPath())
	engine.SetCacheObserver(metrics.NewQueryCacheObserver())

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
			alog.Close()
			if storageEng != nil {
				_ = storageEng.Close()
			}
			return nil, nil, nil, nil, fmt.Errorf("cannot open index %s: %w%s", cliFlags.dbPath, err, mismatchHint(err))
		}
	}

	if storageEng != nil {
		// Replay journal delta — documents indexed since the last Save. The
		// journal is NOT set yet, so these Add/Remove calls do not
		// re-journal. Mirrors cmd/server/main.go's replay loop, except for
		// the dimension check below: a journalled vector is only trusted
		// when it matches the embedder actually in use right now (see the
		// comment on vector below for why).
		replayCtx := context.Background()
		if err := storageEng.Replay(func(key, value []byte, isDelete bool) error {
			id := string(key)
			if isDelete {
				if err := engine.Remove(replayCtx, id); err != nil {
					slog.Warn("storage replay: remove failed", "id", id, "error", err)
				}
				return nil
			}
			text, vector, attrs := index.DecodeJournalValue(value)
			// A carried vector is only trusted when its dimension matches
			// the embedder actually in use right now. Without this check,
			// an unclean exit followed by a model switch (different
			// --model, or --embedder ollama at a different dimension) would
			// replay a vector the current embedder's vector space doesn't
			// match: segment writes would then reject it outright (Save
			// would fail on every run, forever, since the journal entry
			// never gets pruned) or, worse, silently mix incompatible
			// vector spaces when the dimension happens to match but the
			// model doesn't. Falling back to re-embed is exactly today's
			// existing behavior for a legacy (vector-less) entry, so a
			// mismatch degrades to that, not to an error.
			if vector != nil && len(vector) != emb.Dimensions() {
				vector = nil
			}
			var addErr error
			switch {
			case vector != nil:
				addErr = engine.AddWithVectorAttrs(replayCtx, id, text, vector, attrs)
			case len(attrs) > 0:
				addErr = engine.AddWithVectorAttrs(replayCtx, id, text, engine.EmbedText(replayCtx, text), attrs)
			default:
				addErr = engine.Add(replayCtx, id, text)
			}
			if addErr != nil {
				slog.Warn("storage replay: re-index failed", "id", id, "error", addErr)
			}
			return nil
		}); err != nil {
			slog.Warn("storage replay failed", "error", err)
		}
		// Connect the journal — all future mutations are durably recorded first.
		engine.SetDocumentJournal(storageEng)
	}

	teardown := func() {
		var snap *storage.Snapshot
		if storageEng != nil {
			snap = storageEng.Snapshot()
		}
		if err := engine.Save(cliFlags.dbPath); err != nil {
			slog.Error("Failed to save index", "error", err)
		} else {
			alog.Log("SAVED", cliFlags.dbPath)
			if snap != nil {
				// The saved segment is now authoritative for everything this
				// snapshot saw — prune exactly those journal entries, not
				// "everything now" (which would race writes arriving during Save).
				if err := storageEng.Prune(snap); err != nil {
					slog.Warn("storage: prune after save failed (journal will just be larger than necessary)", "error", err)
				}
			}
		}
		if snap != nil {
			_ = snap.Close()
		}
		if err := engine.SaveANN(); err != nil {
			slog.Warn("Could not save the ANN graph; the next open will rebuild it", "error", err)
		}
		if err := engine.Close(); err != nil {
			slog.Error("Failed to release index files", "error", err)
		}
		if storageEng != nil {
			if err := storageEng.Close(); err != nil {
				slog.Error("Failed to close storage engine", "error", err)
			}
		}
		alog.Close()
	}

	return engine, storageEng, alog, teardown, nil
}

// effectiveStorageDir returns cliFlags.storageDir, or, when it was left
// unset, a directory derived from --db. Tying the default to --db (rather
// than one fixed global path regardless of --db) matters because the
// storage directory carries document content, not just a cache: two
// `zenith` invocations against different --db paths sharing one journal by
// default would let documents from one database replay into the other's
// index, and a prune from one invocation would delete journal entries a
// different --db invocation still needed. An explicit --storage-dir always
// wins, for a caller that wants to intentionally share one journal.
func effectiveStorageDir() string {
	if cliFlags.storageDir != "" {
		return cliFlags.storageDir
	}
	return cliFlags.dbPath + ".pebble"
}

// effectiveFSTPath returns cliFlags.fstPath, or, when it was left unset, a
// path derived from --db (same directory/stem as the db path, <db>.fst).
// Tying the default to --db matters because the only guard against loading
// the wrong FST (tryLoadFSTFromDiskLocked's term-count check) is a heuristic:
// two different DBs with coincidentally equal vocabulary sizes could
// otherwise silently load each other's FST under one fixed global default
// path regardless of --db. An explicit --fst always wins.
func effectiveFSTPath() string {
	if cliFlags.fstPath != "" {
		return cliFlags.fstPath
	}
	return cliFlags.dbPath + ".fst"
}

// effectiveFileHashPath returns the content-hash cache path for `zenith
// index`'s/`zenith watch run --index-first`'s skip-unchanged-files
// optimization, derived from --db (<db>.filehashes.json). Previously this
// was one fixed path under ~/.zenith regardless of --db: two different
// --db instances indexing the same directory would share one cache, so the
// second instance could see a file marked up-to-date by the first and skip
// indexing it into its own (different) index entirely.
func effectiveFileHashPath() string {
	return cliFlags.dbPath + ".filehashes.json"
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
