// Command zenith-mcp is an MCP (Model Context Protocol) server that exposes
// ZENITH's persistent collections as retrieval/memory tools to any MCP
// client — Claude Code, Claude Desktop, Cursor, or anything else that
// speaks MCP over stdio.
//
// It embeds internal/collections.Manager directly: there is no separate
// `zenith serve` process to run first and no network hop. Point an MCP
// client at this binary and it gets four tools — list_collections,
// search_collection, upsert_document, get_document — backed by durable,
// on-disk collections under --collections-dir (default
// ~/.zenith/mcp-collections). This is the "zero infra" version of the
// collections API from CLAUDE.md's internal/sidecar section: the same
// Manager internal/sidecar's HTTP routes wrap, used in-process instead.
//
// Usage (manual):
//
//	go run ./cmd/zenith-mcp --collections-dir /path/to/data
//
// Typical MCP client config (e.g. Claude Desktop's claude_desktop_config.json):
//
//	{
//	  "mcpServers": {
//	    "zenith": {
//	      "command": "zenith-mcp",
//	      "args": ["--collections-dir", "/path/to/data"]
//	    }
//	  }
//	}
//
// The server speaks newline-delimited JSON-RPC 2.0 on stdin/stdout (see
// protocol.go) and logs diagnostics to stderr only — stdout is reserved
// for protocol messages. See tools.go for the exact tool schemas and
// main_test.go for example initialize/tools/list/tools/call exchanges.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/shramanb113/ZENITH/internal/collections"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/pkg/zenith"
)

const (
	serverName    = "zenith-mcp"
	serverVersion = "0.1.0"
)

func main() {
	var (
		collectionsDir string
		embedderKind   string
		model          string
		modelsDirFlag  string
		maxCollections int
		defaultMaxDocs int
	)
	flag.StringVar(&collectionsDir, "collections-dir", defaultCollectionsDir(), "Root directory for persistent collections")
	flag.StringVar(&embedderKind, "embedder", "auto", "auto|local (ONNX, falls back to deterministic if unavailable), deterministic (hash-based, non-semantic), or bm25 (lexical only, no vectors)")
	flag.StringVar(&model, "model", "", "Model id for --embedder auto|local (default: the bundled model)")
	flag.StringVar(&modelsDirFlag, "models-dir", "", "Override the directory `zenith models pull` installs non-bundled models into")
	flag.IntVar(&maxCollections, "max-collections", 1000, "Maximum number of persistent collections")
	flag.IntVar(&defaultMaxDocs, "collection-max-docs", 1_000_000, "Default per-collection document quota")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))

	modelsDir := modelsDirFlag
	if modelsDir == "" {
		modelsDir = defaultModelsDir()
	}
	emb, embName := buildEmbedder(embedderKind, model, modelsDir, logger)

	mgr, err := collections.New(collections.Config{
		Root:           collectionsDir,
		Embedder:       emb,
		MaxCollections: maxCollections,
		DefaultMaxDocs: defaultMaxDocs,
		Log:            logger,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "zenith-mcp: "+err.Error())
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go mgr.Run(ctx, time.Minute)

	logger.Info("zenith-mcp ready", "collections_dir", collectionsDir, "embedder", embName)

	tools, handlers := buildTools(mgr)
	srv := &server{name: serverName, version: serverVersion, tools: tools, handlers: handlers}

	runErr := srv.run(ctx, os.Stdin, os.Stdout)
	if cerr := mgr.CloseAll(); cerr != nil {
		logger.Error("closing collections", "error", cerr)
	}
	if runErr != nil {
		fmt.Fprintln(os.Stderr, "zenith-mcp: "+runErr.Error())
		os.Exit(1)
	}
}

// buildEmbedder mirrors the fallback story documented in CLAUDE.md's
// Embedding section: a missing/failed ONNX runtime is never fatal, it
// just degrades --embedder auto|local to lexical-only semantics via the
// deterministic embedder (with a stderr warning), so search still works.
func buildEmbedder(kind, model, modelsDir string, logger *slog.Logger) (zenith.Embedder, string) {
	switch kind {
	case "deterministic":
		return embedding.NewDeterministicEmbedder(384), "deterministic"
	case "bm25":
		return nil, "none" // nil Embedder = BM25-only, per collections.Config's doc comment
	case "", "auto", "local":
		le, err := localembedder.NewByID(model, modelsDir)
		if err != nil {
			if model != "" {
				// An explicitly requested model must fail loudly, not
				// silently swap to a different one.
				fmt.Fprintf(os.Stderr, "zenith-mcp: model %q unavailable: %v\n", model, err)
				os.Exit(1)
			}
			logger.Warn("ONNX embedder unavailable; falling back to deterministic lexical-only embedding", "error", err)
			return embedding.NewDeterministicEmbedder(384), "deterministic"
		}
		if cached, err := embedding.NewCachingEmbedder(le, 10_000); err == nil {
			return cached, le.Spec().ID
		}
		return le, le.Spec().ID
	default:
		fmt.Fprintf(os.Stderr, "zenith-mcp: unknown --embedder %q (want auto|local|deterministic|bm25)\n", kind)
		os.Exit(1)
		return nil, ""
	}
}

func defaultCollectionsDir() string {
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".zenith", "mcp-collections")
	}
	return "./zenith-mcp-collections"
}

func defaultModelsDir() string {
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".zenith", "models")
	}
	return "./zenith-models"
}
