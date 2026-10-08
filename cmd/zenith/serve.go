package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/collections"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/metrics"
	"github.com/shramanb113/ZENITH/internal/server"
	"github.com/shramanb113/ZENITH/internal/sidecar"
	"github.com/shramanb113/ZENITH/pkg/zenith"
	"github.com/spf13/cobra"
	"google.golang.org/grpc"
)

var serveFlags struct {
	port          string
	http          string
	key           string
	synonyms      string
	bind          string
	allowUnauthed bool

	collectionsDir      string
	noCollections       bool
	maxCollections      int
	collectionMaxDocs   int
	collectionMaxOpen   int
	collectionIdleClose time.Duration

	metricsAddr string

	queryCacheRedisAddr string
}

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the gRPC search server",
	Long: `Starts the ZENITH gRPC server and exposes the full search and indexing
API for remote clients. Loads the existing index from zenith.db on startup
and saves it on graceful shutdown (SIGINT / SIGTERM).

Writes received over gRPC are held in memory and saved on graceful shutdown
(SIGINT/SIGTERM) only; a SIGKILL loses them. For write-ahead-logged writes
where every acknowledged write survives a kill, use the Go library
(pkg/zenith) directly.`,

	RunE: func(cmd *cobra.Command, args []string) error {
		setupLogger()
		if serveFlags.http != "" {
			addr, err := server.ResolveListenAddr(serveFlags.http, serveFlags.bind)
			if err != nil {
				return err
			}
			if err := server.CheckExposure(addr, serveFlags.key, serveFlags.allowUnauthed); err != nil {
				return err
			}
			return runHTTP(addr)
		}

		addr := net.JoinHostPort(serveFlags.bind, serveFlags.port)
		if err := server.CheckExposure(addr, serveFlags.key, serveFlags.allowUnauthed); err != nil {
			return err
		}

		lis, err := net.Listen("tcp", addr)
		if err != nil {
			return fmt.Errorf("listen %s: %w", addr, err)
		}

		printHeader("serve", addr)

		engine, _, alog, teardown, err := buildEngine(true)
		if err != nil {
			return fmt.Errorf("engine init: %w", err)
		}
		_ = alog

		metrics.BuildInfo.WithLabelValues(version, cliFlags.model).Set(1)
		metrics.RegisterIndexDocuments(func() float64 { return float64(engine.Count()) })
		metricsSrv, err := startMetrics(serveFlags.metricsAddr, serveFlags.key, serveFlags.allowUnauthed)
		if err != nil {
			return err
		}

		// Metrics first, so a call KeyAuth rejects is still counted.
		grpcServer := grpc.NewServer(
			grpc.ChainUnaryInterceptor(metrics.UnaryServerInterceptor(), server.KeyAuthUnary(serveFlags.key)),
			grpc.ChainStreamInterceptor(metrics.StreamServerInterceptor(), server.KeyAuthStream(serveFlags.key)),
		)
		zenithproto.RegisterSearchServiceServer(grpcServer, &server.ZenithServer{Engine: engine})

		stop := make(chan os.Signal, 1)
		signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

		go func() {
			slog.Info("ZENITH gRPC server live", "addr", lis.Addr())
			if err := grpcServer.Serve(lis); err != nil {
				slog.Error("gRPC serve error", "error", err)
			}
		}()

		fmt.Printf("  %s  gRPC listening on %s\n", green("✓"), bold(addr))
		fmt.Printf("  %s  Press Ctrl-C to stop\n\n", dim("·"))

		<-stop
		fmt.Printf("\n  %s  Shutting down...\n", dim("·"))
		grpcServer.GracefulStop()
		if metricsSrv != nil {
			_ = metricsSrv.Close()
		}
		teardown()
		return nil
	},
}

func init() {
	addEngineFlags(serveCmd)
	serveCmd.Flags().StringVarP(&serveFlags.port, "port", "p", "8080", "gRPC listen port")
	serveCmd.Flags().StringVar(&serveFlags.http, "http", "", "Serve the HTTP/JSON namespace sidecar on this address (e.g. :7700) instead of gRPC")
	serveCmd.Flags().StringVar(&serveFlags.key, "key", os.Getenv("ZENITH_KEY"), "Shared secret: X-Zenith-Key on HTTP /v1/*, x-zenith-key metadata on gRPC (default $ZENITH_KEY)")
	serveCmd.Flags().StringVar(&serveFlags.synonyms, "synonyms", "", "Synonyms file merged into query expansion (HTTP mode)")
	serveCmd.Flags().StringVar(&serveFlags.bind, "bind", "127.0.0.1", "Interface for the gRPC port and for --http values without a host (use 0.0.0.0 inside containers)")
	serveCmd.Flags().BoolVar(&serveFlags.allowUnauthed, "allow-unauthenticated", false, "Allow listening on a non-loopback address with no --key")
	serveCmd.Flags().StringVar(&serveFlags.collectionsDir, "collections-dir", zenithDataPath("collections"), "Root directory for persistent HTTP collections (HTTP mode)")
	serveCmd.Flags().BoolVar(&serveFlags.noCollections, "no-collections", false, "Serve only ephemeral namespaces; disable /v1/collections/*")
	serveCmd.Flags().IntVar(&serveFlags.maxCollections, "max-collections", 1000, "Maximum number of persistent collections")
	serveCmd.Flags().IntVar(&serveFlags.collectionMaxDocs, "collection-max-docs", 1_000_000, "Default per-collection document quota")
	serveCmd.Flags().IntVar(&serveFlags.collectionMaxOpen, "collection-max-open", 64, "Maximum number of collections open (mapped into memory) at once")
	serveCmd.Flags().DurationVar(&serveFlags.collectionIdleClose, "collection-idle-close", 10*time.Minute, "Idle time before an open collection is closed")
	serveCmd.Flags().StringVar(&serveFlags.metricsAddr, "metrics-addr", "", "Prometheus /metrics listen address (e.g. 127.0.0.1:9464); empty disables it")
	serveCmd.Flags().StringVar(&serveFlags.queryCacheRedisAddr, "collection-query-cache-redis-addr", "", "Optional shared L2 Redis address for the query-result cache across every persistent collection (HTTP mode); empty keeps each collection's cache in-process only")
}

// startMetrics starts the Prometheus /metrics listener when addr is
// non-empty, returning (nil, nil) when it's disabled. The listener has no
// auth of its own — collection ids can appear in its labels — so a
// non-loopback address needs the same --key/--allow-unauthenticated
// exposure check as the main server.
func startMetrics(addr, key string, allowUnauthed bool) (*http.Server, error) {
	if addr == "" {
		return nil, nil
	}
	if err := server.CheckExposure(addr, key, allowUnauthed); err != nil {
		return nil, fmt.Errorf("metrics-addr: %w", err)
	}
	mux := http.NewServeMux()
	mux.Handle("GET /metrics", metrics.Handler())
	hs := &http.Server{Addr: addr, Handler: mux}
	go func() {
		if err := hs.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("metrics server error", "error", err)
		}
	}()
	slog.Info("metrics ready", "addr", addr)
	return hs, nil
}

// runHTTP serves the HTTP/JSON API: ephemeral per-request namespaces (/v1/ns/*, in-memory, for
// classify-and-discard workloads) and, unless --no-collections, persistent named collections
// (/v1/collections/*) under --collections-dir, each a durable pkg/zenith DB. It never opens the
// --db index.
func runHTTP(addr string) error {
	emb, model, err := sidecarEmbedder(cliFlags.embedder, cliFlags.model)
	if err != nil {
		return err
	}

	synHash := ""
	if serveFlags.synonyms != "" {
		b, err := os.ReadFile(serveFlags.synonyms)
		if err != nil {
			return fmt.Errorf("synonyms: %w", err)
		}
		n, err := analysis.LoadSynonyms(bytes.NewReader(b), analysis.NewStandardAnalyzer())
		if err != nil {
			return err
		}
		sum := sha256.Sum256(b)
		synHash = hex.EncodeToString(sum[:4])
		slog.Info("synonyms loaded", "mappings", n, "hash", synHash)
	}

	var mgr *collections.Manager
	if !serveFlags.noCollections {
		mgr, err = collections.New(collections.Config{
			Root:                        serveFlags.collectionsDir,
			Embedder:                    emb,
			MaxCollections:              serveFlags.maxCollections,
			DefaultMaxDocs:              serveFlags.collectionMaxDocs,
			MaxOpen:                     serveFlags.collectionMaxOpen,
			IdleClose:                   serveFlags.collectionIdleClose,
			QueryCacheRedisAddr:         serveFlags.queryCacheRedisAddr,
			QueryCacheSize:              cliFlags.queryCacheSize,
			QueryCacheTTL:               cliFlags.queryCacheTTL,
			QueryCacheSemanticThreshold: cliFlags.queryCacheSemanticThreshold,
			ANNThresholdBandPct:         cliFlags.annThresholdBandPct,
		})
		if err != nil {
			return fmt.Errorf("collections: %w", err)
		}
		total, _ := mgr.Counts()
		slog.Info("collections ready", "root", serveFlags.collectionsDir, "found", total)
		if serveFlags.key == "" {
			slog.Warn("collections management API is unauthenticated (no --key)")
		}
	}

	metrics.BuildInfo.WithLabelValues(version, model).Set(1)
	metricsSrv, err := startMetrics(serveFlags.metricsAddr, serveFlags.key, serveFlags.allowUnauthed)
	if err != nil {
		return err
	}

	srv := sidecar.New(sidecar.Config{
		Key: serveFlags.key, Version: version, Model: model, Embedder: emb, Synonyms: synHash, Collections: mgr,
		QueryCacheSize:              cliFlags.queryCacheSize,
		QueryCacheTTL:               cliFlags.queryCacheTTL,
		QueryCacheSemanticThreshold: cliFlags.queryCacheSemanticThreshold,
		ANNThresholdBandPct:         cliFlags.annThresholdBandPct,
	})
	// A collection upsert embeds on the request path, so both timeouts are far
	// more generous than the ephemeral-only defaults were.
	hs := &http.Server{Addr: addr, Handler: srv.Handler(), ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout: 60 * time.Second, WriteTimeout: 120 * time.Second}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go srv.Run(ctx, time.Minute)
	if mgr != nil {
		go mgr.Run(ctx, time.Minute)
	}
	go func() {
		<-ctx.Done()
		// 30s: long enough for hs.Shutdown to drain an in-flight collection
		// upsert before CloseAll below checkpoints every open collection.
		shut, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_ = hs.Shutdown(shut)
		if metricsSrv != nil {
			_ = metricsSrv.Close()
		}
	}()
	printHeader("serve", addr)
	fmt.Printf("  %s  HTTP sidecar on %s (model %s, key %v)\n", green("✓"), bold(addr), model, serveFlags.key != "")
	if err := hs.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	srv.Close()
	if mgr != nil {
		if err := mgr.CloseAll(); err != nil {
			return fmt.Errorf("closing collections: %w", err)
		}
	}
	return nil
}

// sidecarEmbedder returns one embedder shared by every namespace and its model id.
// An explicitly requested --model that cannot load is a hard error — it must never
// silently degrade to a different model (same rule as localEmbedderOrFallback).
func sidecarEmbedder(kind, model string) (zenith.Embedder, string, error) {
	switch kind {
	case "deterministic":
		if model != "" {
			return nil, "", fmt.Errorf("--model requires --embedder auto|local")
		}
		return embedding.NewDeterministicEmbedder(384), "deterministic", nil
	case "", "auto", "local":
		le, err := localembedder.NewByID(model, modelsDir())
		if err != nil {
			if model != "" {
				return nil, "", err
			}
			slog.Warn("ONNX embedder unavailable; sidecar runs BM25 + fuzzy only", "error", err)
			return nil, "none", nil
		}
		// Instrument inside the cache, not outside it, so a cache hit is
		// never counted as inference time.
		if cached, err := embedding.NewCachingEmbedder(metrics.InstrumentEmbedder(le), 10_000); err == nil {
			return cached, le.Spec().ID, nil
		}
		return metrics.InstrumentEmbedder(le), le.Spec().ID, nil
	default:
		slog.Warn("embedder not supported in HTTP mode; using BM25 + fuzzy only", "embedder", kind)
		return nil, "none", nil
	}
}
