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
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/server"
	"github.com/shramanb113/ZENITH/internal/sidecar"
	"github.com/shramanb113/ZENITH/pkg/zenith"
	"github.com/spf13/cobra"
	"google.golang.org/grpc"
)

var serveFlags struct {
	port     string
	http     string
	key      string
	synonyms string
}

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the gRPC search server",
	Long: `Starts the ZENITH gRPC server and exposes the full search and indexing
API for remote clients. Loads the existing index from zenith.db on startup
and saves it on graceful shutdown (SIGINT / SIGTERM).`,

	RunE: func(cmd *cobra.Command, args []string) error {
		setupLogger()
		if serveFlags.http != "" {
			return runHTTP(serveFlags.http)
		}

		lis, err := net.Listen("tcp", ":"+serveFlags.port)
		if err != nil {
			return fmt.Errorf("listen :%s: %w", serveFlags.port, err)
		}

		printHeader("serve", ":"+serveFlags.port)

		engine, alog, teardown, err := buildEngine(true)
		if err != nil {
			return fmt.Errorf("engine init: %w", err)
		}
		_ = alog

		grpcServer := grpc.NewServer()
		zenithproto.RegisterSearchServiceServer(grpcServer, &server.ZenithServer{Engine: engine})

		stop := make(chan os.Signal, 1)
		signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

		go func() {
			slog.Info("ZENITH gRPC server live", "addr", lis.Addr())
			if err := grpcServer.Serve(lis); err != nil {
				slog.Error("gRPC serve error", "error", err)
			}
		}()

		fmt.Printf("  %s  gRPC listening on %s\n", green("✓"), bold(":"+serveFlags.port))
		fmt.Printf("  %s  Press Ctrl-C to stop\n\n", dim("·"))

		<-stop
		fmt.Printf("\n  %s  Shutting down...\n", dim("·"))
		grpcServer.GracefulStop()
		teardown()
		return nil
	},
}

func init() {
	addEngineFlags(serveCmd)
	serveCmd.Flags().StringVarP(&serveFlags.port, "port", "p", "8080", "gRPC listen port")
	serveCmd.Flags().StringVar(&serveFlags.http, "http", "", "Serve the HTTP/JSON namespace sidecar on this address (e.g. :7700) instead of gRPC")
	serveCmd.Flags().StringVar(&serveFlags.key, "key", os.Getenv("ZENITH_KEY"), "Shared secret required in X-Zenith-Key (default $ZENITH_KEY)")
	serveCmd.Flags().StringVar(&serveFlags.synonyms, "synonyms", "", "Synonyms file merged into query expansion (HTTP mode)")
}

// runHTTP serves the per-request namespace sidecar used by Kshetra IQ. It never opens zenith.db.
func runHTTP(addr string) error {
	emb, model := sidecarEmbedder(cliFlags.embedder)

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

	srv := sidecar.New(sidecar.Config{Key: serveFlags.key, Version: version, Model: model, Embedder: emb, Synonyms: synHash})
	hs := &http.Server{Addr: addr, Handler: srv.Handler(), ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout: 15 * time.Second, WriteTimeout: 30 * time.Second}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go srv.Run(ctx, time.Minute)
	go func() {
		<-ctx.Done()
		shut, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = hs.Shutdown(shut)
	}()
	printHeader("serve", addr)
	fmt.Printf("  %s  HTTP sidecar on %s (model %s, key %v)\n", green("✓"), bold(addr), model, serveFlags.key != "")
	if err := hs.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	srv.Close()
	return nil
}

// sidecarEmbedder returns one embedder shared by every namespace and its model id.
func sidecarEmbedder(kind string) (zenith.Embedder, string) {
	switch kind {
	case "deterministic":
		return embedding.NewDeterministicEmbedder(384), "deterministic"
	case "", "auto", "local":
		le, err := localembedder.New()
		if err != nil {
			slog.Warn("ONNX embedder unavailable; sidecar runs BM25 + fuzzy only", "error", err)
			return nil, "none"
		}
		if cached, err := embedding.NewCachingEmbedder(le, 10_000); err == nil {
			return cached, "all-MiniLM-L6-v2"
		}
		return le, "all-MiniLM-L6-v2"
	default:
		slog.Warn("embedder not supported in HTTP mode; using BM25 + fuzzy only", "embedder", kind)
		return nil, "none"
	}
}
