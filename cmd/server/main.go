package main

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
	"github.com/shramanb113/ZENITH/internal/activitylog"
	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/metrics"
	"github.com/shramanb113/ZENITH/internal/pdf"
	"github.com/shramanb113/ZENITH/internal/ranking"
	"github.com/shramanb113/ZENITH/internal/server"
	storage "github.com/shramanb113/ZENITH/internal/storage"
	"github.com/shramanb113/ZENITH/internal/storage/wal"
	"google.golang.org/grpc"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	bind := os.Getenv("ZENITH_BIND")
	if bind == "" {
		bind = "127.0.0.1"
	}
	key := os.Getenv("ZENITH_KEY")
	allowUnauthed := os.Getenv("ZENITH_ALLOW_UNAUTHENTICATED") == "1"
	addr := net.JoinHostPort(bind, "8080")
	if err := server.CheckExposure(addr, key, allowUnauthed); err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}

	lis, err := net.Listen("tcp", addr)
	if err != nil {
		slog.Error("Failed to listen on tcp socket", "error", err)
		os.Exit(1)
	}

	appConfig := config.DefaultConfig()

	storageEng, err := storage.Open(storage.DefaultEngineConfig())
	if err != nil {
		slog.Error("Failed to open storage engine", "error", err)
		os.Exit(1)
	}
	defer func() {
		if err := storageEng.Close(); err != nil {
			slog.Error("Storage engine close failed", "error", err)
		}
	}()

	alog := activitylog.Open()
	defer alog.Close()

	// Embedder: use local ONNX model when CGo is available, deterministic otherwise.
	var emb embedding.Embedder
	if localEmb, err := localembedder.New(); err == nil {
		emb, _ = embedding.NewCachingEmbedder(localEmb, 10_000)
		slog.Info("Local ONNX embedder ready")
	} else {
		slog.Warn("Local embedder unavailable, using deterministic", "error", err)
		emb = embedding.NewDeterministicEmbedder(384)
	}

	tkz := analysis.NewStandardAnalyzer()
	scorer := ranking.NewWeightedRRFRanker(appConfig.RRFConstant, appConfig.MaxResults, 1.0, appConfig.VectorWeight)
	engine := index.NewEngine(appConfig, emb, scorer, tkz)

	engine.SetFSTPath("./data/index.fst")
	engine.SetTermStore(storageEng)
	engine.SetCacheObserver(metrics.NewQueryCacheObserver())

	if err := engine.Load("zenith.db"); err != nil {
		switch {
		case os.IsNotExist(err):
			slog.Info("No existing index found, starting fresh.")
		case errors.Is(err, index.ErrIncompatibleVersion):
			slog.Error("Index file is in an older on-disk format — run `zenith migrate --db zenith.db` (keeps a backup), then restart", "error", err)
			os.Exit(1)
		case len(storageEng.Records()) == 0:
			// The gob snapshot exists but failed to load, and there is no WAL
			// delta to reconstruct from. Starting "fresh" here would silently
			// discard the corrupt file's data on the next Save — surface it
			// instead so the operator can investigate or restore a backup.
			slog.Error("Index file exists but failed to load, and no WAL delta is available to recover from", "error", err)
			os.Exit(1)
		default:
			slog.Warn("Index file failed to load; rebuilding from WAL delta only", "error", err)
		}
	} else {
		slog.Info("Successfully loaded index from disk.")
		alog.Log("LOADED", "zenith.db")
	}

	// Replay WAL delta — documents indexed since the last gob checkpoint.
	// The journal is NOT set yet, so these Add/Remove calls do not re-journal.
	replayCtx := context.Background()
	for _, r := range storageEng.Records() {
		switch r.Op {
		case wal.OpTypePut:
			text, attrs := index.DecodeJournalValue(r.Value)
			var err error
			if len(attrs) > 0 {
				err = engine.AddWithVectorAttrs(replayCtx, string(r.Key), text, engine.EmbedText(replayCtx, text), attrs)
			} else {
				err = engine.Add(replayCtx, string(r.Key), text)
			}
			if err != nil {
				slog.Warn("WAL replay: re-index failed", "id", string(r.Key), "error", err)
			}
		case wal.OpTypeDelete:
			if err := engine.Remove(replayCtx, string(r.Key)); err != nil {
				slog.Warn("WAL replay: remove failed", "id", string(r.Key), "error", err)
			}
		}
	}
	// Connect the journal — all future mutations are durably recorded first.
	engine.SetDocumentJournal(storageEng)

	pdfIndexer := pdf.NewIndexer(engine, alog)
	// IndexPDF is exposed over the network via gRPC with no authentication;
	// file_path comes straight from the caller. Without this, any client
	// that can reach :8080 could make the server open and index arbitrary
	// files on the host's filesystem. Restrict it to a known directory.
	if err := pdfIndexer.SetAllowedRoot("./data/pdfs"); err != nil {
		slog.Error("Failed to set up PDF indexing root", "error", err)
		os.Exit(1)
	}

	grpcServer := grpc.NewServer(
		grpc.ChainUnaryInterceptor(server.KeyAuthUnary(key)),
		grpc.ChainStreamInterceptor(server.KeyAuthStream(key)),
	)
	zenithproto.RegisterSearchServiceServer(grpcServer, &server.ZenithServer{
		Engine:     engine,
		PDFIndexer: pdfIndexer,
		Logger:     alog,
	})

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		slog.Info("ZENITH engine is live", "address", lis.Addr().String())
		if err := grpcServer.Serve(lis); err != nil {
			slog.Error("gRPC serve failed", "error", err)
		}
	}()

	<-stop
	slog.Info("Graceful shutdown initiated")
	grpcServer.GracefulStop()

	if err := engine.Save("zenith.db"); err != nil {
		slog.Error("Failed to save index", "error", err)
	} else {
		slog.Info("Index saved. Goodbye.")
		alog.Log("SAVED", "zenith.db")
		// Checkpoint the WAL — the gob is now authoritative; clear the delta journal.
		if err := storageEng.Checkpoint(); err != nil {
			slog.Error("WAL checkpoint failed", "error", err)
		}
	}
	if err := engine.SaveANN(); err != nil {
		slog.Warn("Could not save the ANN graph; the next start will rebuild it", "error", err)
	}
	if err := engine.Close(); err != nil {
		slog.Error("Failed to release index files", "error", err)
	}
}
