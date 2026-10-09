package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"

	"github.com/shramanb113/ZENITH/internal/crawler"
	"github.com/shramanb113/ZENITH/internal/fileindex"
	"github.com/shramanb113/ZENITH/internal/index"
	imageindexer "github.com/shramanb113/ZENITH/internal/image"
	"github.com/shramanb113/ZENITH/internal/pdf"
	"github.com/spf13/cobra"
)

var indexFlags struct {
	attrs []string
}

var indexCmd = &cobra.Command{
	Use:   "index <directory>",
	Short: "Bulk-index all supported files in a directory",
	Long: `Recursively walks <directory>, extracts text from each supported file,
and adds it to the local index (zenith.db). Running index twice on the same
directory is safe — documents are re-indexed idempotently.

Supported formats:
  .txt .md .log .csv .json .yaml .yml  — raw text
  .go                                  — AST-extracted identifiers + comments
  .py .ts .js .jsx .tsx .rs .java .c   — raw source
  .html .htm                           — tag-stripped visible text`,

	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		setupLogger()
		dir := args[0]

		if info, err := os.Stat(dir); err != nil || !info.IsDir() {
			return fmt.Errorf("not a directory: %s", dir)
		}

		attrs, err := parseAttrs(indexFlags.attrs)
		if err != nil {
			return err
		}

		printHeader("index", dir)

		engine, _, alog, teardown, err := buildEngine(true, true)
		if err != nil {
			return fmt.Errorf("engine init: %w", err)
		}
		defer teardown()

		// Wrap the indexer to count files as they are processed.
		var count atomic.Int64
		var inner crawler.Indexer = engine
		if len(attrs) > 0 {
			inner = &attrIndexer{engine: engine, attrs: attrs}
		}
		counted := &countingIndexer{inner: inner, n: &count}

		w, err := crawler.NewWatcher(counted, alog)
		if err != nil {
			return err
		}
		defer w.Close()

		pi := pdf.NewIndexer(engine, alog)
		w.RegisterFileIndexer(".pdf", pi.WithAttrs(attrs))

		ii := imageindexer.NewIndexer(engine, alog)
		ii.SetAttrs(attrs)
		for _, ext := range []string{".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"} {
			w.RegisterFileIndexer(ext, ii)
		}

		w.SetOnFileIndexed(func(path string) {
			n := count.Add(1)
			printProgress(n, filepath.Base(path))
		})

		// Content-hash deduplication: skip files unchanged since the last run.
		home, _ := os.UserHomeDir()
		if fi, err := fileindex.Open(filepath.Join(home, ".zenith", "file_hashes.json")); err == nil {
			w.SetSkipFile(fi.IsUpToDate)
			w.SetAfterFile(func(path string) { _ = fi.Mark(path) })
			defer fi.Save()
		}

		start := time.Now()
		ctx := context.Background()
		if err := w.IndexDir(ctx, dir); err != nil {
			clearProgress()
			return fmt.Errorf("index: %w", err)
		}

		clearProgress()
		elapsed := time.Since(start)
		n := count.Load()

		printDivider()
		printFooter(
			fmt.Sprintf("%d file%s indexed", n, plural(n)),
			elapsed.Round(time.Millisecond).String(),
		)
		return nil
	},
}

func init() {
	addEngineFlags(indexCmd)
	indexCmd.Flags().StringArrayVar(&indexFlags.attrs, "attr", nil,
		"Attach metadata to every indexed document: key=value or key=[a,b,c] for an array (repeatable), e.g. --attr tenant=acme --attr year=2024 --attr tags=[go,search]. Search with --where / --filter")
}

// attrIndexer indexes text with fixed attributes attached.
type attrIndexer struct {
	engine *index.Engine
	attrs  index.Attrs
}

func (a *attrIndexer) Add(ctx context.Context, id, text string) error {
	return a.engine.AddWithVectorAttrs(ctx, id, text, a.engine.EmbedText(ctx, text), a.attrs)
}

func (a *attrIndexer) Remove(ctx context.Context, id string) error {
	return a.engine.Remove(ctx, id)
}

// ─── helpers ──────────────────────────────────────────────────────────────────

type countingIndexer struct {
	inner crawler.Indexer
	n     *atomic.Int64
}

func (c *countingIndexer) Add(ctx context.Context, id, text string) error {
	err := c.inner.Add(ctx, id, text)
	if err == nil {
		n := c.n.Add(1)
		printProgress(n, filepath.Base(id))
	}
	return err
}

func (c *countingIndexer) Remove(ctx context.Context, id string) error {
	return c.inner.Remove(ctx, id)
}

func plural(n int64) string {
	if n == 1 {
		return ""
	}
	return "s"
}
