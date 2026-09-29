package pdf

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	lpdf "github.com/ledongthuc/pdf"
	"github.com/shramanb113/ZENITH/internal/activitylog"
	"github.com/shramanb113/ZENITH/internal/index"
)

const (
	chunkWords   = 300
	chunkOverlap = 50
)

// PDFIndexer extracts text from PDF files and indexes them into the engine.
// The engine's wired embedder handles vector computation automatically.
type PDFIndexer struct {
	engine      *index.Engine
	logger      *activitylog.Logger
	allowedRoot string // if set, Index rejects any path resolving outside this directory
	attrs       index.Attrs
}

// SetAttrs attaches metadata to every chunk indexed from now on (used by
// `zenith index --attr`); nil clears it.
func (p *PDFIndexer) SetAttrs(a index.Attrs) { p.attrs = a }

// NewIndexer creates a PDFIndexer. An optional logger may be supplied.
func NewIndexer(e *index.Engine, logger ...*activitylog.Logger) *PDFIndexer {
	var l *activitylog.Logger
	if len(logger) > 0 && logger[0] != nil {
		l = logger[0]
	} else {
		l = activitylog.Noop()
	}
	return &PDFIndexer{engine: e, logger: l}
}

// SetAllowedRoot restricts Index to files that resolve (after following ../
// segments and symlinks) inside root. It creates root if missing.
//
// This is intended for indexers reachable from an untrusted, network-facing
// caller (e.g. the gRPC server's IndexPDF RPC), where filePath is
// attacker-controlled and must not be able to reach arbitrary files on the
// host. CLI-driven indexers that only ever receive local, operator-supplied
// paths can leave this unset.
func (p *PDFIndexer) SetAllowedRoot(root string) error {
	abs, err := filepath.Abs(root)
	if err != nil {
		return fmt.Errorf("pdf: resolve allowed root: %w", err)
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return fmt.Errorf("pdf: create allowed root: %w", err)
	}
	p.allowedRoot = filepath.Clean(abs)
	return nil
}

// resolveWithinRoot returns filePath's real, absolute path if and only if it
// falls inside root once symlinks are resolved. Resolving symlinks (rather
// than just filepath.Clean-ing the ".." segments) matters because a symlink
// placed inside an otherwise in-bounds directory could still point outside
// root, and Clean alone would not catch that.
func resolveWithinRoot(root, filePath string) (string, error) {
	abs, err := filepath.Abs(filePath)
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}
	rel, err := filepath.Rel(root, resolved)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q is outside the allowed root", filePath)
	}
	return resolved, nil
}

// Index extracts text from filePath and stores it in the engine.
// Returns the number of chunks indexed.
func (p *PDFIndexer) Index(ctx context.Context, docID, filePath string) (int, error) {
	if p.allowedRoot != "" {
		resolved, err := resolveWithinRoot(p.allowedRoot, filePath)
		if err != nil {
			return 0, fmt.Errorf("pdf: %w", err)
		}
		filePath = resolved
	}

	f, r, err := lpdf.Open(filePath)
	if err != nil {
		return 0, fmt.Errorf("pdf: open %s: %w", filePath, err)
	}
	defer f.Close()

	var docs []index.BatchDoc
	for pageNum := 1; pageNum <= r.NumPage(); pageNum++ {
		page := r.Page(pageNum)
		if page.V.IsNull() {
			continue
		}
		text, err := page.GetPlainText(nil)
		if err != nil || strings.TrimSpace(text) == "" {
			continue
		}
		for chunkIdx, chunk := range splitChunks(text) {
			docs = append(docs, index.BatchDoc{
				ID:   fmt.Sprintf("%s||p%d||c%d||text||0.00,0.00,0.00,0.00", docID, pageNum, chunkIdx),
				Text: chunk, Attrs: p.attrs,
			})
		}
	}

	if len(docs) == 0 {
		return 0, nil
	}
	if err := p.engine.AddBatch(ctx, docs); err != nil {
		return 0, fmt.Errorf("pdf: index %s: %w", docID, err)
	}
	p.logger.Log("PDF", fmt.Sprintf("%s → %d chunks indexed", docID, len(docs)))
	return len(docs), nil
}

func splitChunks(text string) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	var chunks []string
	for i := 0; i < len(words); {
		end := i + chunkWords
		if end > len(words) {
			end = len(words)
		}
		chunks = append(chunks, strings.Join(words[i:end], " "))
		if end == len(words) {
			break
		}
		i += chunkWords - chunkOverlap
	}
	return chunks
}
