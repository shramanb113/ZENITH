package image

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/shramanb113/ZENITH/internal/activitylog"
	"github.com/shramanb113/ZENITH/internal/index"
)

// Indexer indexes image files by decomposing their file path into searchable tokens.
// The engine's wired embedder computes the vector automatically.
type Indexer struct {
	engine *index.Engine
	logger *activitylog.Logger
	attrs  index.Attrs
}

// SetAttrs attaches metadata to every image indexed from now on (used by
// `zenith index --attr`); nil clears it.
func (idx *Indexer) SetAttrs(a index.Attrs) { idx.attrs = a }

// NewIndexer creates an Indexer. An optional logger may be supplied.
func NewIndexer(e *index.Engine, logger ...*activitylog.Logger) *Indexer {
	var l *activitylog.Logger
	if len(logger) > 0 && logger[0] != nil {
		l = logger[0]
	} else {
		l = activitylog.Noop()
	}
	return &Indexer{engine: e, logger: l}
}

// Index derives a text description from the image path and stores it in the engine.
// Returns 1 on success, 0 if the path yields no tokens.
func (idx *Indexer) Index(ctx context.Context, docID, filePath string) (int, error) {
	text := pathToText(filePath)
	if text == "" {
		return 0, nil
	}
	var err error
	if len(idx.attrs) > 0 {
		err = idx.engine.AddWithVectorAttrs(ctx, docID, text, idx.engine.EmbedText(ctx, text), idx.attrs)
	} else {
		err = idx.engine.Add(ctx, docID, text)
	}
	if err != nil {
		return 0, fmt.Errorf("image: index %s: %w", docID, err)
	}
	idx.logger.Log("IMAGE", fmt.Sprintf("%s → filename indexed", docID))
	return 1, nil
}

// pathToText converts a file path into a space-separated string of search tokens.
// Example: /home/user/Photos/2024/vacation/portrait_sunset_beach.jpg
//
//	→ "portrait sunset beach jpg vacation 2024"
func pathToText(filePath string) string {
	if filePath == "" {
		return ""
	}
	base := filepath.Base(filePath)
	ext := strings.TrimPrefix(filepath.Ext(base), ".")
	nameNoExt := strings.TrimSuffix(base, filepath.Ext(base))

	var parts []string
	parts = append(parts, splitTokens(nameNoExt)...)
	if ext != "" {
		parts = append(parts, ext)
	}

	// Include up to 3 ancestor directory names for context.
	dir := filepath.Dir(filePath)
	for i := 0; i < 3; i++ {
		seg := filepath.Base(dir)
		if seg == "." || seg == ".." || seg == string(filepath.Separator) || seg == "/" {
			break
		}
		parts = append(parts, splitTokens(seg)...)
		dir = filepath.Dir(dir)
	}
	return strings.Join(parts, " ")
}

func splitTokens(s string) []string {
	s = strings.NewReplacer("_", " ", "-", " ", ".", " ").Replace(s)
	return strings.Fields(s)
}
