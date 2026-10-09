package image

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/shramanb113/ZENITH/internal/activitylog"
	"github.com/shramanb113/ZENITH/internal/index"
)

const (
	ocrChunkWords   = 300
	ocrChunkOverlap = 50
)

// OCRAvailable reports whether this binary can extract text from image
// pixels (Tesseract OCR, built with -tags ocr against libtesseract). When
// false, images are indexed by filename only — never an error, just a
// smaller amount of searchable text.
func OCRAvailable() bool { return ocrAvailable() }

// OCRFile runs OCR over the image at filePath and returns whatever text is
// found, trimmed ("", nil when OCR support isn't compiled in — see
// OCRAvailable). Exported for internal/pdf's embedded-image OCR feature,
// which stages each decoded image as a temp file and reuses this same
// extraction path rather than duplicating the ocrAvailable/build-tag
// machinery.
func OCRFile(filePath string) (string, error) { return extractOCRText(filePath) }

// Indexer indexes image files by decomposing their file path into searchable
// tokens and, when OCR support is compiled in, any text the pixels contain.
// The engine's wired embedder computes vectors automatically.
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

// Chunk is one chunk of an image's extracted text, with structured fields
// rather than an opaque ID string — a caller indexing through the raw
// engine wants the baked-in ID format Index has always used (see ID
// below); a caller indexing through zenith.DB (internal/sidecar's file
// -ingest route) cannot use that format at all for an OCR chunk (zenith.DB
// reserves "||" for its own internal chunk-ID parsing, see
// pkg/zenith/zenith.go's parseChunkID, and rejects any caller-supplied ID
// containing it) and needs Kind/Index as separate values to build a safe ID
// instead.
type Chunk struct {
	Kind  string // "path" (filename-only) or "ocr"
	Index int    // 0-based chunk index within the OCR text; unused for "path"
	Text  string
}

// ID returns docID for a "path" chunk, docID||ocr||c{index} for an "ocr"
// chunk — the format Index has always stored chunks under.
func (c Chunk) ID(docID string) string {
	if c.Kind == "ocr" {
		return fmt.Sprintf("%s||ocr||c%d", docID, c.Index)
	}
	return docID
}

// ExtractChunks derives chunks from filePath's name/path tokens and, when
// OCR support is compiled in, any text found in the pixels — the same
// decomposition Index stores under, without touching an engine. ocrErr is
// non-nil only when OCR was attempted and failed (missing language data,
// corrupt image, unsupported format); chunks still reflects the
// filename-only fallback in that case. Callers should log ocrErr, never
// treat it as fatal — Index does not.
func ExtractChunks(filePath string) (chunks []Chunk, ocrErr error) {
	pathText := pathToText(filePath)

	ocrText, ocrErr := extractOCRText(filePath)
	ocrText = strings.TrimSpace(ocrText)

	if ocrText == "" {
		if pathText == "" {
			return nil, ocrErr
		}
		return []Chunk{{Kind: "path", Text: pathText}}, ocrErr
	}

	if pathText != "" {
		chunks = append(chunks, Chunk{Kind: "path", Text: pathText})
	}
	for i, chunk := range splitOCRChunks(ocrText) {
		chunks = append(chunks, Chunk{Kind: "ocr", Index: i, Text: chunk})
	}
	return chunks, ocrErr
}

// Index derives a text description from the image path — plus, when OCR
// support is compiled in, any text found in the pixels — and stores it in
// the engine. Returns the number of documents indexed (1 for filename-only;
// 1 + chunk count when OCR text was found), 0 if there is nothing to index.
// An OCR failure (missing language data, corrupt image, unsupported format)
// is logged and never fails the call: image search always falls back to
// filename-only rather than regressing or erroring out.
func (idx *Indexer) Index(ctx context.Context, docID, filePath string) (int, error) {
	chunks, ocrErr := ExtractChunks(filePath)
	if ocrErr != nil {
		idx.logger.Log("IMAGE", fmt.Sprintf("%s → OCR failed, falling back to filename-only: %v", docID, ocrErr))
	}
	if len(chunks) == 0 {
		return 0, nil
	}

	// Filename-only (no OCR text found): preserve the original single-Add
	// path rather than routing a lone document through AddBatch.
	if len(chunks) == 1 && chunks[0].Kind == "path" {
		var err error
		if len(idx.attrs) > 0 {
			err = idx.engine.AddWithVectorAttrs(ctx, docID, chunks[0].Text, idx.engine.EmbedText(ctx, chunks[0].Text), idx.attrs)
		} else {
			err = idx.engine.Add(ctx, docID, chunks[0].Text)
		}
		if err != nil {
			return 0, fmt.Errorf("image: index %s: %w", docID, err)
		}
		idx.logger.Log("IMAGE", fmt.Sprintf("%s → filename indexed", docID))
		return 1, nil
	}

	docs := make([]index.BatchDoc, len(chunks))
	for i, c := range chunks {
		docs[i] = index.BatchDoc{ID: c.ID(docID), Text: c.Text, Attrs: idx.attrs}
	}
	if err := idx.engine.AddBatch(ctx, docs); err != nil {
		return 0, fmt.Errorf("image: index %s: %w", docID, err)
	}
	idx.logger.Log("IMAGE", fmt.Sprintf("%s → filename + OCR text indexed (%d chunk(s))", docID, len(docs)))
	return len(docs), nil
}

// splitOCRChunks splits OCR'd text into overlapping word chunks, matching
// internal/pdf's splitChunks scheme so long scanned documents get the same
// per-chunk embedding relevance as a multi-page PDF instead of one
// embedding diluted across an entire page of text.
func splitOCRChunks(text string) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	var chunks []string
	for i := 0; i < len(words); {
		end := i + ocrChunkWords
		if end > len(words) {
			end = len(words)
		}
		chunks = append(chunks, strings.Join(words[i:end], " "))
		if end == len(words) {
			break
		}
		i += ocrChunkWords - ocrChunkOverlap
	}
	return chunks
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
