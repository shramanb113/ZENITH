package pdf

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"

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
}

// WithAttrs returns a view of p whose Index(ctx, docID, filePath) attaches
// attrs to every chunk — the three-argument shape crawler.FileIndexer expects
// (used by `zenith index --attr`). attrs is bound to the returned value, not
// stored on p, so concurrent callers with different attrs never interfere.
func (p *PDFIndexer) WithAttrs(attrs index.Attrs) AttrBoundIndexer {
	return AttrBoundIndexer{p: p, attrs: attrs}
}

// AttrBoundIndexer is a PDFIndexer with fixed per-chunk attributes.
type AttrBoundIndexer struct {
	p     *PDFIndexer
	attrs index.Attrs
}

// Index is PDFIndexer.Index with the bound attributes.
func (b AttrBoundIndexer) Index(ctx context.Context, docID, filePath string) (int, error) {
	return b.p.Index(ctx, docID, filePath, b.attrs)
}

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

// Chunk is one chunk of a PDF's extracted content — either a windowed
// slice of a page's native text (Kind "text") or OCR'd text from one of a
// page's embedded images (Kind "image", see images.go) — with structured
// position fields rather than an opaque ID string. A caller indexing
// through the raw engine wants the baked-in ID format Index has always
// used (see ID below); a caller indexing through zenith.DB
// (internal/sidecar's file-ingest route) cannot use that format at all
// (zenith.DB reserves "||" for its own internal chunk-ID parsing, see
// pkg/zenith/zenith.go's parseChunkID, and rejects any caller-supplied ID
// containing it) and needs Page/Index/Kind/BBox as separate values to
// build a safe ID and/or attrs instead.
type Chunk struct {
	Page  int    // 1-based page number
	Index int    // 0-based index within Kind on this page
	Kind  string // "text" or "image"
	Text  string
	// BBox fields are only meaningful for Kind=="text" (the chunk's
	// position on the page); an "image" chunk's bbox is not computed — no
	// demand yet to highlight an embedded image's own page position.
	BBoxX, BBoxY, BBoxW, BBoxH float32
}

// ID returns the format Index has always stored a chunk under:
// docID||p{page}||c{index}||text||x,y,w,h for Kind=="text",
// docID||p{page}||img{index}||ocr for Kind=="image".
func (c Chunk) ID(docID string) string {
	if c.Kind == "image" {
		return fmt.Sprintf("%s||p%d||img%d||ocr", docID, c.Page, c.Index)
	}
	return fmt.Sprintf("%s||p%d||c%d||text||%.2f,%.2f,%.2f,%.2f",
		docID, c.Page, c.Index, c.BBoxX, c.BBoxY, c.BBoxW, c.BBoxH)
}

// ExtractChunks parses filePath and returns its page-chunked native text
// plus OCR'd text from embedded images large enough to be worth it (see
// images.go's minImageArea) — a page with little or no native text still
// yields its image chunks, which is the point of the feature: a
// screenshot-only or diagram-only page is otherwise invisible to search.
// It applies no root confinement and touches no engine — callers that need
// SetAllowedRoot's confinement or want the result persisted do that
// themselves (Index wraps this for the local-engine path; internal/sidecar's
// file-ingest route wraps it for the collections path).
func ExtractChunks(filePath string) ([]Chunk, error) {
	f, r, err := lpdf.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("pdf: open %s: %w", filePath, err)
	}
	defer f.Close()

	var chunks []Chunk
	for pageNum := 1; pageNum <= r.NumPage(); pageNum++ {
		page := r.Page(pageNum)
		if page.V.IsNull() {
			continue
		}
		if text, err := page.GetPlainText(nil); err == nil && strings.TrimSpace(text) != "" {
			ranges := splitChunkRanges(text)
			boxes := pageChunkBoxes(page, ranges)
			for chunkIdx, cr := range ranges {
				bx := boxes[chunkIdx]
				chunks = append(chunks, Chunk{
					Page: pageNum, Index: chunkIdx, Kind: "text", Text: cr.text,
					BBoxX: bx.x, BBoxY: bx.y, BBoxW: bx.w, BBoxH: bx.h,
				})
			}
		}
		chunks = append(chunks, extractPageImageChunks(page, pageNum)...)
	}
	return chunks, nil
}

// Index extracts text from filePath and stores it in the engine, attaching
// attrs (nil for none) to every chunk. Returns the number of chunks indexed.
// attrs is a parameter rather than indexer state so one PDFIndexer can serve
// concurrent requests (the gRPC IndexPDF handler) with different metadata.
func (p *PDFIndexer) Index(ctx context.Context, docID, filePath string, attrs index.Attrs) (int, error) {
	if p.allowedRoot != "" {
		resolved, err := resolveWithinRoot(p.allowedRoot, filePath)
		if err != nil {
			return 0, fmt.Errorf("pdf: %w", err)
		}
		filePath = resolved
	}

	chunks, err := ExtractChunks(filePath)
	if err != nil {
		return 0, err
	}
	if len(chunks) == 0 {
		return 0, nil
	}

	docs := make([]index.BatchDoc, len(chunks))
	for i, c := range chunks {
		docs[i] = index.BatchDoc{ID: c.ID(docID), Text: c.Text, Attrs: attrs}
	}
	if err := p.engine.AddBatch(ctx, docs); err != nil {
		return 0, fmt.Errorf("pdf: index %s: %w", docID, err)
	}
	p.logger.Log("PDF", fmt.Sprintf("%s → %d chunks indexed", docID, len(docs)))
	return len(docs), nil
}

// chunkRange is one chunk's text together with the half-open [Start, End)
// word-index range (indexing into strings.Fields(text) of the page the
// chunk came from) it was built from. The index range lets pageChunkBoxes
// line up each chunk with the same slice of the page's position-tagged
// word stream.
type chunkRange struct {
	text       string
	start, end int
}

// splitChunkRanges implements the 300-word/50-overlap windowing and also
// records, for each chunk, the word-index range it was built from.
func splitChunkRanges(text string) []chunkRange {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	var ranges []chunkRange
	for i := 0; i < len(words); {
		end := i + chunkWords
		if end > len(words) {
			end = len(words)
		}
		ranges = append(ranges, chunkRange{text: strings.Join(words[i:end], " "), start: i, end: end})
		if end == len(words) {
			break
		}
		i += chunkWords - chunkOverlap
	}
	return ranges
}

func splitChunks(text string) []string {
	ranges := splitChunkRanges(text)
	if ranges == nil {
		return nil
	}
	chunks := make([]string, len(ranges))
	for i, r := range ranges {
		chunks[i] = r.text
	}
	return chunks
}

// bbox is a chunk's bounding box on its page, in the same units emitted in
// the chunk ID: top-left origin, Y increasing downward (see pageChunkBoxes).
// The zero value is the "no position data" box ("0.00,0.00,0.00,0.00"),
// used whenever real coordinates can't be computed for a chunk.
type bbox struct{ x, y, w, h float32 }

// contentWord is one word reconstructed from the page's content stream
// (lpdf.Page.Content(), which — unlike GetPlainText — exposes one Text
// entry per decoded character, each with its own X/Y/width/font size),
// together with the union box of the characters that made it up. Boxes are
// in the PDF's native coordinate space: bottom-left origin, Y increasing
// upward (see lpdf.Text's doc comment).
type contentWord struct {
	minX, minY, maxX, maxY float64
}

// contentWords walks a page's Content() character stream and groups
// consecutive non-whitespace characters into words, splitting purely on
// literal whitespace characters — the same rule GetPlainText uses to
// separate words in its own output (it never inserts whitespace for a
// position gap between runs). Using the same rule here is what lets
// pageChunkBoxes line this word list up 1:1 with strings.Fields(text) from
// GetPlainText: both merge or split words identically, including PDF
// quirks like two words on adjacent content-stream lines running together
// with no space when the text operator between them is a bare position
// move (Td) rather than a newline op (T*/BT).
func contentWords(c lpdf.Content) []contentWord {
	var words []contentWord
	var cur contentWord
	open := false
	flush := func() {
		if open {
			words = append(words, cur)
			open = false
		}
	}
	for _, t := range c.Text {
		if t.S == "" {
			continue
		}
		r, _ := utf8.DecodeRuneInString(t.S)
		if unicode.IsSpace(r) {
			flush()
			continue
		}
		// Approximate each character's height as its font size: Content()
		// gives only a baseline Y and a width, no ascent/descent metrics.
		h := t.FontSize
		if h <= 0 {
			h = 1
		}
		x0, y0, x1, y1 := t.X, t.Y, t.X+t.W, t.Y+h
		if !open {
			cur = contentWord{x0, y0, x1, y1}
			open = true
			continue
		}
		if x0 < cur.minX {
			cur.minX = x0
		}
		if y0 < cur.minY {
			cur.minY = y0
		}
		if x1 > cur.maxX {
			cur.maxX = x1
		}
		if y1 > cur.maxY {
			cur.maxY = y1
		}
	}
	flush()
	return words
}

// mediaBoxHeight returns a page's MediaBox height in PDF points, walking
// the inherited /Parent chain the way lpdf's own (unexported)
// Page.findInherited does — MediaBox is commonly set once on the Pages
// root and inherited by every leaf page rather than repeated per page.
// Returns 0 if no usable MediaBox is found.
func mediaBoxHeight(v lpdf.Value) float64 {
	for ; !v.IsNull(); v = v.Key("Parent") {
		mb := v.Key("MediaBox")
		if mb.IsNull() || mb.Len() != 4 {
			continue
		}
		if h := mb.Index(3).Float64() - mb.Index(1).Float64(); h > 0 {
			return h
		}
	}
	return 0
}

// unionBox merges a contiguous slice of a page's content-stream words into
// one box and converts it from PDF's native bottom-left-origin, Y-up space
// to a top-left-origin, Y-down space.
//
// Coordinate-system choice: lpdf.Text documents its X/Y as PDF-native
// (origin bottom-left, Y increasing upward — the PDF spec's default user
// space). That's the wrong space for this feature's actual consumer: a
// bbox exists so a UI can draw a highlight rectangle over a *rendered*
// page image, and every common rendering/display surface (HTML, canvas,
// browser PDF viewers' overlay coordinates) places (0,0) at the top-left
// with Y increasing downward. Flipping once here, at the one place that
// has both the raw box and the page height needed to flip it, means every
// downstream consumer (server.go, zenith.go, result.go) can treat bbox.y
// as "distance down from the top of the page" with no further conversion.
func unionBox(words []contentWord, pageHeight float64) bbox {
	if len(words) == 0 {
		return bbox{}
	}
	minX, minY, maxX, maxY := words[0].minX, words[0].minY, words[0].maxX, words[0].maxY
	for _, w := range words[1:] {
		if w.minX < minX {
			minX = w.minX
		}
		if w.minY < minY {
			minY = w.minY
		}
		if w.maxX > maxX {
			maxX = w.maxX
		}
		if w.maxY > maxY {
			maxY = w.maxY
		}
	}
	x := minX
	y := pageHeight - maxY
	w := maxX - minX
	h := maxY - minY
	if y < 0 {
		// Font-size-based height is an estimate, not real glyph ascent, so
		// it can push a box's top very slightly above the page edge for
		// text near the top margin; clamp rather than emit a negative
		// coordinate a consumer wouldn't expect.
		y = 0
	}
	return bbox{float32(x), float32(y), float32(w), float32(h)}
}

// pageChunkBoxes computes a real union bounding box for every chunk range
// on a page, falling back to the zero bbox (same "0.00,0.00,0.00,0.00" a
// chunk got unconditionally before this feature existed) for every chunk
// on that page whenever the computation can't be trusted:
//
//   - no MediaBox (or a degenerate one) — can't convert to top-left space.
//   - the word count contentWords(page.Content()) produces doesn't match
//     the word count GetPlainText/strings.Fields produced (the word list
//     splitChunkRanges actually chunked). A mismatch means some PDF quirk
//     made the two extraction paths disagree on where word boundaries
//     fall, and lining chunk index ranges up against the wrong word
//     stream would silently attach a plausible-looking but wrong box to a
//     chunk — worse than an honest zero.
//   - page.Content() panics on a malformed content stream. Unlike
//     GetPlainText, Content() has no built-in recover(), so one is applied
//     here: a bad page degrades to zero boxes for that page rather than
//     aborting indexing of the rest of the PDF.
//
// The returned slice is always len(ranges) long and index-aligned with it.
func pageChunkBoxes(page lpdf.Page, ranges []chunkRange) []bbox {
	zero := make([]bbox, len(ranges))
	if len(ranges) == 0 {
		return zero
	}

	boxes := zero
	func() {
		defer func() {
			if recover() != nil {
				boxes = zero
			}
		}()

		height := mediaBoxHeight(page.V)
		if height <= 0 {
			return
		}
		words := contentWords(page.Content())
		// ranges is non-empty and built so its last entry's end is always
		// the total word count from strings.Fields(text); comparing
		// against it is equivalent to recomputing that count.
		if len(words) != ranges[len(ranges)-1].end {
			return
		}
		out := make([]bbox, len(ranges))
		for i, cr := range ranges {
			out[i] = unionBox(words[cr.start:cr.end], height)
		}
		boxes = out
	}()
	return boxes
}
