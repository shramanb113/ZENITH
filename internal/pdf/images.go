package pdf

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"os"

	lpdf "github.com/ledongthuc/pdf"
	imageindexer "github.com/shramanb113/ZENITH/internal/image"
)

// minImageArea is the minimum width*height (in pixels) an embedded image
// XObject must have to be considered for OCR. Measured against a real
// 276-page study-guide PDF: 1,354 of 1,911 embedded images were under
// 100x100 (decorative bullets/icons repeated across nearly every page) and
// 381 were 300x300 or larger (screenshots/diagrams, worth OCR). This
// threshold sits between those two populations, erring toward skipping
// decorative noise rather than OCR'ing every icon.
const minImageArea = 150 * 150

// supportedImageFilter reports whether filterName is a PDF image stream
// this package can decode into a Go image.Image. CCITTFaxDecode,
// JBIG2Decode and JPXDecode streams pass through Value.Reader() unchanged
// (see third_party/ledongthuc_pdf's fork) but this package has no decoder
// for any of them (no fax/JBIG2/JPEG2000 decoder in the Go stdlib or this
// project's existing dependencies) — none were found in the one real
// document this feature was built against, so they are a documented gap,
// not a silent guess.
func supportedImageFilter(filterName string) bool {
	switch filterName {
	case "DCTDecode", "FlateDecode", "":
		return true
	}
	return false
}

// pageImageRefs returns every image XObject on page big enough to be worth
// decoding (see minImageArea).
func pageImageRefs(page lpdf.Page) []lpdf.Value {
	res := page.V.Key("Resources")
	if res.IsNull() {
		return nil
	}
	xo := res.Key("XObject")
	if xo.IsNull() {
		return nil
	}
	var refs []lpdf.Value
	for _, key := range xo.Keys() {
		obj := xo.Key(key)
		if obj.Key("Subtype").Name() != "Image" {
			continue
		}
		w := int(obj.Key("Width").Int64())
		h := int(obj.Key("Height").Int64())
		if w <= 0 || h <= 0 || w*h < minImageArea {
			continue
		}
		refs = append(refs, obj)
	}
	return refs
}

// decodeImageXObject decodes one image XObject into a Go image.Image.
// Returns (nil, nil) — not an error — for an image this package
// deliberately does not support (an unrecognised filter or color model): a
// page with such an image should still be indexed for its text and its
// other images, never aborted. A non-nil error means the image claimed a
// supported filter/color model but its actual bytes were corrupt or
// inconsistent with its stated dimensions.
func decodeImageXObject(obj lpdf.Value) (img image.Image, err error) {
	// obj.Reader() can panic on a filter or malformed stream this library
	// doesn't expect (see read.go's applyFilter); one bad embedded image
	// must degrade to "skip this image," never abort indexing the rest of
	// the document.
	defer func() {
		if r := recover(); r != nil {
			img, err = nil, fmt.Errorf("pdf: decoding image: %v", r)
		}
	}()

	filterName := ""
	filter := obj.Key("Filter")
	switch filter.Kind() {
	case lpdf.Name:
		filterName = filter.Name()
	case lpdf.Array:
		if filter.Len() > 0 {
			// A filter chain (e.g. [ASCII85Decode DCTDecode]) — only the
			// last stage determines the final byte format handed to an
			// image decoder.
			filterName = filter.Index(filter.Len() - 1).Name()
		}
	}
	if !supportedImageFilter(filterName) {
		return nil, nil
	}

	data, err := io.ReadAll(obj.Reader())
	if err != nil {
		return nil, fmt.Errorf("pdf: reading image stream: %w", err)
	}

	if filterName == "DCTDecode" {
		im, err := jpeg.Decode(bytes.NewReader(data))
		if err != nil {
			return nil, fmt.Errorf("pdf: decoding JPEG image: %w", err)
		}
		return im, nil
	}
	return decodeRawBitmap(obj, data)
}

// decodeRawBitmap reconstructs an image from a FlateDecode (or unfiltered)
// stream's raw, already-decompressed pixel bytes using the XObject's own
// Width/Height/ColorSpace/BitsPerComponent — the PDF spec's only source of
// truth for how to interpret them (there is no embedded format header like
// a PNG/BMP would have).
func decodeRawBitmap(obj lpdf.Value, data []byte) (image.Image, error) {
	w := int(obj.Key("Width").Int64())
	h := int(obj.Key("Height").Int64())
	bpc := obj.Key("BitsPerComponent").Int64()
	cs := obj.Key("ColorSpace").Name()

	if bpc != 8 || cs != "DeviceRGB" {
		// Indexed palettes, CMYK, 1/4/16-bit depth, ICCBased color spaces,
		// etc. are real PDF possibilities this package does not reconstruct
		// — not found in the one real document this feature was built
		// against (every substantive-sized FlateDecode image there was
		// DeviceRGB/8bpc). Skip, don't guess at a wrong pixel layout.
		return nil, nil
	}
	want := w * h * 3
	if len(data) != want {
		return nil, fmt.Errorf("pdf: raw bitmap %dx%d DeviceRGB/8bpc wants %d bytes, got %d", w, h, want, len(data))
	}

	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < w*h; i++ {
		img.Pix[i*4+0] = data[i*3+0]
		img.Pix[i*4+1] = data[i*3+1]
		img.Pix[i*4+2] = data[i*3+2]
		img.Pix[i*4+3] = 0xff
	}
	return img, nil
}

// ocrImage stages img as a temp PNG file and runs it through the same OCR
// path internal/image uses for standalone image files (imageindexer.OCRFile
// wraps internal/image's build-tag-gated Tesseract extraction), rather than
// duplicating that machinery here. Returns ("", nil) when OCR support isn't
// compiled in.
func ocrImage(img image.Image) (string, error) {
	if !imageindexer.OCRAvailable() {
		return "", nil
	}
	tmp, err := os.CreateTemp("", "zenith-pdf-img-*.png")
	if err != nil {
		return "", fmt.Errorf("pdf: staging image for OCR: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := png.Encode(tmp, img); err != nil {
		tmp.Close()
		return "", fmt.Errorf("pdf: encoding image for OCR: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return "", fmt.Errorf("pdf: staging image for OCR: %w", err)
	}
	return imageindexer.OCRFile(tmpPath)
}

// extractPageImageChunks decodes and OCRs every sufficiently large image
// XObject on page, returning one Chunk per image that yielded non-empty
// text. A single image failing to decode or OCR is skipped, never aborts
// the page.
func extractPageImageChunks(page lpdf.Page, pageNum int) []Chunk {
	refs := pageImageRefs(page)
	if len(refs) == 0 {
		return nil
	}
	var chunks []Chunk
	for i, ref := range refs {
		img, err := decodeImageXObject(ref)
		if err != nil || img == nil {
			continue
		}
		text, err := ocrImage(img)
		if err != nil {
			continue
		}
		if text == "" {
			continue
		}
		chunks = append(chunks, Chunk{Page: pageNum, Index: i, Kind: "image", Text: text})
	}
	return chunks
}
