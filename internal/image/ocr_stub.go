//go:build !cgo || !ocr

package image

// ocrAvailable reports whether this binary was built with Tesseract OCR
// support. The default build (no -tags ocr, or no CGo) never carries the
// gosseract/libtesseract dependency, so image indexing stays filename-only
// unless the binary was deliberately built with -tags ocr against
// libtesseract (see internal/image/ocr_tesseract.go and the Docker image).
func ocrAvailable() bool { return false }

// extractOCRText is a no-op in a build without OCR support: it returns no
// text and no error, so Indexer.Index falls back to filename-only indexing
// exactly as it did before OCR existed.
func extractOCRText(filePath string) (string, error) { return "", nil }
