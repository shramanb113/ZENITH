//go:build cgo && ocr

package image

import (
	"fmt"
	"strings"

	"github.com/otiai10/gosseract/v2"
)

// ocrAvailable reports whether this binary was built with Tesseract OCR
// support (CGo enabled and built with -tags ocr against libtesseract).
func ocrAvailable() bool { return true }

// extractOCRText runs Tesseract over the image at filePath and returns
// whatever text it finds, trimmed. A Tesseract failure (missing language
// data, corrupt image, unsupported format) is returned as an error so the
// caller can log it and fall back to filename-only indexing rather than
// failing the whole Index call.
func extractOCRText(filePath string) (string, error) {
	client := gosseract.NewClient()
	defer client.Close()

	if err := client.SetImage(filePath); err != nil {
		return "", fmt.Errorf("ocr: set image %s: %w", filePath, err)
	}
	text, err := client.Text()
	if err != nil {
		return "", fmt.Errorf("ocr: extract text from %s: %w", filePath, err)
	}
	return strings.TrimSpace(text), nil
}
