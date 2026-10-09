package pdf

import (
	"bytes"
	"compress/zlib"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"path/filepath"
	"testing"

	lpdf "github.com/ledongthuc/pdf"
)

// pdfObj is one numbered indirect object written into a minimal hand-built
// PDF (see buildMinimalPDF). body is everything between "N 0 obj" and
// "endobj"; stream, when non-nil, is written as a PDF stream (dict must
// already include a correct /Length).
type pdfObj struct {
	body   string
	stream []byte
}

// buildMinimalPDF assembles the smallest valid single-page PDF this test
// needs: a Catalog, a Pages tree, one Page (with the given extra resources
// dict entries and content stream), plus whatever additional numbered
// objects objs supplies (e.g. an image XObject). Object numbering: 1
// Catalog, 2 Pages, 3 Page, 4 Contents, 5+ = objs in order. Byte offsets
// for the xref table are computed as each object is written, which is the
// part a hand-built PDF is easy to get wrong — every offset here is
// measured, not guessed.
func buildMinimalPDF(t *testing.T, resourcesDict, contentStream string, objs []pdfObj) []byte {
	t.Helper()
	var buf bytes.Buffer
	offsets := make([]int, 0, 5+len(objs))

	writeObj := func(n int, body string, stream []byte) {
		offsets = append(offsets, buf.Len())
		fmt.Fprintf(&buf, "%d 0 obj\n%s\n", n, body)
		if stream != nil {
			buf.WriteString("stream\n")
			buf.Write(stream)
			buf.WriteString("\nendstream\n")
		}
		buf.WriteString("endobj\n")
	}

	buf.WriteString("%PDF-1.4\n")
	writeObj(1, "<< /Type /Catalog /Pages 2 0 R >>", nil)
	writeObj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>", nil)
	writeObj(3, fmt.Sprintf(
		"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << %s >> /Contents 4 0 R >>",
		resourcesDict), nil)
	writeObj(4, fmt.Sprintf("<< /Length %d >>", len(contentStream)), []byte(contentStream))
	for i, o := range objs {
		n := 5 + i
		body := o.body
		if o.stream != nil {
			body = fmt.Sprintf("%s", o.body) // body already includes /Length
		}
		writeObj(n, body, o.stream)
	}

	total := 5 + len(objs)
	xrefOffset := buf.Len()
	fmt.Fprintf(&buf, "xref\n0 %d\n", total)
	buf.WriteString("0000000000 65535 f \n")
	for _, off := range offsets {
		fmt.Fprintf(&buf, "%010d 00000 n \n", off)
	}
	fmt.Fprintf(&buf, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF", total, xrefOffset)
	return buf.Bytes()
}

func openPDFBytes(t *testing.T, data []byte) (*os.File, *lpdf.Reader) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.pdf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("writing test PDF: %v", err)
	}
	f, r, err := lpdf.Open(path)
	if err != nil {
		t.Fatalf("lpdf.Open: %v", err)
	}
	t.Cleanup(func() { f.Close() })
	return f, r
}

func solidJPEG(t *testing.T, w, h int, c color.RGBA) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
		t.Fatalf("jpeg.Encode: %v", err)
	}
	return buf.Bytes()
}

func TestDecodeImageXObject_DCTDecode(t *testing.T) {
	w, h := 200, 160 // area 32,000 >= minImageArea (22,500) — see pageImageRefs
	jpegBytes := solidJPEG(t, w, h, color.RGBA{200, 50, 50, 255})

	imgObjBody := fmt.Sprintf(
		"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length %d >>",
		w, h, len(jpegBytes))
	data := buildMinimalPDF(t, "/XObject << /Im0 5 0 R >>", "BT ET", []pdfObj{{body: imgObjBody, stream: jpegBytes}})

	_, r := openPDFBytes(t, data)
	page := r.Page(1)
	if page.V.IsNull() {
		t.Fatal("page is null")
	}

	refs := pageImageRefs(page)
	if len(refs) != 1 {
		t.Fatalf("pageImageRefs: got %d refs, want 1", len(refs))
	}

	img, err := decodeImageXObject(refs[0])
	if err != nil {
		t.Fatalf("decodeImageXObject: %v", err)
	}
	if img == nil {
		t.Fatal("decodeImageXObject: got nil image, want decoded JPEG")
	}
	b := img.Bounds()
	if b.Dx() != w || b.Dy() != h {
		t.Fatalf("decoded image bounds = %v, want %dx%d", b, w, h)
	}
}

func TestDecodeImageXObject_FlateDecodeRawRGB(t *testing.T) {
	w, h := 170, 170 // area 28,900 >= minImageArea (22,500) — see pageImageRefs
	raw := make([]byte, w*h*3)
	for i := 0; i < w*h; i++ {
		raw[i*3+0] = 10
		raw[i*3+1] = 20
		raw[i*3+2] = 30
	}
	var compressed bytes.Buffer
	zw := zlib.NewWriter(&compressed)
	if _, err := zw.Write(raw); err != nil {
		t.Fatalf("zlib write: %v", err)
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("zlib close: %v", err)
	}

	imgObjBody := fmt.Sprintf(
		"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length %d >>",
		w, h, compressed.Len())
	data := buildMinimalPDF(t, "/XObject << /Im0 5 0 R >>", "BT ET", []pdfObj{{body: imgObjBody, stream: compressed.Bytes()}})

	_, r := openPDFBytes(t, data)
	page := r.Page(1)
	refs := pageImageRefs(page)
	if len(refs) != 1 {
		t.Fatalf("pageImageRefs: got %d refs, want 1", len(refs))
	}

	img, err := decodeImageXObject(refs[0])
	if err != nil {
		t.Fatalf("decodeImageXObject: %v", err)
	}
	if img == nil {
		t.Fatal("decodeImageXObject: got nil image, want decoded raw bitmap")
	}
	rgba, ok := img.(*image.RGBA)
	if !ok {
		t.Fatalf("decoded image type = %T, want *image.RGBA", img)
	}
	r0, g0, b0, a0 := rgba.At(0, 0).RGBA()
	if r0>>8 != 10 || g0>>8 != 20 || b0>>8 != 30 || a0>>8 != 255 {
		t.Fatalf("pixel (0,0) = %d,%d,%d,%d (>>8), want 10,20,30,255", r0>>8, g0>>8, b0>>8, a0>>8)
	}
}

func TestPageImageRefs_SkipsImagesBelowMinArea(t *testing.T) {
	tiny := solidJPEG(t, 10, 10, color.RGBA{1, 2, 3, 255}) // 100 px, well under minImageArea
	imgObjBody := fmt.Sprintf(
		"<< /Type /XObject /Subtype /Image /Width 10 /Height 10 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length %d >>",
		len(tiny))
	data := buildMinimalPDF(t, "/XObject << /Im0 5 0 R >>", "BT ET", []pdfObj{{body: imgObjBody, stream: tiny}})

	_, r := openPDFBytes(t, data)
	page := r.Page(1)
	if refs := pageImageRefs(page); len(refs) != 0 {
		t.Fatalf("pageImageRefs: got %d refs for a 10x10 image, want 0 (below minImageArea)", len(refs))
	}
}

func TestDecodeImageXObject_UnsupportedFilter_ReturnsNilNotError(t *testing.T) {
	// CCITTFaxDecode passes through Value.Reader() (the fork handles it) but
	// this package has no fax decoder — decodeImageXObject must return
	// (nil, nil), not an error, so the page's other images/text still index.
	fake := []byte{0x00, 0x01, 0x02, 0x03}
	imgObjBody := fmt.Sprintf(
		"<< /Type /XObject /Subtype /Image /Width 200 /Height 200 /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /CCITTFaxDecode /Length %d >>",
		len(fake))
	data := buildMinimalPDF(t, "/XObject << /Im0 5 0 R >>", "BT ET", []pdfObj{{body: imgObjBody, stream: fake}})

	_, r := openPDFBytes(t, data)
	page := r.Page(1)
	refs := pageImageRefs(page)
	if len(refs) != 1 {
		t.Fatalf("pageImageRefs: got %d refs, want 1", len(refs))
	}
	img, err := decodeImageXObject(refs[0])
	if err != nil {
		t.Fatalf("decodeImageXObject: got error %v, want nil (unsupported filter, not a failure)", err)
	}
	if img != nil {
		t.Fatalf("decodeImageXObject: got non-nil image for an unsupported filter, want nil")
	}
}

func TestExtractChunks_EmbeddedImage_NoOCRYieldsNoImageChunksButNoError(t *testing.T) {
	// Without -tags ocr, imageindexer.OCRAvailable() is false, so embedded
	// images never produce a chunk — this proves that degrades cleanly
	// (no error, no panic) rather than testing OCR text itself, which needs
	// a real Tesseract install this test environment may not have.
	w, h := 200, 160 // area 32,000 >= minImageArea (22,500) — see pageImageRefs
	jpegBytes := solidJPEG(t, w, h, color.RGBA{10, 10, 10, 255})
	imgObjBody := fmt.Sprintf(
		"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length %d >>",
		w, h, len(jpegBytes))
	data := buildMinimalPDF(t, "/XObject << /Im0 5 0 R >>", "BT ET", []pdfObj{{body: imgObjBody, stream: jpegBytes}})

	path := filepath.Join(t.TempDir(), "embedded.pdf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("writing test PDF: %v", err)
	}

	chunks, err := ExtractChunks(path)
	if err != nil {
		t.Fatalf("ExtractChunks: %v", err)
	}
	for _, c := range chunks {
		if c.Kind == "image" {
			t.Fatalf("got an image chunk without OCR available: %+v", c)
		}
	}
}
