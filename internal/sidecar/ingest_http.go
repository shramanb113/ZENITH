package sidecar

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	imageindexer "github.com/shramanb113/ZENITH/internal/image"
	"github.com/shramanb113/ZENITH/internal/pdf"
	"github.com/shramanb113/ZENITH/pkg/zenith"
)

// maxIngestMemory bounds how much of a multipart upload's non-file fields
// (id, attrs) are held in memory before Go's multipart reader spills parts
// to temp files on disk; the overall request is already bounded by
// colGuard's MaxBytesReader(maxBody), this just controls the in-memory vs.
// on-disk split for that bounded total.
const maxIngestMemory = 1 << 20

func isIngestableImageExt(ext string) bool {
	switch ext {
	case ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif":
		return true
	}
	return false
}

// ingestCollection serves POST /v1/collections/{id}/ingest: a multipart
// file upload (field "file") is run through the existing PDF/image
// extractors server-side and the resulting chunks are upserted into the
// collection — the file-ingest counterpart to PUT .../docs, which only
// accepts pre-extracted {id, text, attrs} and has no way to hand ZENITH a
// raw PDF or image. Form fields: "id" (required — the document ID chunks
// are derived from, e.g. "doc1||p1||c0||text||..." for a PDF), "attrs"
// (optional JSON object, applied to every resulting chunk).
func (s *Server) ingestCollection(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")

	if err := r.ParseMultipartForm(maxIngestMemory); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeColErr(w, http.StatusRequestEntityTooLarge, "body_too_large", "body too large")
		} else {
			writeColErr(w, http.StatusBadRequest, "invalid_request", "invalid multipart form: "+err.Error())
		}
		return
	}
	defer func() {
		if r.MultipartForm != nil {
			_ = r.MultipartForm.RemoveAll()
		}
	}()

	docID := strings.TrimSpace(r.FormValue("id"))
	if docID == "" {
		writeColErr(w, http.StatusBadRequest, "invalid_request", "id is required")
		return
	}
	var attrs zenith.Attrs
	if raw := r.FormValue("attrs"); raw != "" {
		if err := json.Unmarshal([]byte(raw), &attrs); err != nil {
			writeColErr(w, http.StatusBadRequest, "invalid_request", "attrs must be a JSON object")
			return
		}
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		writeColErr(w, http.StatusBadRequest, "invalid_request", "file is required")
		return
	}
	defer file.Close()

	ext := strings.ToLower(filepath.Ext(header.Filename))
	tmp, err := os.CreateTemp("", "zenith-ingest-*"+ext)
	if err != nil {
		writeColErr(w, http.StatusInternalServerError, "internal", "could not stage upload")
		return
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := io.Copy(tmp, file); err != nil {
		tmp.Close()
		writeColErr(w, http.StatusBadRequest, "invalid_request", "failed reading upload: "+err.Error())
		return
	}
	if err := tmp.Close(); err != nil {
		writeColErr(w, http.StatusInternalServerError, "internal", "could not stage upload")
		return
	}

	docs := make(map[string]string)
	attrsMap := make(map[string]zenith.Attrs)
	addChunk := func(chunkID, text string, extra zenith.Attrs) {
		docs[chunkID] = text
		if len(attrs) == 0 && len(extra) == 0 {
			return
		}
		merged := make(zenith.Attrs, len(attrs)+len(extra))
		for k, v := range attrs {
			merged[k] = v
		}
		for k, v := range extra {
			merged[k] = v
		}
		attrsMap[chunkID] = merged
	}

	switch {
	case ext == ".pdf":
		chunks, err := pdf.ExtractChunks(tmpPath)
		if err != nil {
			writeColErr(w, http.StatusBadRequest, "invalid_request", "could not parse PDF: "+err.Error())
			return
		}
		for _, c := range chunks {
			// "#" instead of pdf.Chunk.ID's "||" (zenith.DB rejects "||" in
			// caller-supplied IDs — see Chunk's doc comment); page/chunk/bbox
			// ride along as attrs instead of being baked into the ID string.
			if c.Kind == "image" {
				chunkID := fmt.Sprintf("%s#p%d#img%d", docID, c.Page, c.Index)
				addChunk(chunkID, c.Text, zenith.Attrs{"_page": c.Page, "_image": c.Index, "_source": "ocr"})
				continue
			}
			chunkID := fmt.Sprintf("%s#p%d#c%d", docID, c.Page, c.Index)
			addChunk(chunkID, c.Text, zenith.Attrs{
				"_page": c.Page, "_chunk": c.Index,
				"_bbox": fmt.Sprintf("%.2f,%.2f,%.2f,%.2f", c.BBoxX, c.BBoxY, c.BBoxW, c.BBoxH),
			})
		}
	case isIngestableImageExt(ext):
		chunks, ocrErr := imageindexer.ExtractChunks(tmpPath)
		if ocrErr != nil {
			s.cfg.Log.Warn("ingest: OCR failed, falling back to filename-only", "collection", id, "doc", docID, "err", ocrErr)
		}
		for _, c := range chunks {
			if c.Kind == "path" {
				addChunk(docID, c.Text, nil)
				continue
			}
			addChunk(fmt.Sprintf("%s#ocr#c%d", docID, c.Index), c.Text, zenith.Attrs{"_chunk": c.Index})
		}
	default:
		writeColErr(w, http.StatusBadRequest, "invalid_request", "unsupported file type: "+ext)
		return
	}
	if len(docs) == 0 {
		writeColErr(w, http.StatusBadRequest, "invalid_request", "no text could be extracted from the file")
		return
	}
	if len(attrsMap) == 0 {
		attrsMap = nil
	}

	start := time.Now()
	res, err := s.cfg.Collections.Upsert(r.Context(), id, docs, attrsMap)
	if err != nil {
		s.colErr(w, err)
		return
	}
	s.cfg.Log.Info("collection ingest", "id", id, "doc", docID, "chunks", len(docs), "ms", time.Since(start).Milliseconds())
	writeJSON(w, http.StatusOK, map[string]int{"upserted": res.Upserted, "new": res.New, "doc_count": res.DocCount})
}
