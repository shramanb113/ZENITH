package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/shramanb113/ZENITH/internal/localembedder"
)

func TestPullVisualFiles_DownloadsAllFourFilesWithExpectedNames(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/vision.onnx", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fake-vision-bytes"))
	})
	mux.HandleFunc("/text.onnx", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fake-text-bytes"))
	})
	mux.HandleFunc("/vocab.json", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"a":1}`))
	})
	mux.HandleFunc("/merges.txt", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("#version: 0.2\n"))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	spec := localembedder.VisualSpec{
		ID:             "fake-clip",
		VisionModelURL: srv.URL + "/vision.onnx",
		TextModelURL:   srv.URL + "/text.onnx",
		VocabURL:       srv.URL + "/vocab.json",
		MergesURL:      srv.URL + "/merges.txt",
	}
	dir := t.TempDir()
	if err := pullVisualFiles(spec, dir); err != nil {
		t.Fatalf("pullVisualFiles: %v", err)
	}

	wantContents := map[string]string{
		"vision_model.onnx": "fake-vision-bytes",
		"text_model.onnx":   "fake-text-bytes",
		"vocab.json":        `{"a":1}`,
		"merges.txt":        "#version: 0.2\n",
	}
	for name, want := range wantContents {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Errorf("reading %s: %v", name, err)
			continue
		}
		if string(got) != want {
			t.Errorf("%s contents = %q, want %q", name, got, want)
		}
	}
}

func TestPullVisualFiles_PropagatesDownloadError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	spec := localembedder.VisualSpec{
		ID:             "fake-clip",
		VisionModelURL: srv.URL + "/missing.onnx",
		TextModelURL:   srv.URL + "/t.onnx",
		VocabURL:       srv.URL + "/v.json",
		MergesURL:      srv.URL + "/m.txt",
	}
	if err := pullVisualFiles(spec, t.TempDir()); err == nil {
		t.Fatal("expected an error when a file 404s, got nil")
	}
}
