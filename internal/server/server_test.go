package server_test

import (
	"context"
	"testing"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/server"
)

type mockPDFIndexer struct {
	lastDocID   string
	lastPath    string
	lastAttrs   index.Attrs
	returnCount int
	returnErr   error
}

func (m *mockPDFIndexer) Index(ctx context.Context, docID, filePath string, attrs index.Attrs) (int, error) {
	m.lastDocID = docID
	m.lastPath = filePath
	m.lastAttrs = attrs
	return m.returnCount, m.returnErr
}

func TestZenithServer_IndexPDF_Success(t *testing.T) {
	mock := &mockPDFIndexer{returnCount: 5}
	srv := &server.ZenithServer{PDFIndexer: mock}

	resp, err := srv.IndexPDF(context.Background(), &zenithproto.IndexPDFRequest{
		DocumentId: "report2024",
		FilePath:   "/data/report.pdf",
	})
	if err != nil {
		t.Fatalf("IndexPDF error: %v", err)
	}
	if !resp.Status {
		t.Errorf("expected Status=true, got false: %s", resp.Message)
	}
	if resp.ChunksIndexed != 5 {
		t.Errorf("ChunksIndexed = %d, want 5", resp.ChunksIndexed)
	}
	if mock.lastDocID != "report2024" {
		t.Errorf("docID not forwarded: got %q", mock.lastDocID)
	}
	if mock.lastPath != "/data/report.pdf" {
		t.Errorf("filePath not forwarded: got %q", mock.lastPath)
	}
}

func TestZenithServer_IndexPDF_EmptyDocID(t *testing.T) {
	srv := &server.ZenithServer{PDFIndexer: &mockPDFIndexer{}}
	resp, err := srv.IndexPDF(context.Background(), &zenithproto.IndexPDFRequest{
		DocumentId: "",
		FilePath:   "/data/report.pdf",
	})
	if err == nil {
		t.Fatal("expected error for empty document_id")
	}
	if resp != nil && resp.Status {
		t.Error("expected Status=false for empty document_id")
	}
}

func TestZenithServer_IndexPDF_EmptyFilePath(t *testing.T) {
	srv := &server.ZenithServer{PDFIndexer: &mockPDFIndexer{}}
	resp, err := srv.IndexPDF(context.Background(), &zenithproto.IndexPDFRequest{
		DocumentId: "doc1",
		FilePath:   "",
	})
	if err == nil {
		t.Fatal("expected error for empty file_path")
	}
	if resp != nil && resp.Status {
		t.Error("expected Status=false for empty file_path")
	}
}

func TestParseChunkFields_ValidID(t *testing.T) {
	// Test the chunk ID parsing via Search result fields
	// We test parseChunkFields indirectly via the exported behavior
	// by checking a known chunk ID format parses correctly
	id := "report2024||p3||c1||text||42.00,100.50,300.00,20.00"
	fields := server.ParseChunkFieldsForTest(id)
	if fields == nil {
		t.Fatal("expected non-nil fields for valid chunk ID")
	}
	if fields["pdf_source"] != "report2024" {
		t.Errorf("pdf_source = %q, want %q", fields["pdf_source"], "report2024")
	}
	if fields["page"] != "3" {
		t.Errorf("page = %q, want %q", fields["page"], "3")
	}
	if fields["chunk"] != "1" {
		t.Errorf("chunk = %q, want %q", fields["chunk"], "1")
	}
	if fields["source_type"] != "text" {
		t.Errorf("source_type = %q, want %q", fields["source_type"], "text")
	}
	if fields["bbox"] != "42.00,100.50,300.00,20.00" {
		t.Errorf("bbox = %q, want %q", fields["bbox"], "42.00,100.50,300.00,20.00")
	}
}

func TestParseChunkFields_RegularID(t *testing.T) {
	fields := server.ParseChunkFieldsForTest("regular-doc-id")
	if fields != nil {
		t.Errorf("expected nil fields for regular (non-PDF) doc ID, got %v", fields)
	}
}
