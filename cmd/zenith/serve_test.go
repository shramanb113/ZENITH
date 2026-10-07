package main

import "testing"

// sidecarEmbedder must never silently swap an explicitly requested --model for
// another one: an unloadable explicit model is a hard error. These three cases
// run with CGO_ENABLED=0 too, since none of them reach the ONNX loader.
func TestSidecarEmbedder_ExplicitModelMismatch(t *testing.T) {
	if _, _, err := sidecarEmbedder("deterministic", "labse"); err == nil {
		t.Fatal("expected an error combining --embedder deterministic with --model, got nil")
	}
}

func TestSidecarEmbedder_UnknownModel(t *testing.T) {
	if _, _, err := sidecarEmbedder("auto", "no-such-model"); err == nil {
		t.Fatal("expected an error for an unregistered model id, got nil")
	}
}

func TestSidecarEmbedder_DeterministicNoModel(t *testing.T) {
	emb, model, err := sidecarEmbedder("deterministic", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model != "deterministic" {
		t.Fatalf("model = %q, want %q", model, "deterministic")
	}
	if emb == nil {
		t.Fatal("expected a non-nil deterministic embedder")
	}
}
