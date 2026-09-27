package embedding

import (
	"context"
	"errors"
	"testing"
)

// DeterministicEmbedder must never fabricate a vector: a hash-seeded random
// unit vector was previously fused into ranking at full VectorWeight as if
// it were a real semantic signal, which could bury a document under its own
// exact-match query. Callers rely on the error to degrade to lexical-only.
func TestDeterministicEmbedder_EmbedReturnsError(t *testing.T) {
	d := NewDeterministicEmbedder(8)
	vec, err := d.Embed(context.Background(), "hello world")
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	if !errors.Is(err, ErrEmbeddingUnavailable) {
		t.Errorf("expected ErrEmbeddingUnavailable, got %v", err)
	}
	if vec != nil {
		t.Errorf("expected nil vector, got %v", vec)
	}
}

func TestDeterministicEmbedder_EmbedBatchReturnsError(t *testing.T) {
	d := NewDeterministicEmbedder(8)
	vecs, err := d.EmbedBatch(context.Background(), []string{"a", "b", "c"})
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	if !errors.Is(err, ErrEmbeddingUnavailable) {
		t.Errorf("expected ErrEmbeddingUnavailable, got %v", err)
	}
	if vecs != nil {
		t.Errorf("expected nil result, got %v", vecs)
	}
}

func TestDeterministicEmbedder_Dimensions(t *testing.T) {
	d := NewDeterministicEmbedder(384)
	if d.Dimensions() != 384 {
		t.Errorf("Dimensions() = %d, want 384", d.Dimensions())
	}
}
