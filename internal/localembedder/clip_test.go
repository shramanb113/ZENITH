package localembedder

import (
	"context"
	"image/color"
	"os"
	"path/filepath"
	"testing"

	"github.com/shramanb113/ZENITH/internal/embedding"
)

func TestNewCLIPByID_NotInstalled(t *testing.T) {
	_, err := NewCLIPByID("clip-vit-base-patch32", t.TempDir())
	if err == nil {
		t.Fatal("NewCLIPByID should fail when the model files are not present")
	}
}

func TestNewCLIPByID_UnknownModel(t *testing.T) {
	_, err := NewCLIPByID("not-a-real-visual-model", t.TempDir())
	if err == nil {
		t.Fatal("NewCLIPByID should fail for an unregistered model id")
	}
}

func TestCLIPEmbedder_SatisfiesVisualEmbedder(t *testing.T) {
	var _ embedding.VisualEmbedder = (*CLIPEmbedder)(nil)
	var _ embedding.Named = (*CLIPEmbedder)(nil)
}

// TestCLIPEmbedder_RealModelSemanticSmoke is gated behind ZENITH_MODEL_EVAL=1
// (the same switch multilingual_test.go already uses for labse/
// distiluse-multilingual — one opt-in-real-model gate for this package, not
// a new one per model) and requires `zenith models pull clip-vit-base-patch32`
// to have already been run into the real default ~/.zenith/models. It is a
// real, not synthetic, test: it loads the actual ~150MB model and proves
// actual cross-modal semantic similarity, not just that the code compiles
// and runs. Explicitly smoke-scale (2 images, 1 query), same honesty
// standard this package already applies to multilingual_test.go — not a
// published benchmark claim.
func TestCLIPEmbedder_RealModelSemanticSmoke(t *testing.T) {
	if os.Getenv("ZENITH_MODEL_EVAL") == "" {
		t.Skip("set ZENITH_MODEL_EVAL=1 to run (needs: zenith models pull clip-vit-base-patch32)")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("cannot determine home dir: %v", err)
	}
	modelsDir := filepath.Join(home, ".zenith", "models")

	clip, err := NewCLIPByID("clip-vit-base-patch32", modelsDir)
	if err != nil {
		t.Fatalf("NewCLIPByID (did you run `zenith models pull clip-vit-base-patch32`?): %v", err)
	}
	defer clip.Close()

	ctx := context.Background()

	// Two distinctly-colored solid images stand in for "a red thing" vs "a
	// blue thing" — not a real photo, but still a real forward pass through
	// the real model, so this at least proves color/caption association
	// isn't completely broken. A real photographic smoke test (the "tree"
	// scenario from the original request) needs an actual photo and is left
	// to the ingestion-wiring plan's testing task, which will use a
	// maintainer-supplied real image per the design spec §10.
	red := solidImage(300, 300, color.RGBA{220, 20, 20, 255})
	blue := solidImage(300, 300, color.RGBA{20, 20, 220, 255})

	redVec, err := clip.EmbedImage(ctx, red)
	if err != nil {
		t.Fatalf("EmbedImage(red): %v", err)
	}
	blueVec, err := clip.EmbedImage(ctx, blue)
	if err != nil {
		t.Fatalf("EmbedImage(blue): %v", err)
	}
	redQueryVec, err := clip.EmbedText(ctx, "a solid red image")
	if err != nil {
		t.Fatalf("EmbedText: %v", err)
	}

	simRedRed := clipDot(redQueryVec, redVec)
	simRedBlue := clipDot(redQueryVec, blueVec)
	t.Logf("sim(red query, red image) = %v, sim(red query, blue image) = %v", simRedRed, simRedBlue)
	if simRedRed <= simRedBlue {
		t.Errorf("expected the red-image query to rank the red image above the blue one: %v vs %v", simRedRed, simRedBlue)
	}
}

// clipDot is a package-local name to avoid colliding with the similarly
// named dot helper in multilingual_test.go (//go:build cgo) when both files
// compile together under CGO_ENABLED=1; this file carries no build tag since
// NewCLIPByID/CLIPEmbedder are available (NewCLIPByID errors cleanly) in
// both cgo and no-cgo builds.
func clipDot(a, b []float32) float64 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}
