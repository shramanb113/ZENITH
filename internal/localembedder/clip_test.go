package localembedder

import (
	"context"
	"image"
	"image/color"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/embedding"
)

// realCLIPForEval loads the real pulled clip-vit-base-patch32 model, or
// skips the calling test with an explanatory message if ZENITH_MODEL_EVAL
// is unset or the model isn't installed. Shared by every ZENITH_MODEL_EVAL
// gated test in this file so each one doesn't repeat the lookup.
func realCLIPForEval(t *testing.T) *CLIPEmbedder {
	t.Helper()
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
	return clip
}

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
	clip := realCLIPForEval(t)
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
	redQueryVec, err := clip.EmbedText(ctx, "the color red")
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

// TestCLIPEmbedder_DoubleCloseDoesNotPanic is gated behind ZENITH_MODEL_EVAL
// (see realCLIPForEval) because NewCLIPByID needs the real model files to
// construct a CLIPEmbedder at all — there is no lighter-weight constructor
// that skips loading real ONNX sessions. This is the pragmatic choice
// documented in the final-review fix list, not a design preference: the
// real model is already pulled, so gating here costs nothing extra.
func TestCLIPEmbedder_DoubleCloseDoesNotPanic(t *testing.T) {
	clip := realCLIPForEval(t)

	clip.Close()
	clip.Close() // must be a safe no-op, not a second close-of-closed-channel panic.
}

// TestCLIPEmbedder_EmbedAfterCloseReturnsError proves every Embed* method
// returns errCLIPClosed (not a panic, not undefined behavior against a
// destroyed session) once Close has run.
func TestCLIPEmbedder_EmbedAfterCloseReturnsError(t *testing.T) {
	clip := realCLIPForEval(t)
	clip.Close()

	ctx := context.Background()
	img := solidImage(32, 32, color.RGBA{10, 20, 30, 255})

	if _, err := clip.EmbedImage(ctx, img); err == nil {
		t.Error("EmbedImage after Close: want error, got nil")
	} else if !strings.Contains(err.Error(), "closed") {
		t.Errorf("EmbedImage after Close: err = %v, want it to mention \"closed\"", err)
	}

	if _, err := clip.EmbedImageBatch(ctx, []image.Image{img, img}); err == nil {
		t.Error("EmbedImageBatch after Close: want error, got nil")
	} else if !strings.Contains(err.Error(), "closed") {
		t.Errorf("EmbedImageBatch after Close: err = %v, want it to mention \"closed\"", err)
	}

	if _, err := clip.EmbedText(ctx, "a query"); err == nil {
		t.Error("EmbedText after Close: want error, got nil")
	} else if !strings.Contains(err.Error(), "closed") {
		t.Errorf("EmbedText after Close: err = %v, want it to mention \"closed\"", err)
	}
}

// TestCLIPEmbedder_RejectsNilAndZeroDimensionImages proves Fix 5's input
// validation: a nil image.Image, or one whose Bounds() has zero width or
// height, must return a clear error from EmbedImageBatch (and therefore
// EmbedImage) instead of panicking inside clipPreprocess or silently
// producing a meaningless constant vector.
func TestCLIPEmbedder_RejectsNilAndZeroDimensionImages(t *testing.T) {
	clip := realCLIPForEval(t)
	defer clip.Close()

	ctx := context.Background()
	good := solidImage(32, 32, color.RGBA{10, 20, 30, 255})
	zeroDim := image.NewRGBA(image.Rect(0, 0, 0, 0))

	if _, err := clip.EmbedImage(ctx, nil); err == nil {
		t.Error("EmbedImage(nil): want error, got nil")
	}
	if _, err := clip.EmbedImage(ctx, zeroDim); err == nil {
		t.Error("EmbedImage(zero-dimension image): want error, got nil")
	}
	if _, err := clip.EmbedImageBatch(ctx, []image.Image{good, nil}); err == nil {
		t.Error("EmbedImageBatch([good, nil]): want error, got nil")
	}
	if _, err := clip.EmbedImageBatch(ctx, []image.Image{good, zeroDim}); err == nil {
		t.Error("EmbedImageBatch([good, zero-dimension]): want error, got nil")
	}
}

// TestCLIPEmbedder_BatchMatchesSingleEmbed is a nice-to-have proving Fix 2's
// reproducibility claim against the real model: EmbedImageBatch now
// dispatches each image as its own independent job, so a given image's
// vector must no longer depend on what else shared its batch call.
func TestCLIPEmbedder_BatchMatchesSingleEmbed(t *testing.T) {
	clip := realCLIPForEval(t)
	defer clip.Close()

	ctx := context.Background()
	red := solidImage(300, 300, color.RGBA{220, 20, 20, 255})
	blue := solidImage(300, 300, color.RGBA{20, 20, 220, 255})

	solo, err := clip.EmbedImage(ctx, red)
	if err != nil {
		t.Fatalf("EmbedImage(red): %v", err)
	}
	batch, err := clip.EmbedImageBatch(ctx, []image.Image{red, blue})
	if err != nil {
		t.Fatalf("EmbedImageBatch([red, blue]): %v", err)
	}

	cos := clipDot(solo, batch[0])
	t.Logf("cosine(EmbedImage(red), EmbedImageBatch([red, blue])[0]) = %v", cos)
	if cos < 0.9999 {
		t.Errorf("batch-vs-solo cosine = %v, want > 0.9999 (independent per-image jobs)", cos)
	}
}
