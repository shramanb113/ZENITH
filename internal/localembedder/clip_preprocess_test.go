package localembedder

import (
	"image"
	"image/color"
	"math"
	"runtime"
	"testing"
)

func solidImage(w, h int, c color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetRGBA(x, y, c)
		}
	}
	return img
}

func almostEqual(a, b float32) bool {
	return math.Abs(float64(a-b)) < 1e-4
}

func TestCLIPPreprocess_SolidColorExactNormalization(t *testing.T) {
	// For a uniform-color image, resize and center-crop are lossless by
	// construction (every interpolation kernel reproduces a constant
	// function exactly), so the output must equal the hand-computed
	// normalization exactly — this is a real numeric fixture, not a
	// shape/range sanity check. Values computed directly from the real
	// preprocessor_config.json mean/std (2026-10-09).
	img := solidImage(300, 300, color.RGBA{128, 64, 32, 255})
	mean := [3]float32{0.48145466, 0.4578275, 0.40821073}
	std := [3]float32{0.26862954, 0.26130258, 0.27577711}
	out := clipPreprocess(img, 224, mean, std)

	wantR := float32(0.07633607)
	wantG := float32(-0.79159994)
	wantB := float32(-1.02517766)

	plane := 224 * 224
	// Spot-check several positions, not just [0], in case of an off-by-one
	// in the crop math that happens to leave pixel 0 correct.
	for _, idx := range []int{0, 100, plane - 1, plane/2 + 50} {
		if !almostEqual(out[0*plane+idx], wantR) {
			t.Errorf("R[%d] = %v, want %v", idx, out[0*plane+idx], wantR)
		}
		if !almostEqual(out[1*plane+idx], wantG) {
			t.Errorf("G[%d] = %v, want %v", idx, out[1*plane+idx], wantG)
		}
		if !almostEqual(out[2*plane+idx], wantB) {
			t.Errorf("B[%d] = %v, want %v", idx, out[2*plane+idx], wantB)
		}
	}
}

func TestCLIPPreprocess_OutputShape(t *testing.T) {
	mean := [3]float32{0.5, 0.5, 0.5}
	std := [3]float32{0.5, 0.5, 0.5}
	for _, dims := range [][2]int{{300, 300}, {100, 400}, {400, 100}, {50, 50}} {
		img := solidImage(dims[0], dims[1], color.RGBA{10, 20, 30, 255})
		out := clipPreprocess(img, 224, mean, std)
		if len(out) != 3*224*224 {
			t.Errorf("size %v: len(out) = %d, want %d", dims, len(out), 3*224*224)
		}
	}
}

func TestCLIPPreprocess_UpscalesSmallImages(t *testing.T) {
	// A source image smaller than the target size on both axes must still
	// produce a full-size output (upscaling), not panic or short-circuit.
	mean := [3]float32{0.5, 0.5, 0.5}
	std := [3]float32{0.5, 0.5, 0.5}
	img := solidImage(10, 8, color.RGBA{200, 200, 200, 255})
	out := clipPreprocess(img, 224, mean, std)
	if len(out) != 3*224*224 {
		t.Errorf("len(out) = %d, want %d", len(out), 3*224*224)
	}
}

// TestCLIPPreprocess_ExtremeAspectRatioStaysBounded proves the Fix 1 memory
// bound: resizing the WHOLE image to shortest-edge=224 before cropping would,
// for a 1x5000px image, allocate a ~224 x 1,120,000px intermediate (measured
// ~1.16GB) even though only a 224x224 result is ever used. The fixed
// clipPreprocess must compute the crop region in source coordinates first and
// scale only that region directly, so peak allocation for this call stays
// close to the true cost (size*size*3*4 bytes for the output slice, plus a
// small constant number of size x size RGBA buffers) — comfortably under the
// 50MB bound asserted here, and nowhere near the old ~1.16GB.
func TestCLIPPreprocess_ExtremeAspectRatioStaysBounded(t *testing.T) {
	mean := [3]float32{0.5, 0.5, 0.5}
	std := [3]float32{0.5, 0.5, 0.5}

	for _, dims := range [][2]int{{1, 5000}, {5, 5000}, {5000, 1}} {
		img := solidImage(dims[0], dims[1], color.RGBA{10, 20, 30, 255})

		runtime.GC()
		var before runtime.MemStats
		runtime.ReadMemStats(&before)

		out := clipPreprocess(img, 224, mean, std)

		var after runtime.MemStats
		runtime.ReadMemStats(&after)

		if len(out) != 3*224*224 {
			t.Errorf("size %v: len(out) = %d, want %d", dims, len(out), 3*224*224)
		}

		const bound = 50 * 1024 * 1024 // 50MB; true cost is well under 1MB.
		allocated := after.TotalAlloc - before.TotalAlloc
		if allocated > bound {
			t.Errorf("size %v: clipPreprocess allocated %d bytes, want <= %d (bounded preprocessing)", dims, allocated, bound)
		}
	}
}

func TestCLIPPreprocess_DegenerateTinyImageDoesNotPanic(t *testing.T) {
	mean := [3]float32{0.5, 0.5, 0.5}
	std := [3]float32{0.5, 0.5, 0.5}
	img := solidImage(1, 1, color.RGBA{1, 2, 3, 255})
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("clipPreprocess panicked on a 1x1 image: %v", r)
		}
	}()
	out := clipPreprocess(img, 224, mean, std)
	if len(out) != 3*224*224 {
		t.Errorf("len(out) = %d, want %d", len(out), 3*224*224)
	}
}
