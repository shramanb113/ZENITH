package modelspec

import "testing"

func TestLookupVisual_Found(t *testing.T) {
	spec, err := LookupVisual("clip-vit-base-patch32")
	if err != nil {
		t.Fatalf("LookupVisual: %v", err)
	}
	if spec.Dims != 512 {
		t.Errorf("Dims = %d, want 512", spec.Dims)
	}
	if spec.ImageSize != 224 {
		t.Errorf("ImageSize = %d, want 224", spec.ImageSize)
	}
	if spec.ContextLength != 77 {
		t.Errorf("ContextLength = %d, want 77", spec.ContextLength)
	}
	if spec.VisionModelURL == "" || spec.TextModelURL == "" {
		t.Error("VisionModelURL/TextModelURL must not be empty")
	}
	if spec.VocabURL == "" || spec.MergesURL == "" {
		t.Error("VocabURL/MergesURL must not be empty")
	}
	wantMean := [3]float32{0.48145466, 0.4578275, 0.40821073}
	if spec.ImageMean != wantMean {
		t.Errorf("ImageMean = %v, want %v", spec.ImageMean, wantMean)
	}
	wantStd := [3]float32{0.26862954, 0.26130258, 0.27577711}
	if spec.ImageStd != wantStd {
		t.Errorf("ImageStd = %v, want %v", spec.ImageStd, wantStd)
	}
	if spec.IndexName() != "clip:clip-vit-base-patch32" {
		t.Errorf("IndexName() = %q, want %q", spec.IndexName(), "clip:clip-vit-base-patch32")
	}
}

func TestLookupVisual_CaseInsensitive(t *testing.T) {
	if _, err := LookupVisual("CLIP-VIT-BASE-PATCH32"); err != nil {
		t.Errorf("LookupVisual should be case-insensitive: %v", err)
	}
}

func TestLookupVisual_Unknown(t *testing.T) {
	if _, err := LookupVisual("not-a-real-model"); err == nil {
		t.Error("LookupVisual(unknown) should return an error")
	}
}

func TestVisualModels_IncludesDefault(t *testing.T) {
	found := false
	for _, m := range VisualModels() {
		if m.ID == DefaultVisualID {
			found = true
		}
	}
	if !found {
		t.Errorf("VisualModels() does not include DefaultVisualID %q", DefaultVisualID)
	}
}
