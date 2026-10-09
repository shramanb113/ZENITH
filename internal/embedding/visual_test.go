package embedding

import (
	"context"
	"image"
	"testing"
)

type fakeVisualEmbedder struct{ dims int }

func (f *fakeVisualEmbedder) EmbedImage(_ context.Context, _ image.Image) ([]float32, error) {
	return make([]float32, f.dims), nil
}
func (f *fakeVisualEmbedder) EmbedImageBatch(_ context.Context, imgs []image.Image) ([][]float32, error) {
	out := make([][]float32, len(imgs))
	for i := range imgs {
		out[i] = make([]float32, f.dims)
	}
	return out, nil
}
func (f *fakeVisualEmbedder) EmbedText(_ context.Context, _ string) ([]float32, error) {
	return make([]float32, f.dims), nil
}
func (f *fakeVisualEmbedder) Dimensions() int { return f.dims }
func (f *fakeVisualEmbedder) Name() string    { return "fake-visual" }

func TestVisualEmbedder_InterfaceSatisfiedByFake(t *testing.T) {
	var _ VisualEmbedder = (*fakeVisualEmbedder)(nil)
	var _ Named = (*fakeVisualEmbedder)(nil)

	f := &fakeVisualEmbedder{dims: 512}
	ctx := context.Background()
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))

	v, err := f.EmbedImage(ctx, img)
	if err != nil || len(v) != 512 {
		t.Fatalf("EmbedImage: v=%v err=%v", v, err)
	}
	batch, err := f.EmbedImageBatch(ctx, []image.Image{img, img})
	if err != nil || len(batch) != 2 {
		t.Fatalf("EmbedImageBatch: batch=%v err=%v", batch, err)
	}
	tv, err := f.EmbedText(ctx, "a photo of a tree")
	if err != nil || len(tv) != 512 {
		t.Fatalf("EmbedText: v=%v err=%v", tv, err)
	}
	if f.Dimensions() != 512 {
		t.Errorf("Dimensions() = %d, want 512", f.Dimensions())
	}
}
