package localembedder

import (
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"runtime"

	"github.com/shramanb113/ZENITH/internal/embedding"
)

// CLIPEmbedder implements embedding.VisualEmbedder using in-process ONNX
// inference over CLIP's two independent towers. Never bundled into the
// binary — always loaded from modelsDir (the `zenith models pull` layout),
// unlike the text Embedder's bundled default. Safe for concurrent use.
type CLIPEmbedder struct {
	spec VisualSpec
	tok  *clipTokenizer
	pool *visionPool
	text *clipTextModel
}

// NewCLIPByID loads a registered visual model from
// modelsDir/<id>/{vision_model.onnx,text_model.onnx,vocab.json,merges.txt}
// (the layout `zenith models pull <id>` produces for a visual model — see
// Task 7). Requires CGo; without it, every call returns errNoCGo.
func NewCLIPByID(id, modelsDir string) (*CLIPEmbedder, error) {
	spec, err := LookupVisual(id)
	if err != nil {
		return nil, err
	}
	dir := filepath.Join(modelsDir, spec.ID)

	visionBytes, err := os.ReadFile(filepath.Join(dir, "vision_model.onnx"))
	if err != nil {
		return nil, fmt.Errorf("localembedder: visual model %q is not installed (%w) — run: zenith models pull %s", spec.ID, err, spec.ID)
	}
	textBytes, err := os.ReadFile(filepath.Join(dir, "text_model.onnx"))
	if err != nil {
		return nil, fmt.Errorf("localembedder: visual model %q is missing its text tower (%w) — run: zenith models pull %s", spec.ID, err, spec.ID)
	}
	vocabBytes, err := os.ReadFile(filepath.Join(dir, "vocab.json"))
	if err != nil {
		return nil, fmt.Errorf("localembedder: visual model %q is missing vocab.json (%w) — run: zenith models pull %s", spec.ID, err, spec.ID)
	}
	mergesBytes, err := os.ReadFile(filepath.Join(dir, "merges.txt"))
	if err != nil {
		return nil, fmt.Errorf("localembedder: visual model %q is missing merges.txt (%w) — run: zenith models pull %s", spec.ID, err, spec.ID)
	}

	tok, err := newCLIPTokenizer(vocabBytes, mergesBytes)
	if err != nil {
		return nil, fmt.Errorf("localembedder: clip tokenizer: %w", err)
	}

	libPath, err := extractToTemp(ortLibBytes, ortLibFilename)
	if err != nil {
		return nil, fmt.Errorf("localembedder: extract ort lib: %w", err)
	}

	pool, err := newVisionPool(visionBytes, libPath, clipPoolSize(runtime.NumCPU()), spec.Dims)
	if err != nil {
		return nil, fmt.Errorf("localembedder: clip vision pool: %w", err)
	}
	textModel, err := newClipTextModel(textBytes, libPath)
	if err != nil {
		pool.close()
		return nil, fmt.Errorf("localembedder: clip text model: %w", err)
	}

	return &CLIPEmbedder{spec: spec, tok: tok, pool: pool, text: textModel}, nil
}

// EmbedImage returns a 512-dim (spec.Dims) embedding for one decoded image.
func (c *CLIPEmbedder) EmbedImage(ctx context.Context, img image.Image) ([]float32, error) {
	out, err := c.EmbedImageBatch(ctx, []image.Image{img})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// EmbedImageBatch preprocesses every image and runs one ONNX call per batch
// through the vision session pool.
func (c *CLIPEmbedder) EmbedImageBatch(_ context.Context, imgs []image.Image) ([][]float32, error) {
	if len(imgs) == 0 {
		return nil, nil
	}
	size := c.spec.ImageSize
	flat := make([]float32, 0, len(imgs)*3*size*size)
	for _, img := range imgs {
		flat = append(flat, clipPreprocess(img, size, c.spec.ImageMean, c.spec.ImageStd)...)
	}
	embeds, err := c.pool.embedBatch(flat, len(imgs), size)
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(imgs))
	for i := range imgs {
		vec := make([]float32, c.spec.Dims)
		copy(vec, embeds[i*c.spec.Dims:(i+1)*c.spec.Dims])
		out[i] = l2Normalize(vec)
	}
	return out, nil
}

// EmbedText embeds a text query into the same space EmbedImage populates.
func (c *CLIPEmbedder) EmbedText(_ context.Context, text string) ([]float32, error) {
	ids := c.tok.encode(text, c.spec.ContextLength)
	embeds, err := c.text.infer(ids, 1, c.spec.ContextLength, c.spec.Dims)
	if err != nil {
		return nil, err
	}
	return l2Normalize(embeds), nil
}

// Dimensions returns the shared output size of both towers.
func (c *CLIPEmbedder) Dimensions() int { return c.spec.Dims }

// Name identifies the visual model for index-file compatibility checks (see
// embedding.Named and the design spec's §4 extended index header).
func (c *CLIPEmbedder) Name() string { return c.spec.IndexName() }

// Close releases both ONNX sessions/pools. Best-effort cleanup — like the
// text Embedder, callers are not required to call this before process exit.
func (c *CLIPEmbedder) Close() {
	c.pool.close()
	c.text.close()
}

// Verify interfaces at compile time.
var (
	_ embedding.VisualEmbedder = (*CLIPEmbedder)(nil)
	_ embedding.Named          = (*CLIPEmbedder)(nil)
)
