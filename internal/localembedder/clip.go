package localembedder

import (
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/shramanb113/ZENITH/internal/embedding"
)

// errCLIPClosed is returned by every Embed* method once Close has been
// called, instead of proceeding against a destroyed/closing pool or
// panicking on a closed channel.
var errCLIPClosed = fmt.Errorf("localembedder: CLIPEmbedder is closed")

// CLIPEmbedder implements embedding.VisualEmbedder using in-process ONNX
// inference over CLIP's two independent towers. Never bundled into the
// binary — always loaded from modelsDir (the `zenith models pull` layout),
// unlike the text Embedder's bundled default. Safe for concurrent use,
// including concurrent Close/Embed* races: Close is idempotent (a second
// call is a no-op) and every Embed* call after Close returns errCLIPClosed
// instead of panicking or running against a destroyed session.
type CLIPEmbedder struct {
	spec VisualSpec
	tok  *clipTokenizer
	pool *visionPool
	text *clipTextModel

	mu     sync.RWMutex
	closed bool
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

// EmbedImageBatch preprocesses every image and dispatches each one to the
// vision session pool as its own separate job (batchSize=1 each), run
// concurrently across goroutines. This naturally load-balances across every
// pool worker (the pool's job channel is work-stealing — whichever worker is
// next free picks up the next job), unlike building one combined
// flat/batched tensor for the whole call, which only one worker would ever
// process. Splitting into independent single-image jobs also makes each
// image's output vector depend only on itself: measured against the real
// quantized model, a shared-batch tensor made a given image's embedding
// differ slightly depending on which other images shared its batch (cosine
// 0.988-0.994 vs ~1.0), which breaks reproducibility for any downstream
// embedding cache.
//
// Unbounded goroutines-per-call is intentional, not an oversight: this
// mirrors EmbedImage's existing single-image cost (one goroutine, one real
// inference call), and the pool's fixed worker count is already the
// bottleneck for a large batch, so no additional semaphore is needed.
func (c *CLIPEmbedder) EmbedImageBatch(_ context.Context, imgs []image.Image) ([][]float32, error) {
	if len(imgs) == 0 {
		return nil, nil
	}

	// Held for the whole call, not just the initial check: multiple
	// concurrent Embed* calls can all hold this RLock at once (that's the
	// point of RWMutex — readers don't serialize each other), but Close's
	// write lock cannot proceed until every in-flight embed here has
	// returned, so the pool/session can never be torn down out from under
	// a call already past the closed check.
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return nil, errCLIPClosed
	}

	for i, img := range imgs {
		if img == nil {
			return nil, fmt.Errorf("localembedder: image at index %d is nil or has zero dimensions", i)
		}
		b := img.Bounds()
		if b.Dx() <= 0 || b.Dy() <= 0 {
			return nil, fmt.Errorf("localembedder: image at index %d is nil or has zero dimensions", i)
		}
	}

	size := c.spec.ImageSize
	out := make([][]float32, len(imgs))
	errs := make([]error, len(imgs))
	var wg sync.WaitGroup
	for i, img := range imgs {
		wg.Add(1)
		go func(i int, img image.Image) {
			defer wg.Done()
			flat := clipPreprocess(img, size, c.spec.ImageMean, c.spec.ImageStd)
			embeds, err := c.pool.embedBatch(flat, 1, size)
			if err != nil {
				errs[i] = err
				return
			}
			vec := make([]float32, c.spec.Dims)
			copy(vec, embeds)
			out[i] = l2Normalize(vec)
		}(i, img)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// EmbedText embeds a text query into the same space EmbedImage populates.
func (c *CLIPEmbedder) EmbedText(_ context.Context, text string) ([]float32, error) {
	// See EmbedImageBatch: held for the whole call so Close cannot tear
	// down the text session while this inference is in flight.
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return nil, errCLIPClosed
	}

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
// Idempotent: a second (or later) call is a safe no-op rather than double-
// closing the pool/session. Race-safe against concurrent Embed* calls too:
// every Embed* method holds mu's read lock for its whole duration (not just
// an initial check), so Close's write-lock acquisition blocks until every
// in-flight embed has finished before it tears anything down, and any
// Embed* call that starts after Close has set closed=true observes that
// under its own read lock and returns errCLIPClosed immediately, never
// touching the (possibly already-destroyed) pool/session.
func (c *CLIPEmbedder) Close() {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return
	}
	c.closed = true
	c.mu.Unlock()

	c.pool.close()
	c.text.close()
}

// Verify interfaces at compile time.
var (
	_ embedding.VisualEmbedder = (*CLIPEmbedder)(nil)
	_ embedding.Named          = (*CLIPEmbedder)(nil)
)
