//go:build cgo

package localembedder

import (
	"fmt"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// clipPoolSize picks the vision-tower session-pool size from the number of
// CPUs available. One session per CPU would reintroduce the exact
// oversubscription ROADMAP items K/L already measured (naive concurrent
// ONNX sessions run 5x slower, not faster) — this plan's default is a small
// fixed pool (min(numCPU, 4)) so each session still gets a real thread
// budget; the real knee is a benchmark sweep left to a later pass (per the
// design spec §6/§10 — this plan fixes the mechanism, not the tuned default).
func clipPoolSize(numCPU int) int {
	if numCPU < 1 {
		return 1
	}
	if numCPU > 4 {
		return 4
	}
	return numCPU
}

// clipThreadsPerSession divides numCPU across poolSize sessions, clamped to
// at least 1 so SetIntraOpNumThreads is never called with 0 or a negative
// value (which onnxruntime would reject).
func clipThreadsPerSession(poolSize, numCPU int) int {
	if poolSize < 1 {
		poolSize = 1
	}
	if numCPU < 1 {
		numCPU = 1
	}
	n := numCPU / poolSize
	if n < 1 {
		n = 1
	}
	return n
}

type visionJob struct {
	pixelValues []float32
	batchSize   int
	imageSize   int
	resultCh    chan visionResult
}

type visionResult struct {
	embeds []float32
	err    error
}

type visionWorker struct {
	session *ort.DynamicAdvancedSession
}

func (w *visionWorker) infer(pixelValues []float32, batchSize, imageSize, dims int) ([]float32, error) {
	inShape := ort.NewShape(int64(batchSize), 3, int64(imageSize), int64(imageSize))
	inTensor, err := ort.NewTensor(inShape, pixelValues)
	if err != nil {
		return nil, fmt.Errorf("clip vision input tensor: %w", err)
	}
	defer inTensor.Destroy()

	outData := make([]float32, batchSize*dims)
	outShape := ort.NewShape(int64(batchSize), int64(dims))
	outTensor, err := ort.NewTensor(outShape, outData)
	if err != nil {
		return nil, fmt.Errorf("clip vision output tensor: %w", err)
	}
	defer outTensor.Destroy()

	if err := w.session.Run([]ort.Value{inTensor}, []ort.Value{outTensor}); err != nil {
		return nil, fmt.Errorf("clip vision run: %w", err)
	}
	result := make([]float32, len(outTensor.GetData()))
	copy(result, outTensor.GetData())
	return result, nil
}

// visionPool is N ONNX sessions for CLIP's vision tower, each thread-capped
// so total intra-op thread usage stays ~= NumCPU regardless of N (see
// clipThreadsPerSession), fed by N goroutines pulling jobs from one channel.
// Each session is touched by exactly one goroutine (the one running
// runWorker for it), so no per-session mutex is needed — concurrency safety
// comes from the channel handoff, not locking.
type visionPool struct {
	jobs    chan visionJob
	workers []*visionWorker
	dims    int
	wg      sync.WaitGroup
}

func newVisionPool(modelBytes []byte, libPath string, poolSize, dims int) (*visionPool, error) {
	if err := initORT(libPath); err != nil {
		return nil, fmt.Errorf("ort init: %w", err)
	}
	if poolSize < 1 {
		poolSize = 1
	}
	threads := clipThreadsPerSession(poolSize, runtime.NumCPU())

	p := &visionPool{jobs: make(chan visionJob), dims: dims}
	for i := 0; i < poolSize; i++ {
		opts, err := ort.NewSessionOptions()
		if err != nil {
			p.closePartial()
			return nil, fmt.Errorf("clip vision session options: %w", err)
		}
		if err := opts.SetIntraOpNumThreads(threads); err != nil {
			opts.Destroy()
			p.closePartial()
			return nil, fmt.Errorf("clip vision SetIntraOpNumThreads: %w", err)
		}
		if err := opts.SetInterOpNumThreads(1); err != nil {
			opts.Destroy()
			p.closePartial()
			return nil, fmt.Errorf("clip vision SetInterOpNumThreads: %w", err)
		}
		session, err := ort.NewDynamicAdvancedSessionWithONNXData(
			modelBytes, []string{"pixel_values"}, []string{"image_embeds"}, opts)
		opts.Destroy()
		if err != nil {
			p.closePartial()
			return nil, fmt.Errorf("clip vision session %d: %w", i, err)
		}
		w := &visionWorker{session: session}
		p.workers = append(p.workers, w)
		p.wg.Add(1)
		go p.runWorker(w)
	}
	return p, nil
}

func (p *visionPool) runWorker(w *visionWorker) {
	defer p.wg.Done()
	for job := range p.jobs {
		embeds, err := w.infer(job.pixelValues, job.batchSize, job.imageSize, p.dims)
		job.resultCh <- visionResult{embeds: embeds, err: err}
	}
}

// embedBatch sends one batch through whichever worker is free next.
func (p *visionPool) embedBatch(pixelValues []float32, batchSize, imageSize int) ([]float32, error) {
	resultCh := make(chan visionResult, 1)
	p.jobs <- visionJob{pixelValues: pixelValues, batchSize: batchSize, imageSize: imageSize, resultCh: resultCh}
	res := <-resultCh
	return res.embeds, res.err
}

// closePartial destroys whatever sessions were created before a mid-
// construction error, without starting worker goroutines for them.
func (p *visionPool) closePartial() {
	for _, w := range p.workers {
		_ = w.session.Destroy()
	}
}

func (p *visionPool) close() {
	close(p.jobs)
	p.wg.Wait()
	for _, w := range p.workers {
		_ = w.session.Destroy()
	}
}

// clipTextModel is a single ONNX session for CLIP's text tower (one
// session + mutex, mirroring internal/localembedder/reranker_cgo.go's
// existing per-model-session pattern) — queries are embedded one at a time,
// not in bulk, so this tower does not need the vision tower's N-session
// pool. The real export has no attention_mask input (verified 2026-10-09 by
// inspecting the graph directly) — only input_ids.
type clipTextModel struct {
	session *ort.DynamicAdvancedSession
	mu      sync.Mutex
}

func newClipTextModel(modelBytes []byte, libPath string) (*clipTextModel, error) {
	if err := initORT(libPath); err != nil {
		return nil, fmt.Errorf("ort init: %w", err)
	}
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(
		modelBytes, []string{"input_ids"}, []string{"text_embeds"}, nil)
	if err != nil {
		return nil, fmt.Errorf("clip text session: %w", err)
	}
	return &clipTextModel{session: session}, nil
}

func (m *clipTextModel) infer(inputIDs []int64, batchSize, seqLen, dims int) ([]float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	shape := ort.NewShape(int64(batchSize), int64(seqLen))
	idTensor, err := ort.NewTensor(shape, inputIDs)
	if err != nil {
		return nil, fmt.Errorf("clip text input tensor: %w", err)
	}
	defer idTensor.Destroy()

	outData := make([]float32, batchSize*dims)
	outShape := ort.NewShape(int64(batchSize), int64(dims))
	outTensor, err := ort.NewTensor(outShape, outData)
	if err != nil {
		return nil, fmt.Errorf("clip text output tensor: %w", err)
	}
	defer outTensor.Destroy()

	if err := m.session.Run([]ort.Value{idTensor}, []ort.Value{outTensor}); err != nil {
		return nil, fmt.Errorf("clip text run: %w", err)
	}
	result := make([]float32, len(outTensor.GetData()))
	copy(result, outTensor.GetData())
	return result, nil
}

func (m *clipTextModel) close() { _ = m.session.Destroy() }
