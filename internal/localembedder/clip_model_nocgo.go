//go:build !cgo

package localembedder

func clipPoolSize(numCPU int) int { return 1 }

func clipThreadsPerSession(poolSize, numCPU int) int { return 1 }

type visionPool struct{}

func newVisionPool(_ []byte, _ string, _, _ int) (*visionPool, error) {
	return nil, errNoCGo
}

func (p *visionPool) embedBatch(_ []float32, _, _ int) ([]float32, error) {
	return nil, errNoCGo
}

func (p *visionPool) close() {}

type clipTextModel struct{}

func newClipTextModel(_ []byte, _ string) (*clipTextModel, error) {
	return nil, errNoCGo
}

func (m *clipTextModel) infer(_ []int64, _, _, _ int) ([]float32, error) {
	return nil, errNoCGo
}

func (m *clipTextModel) close() {}
