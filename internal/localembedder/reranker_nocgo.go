//go:build !cgo

package localembedder

// rerankModel mirrors onnxModel's no-CGo stub: every method returns errNoCGo
// so Reranker's shared code compiles without an ONNX runtime.
type rerankModel struct{}

func newRerankModel(_ []byte, _ string) (*rerankModel, error) {
	return nil, errNoCGo
}

func (m *rerankModel) score(_, _, _ []int64, _, _ int) ([]float32, error) {
	return nil, errNoCGo
}

func (m *rerankModel) close() {}
