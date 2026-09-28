//go:build !cgo

package localembedder

import "errors"

// errNoCGo is returned when New() is called in a binary built without CGo.
// Rebuild with CGO_ENABLED=1 and MinGW-w64 (Windows) or gcc (Linux/macOS).
var errNoCGo = errors.New("localembedder: CGo required — rebuild with CGO_ENABLED=1 and a C compiler")

type onnxModel struct{}

func newOnnxModel(_ []byte, _ string) (*onnxModel, error) {
	return nil, errNoCGo
}

func (m *onnxModel) infer(_ []int64, _ []int64, _ []int64, _, _, _ int) ([]float32, error) {
	return nil, errNoCGo
}

func (m *onnxModel) close() {}
