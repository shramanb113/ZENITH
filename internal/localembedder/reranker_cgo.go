//go:build cgo

package localembedder

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// rerankModel wraps an ONNX cross-encoder session. Its output is a single
// relevance logit per row ("logits", shape [batch, 1]), not the per-token
// last_hidden_state onnxModel produces, so it gets its own thin session
// wrapper instead of reusing onnxModel.
type rerankModel struct {
	session *ort.DynamicAdvancedSession
	mu      sync.Mutex
}

func newRerankModel(modelBytes []byte, libPath string) (*rerankModel, error) {
	if err := initORT(libPath); err != nil {
		return nil, fmt.Errorf("ort init: %w", err)
	}
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(
		modelBytes,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"logits"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("ort session: %w", err)
	}
	return &rerankModel{session: session}, nil
}

// score runs a forward pass over a batch of pre-padded pair encodings and
// returns one relevance logit per row.
func (m *rerankModel) score(inputIDs, attnMask, typeIDs []int64, batchSize, seqLen int) ([]float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	shape2D := ort.NewShape(int64(batchSize), int64(seqLen))

	idTensor, err := ort.NewTensor(shape2D, inputIDs)
	if err != nil {
		return nil, fmt.Errorf("ort input_ids tensor: %w", err)
	}
	defer idTensor.Destroy()

	maskTensor, err := ort.NewTensor(shape2D, attnMask)
	if err != nil {
		return nil, fmt.Errorf("ort attention_mask tensor: %w", err)
	}
	defer maskTensor.Destroy()

	typeTensor, err := ort.NewTensor(shape2D, typeIDs)
	if err != nil {
		return nil, fmt.Errorf("ort token_type_ids tensor: %w", err)
	}
	defer typeTensor.Destroy()

	outData := make([]float32, batchSize)
	outShape := ort.NewShape(int64(batchSize), 1)
	outTensor, err := ort.NewTensor(outShape, outData)
	if err != nil {
		return nil, fmt.Errorf("ort output tensor: %w", err)
	}
	defer outTensor.Destroy()

	if err := m.session.Run(
		[]ort.Value{idTensor, maskTensor, typeTensor},
		[]ort.Value{outTensor},
	); err != nil {
		return nil, fmt.Errorf("ort run: %w", err)
	}

	result := make([]float32, len(outTensor.GetData()))
	copy(result, outTensor.GetData())
	return result, nil
}

func (m *rerankModel) close() { _ = m.session.Destroy() }
