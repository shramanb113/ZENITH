//go:build cgo

package localembedder

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	ortOnce sync.Once
	ortErr  error
)

func initORT(libPath string) error {
	ortOnce.Do(func() {
		ort.SetSharedLibraryPath(libPath)
		ortErr = ort.InitializeEnvironment()
	})
	return ortErr
}

type onnxModel struct {
	session    *ort.DynamicAdvancedSession
	hasTypeIDs bool
	mu         sync.Mutex
}

// newOnnxModel loads the ONNX model from bytes and initialises an inference
// session. libPath must point to the extracted onnxruntime shared library on
// disk. typeIDs must match the graph's real input signature — DistilBERT-
// family models have no token_type_ids input at all (see Spec.NoTokenTypeIDs)
// and onnxruntime refuses the run ("Invalid input name") if one is offered.
func newOnnxModel(modelBytes []byte, libPath string, typeIDs bool) (*onnxModel, error) {
	if err := initORT(libPath); err != nil {
		return nil, fmt.Errorf("ort init: %w", err)
	}
	inputs := []string{"input_ids", "attention_mask"}
	if typeIDs {
		inputs = append(inputs, "token_type_ids")
	}
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(
		modelBytes,
		inputs,
		[]string{"last_hidden_state"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("ort session: %w", err)
	}
	return &onnxModel{session: session, hasTypeIDs: typeIDs}, nil
}

// infer runs a forward pass and returns the raw last_hidden_state as a flat
// []float32 of shape [batchSize * seqLen * hiddenSize].
func (m *onnxModel) infer(inputIDs, attnMask, typeIDs []int64, batchSize, seqLen, hiddenSize int) ([]float32, error) {
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

	inputs := make([]ort.Value, 0, 3) // cap 3: avoids a realloc when token_type_ids is appended below
	inputs = append(inputs, idTensor, maskTensor)
	if m.hasTypeIDs {
		typeTensor, err := ort.NewTensor(shape2D, typeIDs)
		if err != nil {
			return nil, fmt.Errorf("ort token_type_ids tensor: %w", err)
		}
		defer typeTensor.Destroy()
		inputs = append(inputs, typeTensor)
	}

	outData := make([]float32, batchSize*seqLen*hiddenSize)
	outShape := ort.NewShape(int64(batchSize), int64(seqLen), int64(hiddenSize))
	outTensor, err := ort.NewTensor(outShape, outData)
	if err != nil {
		return nil, fmt.Errorf("ort output tensor: %w", err)
	}
	defer outTensor.Destroy()

	if err := m.session.Run(
		inputs,
		[]ort.Value{outTensor},
	); err != nil {
		return nil, fmt.Errorf("ort run: %w", err)
	}

	result := make([]float32, len(outTensor.GetData()))
	copy(result, outTensor.GetData())
	return result, nil
}

func (m *onnxModel) close() {
	_ = m.session.Destroy()
}
