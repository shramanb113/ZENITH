package localembedder

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
)

// denseLayer is a trained linear projection + tanh activation applied after
// pooling, matching the sentence-transformers Dense module (e.g. LaBSE's
// 2_Dense, trained jointly with the translation-ranking loss — unlike a
// generic BERT pooler, which is NSP-trained and not meaningful for semantic
// similarity): y = tanh(W·x + b).
type denseLayer struct {
	weight []float32 // out x in, row-major
	bias   []float32
	in, out int
}

type safetensorsEntry struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// parseDenseSafetensors reads a single-file safetensors blob containing
// "linear.weight" ([out, in], F32) and "linear.bias" ([out], F32) tensors —
// the layout sentence-transformers saves its Dense module in.
func parseDenseSafetensors(data []byte) (*denseLayer, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("safetensors: file too short")
	}
	headerLen := binary.LittleEndian.Uint64(data[:8])
	if headerLen > uint64(len(data)-8) {
		return nil, fmt.Errorf("safetensors: truncated header")
	}
	var header map[string]safetensorsEntry
	if err := json.Unmarshal(data[8:8+headerLen], &header); err != nil {
		return nil, fmt.Errorf("safetensors: header: %w", err)
	}
	body := data[8+headerLen:]

	readF32 := func(name string) ([]float32, []int, error) {
		t, ok := header[name]
		if !ok {
			return nil, nil, fmt.Errorf("safetensors: missing tensor %q", name)
		}
		if t.DType != "F32" {
			return nil, nil, fmt.Errorf("safetensors: tensor %q has dtype %q, want F32", name, t.DType)
		}
		start, end := t.DataOffsets[0], t.DataOffsets[1]
		if start < 0 || end > len(body) || end < start || (end-start)%4 != 0 {
			return nil, nil, fmt.Errorf("safetensors: tensor %q has invalid data offsets %v", name, t.DataOffsets)
		}
		raw := body[start:end]
		out := make([]float32, (end-start)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, t.Shape, nil
	}

	bias, biasShape, err := readF32("linear.bias")
	if err != nil {
		return nil, err
	}
	if len(biasShape) != 1 {
		return nil, fmt.Errorf("safetensors: linear.bias has shape %v, want rank 1", biasShape)
	}
	out := biasShape[0]

	weight, weightShape, err := readF32("linear.weight")
	if err != nil {
		return nil, err
	}
	if len(weightShape) != 2 || weightShape[0] != out {
		return nil, fmt.Errorf("safetensors: linear.weight has shape %v, want [%d, in]", weightShape, out)
	}
	in := weightShape[1]
	if in*out != len(weight) {
		return nil, fmt.Errorf("safetensors: linear.weight data length %d does not match shape %v", len(weight), weightShape)
	}

	return &denseLayer{weight: weight, bias: bias, in: in, out: out}, nil
}

// apply computes tanh(W·x + b).
func (d *denseLayer) apply(x []float32) []float32 {
	out := make([]float32, d.out)
	for o := 0; o < d.out; o++ {
		var sum float32
		row := d.weight[o*d.in : (o+1)*d.in]
		for i, xi := range x {
			sum += row[i] * xi
		}
		out[o] = float32(math.Tanh(float64(sum + d.bias[o])))
	}
	return out
}
