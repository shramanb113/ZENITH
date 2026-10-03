package localembedder

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"
)

// buildSafetensors assembles a minimal valid safetensors blob (header + raw
// little-endian F32 data) for the given named tensors, in the same layout
// sentence-transformers writes its Dense module in.
func buildSafetensors(t *testing.T, tensors map[string][]float32, shapes map[string][]int) []byte {
	t.Helper()
	header := map[string]safetensorsEntry{}
	var body []byte
	for name, data := range tensors {
		start := len(body)
		for _, f := range data {
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(f))
			body = append(body, buf[:]...)
		}
		header[name] = safetensorsEntry{
			DType:       "F32",
			Shape:       shapes[name],
			DataOffsets: [2]int{start, len(body)},
		}
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	var out []byte
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerJSON)))
	out = append(out, lenBuf[:]...)
	out = append(out, headerJSON...)
	out = append(out, body...)
	return out
}

func TestParseDenseSafetensors(t *testing.T) {
	// 2x2 for a small, hand-checkable case: weight is [[1,0],[0,1]] (identity),
	// bias is [0,0], so apply(x) == tanh(x) elementwise.
	data := buildSafetensors(t,
		map[string][]float32{
			"linear.weight": {1, 0, 0, 1},
			"linear.bias":   {0, 0},
		},
		map[string][]int{
			"linear.weight": {2, 2},
			"linear.bias":   {2},
		},
	)
	d, err := parseDenseSafetensors(data)
	if err != nil {
		t.Fatalf("parseDenseSafetensors: %v", err)
	}
	if d.in != 2 || d.out != 2 {
		t.Fatalf("got in=%d out=%d, want 2,2", d.in, d.out)
	}
	got := d.apply([]float32{0.5, -1.0})
	want := []float32{float32(math.Tanh(0.5)), float32(math.Tanh(-1.0))}
	for i := range want {
		if diff := got[i] - want[i]; diff > 1e-6 || diff < -1e-6 {
			t.Errorf("apply()[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestParseDenseSafetensorsMissingTensor(t *testing.T) {
	data := buildSafetensors(t,
		map[string][]float32{"linear.weight": {1, 0, 0, 1}},
		map[string][]int{"linear.weight": {2, 2}},
	)
	if _, err := parseDenseSafetensors(data); err == nil {
		t.Fatal("expected error for missing linear.bias, got nil")
	}
}

func TestParseDenseSafetensorsShapeMismatch(t *testing.T) {
	data := buildSafetensors(t,
		map[string][]float32{
			"linear.weight": {1, 0, 0, 1, 0, 0}, // [3,2], doesn't match bias len 2
			"linear.bias":   {0, 0},
		},
		map[string][]int{
			"linear.weight": {3, 2},
			"linear.bias":   {2},
		},
	)
	if _, err := parseDenseSafetensors(data); err == nil {
		t.Fatal("expected error for shape mismatch, got nil")
	}
}
