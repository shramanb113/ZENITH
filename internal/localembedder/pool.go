package localembedder

import "math"



// meanPool computes the attention-mask-weighted mean over the sequence dimension.
// lastHiddenState is a flat []float32 of shape [seqLen * hidden].
// attentionMask is []int64 of length seqLen (1=real token, 0=padding).
func meanPool(lastHiddenState []float32, attentionMask []int64, seqLen, hidden int) []float32 {
	out := make([]float32, hidden)
	var count float64
	for i := 0; i < seqLen; i++ {
		if attentionMask[i] == 0 {
			continue
		}
		count++
		base := i * hidden
		for j := 0; j < hidden; j++ {
			out[j] += lastHiddenState[base+j]
		}
	}
	if count > 0 {
		for j := range out {
			out[j] = float32(float64(out[j]) / count)
		}
	}
	return out
}

// l2Normalize normalizes vec to unit length in-place and returns it.
// Returns the vector unchanged if its magnitude is zero.
func l2Normalize(vec []float32) []float32 {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	if sum == 0 {
		return vec
	}
	norm := float32(math.Sqrt(sum))
	for i := range vec {
		vec[i] /= norm
	}
	return vec
}
