package analysis

import "math"

func CosineSimilarity(a, b []float32) float32 {

	if len(a) != len(b) || len(a) == 0 || len(b) == 0 {
		return 0.0
	}

	var dot float64
	var selfA float64
	var selfB float64

	for i := 0; i < len(a); i++ {
		dot += float64(a[i] * b[i])
		selfA += float64(a[i] * a[i])
		selfB += float64(b[i] * b[i])
	}

	if selfA == 0.0 || selfB == 0.0 {
		return 0.0
	}

	return float32(dot / (math.Sqrt(selfA) * math.Sqrt(selfB)))
}
