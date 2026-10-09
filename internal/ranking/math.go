package ranking

import "math"

// DotProduct calculates the dot product using float64 and 4-way loop unrolling for speed
func DotProduct(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 || len(b) == 0 {
		return 0.0
	}

	var dot float64
	limit := len(a) - len(a)%4

	for i := 0; i < limit; i += 4 {
		dot += float64(a[i])*float64(b[i]) +
			float64(a[i+1])*float64(b[i+1]) +
			float64(a[i+2])*float64(b[i+2]) +
			float64(a[i+3])*float64(b[i+3])
	}

	for i := limit; i < len(a); i++ {
		dot += float64(a[i]) * float64(b[i])
	}

	return dot
}

// Magnitude calculates the vector magnitude using float64 and loop unrolling
func Magnitude(a []float32) float64 {
	var selfA float64
	limit := len(a) - len(a)%4

	for i := 0; i < limit; i += 4 {
		selfA += float64(a[i])*float64(a[i]) +
			float64(a[i+1])*float64(a[i+1]) +
			float64(a[i+2])*float64(a[i+2]) +
			float64(a[i+3])*float64(a[i+3])
	}

	for i := limit; i < len(a); i++ {
		selfA += float64(a[i]) * float64(a[i])
	}

	return math.Sqrt(selfA)
}
