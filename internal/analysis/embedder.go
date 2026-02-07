package analysis

import (
	"hash/fnv"
	"log"
	"math/rand"
)

func ConvertFromStringToVector(s string) []float32 {
	h := fnv.New64()
	h.Write([]byte(s))
	uint64Hash := h.Sum64()

	vector := make([]float32, 128)
	r := rand.New(rand.NewSource(int64(uint64Hash)))

	for i := 0; i < 128; i++ {
		vector[i] = r.Float32()
	}

	log.Printf("🧠 Vectorized: '%s' into %d-dim space (Hash: %d)", s, len(vector), uint64Hash)

	return vector
}
