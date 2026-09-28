package index

import (
	"math"
	"sync"

	"github.com/x448/float16"
)

type VectorEntry struct {
	Vector    []uint16 // Stored as float16 representation
	Magnitude float64
}

// Memory optimization helpers
func FloatsToFloat16(vec []float32) []uint16 {
	out := make([]uint16, len(vec))
	for i, v := range vec {
		out[i] = float16.Fromfloat32(v).Bits()
	}
	return out
}

func Float16ToFloats(vec []uint16) []float32 {
	out := make([]float32, len(vec))
	for i, v := range vec {
		out[i] = float16.Frombits(v).Float32()
	}
	return out
}

// normalizeVector validates that vec has at least one dimension, contains
// only finite values, and has nonzero length, then returns an independent
// L2-normalized copy. It returns nil for anything else (empty/nil input,
// any NaN/Inf element, or a zero vector with no direction) so the caller
// degrades to lexical-only for that vector rather than storing or scoring
// against garbage.
//
// Scoring throughout the engine is a raw dot product (see ranking.DotProduct
// / vectorPass), which is only a valid cosine similarity when both operands
// are unit vectors. Not every source reliably provides that — Ollama and
// caller-supplied vectors aren't normalized — so we normalize once here at
// ingest instead of trusting every embedder.
func normalizeVector(vec []float32) []float32 {
	if len(vec) == 0 {
		return nil
	}
	var sumSq float64
	for _, v := range vec {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil
		}
		sumSq += f * f
	}
	if sumSq == 0 {
		return nil
	}
	norm := float32(math.Sqrt(sumSq))
	out := make([]float32, len(vec))
	for i, v := range vec {
		out[i] = v / norm
	}
	return out
}

type VectorStore struct {
	mu          sync.RWMutex
	vectors     map[uint64]VectorEntry
	wordVectors map[string]VectorEntry
}

func NewVectorStore() *VectorStore {
	return &VectorStore{
		vectors:     make(map[uint64]VectorEntry),
		wordVectors: make(map[string]VectorEntry),
	}
}

func (vs *VectorStore) RLock()   { vs.mu.RLock() }
func (vs *VectorStore) RUnlock() { vs.mu.RUnlock() }
func (vs *VectorStore) Lock()    { vs.mu.Lock() }
func (vs *VectorStore) Unlock()  { vs.mu.Unlock() }

func (vs *VectorStore) GetVectors() map[uint64]VectorEntry     { return vs.vectors }
func (vs *VectorStore) GetWordVectors() map[string]VectorEntry { return vs.wordVectors }

// ReplaceAll atomically swaps both backing maps for freshly decoded ones.
// Callers must hold vs.Lock() for the duration of the swap.
func (vs *VectorStore) ReplaceAll(vectors map[uint64]VectorEntry, wordVectors map[string]VectorEntry) {
	vs.vectors = vectors
	vs.wordVectors = wordVectors
}
func (vs *VectorStore) HasWordVector(word string) bool {
	vs.mu.RLock()
	defer vs.mu.RUnlock()
	_, exists := vs.wordVectors[word]
	return exists
}
