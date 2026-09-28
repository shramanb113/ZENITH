package index

import "sync"

// InvertedIndex holds the postings of the mutable in-memory delta (documents
// added since the last flush). Documents that live in segments are read from
// the segments instead; see layers.go.
type InvertedIndex struct {
	mu         sync.RWMutex
	data       map[string][]uint64 // edge-n-gram fragment -> internal doc IDs
	globalSeen map[string]int      // term → occurrence count in the delta; 0 means deleted
	docTokens  map[uint64][]string // raw tokens per doc for globalSeen ref-counting and re-linking
}

func NewInvertedIndex() *InvertedIndex {
	return &InvertedIndex{
		data:       make(map[string][]uint64),
		globalSeen: make(map[string]int),
		docTokens:  make(map[uint64][]string),
	}
}

func (idx *InvertedIndex) RLock()   { idx.mu.RLock() }
func (idx *InvertedIndex) RUnlock() { idx.mu.RUnlock() }
func (idx *InvertedIndex) Lock()    { idx.mu.Lock() }
func (idx *InvertedIndex) Unlock()  { idx.mu.Unlock() }

func (idx *InvertedIndex) GetGlobalSeen() map[string]int     { return idx.globalSeen }
func (idx *InvertedIndex) GetDocTokens() map[uint64][]string { return idx.docTokens }
func (idx *InvertedIndex) GetData() map[string][]uint64      { return idx.data }

// ReplaceAll atomically swaps every backing map. Callers must hold idx.Lock()
// for the duration of the swap.
func (idx *InvertedIndex) ReplaceAll(
	data map[string][]uint64,
	globalSeen map[string]int,
	docTokens map[uint64][]string,
) {
	idx.data = data
	idx.globalSeen = globalSeen
	idx.docTokens = docTokens
}

// Reset empties the delta.
func (idx *InvertedIndex) Reset() {
	idx.ReplaceAll(make(map[string][]uint64), make(map[string]int), make(map[uint64][]string))
}
