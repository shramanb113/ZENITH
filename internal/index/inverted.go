package index

import "sync"

type InvertedIndex struct {
	mu           sync.RWMutex
	data         map[string][]uint64
	tokenCounts  map[string]int
	vocabulary   map[int][]string
	globalSeen   map[string]int     // term → reference count; 0 means deleted
	docFragments map[uint64][]string // fragment tracking for idempotency
	docTokens    map[uint64][]string // raw tokens per doc for globalSeen ref-counting
}

func NewInvertedIndex() *InvertedIndex {
	return &InvertedIndex{
		data:         make(map[string][]uint64),
		tokenCounts:  make(map[string]int),
		vocabulary:   make(map[int][]string),
		globalSeen:   make(map[string]int),
		docFragments: make(map[uint64][]string),
		docTokens:    make(map[uint64][]string),
	}
}

func (idx *InvertedIndex) RLock()   { idx.mu.RLock() }
func (idx *InvertedIndex) RUnlock() { idx.mu.RUnlock() }
func (idx *InvertedIndex) Lock()    { idx.mu.Lock() }
func (idx *InvertedIndex) Unlock()  { idx.mu.Unlock() }

func (idx *InvertedIndex) GetTokenCounts() map[string]int      { return idx.tokenCounts }
func (idx *InvertedIndex) GetGlobalSeen() map[string]int       { return idx.globalSeen }
func (idx *InvertedIndex) GetDocFragments() map[uint64][]string { return idx.docFragments }
func (idx *InvertedIndex) GetDocTokens() map[uint64][]string   { return idx.docTokens }
func (idx *InvertedIndex) GetVocabulary() map[int][]string     { return idx.vocabulary }
func (idx *InvertedIndex) GetData() map[string][]uint64        { return idx.data }

// ReplaceAll atomically swaps every backing map for freshly decoded ones.
// Callers must hold idx.Lock() for the duration of the swap.
func (idx *InvertedIndex) ReplaceAll(
	data map[string][]uint64,
	tokenCounts map[string]int,
	vocabulary map[int][]string,
	globalSeen map[string]int,
	docFragments map[uint64][]string,
	docTokens map[uint64][]string,
) {
	idx.data = data
	idx.tokenCounts = tokenCounts
	idx.vocabulary = vocabulary
	idx.globalSeen = globalSeen
	idx.docFragments = docFragments
	idx.docTokens = docTokens
}
