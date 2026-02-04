package index

import "sync"

type InMemoryIndex struct {
	mu              sync.RWMutex
	data            map[string][]uint32
	idMapping       map[uint32]string
	internalCounter uint32
}

func NewInMemoryIndex() *InMemoryIndex {
	return &InMemoryIndex{
		data:            make(map[string][]uint32),
		idMapping:       make(map[uint32]string),
		internalCounter: 0,
	}
}

/* internal counter is for easier mapping of any document id to just a integer and the data holds the words and the slice of document id ( which is internalcounter) appearing on*/
func (idx *InMemoryIndex) Add(originalID string, tokens []string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.internalCounter += 1
	idx.idMapping[idx.internalCounter] = originalID

	for _, token := range tokens {

		// avoiding duplication

		if value, ok := idx.data[token]; ok {
			if len(value) > 0 && value[len(value)-1] == idx.internalCounter {
				continue
			}
		}
		idx.data[token] = append(idx.data[token], idx.internalCounter)
	}

}

func (idx *InMemoryIndex) Search(queryTokens []string) []string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	foundIDs := make(map[uint32]struct{})

	for _, queryToken := range queryTokens {
		if ids, ok := idx.data[queryToken]; ok {
			for _, id := range ids {
				foundIDs[id] = struct{}{}
			}
		}
	}

	results := []string{}

	for ids, _ := range foundIDs {
		results = append(results, idx.idMapping[ids])
	}

	return results
}
