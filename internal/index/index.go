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

func (idx *InMemoryIndex) SearchAND(queryTokens []string) []string {
	if len(queryTokens) == 0 {
		return nil
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	candidate, ok := idx.data[queryTokens[0]]
	if !ok {
		return nil
	}

	for i := 1; i < len(queryTokens); i++ {
		nextList, ok := idx.data[queryTokens[i]]
		if !ok || len(candidate) == 0 {
			return nil
		}

		var commonList []uint32
		j, k := 0, 0

		for j < len(candidate) && k < len(nextList) {
			if candidate[j] == nextList[k] {
				commonList = append(commonList, candidate[j])
				j++
				k++
			} else if candidate[j] < nextList[k] {
				j++
			} else if candidate[j] > nextList[k] {
				k++
			}

		}

		candidate = commonList
	}

	results := []string{}

	for _, ids := range candidate {
		results = append(results, idx.idMapping[uint32(ids)])
	}

	return results
}
