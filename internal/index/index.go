package index

import (
	"encoding/gob"
	"log"
	"os"
	"slices"
	"sync"
	"time"
)

type InMemoryIndex struct {
	mu              sync.RWMutex
	data            map[string][]uint32
	idMapping       map[uint32]string
	internalCounter uint32
}

type SearchResult struct {
	ID    string
	Score float64
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

func (idx *InMemoryIndex) Search(queryTokens []string) []SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	scores := make(map[uint32]float64)

	for _, queryToken := range queryTokens {
		if ids, ok := idx.data[queryToken]; ok {
			for _, id := range ids {
				scores[id] += 1.0
			}
		}
	}

	results := make([]SearchResult, 0)

	for id, score := range scores {
		searchResult := &SearchResult{
			ID:    idx.idMapping[id],
			Score: score,
		}
		results = append(results, *searchResult)
	}

	slices.SortFunc(results, func(a SearchResult, b SearchResult) int {
		if a.Score > b.Score {
			return -1
		}
		if a.Score < b.Score {
			return 1
		}
		return 0
	})

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

func (idx *InMemoryIndex) Save(filepath string) error {
	start := time.Now()
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	log.Printf("💾 Saving index to %s...", filepath)

	file, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)

	if err := encoder.Encode(idx.data); err != nil {
		return err
	}

	if err := encoder.Encode(idx.idMapping); err != nil {
		return err
	}

	log.Printf("✅ Index saved. Entries: %d. Duration: %v", len(idx.data), time.Since(start))
	return nil
}

func (idx *InMemoryIndex) Load(filepath string) error {
	start := time.Now()
	idx.mu.Lock()
	defer idx.mu.Unlock()

	osClient, err := os.Open(filepath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("⚠️ No persistence file found at %s. Starting fresh.", filepath)
			return err
		}
		return err
	}
	defer osClient.Close()

	info := gob.NewDecoder(osClient)

	if err := info.Decode(&idx.data); err != nil {
		return err
	}

	if err := info.Decode(&idx.idMapping); err != nil {
		return err
	}

	log.Printf("Successfully loaded %d internal IDs from disk in %v", len(idx.idMapping), time.Since(start))
	return nil
}
