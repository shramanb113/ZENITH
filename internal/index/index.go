package index

import (
	"encoding/gob"
	"log"
	"math"
	"os"
	"slices"
	"sync"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
)

type InMemoryIndex struct {
	mu              sync.RWMutex
	data            map[string][]uint32
	idMapping       map[uint32]string
	internalCounter uint32
	vectors         map[uint32][]float32
	tokenCounts     map[string]int
	phoneticData    map[string][]uint32
}

const (
	MinGram = 3
	MaxGram = 10
)

type SearchResult struct {
	ID           string
	KeywordScore float64
	VectorScore  float64
	FinalScore   float64
}

type SearchResponse struct {
	ID    string
	Score float64
}

func NewInMemoryIndex() *InMemoryIndex {
	return &InMemoryIndex{
		data:            make(map[string][]uint32),
		idMapping:       make(map[uint32]string),
		internalCounter: 0,
		vectors:         make(map[uint32][]float32),
		tokenCounts:     make(map[string]int),
		phoneticData:    make(map[string][]uint32),
	}
}

/* internal counter is for easier mapping of any document id to just a integer and the data holds the words and the slice of document id ( which is internalcounter) appearing on*/
func (idx *InMemoryIndex) Add(originalID string, fullText string, tokens []string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.internalCounter += 1
	idx.idMapping[idx.internalCounter] = originalID

	idx.vectors[idx.internalCounter], _ = analysis.GetEmbedding(fullText)
	seenInDoc := make(map[string]bool)

	for _, token := range tokens {

		idx.tokenCounts[token]++

		// avoiding duplication

		fragments := generateEdgeNgrams(token)

		for _, frag := range fragments {
			log.Printf("🔨 Indexing Fragment: %s for %s", frag, originalID)

			if seenInDoc[frag] {
				continue
			}
			seenInDoc[frag] = true

			idx.data[frag] = append(idx.data[frag], idx.internalCounter)
			log.Printf("🔨 Indexing Fragment: %s for %s", frag, originalID)
		}

		phon := analysis.Soundex(token)
		if phon != "" && !seenInDoc[phon] {
			log.Printf("Indexing Phonetic : %s for %s", phon, originalID)
			idx.phoneticData[phon] = append(idx.phoneticData[phon], idx.internalCounter)
			seenInDoc[phon] = true
		}

	}

}

func (idx *InMemoryIndex) Search(query string, queryTokens []string) []SearchResponse {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	log.Printf("the query token is  :")

	for _, word := range queryTokens {
		log.Printf("%s\n", word)
	}
	queryVec, _ := analysis.GetEmbedding(query)
	KeywordScores := make(map[uint32]float64)
	matchCounts := make(map[uint32]int)

	for _, queryToken := range queryTokens {

		var searchFragments []string

		if len(queryToken) >= 3 {
			searchFragments = generateEdgeNgrams(queryToken)
		} else {
			searchFragments = []string{queryToken}
		}

		log.Printf("DEBUG: Searching fragments for token [%s]: %v", queryToken, searchFragments)

		for _, frag := range searchFragments {
			if ids, ok := idx.data[frag]; ok {
				idf := 1.0 / math.Log(2.0+float64(idx.tokenCounts[frag]))
				lengthBonus := float64(len(frag)) / float64(len(queryToken))

				for _, id := range ids {
					KeywordScores[id] += idf * lengthBonus * 100.0
					matchCounts[id] += 1
				}
			}
		}
	}

	VectorScores := make(map[uint32]float64)

	for id, docVec := range idx.vectors {
		sim := analysis.CosineSimilarity(queryVec, docVec)

		VectorScores[id] += (float64(sim))
	}

	// Reciprocal rank fusion

	rrfScores := make(map[uint32]float64)

	for id, count := range matchCounts {
		KeywordScores[id] += 10000.0

		if count >= len(queryTokens) {
			rrfScores[id] += 2.0
		}

		if count > 1 {
			KeywordScores[id] += math.Pow(float64(count), 10)
		}

		if count == len(queryTokens) {
			KeywordScores[id] += 100000.0
		}
	}

	keywordIDs := make([]uint32, 0, len(KeywordScores))
	for id := range KeywordScores {
		keywordIDs = append(keywordIDs, id)
		log.Printf("DEBUG: Keyword Match Found for DocID %d", id)
	}

	slices.SortFunc(keywordIDs, func(a, b uint32) int {
		if KeywordScores[a] > KeywordScores[b] {
			return -1
		}
		if KeywordScores[a] < KeywordScores[b] {
			return 1
		}
		return 0
	})

	vectorIDs := make([]uint32, 0, len(VectorScores))
	for id := range VectorScores {
		vectorIDs = append(vectorIDs, id)
	}

	slices.SortFunc(vectorIDs, func(a, b uint32) int {
		if VectorScores[a] > VectorScores[b] {
			return -1
		}
		if VectorScores[a] < VectorScores[b] {
			return 1
		}
		return 0

	})

	k := 10.0

	for rank, id := range keywordIDs {
		rrfScores[id] += (1.0 / (k + float64(rank+1))) * 10.0
	}

	for rank, id := range vectorIDs {
		rrfScores[id] += 1.0 / (k + float64(rank+1))
	}

	searchResponse := make([]SearchResponse, 0, len(rrfScores))

	for id, score := range rrfScores {
		searchResponse = append(searchResponse, SearchResponse{
			ID:    idx.idMapping[id],
			Score: score,
		})
	}

	slices.SortFunc(searchResponse, func(a, b SearchResponse) int {
		if a.Score > b.Score {
			return -1
		}
		if a.Score < b.Score {
			return 1
		}
		return 0
	})

	return searchResponse

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

	if err := encoder.Encode(idx.vectors); err != nil {
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

	if err := info.Decode(&idx.vectors); err != nil {
		return err
	}

	log.Printf("Successfully loaded %d internal IDs from disk in %v", len(idx.idMapping), time.Since(start))
	return nil
}

func generateEdgeNgrams(token string) []string {
	runes := []rune(token)
	n := len(runes)

	if n < MinGram {
		return []string{token}
	}

	limit := n
	if limit > MaxGram {
		limit = MaxGram
	}

	results := make([]string, 0, limit-MinGram+1)
	results = append(results, token)
	for i := MinGram; i < limit; i++ {
		results = append(results, string(runes[0:i]))
	}

	if n > MaxGram {
		results = append(results, token)
	}

	return results
}
