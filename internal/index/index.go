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
	}
}

/* internal counter is for easier mapping of any document id to just a integer and the data holds the words and the slice of document id ( which is internalcounter) appearing on*/
func (idx *InMemoryIndex) Add(originalID string, fullText string, tokens []string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.internalCounter += 1
	idx.idMapping[idx.internalCounter] = originalID

	if originalID == "TECH-01" {
		log.Printf("🛠️ [INDEX DEBUG] Tokens for TECH-01: %v", tokens)
	}

	idx.vectors[idx.internalCounter], _ = analysis.GetEmbedding(fullText)
	seenInDoc := make(map[string]bool)

	for _, token := range tokens {

		// avoiding duplication

		if value, ok := idx.data[token]; ok {
			if len(value) > 0 && value[len(value)-1] == idx.internalCounter {
				continue
			}
		}
		if !seenInDoc[token] {
			idx.tokenCounts[token]++
			seenInDoc[token] = true
		}

		idx.data[token] = append(idx.data[token], idx.internalCounter)
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
		if ids, ok := idx.data[queryToken]; ok {
			for _, id := range ids {
				KeywordScores[id] += 1.0 / math.Log(1.0+float64(idx.tokenCounts[queryToken]))
				matchCounts[id] += 1

			}
		}
	}

	VectorScores := make(map[uint32]float64)

	for id, docVec := range idx.vectors {
		sim := analysis.CosineSimilarity(queryVec, docVec)

		VectorScores[id] += (float64(sim))
	}

	/*

		// NORMALIZATION

			allIDs := make(map[uint32]struct{})
			for id := range KeywordScores {
				allIDs[id] = struct{}{}
			}
			for id := range VectorScores {
				allIDs[id] = struct{}{}
			}

			results := make([]SearchResult, 0, len(allIDs))

			maxK, minK := -1.0, math.MaxFloat64
			maxV, minV := -1.0, math.MaxFloat64

			for id := range allIDs {
				kScore := KeywordScores[id]
				vScore := VectorScores[id]

				if matchCounts[id] == len(queryTokens) {
					kScore *= 2.0
				}

				// Update Keyword boundaries
				if kScore > maxK {
					maxK = kScore
				}
				if kScore < minK {
					minK = kScore
				}

				// Update Vector boundaries
				if vScore > maxV {
					maxV = vScore
				}
				if vScore < minV {
					minV = vScore
				}

				results = append(results, SearchResult{
					ID:           idx.idMapping[id],
					KeywordScore: kScore,
					VectorScore:  vScore,
				})
			}

			alpha := 0.3
			beta := 0.7

			for i := range results {
				normK := 0.0
				if maxK > minK {
					normK = (results[i].KeywordScore - minK) / (maxK - minK)
				}

				normV := 0.0
				if maxV > minV {
					normV = (results[i].VectorScore - minV) / (maxV - minV)
				}

				results[i].KeywordScore = normK
				results[i].VectorScore = normV
				results[i].FinalScore = (alpha * normK) + (beta * normV)
			}

			slices.SortFunc(results, func(a SearchResult, b SearchResult) int {
				if a.FinalScore > b.FinalScore {
					return -1
				}
				if a.FinalScore < b.FinalScore {
					return 1
				}
				return 0
			})

	*/

	// Reciprocal rank fusion

	for id := range KeywordScores {
		if matchCounts[id] == len(queryTokens) {
			KeywordScores[id] += 1000
		}
	}

	keywordIDs := make([]uint32, 0, len(KeywordScores))
	for id, score := range KeywordScores {
		if score > 0 {
			keywordIDs = append(keywordIDs, id)
		}
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
	for id, score := range VectorScores {

		if score > 0.0 {
			vectorIDs = append(vectorIDs, id)
		}

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

	rrfScores := make(map[uint32]float64)
	k := 60.0

	for rank, id := range keywordIDs {
		rrfScores[id] += (1.0 / (k + float64(rank+1))) * 2.0
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
	for i := MinGram; i <= limit; i++ {
		results = append(results, string(runes[0:i]))
	}

	if n > MaxGram {
		results = append(results, token)
	}

	return results
}
