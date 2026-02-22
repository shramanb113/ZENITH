package index

import (
	"encoding/gob"
	"log"
	"maps"
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
	vocabulary      map[int][]string
	globalSeen      map[string]bool
	wordVectors     map[string][]float32
}

const (
	MinGram       = 3
	MaxGram       = 10
	SynonymWeight = 0.4
)

type synonymCandidate struct {
	word  string
	score float32
}

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
		vocabulary:      make(map[int][]string),
		globalSeen:      make(map[string]bool),
		wordVectors:     make(map[string][]float32),
	}
}

/* internal counter is for easier mapping of any document id to just a integer and the data holds the words and the slice of document id ( which is internalcounter) appearing on*/
func (idx *InMemoryIndex) Add(originalID string, fullText string, tokens []string) {

	docVec, _ := analysis.GetEmbedding(fullText)
	tempWordVectors := make(map[string][]float32)
	for _, t := range tokens {

		_, exists := tempWordVectors[t]
		if !idx.HasWordVector(t) && !exists {
			vec := idx.RegisterWordVector(t)
			tempWordVectors[t] = vec
		}
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.internalCounter += 1
	idx.idMapping[idx.internalCounter] = originalID

	idx.vectors[idx.internalCounter] = docVec

	maps.Copy(idx.wordVectors, tempWordVectors)

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
		}

		phon := analysis.Soundex(token)
		if phon != "" && !seenInDoc[phon] {
			log.Printf("Indexing Phonetic : %s for %s", phon, originalID)
			idx.phoneticData[phon] = append(idx.phoneticData[phon], idx.internalCounter)
			seenInDoc[phon] = true
		}

		if !idx.globalSeen[token] {
			L := len(token)
			idx.vocabulary[L] = append(idx.vocabulary[L], token)
			idx.globalSeen[token] = true

		}

	}

}

func (idx *InMemoryIndex) Search(query string, queryTokens []string) []SearchResponse {
	idx.mu.RLock()

	queryVec, _ := analysis.GetEmbedding(query)
	keywordScores := make(map[uint32]float64)
	matchCounts := make(map[uint32]int)

	// --- Pass 1: Lexical, Phonetic, and Fuzzy ---
	for _, token := range queryTokens {
		Q := len(token)

		// 1. N-Grams
		var searchFragments []string
		if Q >= 3 {
			searchFragments = generateEdgeNgrams(token)
		} else {
			searchFragments = []string{token}
		}

		for _, frag := range searchFragments {
			if ids, ok := idx.data[frag]; ok {
				for _, id := range ids {
					keywordScores[id] += (float64(len(frag)) / float64(Q)) * 100.0
					matchCounts[id]++
				}
			}
		}

		// 2. Phonetic
		phon := analysis.Soundex(token)
		if ids, ok := idx.phoneticData[phon]; ok {
			for _, id := range ids {
				keywordScores[id] += 50.0
				matchCounts[id]++
			}
		}

		// 3. Fuzzy (Levenshtein)
		if Q > 3 {
			minL, maxL := Q-1, Q+1
			for size, list := range idx.vocabulary {
				if size >= minL && size <= maxL {
					for _, candidate := range list {
						if dist, ok := analysis.Levenshtein(token, candidate); ok && dist > 0 && dist <= 2 {
							if ids, exists := idx.data[candidate]; exists {
								for _, id := range ids {
									keywordScores[id] += 60.0 / float64(dist)
									matchCounts[id]++
								}
							}
						}
					}
				}
			}
		}
	}

	// Calculate Vector Scores
	vectorScores := make(map[uint32]float64)
	for id, docVec := range idx.vectors {
		vectorScores[id] = float64(analysis.CosineSimilarity(queryVec, docVec))
	}

	// Initial Anchor: Only documents with keyword/fuzzy/phonetic matches get the "Top Tier" boost
	for id, score := range keywordScores {
		if score > 0 {
			keywordScores[id] += 10000.0
			if matchCounts[id] >= len(queryTokens) {
				keywordScores[id] += 50000.0
			}
		}
	}

	searchResponse := idx.finalizeRanks(keywordScores, vectorScores)

	// --- Pass 2: Neural Expansion (The "Rank 5" Killer) ---
	// If the top result is weak (Score < 5.0), it means we have no keyword matches.
	// --- Pass 2: Neural Expansion ---
	if len(searchResponse) == 0 || (len(searchResponse) > 0 && searchResponse[0].Score < 5.0) {
		idx.mu.RUnlock()

		expandedTerms := []string{}
		for _, token := range queryTokens {
			if len(token) < 3 {
				continue
			}
			// Aggressive neighbor search
			neighbors := idx.GetSemanticNeighbors(token, 5, 0.70)
			expandedTerms = append(expandedTerms, neighbors...)
		}

		idx.mu.RLock()
		analyzer := analysis.New() // Ensure we use the same rules as the indexer

		for _, term := range expandedTerms {
			// CRITICAL FIX: Stem the synonym to match the index keys!
			stemmedSynonym := analyzer.Stem(term)

			targets := make(map[uint32]bool)

			// 1. Check the exact stemmed synonym (e.g., "environment")
			if ids, ok := idx.data[stemmedSynonym]; ok {
				for _, id := range ids {
					targets[id] = true
				}
			}

			// 2. Check the prefix fallback (e.g., "env")
			if len(stemmedSynonym) > 3 {
				prefix := stemmedSynonym[:3]
				if ids, ok := idx.data[prefix]; ok {
					for _, id := range ids {
						targets[id] = true
					}
				}
			}

			for id := range targets {
				// We give them a massive Anchor + Bonus
				keywordScores[id] += 20000.0
				matchCounts[id]++ // Count it so RRF sees it as a keyword hit
			}
		}

		// RE-RANK: This forces the RRF to evaluate these "Neural Hits" as Keyword matches
		searchResponse = idx.finalizeRanks(keywordScores, vectorScores)
	}

	idx.mu.RUnlock()
	return searchResponse
}

func (idx *InMemoryIndex) finalizeRanks(keywordScores map[uint32]float64, vectorScores map[uint32]float64) []SearchResponse {
	const k = 10.0
	rrfScores := make(map[uint32]float64)

	// 1. Keyword Ranking (Only for those with anchor scores)
	keywordIDs := make([]uint32, 0)
	for id, score := range keywordScores {
		if score > 0 {
			keywordIDs = append(keywordIDs, id)
		}
	}
	slices.SortFunc(keywordIDs, func(a, b uint32) int {
		return compareScores(keywordScores[b], keywordScores[a])
	})

	// 2. Vector Ranking (Global)
	vectorIDs := make([]uint32, 0, len(vectorScores))
	for id := range vectorScores {
		vectorIDs = append(vectorIDs, id)
	}
	slices.SortFunc(vectorIDs, func(a, b uint32) int {
		return compareScores(vectorScores[b], vectorScores[a])
	})

	// 3. RRF Blending
	// Boost documents that appear in the keywordIDs (which now includes synonyms)
	for rank, id := range keywordIDs {
		rrfScores[id] += (1.0 / (k + float64(rank+1))) * 100.0
	}
	for rank, id := range vectorIDs {
		rrfScores[id] += (1.0 / (k + float64(rank+1)))
	}

	results := make([]SearchResponse, 0, len(rrfScores))
	for id, score := range rrfScores {
		results = append(results, SearchResponse{
			ID:    idx.idMapping[id],
			Score: score,
		})
	}

	slices.SortFunc(results, func(a, b SearchResponse) int {
		return compareScores(b.Score, a.Score)
	})

	return results
}

func compareScores(a, b float64) int {
	if a > b {
		return 1
	}
	if a < b {
		return -1
	}
	return 0
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

func (idx *InMemoryIndex) RegisterWordVector(word string) []float32 {

	vector, err := analysis.GetEmbedding(word)

	if err != nil {
		return nil
	}

	return vector
}

func (idx *InMemoryIndex) HasWordVector(t string) bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	_, exists := idx.wordVectors[t]

	return exists
}

func (idx *InMemoryIndex) GetSemanticNeighbors(token string, topN int, threshold float32) []string {
	tokenVec, err := analysis.GetEmbedding(token)
	if err != nil || tokenVec == nil {
		return []string{}
	}

	idx.mu.RLock()
	wordAndVec := make(map[string][]float32, len(idx.wordVectors))
	for k, v := range idx.wordVectors {
		wordAndVec[k] = v
	}
	idx.mu.RUnlock()

	var candidates []synonymCandidate

	for word, vec := range wordAndVec {
		if word == token {
			continue
		}

		score := analysis.CosineSimilarity(tokenVec, vec)
		if score >= threshold {
			candidates = append(candidates, synonymCandidate{
				word:  word,
				score: score,
			})
		}
	}

	if len(candidates) == 0 {
		return []string{}
	}

	slices.SortFunc(candidates, func(a, b synonymCandidate) int {
		if a.score > b.score {
			return -1
		}
		if a.score < b.score {
			return 1
		}
		return 0
	})

	limit := min(topN, len(candidates))

	results := make([]string, 0, limit)
	for i := 0; i < limit; i++ {
		results = append(results, candidates[i].word)
	}

	return results
}
