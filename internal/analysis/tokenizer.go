package analysis

import (
	"regexp"
	"strings"
)

type Tokenizer interface {
	Tokenize(text string) []string
}

type StandardTokenizer struct {
	stopWords map[string]struct{}
}

func NewStandardTokenizer() *StandardTokenizer {
	stopList := []string{
		"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
	}

	stopMap := make(map[string]struct{})
	for _, s := range stopList {
		stopMap[s] = struct{}{}
	}
	return &StandardTokenizer{stopWords: stopMap}
}

func (t *StandardTokenizer) Tokenize(text string) []string {
	PorterStem := New()

	re := regexp.MustCompile(`[A-Z][a-z0-9]*|[a-z0-9]+|[A-Z]+`)
	rawTokens := re.FindAllString(text, -1)

	var filtered []string

	for _, token := range rawTokens {
		token = strings.ToLower(token)

		if _, ok := t.stopWords[token]; ok {
			continue
		}

		stemmed := PorterStem.Stem(token)
		if stemmed != "" {
			filtered = append(filtered, stemmed)
		}
	}
	return filtered
}
