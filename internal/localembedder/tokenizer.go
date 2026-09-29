package localembedder

import (
	"bufio"
	"bytes"
	"strings"
	"unicode"
)

const (
	tokenPAD = int64(0)
	tokenUNK = int64(100)
	tokenCLS = int64(101)
	tokenSEP = int64(102)
	maxLen   = 256
)

type tokenizer struct {
	vocab map[string]int64
}

func newTokenizerFromBytes(data []byte) (*tokenizer, error) {
	vocab := make(map[string]int64, 32000)
	scanner := bufio.NewScanner(bytes.NewReader(data))
	var id int64
	for scanner.Scan() {
		vocab[scanner.Text()] = id
		id++
	}
	return &tokenizer{vocab: vocab}, scanner.Err()
}

// tokenize returns (inputIDs, attentionMask, tokenTypeIDs) each of length maxLength.
func (t *tokenizer) tokenize(text string, maxLength int) ([]int64, []int64, []int64) {
	pieces := t.wordpieceTokenize(strings.ToLower(text))

	// Reserve 2 slots for [CLS] and [SEP]; truncate if necessary.
	cap := maxLength - 2
	if len(pieces) > cap {
		pieces = pieces[:cap]
	}

	ids := make([]int64, maxLength)
	mask := make([]int64, maxLength)
	typeIDs := make([]int64, maxLength) // all zero: single-sequence input

	ids[0] = tokenCLS
	mask[0] = 1

	for i, tok := range pieces {
		id, ok := t.vocab[tok]
		if !ok {
			id = tokenUNK
		}
		ids[i+1] = id
		mask[i+1] = 1
	}

	sep := len(pieces) + 1
	ids[sep] = tokenSEP
	mask[sep] = 1
	// Positions sep+1…maxLength-1 remain 0 (PAD) in ids, mask, and typeIDs.

	return ids, mask, typeIDs
}

// encodeIDs returns the input token IDs for text — [CLS] + wordpieces + [SEP] —
// truncated so the total never exceeds maxTokens. No padding is applied; the
// caller pads to its chosen sequence length. Padding to a fixed 256 here cost
// 3–60× wasted ONNX compute per input (attention masks make padded positions
// correct, not free).
func (t *tokenizer) encodeIDs(text string, maxTokens int) []int64 {
	pieces := t.wordpieceTokenize(strings.ToLower(text))

	cap := maxTokens - 2
	if len(pieces) > cap {
		pieces = pieces[:cap]
	}

	ids := make([]int64, 0, len(pieces)+2)
	ids = append(ids, tokenCLS)
	for _, tok := range pieces {
		id, ok := t.vocab[tok]
		if !ok {
			id = tokenUNK
		}
		ids = append(ids, id)
	}
	return append(ids, tokenSEP)
}

// encodePair returns (inputIDs, attentionMask, tokenTypeIDs) for a cross-encoder
// pair: [CLS] query [SEP] doc [SEP], with tokenTypeIDs 0 over the query+first
// [SEP] and 1 over the doc+second [SEP]. The query is capped at maxQueryTokens
// so a long document can't crowd it out entirely; no padding is applied here —
// callers batch-pad to their chosen sequence length, same as encodeIDs.
func (t *tokenizer) encodePair(query, doc string, maxLength int) ([]int64, []int64, []int64) {
	const maxQueryTokens = 64
	qPieces := t.wordpieceTokenize(strings.ToLower(query))
	if len(qPieces) > maxQueryTokens {
		qPieces = qPieces[:maxQueryTokens]
	}
	dPieces := t.wordpieceTokenize(strings.ToLower(doc))
	budget := maxLength - 3 - len(qPieces) // [CLS] + [SEP] + [SEP]
	if budget < 0 {
		budget = 0
	}
	if len(dPieces) > budget {
		dPieces = dPieces[:budget]
	}

	n := 3 + len(qPieces) + len(dPieces)
	ids := make([]int64, 0, n)
	mask := make([]int64, 0, n)
	typeIDs := make([]int64, 0, n)

	ids = append(ids, tokenCLS)
	mask = append(mask, 1)
	typeIDs = append(typeIDs, 0)
	for _, tok := range qPieces {
		id, ok := t.vocab[tok]
		if !ok {
			id = tokenUNK
		}
		ids = append(ids, id)
		mask = append(mask, 1)
		typeIDs = append(typeIDs, 0)
	}
	ids = append(ids, tokenSEP)
	mask = append(mask, 1)
	typeIDs = append(typeIDs, 0)
	for _, tok := range dPieces {
		id, ok := t.vocab[tok]
		if !ok {
			id = tokenUNK
		}
		ids = append(ids, id)
		mask = append(mask, 1)
		typeIDs = append(typeIDs, 1)
	}
	ids = append(ids, tokenSEP)
	mask = append(mask, 1)
	typeIDs = append(typeIDs, 1)

	return ids, mask, typeIDs
}

func (t *tokenizer) wordpieceTokenize(text string) []string {
	var out []string
	for _, word := range splitOnWhitespaceAndPunct(text) {
		if word == "" {
			continue
		}
		out = append(out, t.tokenizeWord(word)...)
	}
	return out
}

func (t *tokenizer) tokenizeWord(word string) []string {
	runes := []rune(word)
	var sub []string
	start := 0
	for start < len(runes) {
		end := len(runes)
		cur := ""
		for end > start {
			candidate := string(runes[start:end])
			if start > 0 {
				candidate = "##" + candidate
			}
			if _, ok := t.vocab[candidate]; ok {
				cur = candidate
				break
			}
			end--
		}
		if cur == "" {
			return []string{"[UNK]"}
		}
		sub = append(sub, cur)
		start = end
	}
	return sub
}

func splitOnWhitespaceAndPunct(text string) []string {
	var words []string
	var cur strings.Builder
	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			if cur.Len() > 0 {
				words = append(words, cur.String())
				cur.Reset()
			}
		case isPunct(r):
			if cur.Len() > 0 {
				words = append(words, cur.String())
				cur.Reset()
			}
			words = append(words, string(r))
		default:
			cur.WriteRune(r)
		}
	}
	if cur.Len() > 0 {
		words = append(words, cur.String())
	}
	return words
}

func isPunct(r rune) bool {
	return (r >= 33 && r <= 47) ||
		(r >= 58 && r <= 64) ||
		(r >= 91 && r <= 96) ||
		(r >= 123 && r <= 126) ||
		unicode.IsPunct(r) || unicode.IsSymbol(r)
}
