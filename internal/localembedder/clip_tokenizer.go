package localembedder

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

const (
	clipBOS = int64(49406) // <|startoftext|>
	clipEOS = int64(49407) // <|endoftext|> — also pad and unk
)

// clipPretokenizeRE is CLIP's own pre-tokenization pattern, taken verbatim
// from the real tokenizer.json's pre_tokenizer config (not GPT-2's pattern,
// which differs) — text is already lowercased by the time this runs, so no
// case-insensitive flag is needed. Non-matching spans (in practice, just
// whitespace, which the normalizer has already collapsed to single spaces)
// are dropped, not kept, matching "Removed"+invert=true in the real config.
var clipPretokenizeRE = regexp.MustCompile(`<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`)

var whitespaceRunRE = regexp.MustCompile(`\s+`)

// clipByteEncoder is the GPT-2/CLIP byte-level BPE byte->rune bijection:
// printable Latin-1 bytes map to themselves; every other byte value (control
// characters, etc.) maps to a rune starting at 256, so every possible byte
// has a stable, printable rune representation for BPE merges to operate on.
// This exact construction (ranges '!'-'~', 0xA1-0xAC, 0xAE-0xFF, then the
// remaining bytes in ascending order) matches the reference implementation —
// verified by generating it and round-tripping real vocab.json lookups.
func clipByteEncoder() map[byte]rune {
	var bs []int
	isIn := map[int]bool{}
	add := func(lo, hi int) {
		for b := lo; b <= hi; b++ {
			bs = append(bs, b)
			isIn[b] = true
		}
	}
	add('!', '~')
	add(0xA1, 0xAC)
	add(0xAE, 0xFF)
	cs := append([]int(nil), bs...)
	n := 0
	for b := 0; b < 256; b++ {
		if !isIn[b] {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	m := make(map[byte]rune, 256)
	for i, b := range bs {
		m[byte(b)] = rune(cs[i])
	}
	return m
}

type clipTokenizer struct {
	vocab     map[string]int64
	mergeRank map[[2]string]int
	byteEnc   map[byte]rune
}

// newCLIPTokenizer builds a tokenizer from CLIP's own vocab.json (a flat
// string->id map) and merges.txt (one "tokenA tokenB" pair per line, rank =
// line index, with a leading "#version: ..." comment line skipped) — a
// wholly separate format and algorithm from the WordPiece tokenizer the rest
// of this package uses for the bert-base-uncased-vocabulary text models.
func newCLIPTokenizer(vocabJSON, mergesTxt []byte) (*clipTokenizer, error) {
	var vocab map[string]int64
	if err := json.Unmarshal(vocabJSON, &vocab); err != nil {
		return nil, fmt.Errorf("clip tokenizer: parsing vocab.json: %w", err)
	}

	mergeRank := make(map[[2]string]int)
	sc := bufio.NewScanner(bytes.NewReader(mergesTxt))
	rank := 0
	first := true
	for sc.Scan() {
		line := sc.Text()
		if first {
			first = false
			if strings.HasPrefix(line, "#version") {
				continue
			}
		}
		parts := strings.Fields(line)
		if len(parts) != 2 {
			continue
		}
		mergeRank[[2]string{parts[0], parts[1]}] = rank
		rank++
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("clip tokenizer: parsing merges.txt: %w", err)
	}

	return &clipTokenizer{vocab: vocab, mergeRank: mergeRank, byteEnc: clipByteEncoder()}, nil
}

// bpe applies BPE merges to word (a list of single-rune-string symbols, the
// last already carrying the "</w>" end-of-word suffix) until no mergeable
// adjacent pair remains, always merging the lowest-rank pair first.
func (t *clipTokenizer) bpe(word []string) []string {
	if len(word) <= 1 {
		return word
	}
	for {
		bestRank := -1
		bestIdx := -1
		for i := 0; i < len(word)-1; i++ {
			if r, ok := t.mergeRank[[2]string{word[i], word[i+1]}]; ok {
				if bestRank == -1 || r < bestRank {
					bestRank = r
					bestIdx = i
				}
			}
		}
		if bestIdx == -1 {
			break
		}
		merged := word[bestIdx] + word[bestIdx+1]
		next := make([]string, 0, len(word)-1)
		next = append(next, word[:bestIdx]...)
		next = append(next, merged)
		next = append(next, word[bestIdx+2:]...)
		word = next
	}
	return word
}

func (t *clipTokenizer) byteLevelEncode(s string) []string {
	b := []byte(s)
	out := make([]string, len(b))
	for i, by := range b {
		out[i] = string(t.byteEnc[by])
	}
	return out
}

// encodeIDs returns [bos, ...content ids, eos] with no padding and no
// truncation — the caller (encode) applies the context-length limit.
func (t *clipTokenizer) encodeIDs(text string) []int64 {
	text = whitespaceRunRE.ReplaceAllString(text, " ")
	text = strings.ToLower(text)

	pretoks := clipPretokenizeRE.FindAllString(text, -1)
	ids := make([]int64, 0, len(pretoks)+2)
	ids = append(ids, clipBOS)
	for _, pt := range pretoks {
		symbols := t.byteLevelEncode(pt)
		if len(symbols) > 0 {
			symbols[len(symbols)-1] += "</w>"
		}
		for _, piece := range t.bpe(symbols) {
			id, ok := t.vocab[piece]
			if !ok {
				id = clipEOS // unk_token is <|endoftext|> for this model
			}
			ids = append(ids, id)
		}
	}
	return append(ids, clipEOS)
}

// encode returns exactly contextLength ids: bos, content (truncated so the
// trailing eos always fits), eos, then eos used again as the pad id up to
// contextLength — matching this model's tokenizer_config.json, where
// pad_token == eos_token.
func (t *clipTokenizer) encode(text string, contextLength int) []int64 {
	ids := t.encodeIDs(text)
	if len(ids) > contextLength {
		// Keep bos, truncate content, force the last slot back to eos.
		ids = ids[:contextLength]
		ids[contextLength-1] = clipEOS
	}
	out := make([]int64, contextLength)
	copy(out, ids)
	for i := len(ids); i < contextLength; i++ {
		out[i] = clipEOS
	}
	return out
}
