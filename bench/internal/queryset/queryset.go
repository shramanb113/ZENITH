// Package queryset derives harder query sets from a clean one so the
// benchmark measures more than exact-match retrieval. Every transform is
// seeded and deterministic: the same seed produces the same queries, so
// published numbers can be reproduced.
//
// Ground truth is unchanged. A perturbed query is graded against the same
// relevant passages as the clean query it came from; the question being
// asked is "does the engine still find the answer when the query is messy?".
package queryset

import (
	"math/rand"
	"strings"
	"unicode"
)

// Kind names a query set.
type Kind string

const (
	Clean     Kind = "clean"
	Typo      Kind = "typo"
	CodeMixed Kind = "codemixed"
)

// Valid reports whether k is a known query set.
func (k Kind) Valid() bool { return k == Clean || k == Typo || k == CodeMixed }

// Apply transforms text according to kind using rng.
func Apply(kind Kind, text string, rng *rand.Rand) string {
	switch kind {
	case Typo:
		return Perturb(text, rng, 0.35)
	case CodeMixed:
		return Mix(text, rng)
	default:
		return text
	}
}

// Perturb applies at most one character-level edit (delete, insert,
// substitute, or adjacent transposition) to each word of length >= 4 with
// probability rate, and always to at least one eligible word so no query
// silently stays clean. Short words are left alone: a one-character edit to
// a 3-letter word is usually another word, not a typo.
func Perturb(text string, rng *rand.Rand, rate float64) string {
	words := strings.Fields(text)
	var eligible []int
	for i, w := range words {
		if len([]rune(w)) >= 4 && isWordish(w) {
			eligible = append(eligible, i)
		}
	}
	if len(eligible) == 0 {
		return text
	}
	changed := false
	for _, i := range eligible {
		if rng.Float64() < rate {
			words[i] = editWord(words[i], rng)
			changed = true
		}
	}
	if !changed {
		i := eligible[rng.Intn(len(eligible))]
		words[i] = editWord(words[i], rng)
	}
	return strings.Join(words, " ")
}

func isWordish(w string) bool {
	for _, r := range w {
		if !unicode.IsLetter(r) {
			return false
		}
	}
	return true
}

func editWord(w string, rng *rand.Rand) string {
	r := []rune(w)
	// Never touch the first rune: real typos rarely change the first letter,
	// and it keeps the perturbation realistic rather than adversarial.
	pos := 1 + rng.Intn(len(r)-1)
	switch rng.Intn(4) {
	case 0: // delete
		return string(append(r[:pos:pos], r[pos+1:]...))
	case 1: // insert a duplicate of a neighbouring letter (fat-finger repeat)
		out := append([]rune{}, r[:pos]...)
		out = append(out, r[pos-1])
		return string(append(out, r[pos:]...))
	case 2: // substitute with a nearby letter
		r[pos] = neighbour(r[pos], rng)
		return string(r)
	default: // transpose two adjacent letters, never involving the first
		pos = 2 + rng.Intn(len(r)-2)
		r[pos-1], r[pos] = r[pos], r[pos-1]
		return string(r)
	}
}

func neighbour(c rune, rng *rand.Rand) rune {
	const rows = "qwertyuiop asdfghjkl zxcvbnm"
	lc := unicode.ToLower(c)
	i := strings.IndexRune(rows, lc)
	if i < 0 {
		return c
	}
	var opts []rune
	rr := []rune(rows)
	if i > 0 && rr[i-1] != ' ' {
		opts = append(opts, rr[i-1])
	}
	if i+1 < len(rr) && rr[i+1] != ' ' {
		opts = append(opts, rr[i+1])
	}
	if len(opts) == 0 {
		return c
	}
	return opts[rng.Intn(len(opts))]
}

// hinglishFillers are common romanised-Hindi function words and question
// particles that appear interleaved with English content words in
// code-mixed queries ("kubernetes ka memory issue kaise fix kare").
var hinglishFillers = []string{
	"ka", "ki", "ke", "hai", "hain", "kya", "kaise", "kyun", "mein", "me",
	"ko", "se", "aur", "nahi", "kare", "karna", "batao", "chahiye", "wala", "ye",
}

// Mix interleaves romanised-Hindi function words among the query's English
// words. IMPORTANT LIMITATION: this is a synthetic *noise-robustness* proxy
// for code-mixed input, not real Hinglish data. The corpus is English, so the
// content words stay English and only function words are added; it cannot
// test romanised-spelling variation of content words or Devanagari. Treat
// results as "does extra foreign-language noise break retrieval?", and replace
// this with a human-authored code-mixed set before making claims about
// Hinglish quality.
func Mix(text string, rng *rand.Rand) string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return text
	}
	out := make([]string, 0, len(words)*2)
	for i, w := range words {
		out = append(out, w)
		if i < len(words)-1 && rng.Float64() < 0.6 {
			out = append(out, hinglishFillers[rng.Intn(len(hinglishFillers))])
		}
	}
	// Question-style tail, as in "... kaise kare".
	if rng.Float64() < 0.5 {
		out = append(out, hinglishFillers[rng.Intn(len(hinglishFillers))])
	}
	return strings.Join(out, " ")
}
