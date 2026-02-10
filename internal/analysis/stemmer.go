package analysis

import "strings"

type Stemmer struct{}

func New() *Stemmer {
	return &Stemmer{}
}

func (s *Stemmer) Stem(word string) string {

	if len(word) <= 2 {
		return word
	}

	runes := []rune(strings.ToLower(word))

	// handles plurals and past participles
	runes = s.step1a(runes)

	// handles other endings
	runes = s.step1b(runes)

	// handle words ending with 'y'
	runes = s.step1c(runes)

	// handling double suffixes
	runes = s.step2(runes)

	// handling words ending with -ic,-full,-ness maybe some more
	runes = s.step3(runes)

	// handling -ant , -ence , etc.
	runes = s.step4(runes)

	// handling last letters which is e
	runes = s.step5a(runes)

	// hanadling with double l
	runes = s.step5b(runes)

	return string(runes)
}

// Helper functions

// In Porter's algorithm , y is considered a vowel if preceded by a constant

func (s *Stemmer) isConsonant(runes []rune, i int) bool {

	r := runes[i]

	if r == 'a' || r == 'e' || r == 'i' || r == 'o' || r == 'u' {
		return false
	}

	if r == 'y' {
		if i == 0 {
			return true
		}

		return !s.isConsonant(runes, i-1)
	}
	return true
}

func (s *Stemmer) isVowel(runes []rune, i int) bool {
	return !s.isConsonant(runes, i)
}

// to check the Vowel - consonant sequence count
func (s *Stemmer) m(runes []rune) int {

	count := 0
	i := 0
	n := len(runes)

	for i < n && s.isConsonant(runes, i) {
		i++
	}

	for i < n {

		for i < n && s.isVowel(runes, i) {
			i++
		}

		if i >= n {
			break
		}

		for i < n && s.isConsonant(runes, i) {
			i++
		}
		count++
	}

	return count
}

// endsWith checks if word ends wth suffix

func (s *Stemmer) endsWith(runes []rune, suffix string) bool {

	suffixRune := []rune(suffix)

	if len(runes) < len(suffixRune) {
		return false
	}

	for i := 0; i < len(suffixRune); i++ {

		if runes[len(runes)-len(suffixRune)+i] != suffixRune[i] {
			return false
		}
	}

	return true
}

func (s *Stemmer) replaceSuffix(runes []rune, suffix string, replacement string) []rune {

	if s.endsWith(runes, suffix) {

		return append(runes[:len(runes)-len([]rune(suffix))], []rune(replacement)...)

	}

	return runes
}

func (s *Stemmer) containsVowel(runes []rune) bool {

	for i := 0; i < len(runes); i++ {
		if s.isVowel(runes, i) {
			return true
		}
	}
	return false
}

func (s *Stemmer) endsDoubleConsonant(runes []rune) bool {

	if len(runes) < 2 {
		return false
	}

	last := runes[len(runes)-1]
	secondLast := runes[len(runes)-2]

	return last == secondLast && s.isConsonant(runes, len(runes)-1)
}

// endsCVC checks if word ends with consonant-vowel-consonant where second C is not w, x, or y.
func (s *Stemmer) endsCVC(runes []rune) bool {
	if len(runes) < 3 {
		return false
	}

	a := len(runes) - 3
	b := len(runes) - 2
	c := len(runes) - 1

	if !s.isConsonant(runes, a) || !s.isVowel(runes, b) || !s.isConsonant(runes, c) {
		return false
	}

	last := runes[c]
	return last != 'w' && last != 'x' && last != 'y'
}

// actual steps

func (s *Stemmer) step1a(runes []rune) []rune {

	// english suffixes mappped with their plurals
	replacements := []struct {
		suffix      string
		replacement string
	}{
		{"sses", "ss"},
		{"s", ""},
		{"ies", "i"},
		{"ss", "ss"},
	}

	for _, r := range replacements {

		if s.endsWith(runes, r.suffix) {
			return s.replaceSuffix(runes, r.suffix, r.replacement)
		}
	}

	return runes
}

func (s *Stemmer) step1b(runes []rune) []rune {

	return runes
}
func (s *Stemmer) step1c(runes []rune) []rune {

	if s.endsWith(runes, "y") && s.containsVowel(runes[:len(runes)-1]) {
		return append(runes[:len(runes)-1], []rune("i")...)
	}

	return runes
}

func (s *Stemmer) step2(runes []rune) []rune {

	return runes
}
func (s *Stemmer) step3(runes []rune) []rune {
	suffixes := []struct {
		suffix      string
		replacement string
	}{
		{"icate", "ic"},
		{"ative", ""},
		{"alize", "al"},
		{"iciti", "ic"},
		{"ical", "ic"},
		{"ful", ""},
		{"ness", ""},
	}

	for _, suffix := range suffixes {
		if s.endsWith(runes, suffix.suffix) {
			stem := runes[:len(runes)-len([]rune(suffix.suffix))]
			if s.m(stem) > 0 {
				return s.replaceSuffix(runes, suffix.suffix, suffix.replacement)
			}
			break
		}
	}

	return runes
}
func (s *Stemmer) step4(runes []rune) []rune {

	return runes
}
func (s *Stemmer) step5a(runes []rune) []rune {
	if s.endsWith(runes, "e") {
		stem := runes[:len(runes)-1]
		measure := s.m(runes)
		if measure > 1 || (measure == 1 && !s.endsCVC(stem)) {
			return stem
		}
	}
	return runes
}

func (s *Stemmer) step5b(runes []rune) []rune {
	if s.m(runes) > 1 && s.endsDoubleConsonant(runes) && runes[len(runes)-1] == 'l' {
		return runes[:len(runes)-1]
	}
	return runes
}
