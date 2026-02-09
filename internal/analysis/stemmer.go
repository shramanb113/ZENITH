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

	runes := s.step1a(runes)

	runes := s.step1b(runes)

	runes := s.step1c(runes)

	runes := s.step2(runes)

	runes := s.step3(runes)

	runes := s.step4(runes)

	runes := s.step5a(runes)

	runes := s.step5b(runes)

	return string(runes)
}
