package analysis

import "testing"

func TestSoundex(t *testing.T) {
	cases := []struct {
		input string
		want  string
	}{
		{"Robert", "R163"},
		{"Rupert", "R163"},   // same Soundex as Robert
		{"Euler", "E460"},
		{"Ellery", "E460"},   // same as Euler
		{"Gauss", "G200"},
		{"Ghosh", "G200"},    // same as Gauss
		{"Hilbert", "H416"},
		{"Heilbronn", "H416"}, // same as Hilbert
		{"Knuth", "K530"},
		{"Kant", "K530"},     // same as Knuth
		{"Thompson", "T512"},
		{"Thomson", "T525"},  // different
		{"", ""},
		{"A", "A000"},
		{"Go", "G000"},
		// H/W rule: H and W don't break adjacency between the consonants
		// on either side (unlike a true vowel), so C is treated as
		// adjacent to S through the H.
		{"Ashcraft", "A261"},
		{"Ashcroft", "A261"},
	}
	for _, c := range cases {
		got := Soundex(c.input)
		if got != c.want {
			t.Errorf("Soundex(%q) = %q, want %q", c.input, got, c.want)
		}
	}
}

func TestSoundexMatchesSimilarWords(t *testing.T) {
	// These pairs must produce the same code for phonetic matching to work.
	pairs := [][2]string{
		{"Robert", "Rupert"},
		{"Euler", "Ellery"},
		{"Gauss", "Ghosh"},
	}
	for _, p := range pairs {
		a, b := Soundex(p[0]), Soundex(p[1])
		if a != b {
			t.Errorf("Soundex(%q)=%q != Soundex(%q)=%q — expected a match", p[0], a, p[1], b)
		}
	}
}
