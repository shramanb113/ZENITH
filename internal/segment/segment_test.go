package segment

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

type fixture struct {
	docs  []DocIn
	terms []TermIn
	frags []KeyIn
	phon  []KeyIn
	fwd   [][]TermFreq
	words []string
	wvecs [][]uint16
	dels  []uint64
	dims  int
}

func makeFixture(nDocs, dims int) fixture {
	f := fixture{dims: dims}
	for i := 0; i < nDocs; i++ {
		d := DocIn{
			ID:   uint64(1000 + i*7), // ascending, gaps
			Orig: fmt.Sprintf("doc-%d", i),
			Text: fmt.Sprintf("text body number %d with ünïcode ✓", i),
			Len:  3 + i%5,
		}
		if i%4 != 3 { // every 4th doc has no vector
			d.Vec = make([]uint16, dims)
			for k := range d.Vec {
				d.Vec[k] = uint16(i*31 + k)
			}
		}
		if i%3 == 0 {
			d.Attrs = []byte(fmt.Sprintf(`{"k":%d}`, i))
		}
		f.docs = append(f.docs, d)
	}
	// terms: t000..t019, each in a stride of docs
	for t := 0; t < 20 && t < nDocs; t++ {
		ti := TermIn{Term: fmt.Sprintf("t%03d", t), CF: 0}
		for r := t % 3; r < nDocs; r += 3 + t%4 {
			ti.Posts = append(ti.Posts, Posting{Doc: r, TF: 1 + (r+t)%3})
			ti.CF += 1 + (r+t)%3
		}
		f.terms = append(f.terms, ti)
	}
	f.fwd = make([][]TermFreq, nDocs)
	for ti, t := range f.terms {
		for _, p := range t.Posts {
			f.fwd[p.Doc] = append(f.fwd[p.Doc], TermFreq{Term: ti, TF: p.TF})
		}
	}
	for _, key := range []string{"abc", "abcd", "xyz"} {
		k := KeyIn{Key: key}
		for r := len(key) % 2; r < nDocs; r += 2 + len(key)%3 {
			k.Docs = append(k.Docs, r)
		}
		f.frags = append(f.frags, k)
	}
	sort.Slice(f.frags, func(i, j int) bool { return f.frags[i].Key < f.frags[j].Key })
	f.phon = []KeyIn{{Key: "A100", Docs: []int{0, 2}}, {Key: "B200", Docs: []int{1}}}
	f.words = []string{"alpha", "beta", "gamma"}
	for i := range f.words {
		v := make([]uint16, dims)
		for k := range v {
			v[k] = uint16(100*i + k)
		}
		f.wvecs = append(f.wvecs, v)
	}
	f.dels = []uint64{5, 9, 4242}
	return f
}

func (f fixture) write(t testing.TB, path string) {
	t.Helper()
	w, err := Create(path, f.dims)
	if err != nil {
		t.Fatal(err)
	}
	w.WriteDocs(len(f.docs), func(i int) DocIn { return f.docs[i] })
	ti := 0
	w.WriteTerms(func() (TermIn, bool) {
		if ti >= len(f.terms) {
			return TermIn{}, false
		}
		ti++
		return f.terms[ti-1], true
	})
	fi := 0
	w.WriteFrags(func() (KeyIn, bool) {
		if fi >= len(f.frags) {
			return KeyIn{}, false
		}
		fi++
		return f.frags[fi-1], true
	})
	pi := 0
	w.WritePhon(func() (KeyIn, bool) {
		if pi >= len(f.phon) {
			return KeyIn{}, false
		}
		pi++
		return f.phon[pi-1], true
	})
	w.WriteForward(len(f.docs), func(i int) []TermFreq { return f.fwd[i] })
	w.WriteWords(len(f.words), func(i int) (string, []uint16) { return f.words[i], f.wvecs[i] })
	w.WriteDels(f.dels)
	if err := w.Finish([]byte(`{"gen":1}`)); err != nil {
		t.Fatal(err)
	}
}

func TestRoundTrip(t *testing.T) {
	f := makeFixture(50, 8)
	path := filepath.Join(t.TempDir(), "a.seg")
	f.write(t, path)

	s, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Verify(); err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if s.NumDocs() != 50 || s.Dims() != 8 {
		t.Fatalf("docs=%d dims=%d", s.NumDocs(), s.Dims())
	}
	if string(s.Meta()) != `{"gen":1}` {
		t.Fatalf("meta = %q", s.Meta())
	}

	for i, d := range f.docs {
		if s.DocID(i) != d.ID || s.Orig(i) != d.Orig || s.Text(i) != d.Text || s.DocLen(i) != d.Len {
			t.Fatalf("doc %d mismatch", i)
		}
		if got, ok := s.FindDoc(d.ID); !ok || got != i {
			t.Fatalf("FindDoc(%d) = %d,%v", d.ID, got, ok)
		}
		if string(s.Attrs(i)) != string(d.Attrs) {
			t.Fatalf("attrs %d: %q vs %q", i, s.Attrs(i), d.Attrs)
		}
		v := s.Vec(i)
		if (v == nil) != (d.Vec == nil) {
			t.Fatalf("vec presence %d", i)
		}
		for k := range d.Vec {
			if v[k] != d.Vec[k] {
				t.Fatalf("vec %d[%d]", i, k)
			}
		}
	}
	if _, ok := s.FindDoc(1001); ok {
		t.Fatal("FindDoc found a missing ID")
	}
	wantVecs := 0
	for _, d := range f.docs {
		if d.Vec != nil {
			wantVecs++
		}
	}
	if s.NumVecs() != wantVecs {
		t.Fatalf("NumVecs = %d, want %d", s.NumVecs(), wantVecs)
	}

	for ti, want := range f.terms {
		idx := s.FindTerm(want.Term)
		if idx != ti {
			t.Fatalf("FindTerm(%q) = %d, want %d", want.Term, idx, ti)
		}
		if string(s.TermKey(idx)) != want.Term || s.TermCF(idx) != want.CF || s.TermDF(idx) != len(want.Posts) {
			t.Fatalf("term %q stats mismatch", want.Term)
		}
		var got []Posting
		s.TermPostings(idx, func(row, tf int) bool { got = append(got, Posting{row, tf}); return true })
		if fmt.Sprint(got) != fmt.Sprint(want.Posts) {
			t.Fatalf("term %q postings %v want %v", want.Term, got, want.Posts)
		}
	}
	if s.FindTerm("nope") != -1 || s.FindTerm("") != -1 || s.FindTerm("zzzz") != -1 {
		t.Fatal("FindTerm found a missing term")
	}

	for _, want := range f.frags {
		idx := s.FindFrag(want.Key)
		if idx < 0 {
			t.Fatalf("frag %q missing", want.Key)
		}
		var got []int
		s.FragPostings(idx, func(r int) bool { got = append(got, r); return true })
		if fmt.Sprint(got) != fmt.Sprint(want.Docs) || s.FragCount(idx) != len(want.Docs) {
			t.Fatalf("frag %q postings %v want %v", want.Key, got, want.Docs)
		}
	}
	if s.FindFrag("ab") != -1 {
		t.Fatal("FindFrag matched a prefix")
	}
	for _, want := range f.phon {
		idx := s.FindPhon(want.Key)
		var got []int
		s.PhonPostings(idx, func(r int) bool { got = append(got, r); return true })
		if fmt.Sprint(got) != fmt.Sprint(want.Docs) {
			t.Fatalf("phon %q postings %v want %v", want.Key, got, want.Docs)
		}
	}

	for i := range f.docs {
		var got []TermFreq
		s.Forward(i, func(term, tf int) { got = append(got, TermFreq{term, tf}) })
		if fmt.Sprint(got) != fmt.Sprint(f.fwd[i]) {
			t.Fatalf("forward %d = %v want %v", i, got, f.fwd[i])
		}
	}

	for i, w := range f.words {
		idx := s.FindWord(w)
		if idx != i || s.Word(idx) != w {
			t.Fatalf("word %q idx %d", w, idx)
		}
		v := s.WordVec(idx)
		for k := range v {
			if v[k] != f.wvecs[i][k] {
				t.Fatalf("word vec %q[%d]", w, k)
			}
		}
	}
	if fmt.Sprint(s.Dels()) != fmt.Sprint(f.dels) {
		t.Fatalf("dels = %v", s.Dels())
	}
}

func TestEmptySegment(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.seg")
	f := fixture{dims: 4}
	f.write(t, path)
	s, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if s.NumDocs() != 0 || s.NumTerms() != 0 || s.NumFrags() != 0 || s.NumWords() != 0 || len(s.Dels()) != 0 {
		t.Fatal("empty segment should have nothing")
	}
	if _, ok := s.FindDoc(1); ok || s.FindTerm("x") != -1 || s.FindFrag("x") != -1 {
		t.Fatal("lookups on an empty segment must miss")
	}
	if err := s.Verify(); err != nil {
		t.Fatal(err)
	}
}

func TestDetectsTruncationAndCorruption(t *testing.T) {
	f := makeFixture(30, 8)
	dir := t.TempDir()
	path := filepath.Join(dir, "a.seg")
	f.write(t, path)
	orig, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	// Truncation at any point (a crash mid-write) must never open as valid.
	for _, cut := range []int{0, 10, headerSize, len(orig) / 2, len(orig) - 1, len(orig) - footerSize} {
		p := filepath.Join(dir, fmt.Sprintf("cut%d.seg", cut))
		if cut == 0 {
			os.WriteFile(p, nil, 0o644)
		} else {
			os.WriteFile(p, orig[:cut], 0o644)
		}
		if s, err := Open(p); err == nil {
			s.Close()
			t.Fatalf("truncated to %d bytes opened successfully", cut)
		}
	}

	// A flipped bit in the header/directory is caught at Open; one inside a
	// data section is caught by Verify.
	flip := func(off int) string {
		b := append([]byte(nil), orig...)
		b[off] ^= 0x40
		p := filepath.Join(dir, fmt.Sprintf("flip%d.seg", off))
		os.WriteFile(p, b, 0o644)
		return p
	}
	if s, err := Open(flip(20)); err == nil {
		s.Close()
		t.Fatal("corrupt header opened")
	} else if !errors.Is(err, ErrCorrupt) {
		t.Fatalf("header corruption error = %v, want ErrCorrupt", err)
	}

	s, err := Open(flip(len(orig) / 2))
	if err != nil {
		t.Fatalf("data-section corruption should still Open (checked lazily): %v", err)
	}
	defer s.Close()
	if err := s.Verify(); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("Verify after data corruption = %v, want ErrCorrupt", err)
	}
}

func TestRejectsUnorderedDocs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.seg")
	w, err := Create(path, 2)
	if err != nil {
		t.Fatal(err)
	}
	docs := []DocIn{{ID: 5}, {ID: 5}}
	w.WriteDocs(2, func(i int) DocIn { return docs[i] })
	if err := w.Finish(nil); err == nil {
		t.Fatal("expected an error for duplicate/unordered doc IDs")
	}
	if _, err := os.Stat(path); err == nil {
		t.Fatal("a failed write must not leave a file behind")
	}
}

func BenchmarkOpenLarge(b *testing.B) {
	f := makeFixture(20000, 64)
	path := filepath.Join(b.TempDir(), "big.seg")
	f.write(b, path)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, err := Open(path)
		if err != nil {
			b.Fatal(err)
		}
		s.Close()
	}
}
