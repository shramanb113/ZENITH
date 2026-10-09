package localembedder

import (
	"reflect"
	"testing"
)

// testCLIPVocabJSON/testCLIPMergesTxt hold a tiny fixture vocabulary built
// from real CLIP vocab.json/merges.txt entries (not synthetic ids) covering
// exactly the pieces the test cases below need, so the test doesn't require
// network access or the full 862KB vocab.json file. The ids match the real
// registry's vocab.json exactly for every token used here.
var (
	testCLIPVocabJSON = []byte(`{
		"a</w>":320,"photo</w>":1125,"of</w>":539,"tree</w>":2677,"!</w>":256,
		"it</w>":585,"'s</w>":568,"cat</w>":2368,"toy</w>":5988,",</w>":267,"running</w>":2761,"fast</w>":1953,
		"2</w>":273,"dogs</w>":3255,"and</w>":537,"3</w>":274,"cats</w>":3989,"hello</w>":3306,
		"<|startoftext|>":49406,"<|endoftext|>":49407
	}`)
	testCLIPMergesTxt = []byte("#version: 0.2\n" +
		"i n\n" +
		"a n\n" +
		"r e\n" +
		"in g</w>\n" +
		"a t\n" +
		"t o</w>\n" +
		"a t</w>\n" +
		"an d</w>\n" +
		"o f</w>\n" +
		"e l\n" +
		"s t</w>\n" +
		"' s</w>\n" +
		"u n\n" +
		"t o\n" +
		"i t</w>\n" +
		"h o\n" +
		"d o\n" +
		"n ing</w>\n" +
		"f a\n" +
		"p ho\n" +
		"el l\n" +
		"t re\n" +
		"at s</w>\n" +
		"pho to</w>\n" +
		"g s</w>\n" +
		"r un\n" +
		"fa st</w>\n" +
		"c at</w>\n" +
		"ell o</w>\n" +
		"tre e</w>\n" +
		"run ning</w>\n" +
		"do gs</w>\n" +
		"h ello</w>\n" +
		"c ats</w>\n" +
		"to y</w>\n")
)

func mustNewTestCLIPTokenizer(t *testing.T) *clipTokenizer {
	t.Helper()
	tok, err := newCLIPTokenizer(testCLIPVocabJSON, testCLIPMergesTxt)
	if err != nil {
		t.Fatalf("newCLIPTokenizer: %v", err)
	}
	return tok
}

func TestCLIPTokenizer_RealFixtures(t *testing.T) {
	tok := mustNewTestCLIPTokenizer(t)
	cases := []struct {
		text string
		want []int64
	}{
		{"a photo of a tree", []int64{49406, 320, 1125, 539, 320, 2677, 49407}},
		{"A Photo of a TREE!", []int64{49406, 320, 1125, 539, 320, 2677, 256, 49407}},
		{"it's a cat's toy, running   fast", []int64{49406, 585, 568, 320, 2368, 568, 5988, 267, 2761, 1953, 49407}},
		{"2 dogs and 3 cats", []int64{49406, 273, 3255, 537, 274, 3989, 49407}},
		{"", []int64{49406, 49407}},
		{"hello", []int64{49406, 3306, 49407}},
	}
	for _, c := range cases {
		got := tok.encodeIDs(c.text)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("encodeIDs(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

func TestCLIPTokenizer_EncodePadsToContextLength(t *testing.T) {
	tok := mustNewTestCLIPTokenizer(t)
	ids := tok.encode("hello", 8)
	want := []int64{49406, 3306, 49407, 49407, 49407, 49407, 49407, 49407}
	if !reflect.DeepEqual(ids, want) {
		t.Errorf("encode(\"hello\", 8) = %v, want %v", ids, want)
	}
}

func TestCLIPTokenizer_EncodeTruncatesLongText(t *testing.T) {
	tok := mustNewTestCLIPTokenizer(t)
	// "a" repeated 20 times pre-tokenizes to 20 content pieces; context
	// length 5 leaves room for bos + 3 content + eos.
	long := ""
	for i := 0; i < 20; i++ {
		long += "a "
	}
	ids := tok.encode(long, 5)
	if len(ids) != 5 {
		t.Fatalf("encode truncated length = %d, want 5", len(ids))
	}
	if ids[0] != 49406 {
		t.Errorf("ids[0] = %d, want bos 49406", ids[0])
	}
	if ids[4] != 49407 {
		t.Errorf("ids[4] (last slot) = %d, want eos 49407 — truncation must never drop eos", ids[4])
	}
}

func TestCLIPTokenizer_EmptyAndWhitespaceOnly(t *testing.T) {
	tok := mustNewTestCLIPTokenizer(t)
	for _, text := range []string{"", "   ", "\t\n "} {
		got := tok.encodeIDs(text)
		want := []int64{49406, 49407}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("encodeIDs(%q) = %v, want %v", text, got, want)
		}
	}
}
