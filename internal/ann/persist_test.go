package ann

import (
	"bytes"
	"sync"
	"testing"
)

// roundTrip writes c.idx and reads it back, binding every vector with tag 1
// except those in skip (unbound: as if deleted since the file was written).
func roundTrip(t *testing.T, c *corpus, tagOf func(id uint64) (uint64, bool), skip map[uint64]bool) (*Index, []byte) {
	t.Helper()
	var buf bytes.Buffer
	if _, err := c.idx.WriteTo(&buf, WriteOptions{Chunk: 97, Tag: tagOf, Identity: []byte("id-1")}); err != nil {
		t.Fatal(err)
	}
	raw := append([]byte(nil), buf.Bytes()...)
	g, ident, err := Read(raw)
	if err != nil {
		t.Fatal(err)
	}
	if string(ident) != "id-1" {
		t.Fatalf("identity %q", ident)
	}
	for id, v := range c.vecs {
		if !skip[id] {
			g.SetVec(id, v)
		}
	}
	g.Finish()
	return g, raw
}

func always(uint64) (uint64, bool) { return 1, true }

// A graph that went through a file must answer like the original: same recall
// against brute force, and near-identical result lists.
func TestPersist_RoundTripKeepsRecall(t *testing.T) {
	c := buildCorpus(6000, 64, 1)
	g, raw := roundTrip(t, c, always, nil)
	t.Logf("6000 nodes -> %d bytes (%.0f B/node)", len(raw), float64(len(raw))/6000)
	if g.Len() != c.idx.Len() {
		t.Fatalf("live nodes %d, want %d", g.Len(), c.idx.Len())
	}
	qs := queries(c, 150, 9)
	var sumOrig, sumLoaded float64
	same := 0
	for _, q := range qs {
		want := c.brute(q, 10, nil, nil)
		o, l := c.idx.Search(q, 10, 64, nil), g.Search(q, 10, 64, nil)
		sumOrig += recall(o, want)
		sumLoaded += recall(l, want)
		if len(o) == len(l) {
			eq := true
			for i := range o {
				if o[i].ID != l[i].ID {
					eq = false
				}
			}
			if eq {
				same++
			}
		}
	}
	ro, rl := sumOrig/float64(len(qs)), sumLoaded/float64(len(qs))
	t.Logf("recall@10 original %.4f, restored %.4f; identical result lists for %d/%d queries", ro, rl, same, len(qs))
	if rl < 0.95 || rl < ro-0.01 {
		t.Fatalf("restored recall %.3f (original %.3f)", rl, ro)
	}
	if same < len(qs)*9/10 {
		t.Fatalf("only %d/%d queries return the same list after a round trip", same, len(qs))
	}
}

// Documents deleted before writing are dropped (with their links); documents
// deleted after writing are tombstoned at load; neither is ever returned.
func TestPersist_DeletedNodesAreNeverReturned(t *testing.T) {
	c := buildCorpus(3000, 32, 4)
	deletedBefore := map[uint64]bool{}
	for id := uint64(1); id <= 3000; id += 7 {
		c.idx.Delete(id)
		deletedBefore[id] = true
	}
	deletedAfter := map[uint64]bool{}
	for id := uint64(2); id <= 3000; id += 11 {
		if !deletedBefore[id] {
			deletedAfter[id] = true
		}
	}
	g, _ := roundTrip(t, c, always, deletedAfter)
	wantLive := 3000 - len(deletedBefore) - len(deletedAfter)
	if g.Len() != wantLive {
		t.Fatalf("Len = %d, want %d", g.Len(), wantLive)
	}
	for _, q := range queries(c, 60, 5) {
		for _, h := range g.Search(q, 10, 64, nil) {
			if deletedBefore[h.ID] || deletedAfter[h.ID] {
				t.Fatalf("deleted document %d returned", h.ID)
			}
		}
	}
	// Recall against brute force over the survivors stays high.
	dead := map[uint64]bool{}
	for id := range deletedBefore {
		dead[id] = true
	}
	for id := range deletedAfter {
		dead[id] = true
	}
	var sum float64
	qs := queries(c, 100, 6)
	for _, q := range qs {
		sum += recall(g.Search(q, 10, 64, nil), c.brute(q, 10, nil, dead))
	}
	if avg := sum / float64(len(qs)); avg < 0.9 {
		t.Fatalf("recall after dropping deleted nodes = %.3f", avg)
	}
}

// Nodes the tag callback rejects (vector not in a segment yet) are left out.
func TestPersist_RejectedNodesAreLeftOut(t *testing.T) {
	c := buildCorpus(500, 32, 2)
	reject := func(id uint64) (uint64, bool) { return 1, id%2 == 0 }
	var buf bytes.Buffer
	n, err := c.idx.WriteTo(&buf, WriteOptions{Tag: reject})
	if err != nil {
		t.Fatal(err)
	}
	if n != 250 {
		t.Fatalf("wrote %d nodes, want 250", n)
	}
	g, _, err := Read(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := g.NodeTag(3); ok {
		t.Fatal("a rejected node is present in the file")
	}
	if _, ok := g.NodeTag(4); !ok {
		t.Fatal("an accepted node is missing")
	}
}

// New documents can be inserted into a restored graph and are found.
func TestPersist_InsertAfterRestore(t *testing.T) {
	c := buildCorpus(2000, 32, 8)
	g, _ := roundTrip(t, c, always, nil)
	extra := buildCorpus(200, 32, 99)
	newVecs := map[uint64][]uint16{}
	for id, v := range extra.f32 {
		nid := id + 100000
		newVecs[nid] = extra.vecs[id]
		g.Insert(nid, v, extra.vecs[id])
	}
	found := 0
	for nid, v16 := range newVecs {
		q := make([]float32, 32)
		for i := range q {
			q[i] = extra.f32[nid-100000][i]
		}
		_ = v16
		for _, h := range g.Search(q, 3, 64, nil) {
			if h.ID == nid {
				found++
				break
			}
		}
	}
	if found < 190 {
		t.Fatalf("only %d/200 documents inserted after a restore are found by their own vector", found)
	}
}

// Damage must be detected, not half-loaded.
func TestPersist_CorruptAndTruncatedFilesAreRejected(t *testing.T) {
	c := buildCorpus(300, 16, 3)
	var buf bytes.Buffer
	if _, err := c.idx.WriteTo(&buf, WriteOptions{Tag: always}); err != nil {
		t.Fatal(err)
	}
	raw := buf.Bytes()
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{"empty", nil},
		{"truncated", raw[:len(raw)/2]},
		{"flipped byte", func() []byte { b := append([]byte(nil), raw...); b[len(b)/2] ^= 0x55; return b }()},
		{"bad magic", func() []byte { b := append([]byte(nil), raw...); b[0] = 'X'; return b }()},
	} {
		if _, _, err := Read(tc.data); err == nil {
			t.Errorf("%s: Read accepted a damaged file", tc.name)
		}
	}
}

// Writing while another goroutine mutates the graph must be safe when the lock
// callback is honoured, and the file must stay readable.
func TestPersist_WriteWhileMutatingWithLock(t *testing.T) {
	c := buildCorpus(3000, 32, 12)
	var mu lockedGraph
	done := make(chan struct{})
	go func() {
		defer close(done)
		extra := buildCorpus(400, 32, 77)
		for id, v := range extra.f32 {
			mu.Lock()
			c.idx.Insert(id+500000, v, extra.vecs[id])
			mu.Unlock()
		}
	}()
	var buf bytes.Buffer
	_, err := c.idx.WriteTo(&buf, WriteOptions{
		Chunk: 64,
		Lock:  func() func() { mu.RLock(); return mu.RUnlock },
		Tag:   always,
	})
	<-done
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := Read(buf.Bytes()); err != nil {
		t.Fatalf("file written during mutation is unreadable: %v", err)
	}
}

// lockedGraph is the engine's RWMutex stand-in for the test above.
type lockedGraph struct{ sync.RWMutex }
