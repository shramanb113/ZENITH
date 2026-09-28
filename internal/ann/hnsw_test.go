package ann

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/x448/float16"
)

func unit(r *rand.Rand, center []float32, noise float64) []float32 {
	v := make([]float32, len(center))
	var ss float64
	for i := range v {
		v[i] = center[i] + float32(r.NormFloat64()*noise)
		ss += float64(v[i]) * float64(v[i])
	}
	n := float32(math.Sqrt(ss))
	for i := range v {
		v[i] /= n
	}
	return v
}

func toF16(v []float32) []uint16 {
	out := make([]uint16, len(v))
	for i, x := range v {
		out[i] = float16.Fromfloat32(x).Bits()
	}
	return out
}

type corpus struct {
	vecs map[uint64][]uint16
	f32  map[uint64][]float32
	idx  *Index
}

func buildCorpus(n, dim int, seed int64) *corpus {
	r := rand.New(rand.NewSource(seed))
	centers := make([][]float32, 40)
	for i := range centers {
		c := make([]float32, dim)
		for j := range c {
			c[j] = float32(r.NormFloat64())
		}
		centers[i] = c
	}
	c := &corpus{vecs: map[uint64][]uint16{}, f32: map[uint64][]float32{}}
	c.idx = New(16, 100)
	for i := 0; i < n; i++ {
		id := uint64(i + 1)
		v := unit(r, centers[r.Intn(len(centers))], 0.35)
		c.f32[id] = v
		c.vecs[id] = toF16(v)
		c.idx.Insert(id, v, c.vecs[id])
	}
	return c
}

func (c *corpus) brute(q []float32, k int, allow func(uint64) bool, dead map[uint64]bool) []uint64 {
	type s struct {
		id  uint64
		sim float64
	}
	var all []s
	for id, v := range c.vecs {
		if dead[id] || (allow != nil && !allow(id)) {
			continue
		}
		all = append(all, s{id, DotF32F16(q, v)})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].sim > all[j].sim })
	if len(all) > k {
		all = all[:k]
	}
	out := make([]uint64, len(all))
	for i, a := range all {
		out[i] = a.id
	}
	return out
}

func recall(got []Hit, want []uint64) float64 {
	if len(want) == 0 {
		return 1
	}
	w := map[uint64]bool{}
	for _, id := range want {
		w[id] = true
	}
	hit := 0
	for _, h := range got {
		if w[h.ID] {
			hit++
		}
	}
	return float64(hit) / float64(len(want))
}

func queries(c *corpus, n int, seed int64) [][]float32 {
	r := rand.New(rand.NewSource(seed))
	ids := make([]uint64, 0, len(c.f32))
	for id := range c.f32 {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	out := make([][]float32, n)
	for i := range out {
		out[i] = unit(r, c.f32[ids[r.Intn(len(ids))]], 0.15)
	}
	return out
}

func TestHNSW_RecallAtTen(t *testing.T) {
	c := buildCorpus(6000, 64, 1)
	var sum float64
	qs := queries(c, 150, 9)
	for _, q := range qs {
		sum += recall(c.idx.Search(q, 10, 64, nil), c.brute(q, 10, nil, nil))
	}
	avg := sum / float64(len(qs))
	t.Logf("recall@10 (ef=64, 6k vecs, 64-dim): %.4f", avg)
	if avg < 0.95 {
		t.Fatalf("recall@10 = %.3f, want >= 0.95", avg)
	}
}

func TestHNSW_FilteredSearchOnlyReturnsAllowed(t *testing.T) {
	c := buildCorpus(4000, 32, 2)
	allow := func(id uint64) bool { return id%4 == 0 }
	var sum float64
	qs := queries(c, 100, 5)
	for _, q := range qs {
		got := c.idx.Search(q, 10, 128, allow)
		for _, h := range got {
			if !allow(h.ID) {
				t.Fatalf("returned disallowed id %d", h.ID)
			}
		}
		sum += recall(got, c.brute(q, 10, allow, nil))
	}
	if avg := sum / float64(len(qs)); avg < 0.9 {
		t.Fatalf("filtered recall@10 = %.3f, want >= 0.9", avg)
	}
}

func TestHNSW_DeletedNeverReturnedAndReinsertReplaces(t *testing.T) {
	c := buildCorpus(2000, 32, 3)
	dead := map[uint64]bool{}
	for id := uint64(1); id <= 2000; id += 3 {
		c.idx.Delete(id)
		dead[id] = true
	}
	if c.idx.Len() != 2000-len(dead) {
		t.Fatalf("Len = %d, want %d", c.idx.Len(), 2000-len(dead))
	}
	var sum float64
	qs := queries(c, 80, 6)
	for _, q := range qs {
		got := c.idx.Search(q, 10, 96, nil)
		for _, h := range got {
			if dead[h.ID] {
				t.Fatalf("returned deleted id %d", h.ID)
			}
		}
		sum += recall(got, c.brute(q, 10, nil, dead))
	}
	if avg := sum / float64(len(qs)); avg < 0.93 {
		t.Fatalf("recall after deletes = %.3f, want >= 0.93", avg)
	}

	// Re-inserting a deleted id with a new vector must make it findable again.
	r := rand.New(rand.NewSource(11))
	nv := unit(r, c.f32[2], 0.05)
	c.f32[1], c.vecs[1] = nv, toF16(nv)
	c.idx.Insert(1, nv, c.vecs[1])
	top := c.idx.Search(nv, 1, 64, nil)
	if len(top) != 1 || top[0].ID != 1 {
		t.Fatalf("re-inserted id not returned as its own nearest neighbour: %+v", top)
	}
}

func TestHNSW_EmptyAndTiny(t *testing.T) {
	idx := New(16, 100)
	if got := idx.Search([]float32{1, 0}, 5, 10, nil); got != nil {
		t.Fatalf("empty index returned %v", got)
	}
	idx = New(16, 100)
	idx.Insert(1, []float32{1, 0}, toF16([]float32{1, 0}))
	if got := idx.Search([]float32{1, 0}, 5, 10, nil); len(got) != 1 || got[0].ID != 1 {
		t.Fatalf("single-node search = %v", got)
	}
}
