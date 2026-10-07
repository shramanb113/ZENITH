package collections

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/pkg/zenith"
)

// fakeClock lets tests advance time deterministically for idle-close tests.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newFakeClock() *fakeClock { return &fakeClock{t: time.Now()} }

func (c *fakeClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

func newTestManager(t *testing.T, mutate func(*Config)) (*Manager, string) {
	dir := t.TempDir()
	cfg := Config{
		Root:     dir,
		Embedder: embedding.NewDeterministicEmbedder(384),
	}
	if mutate != nil {
		mutate(&cfg)
	}
	m, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = m.CloseAll() })
	return m, dir
}

func TestManager_CreateUpsertSearch(t *testing.T) {
	m, _ := newTestManager(t, nil)
	ctx := context.Background()

	if _, _, err := m.Create("acme", CreateOptions{}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	res, err := m.Upsert(ctx, "acme", map[string]string{
		"d1": "the quick brown fox",
		"d2": "lazy dog sleeps",
		"d3": "jumping over obstacles",
	}, nil)
	if err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if res.Upserted != 3 || res.New != 3 || res.DocCount != 3 {
		t.Fatalf("unexpected UpsertResult: %+v", res)
	}

	hits, err := m.Search(ctx, "acme", "quick fox")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) == 0 || hits[0].ID != "d1" {
		t.Fatalf("expected d1 top hit, got %+v", hits)
	}

	info, err := m.Stat("acme")
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if info.DocCount != 3 || !info.DocCountExact {
		t.Fatalf("unexpected Stat: %+v", info)
	}
}

func TestManager_ReopenAfterRestart(t *testing.T) {
	dir := t.TempDir()
	emb := embedding.NewDeterministicEmbedder(384)
	ctx := context.Background()

	m1, err := New(Config{Root: dir, Embedder: emb})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, _, err := m1.Create("c1", CreateOptions{}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	docs := map[string]string{}
	for i := 0; i < 5; i++ {
		docs[fmt.Sprintf("d%d", i)] = fmt.Sprintf("document body number %d", i)
	}
	if _, err := m1.Upsert(ctx, "c1", docs, nil); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := m1.CloseAll(); err != nil {
		t.Fatalf("CloseAll: %v", err)
	}

	m2, err := New(Config{Root: dir, Embedder: emb})
	if err != nil {
		t.Fatalf("New (reopen): %v", err)
	}
	t.Cleanup(func() { _ = m2.CloseAll() })

	list := m2.List()
	if len(list) != 1 || list[0].ID != "c1" {
		t.Fatalf("expected one collection c1, got %+v", list)
	}
	if list[0].DocCount != 5 || list[0].Open {
		t.Fatalf("expected DocCount=5, Open=false before first use, got %+v", list[0])
	}
	for id, text := range docs {
		got, err := m2.GetDoc(ctx, "c1", id)
		if err != nil || got != text {
			t.Fatalf("GetDoc(%s) = %q, %v; want %q, nil", id, got, err, text)
		}
	}
	if _, err := m2.Search(ctx, "c1", "document"); err != nil {
		t.Fatalf("Search after reopen: %v", err)
	}
}

func TestManager_Delete(t *testing.T) {
	m, dir := newTestManager(t, nil)
	ctx := context.Background()

	_, key1, err := m.Create("c1", CreateOptions{})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := m.Delete("c1"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "c1")); !os.IsNotExist(err) {
		t.Fatalf("expected c1 dir gone, stat err = %v", err)
	}
	if _, err := m.GetDoc(ctx, "c1", "x"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetDoc after delete: %v", err)
	}

	_, key2, err := m.Create("c1", CreateOptions{})
	if err != nil {
		t.Fatalf("re-Create: %v", err)
	}
	if key1 == key2 {
		t.Fatalf("re-created collection got the same key")
	}
	if m.CheckKey("c1", key1) {
		t.Fatalf("old key still matches after re-create")
	}
	if !m.CheckKey("c1", key2) {
		t.Fatalf("new key does not match")
	}
}

func TestManager_Quota(t *testing.T) {
	m, _ := newTestManager(t, nil)
	ctx := context.Background()

	if _, _, err := m.Create("c1", CreateOptions{MaxDocs: 3}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := m.Upsert(ctx, "c1", map[string]string{"a": "aa", "b": "bb"}, nil); err != nil {
		t.Fatalf("Upsert a,b: %v", err)
	}
	if _, err := m.Upsert(ctx, "c1", map[string]string{"c": "cc", "d": "dd"}, nil); !errors.Is(err, ErrQuota) {
		t.Fatalf("Upsert c,d: want ErrQuota, got %v", err)
	}
	info, err := m.Stat("c1")
	if err != nil || info.DocCount != 2 {
		t.Fatalf("DocCount after failed quota upsert: %+v, %v", info, err)
	}
	if _, err := m.GetDoc(ctx, "c1", "c"); !errors.Is(err, ErrDocNotFound) {
		t.Fatalf("GetDoc(c): want ErrDocNotFound, got %v", err)
	}
	if _, err := m.Upsert(ctx, "c1", map[string]string{"a": "aa2", "b": "bb2"}, nil); err != nil {
		t.Fatalf("Upsert replace a,b: %v", err)
	}
	if _, err := m.Upsert(ctx, "c1", map[string]string{"c": "cc"}, nil); err != nil {
		t.Fatalf("Upsert c: %v", err)
	}
	info, err = m.Stat("c1")
	if err != nil || info.DocCount != 3 {
		t.Fatalf("final DocCount: %+v, %v", info, err)
	}
}

func TestManager_InvalidIDs(t *testing.T) {
	invalid := []string{"", "CON", "con", "nul", "com1", "Upper", "../x", ".trash", stringsRepeat("a", 64), "a/b"}
	for _, id := range invalid {
		if ValidID(id) {
			t.Errorf("ValidID(%q) = true, want false", id)
		}
	}
	valid := []string{"a", stringsRepeat("a", 63)}
	for _, id := range valid {
		if !ValidID(id) {
			t.Errorf("ValidID(%q) = false, want true", id)
		}
	}
}

func stringsRepeat(s string, n int) string {
	out := make([]byte, 0, n*len(s))
	for i := 0; i < n; i++ {
		out = append(out, s...)
	}
	return string(out)
}

func TestManager_InvalidOptions(t *testing.T) {
	m, _ := newTestManager(t, nil)
	cases := []CreateOptions{
		{MaxDocs: 10_000_001},
		{MaxBodyBytes: 1023},
		{MaxBodyBytes: 64<<20 + 1},
	}
	for i, o := range cases {
		id := fmt.Sprintf("c%d", i)
		if _, _, err := m.Create(id, o); err == nil {
			t.Errorf("Create(%+v): want error, got nil", o)
		}
		if _, err := os.Stat(filepath.Join(m.cfg.Root, id)); !os.IsNotExist(err) {
			t.Errorf("Create(%+v): dir created despite invalid options", o)
		}
	}
}

func TestManager_DuplicateCreate(t *testing.T) {
	m, _ := newTestManager(t, nil)
	if _, _, err := m.Create("c1", CreateOptions{}); err != nil {
		t.Fatalf("first Create: %v", err)
	}
	if _, _, err := m.Create("c1", CreateOptions{}); !errors.Is(err, ErrExists) {
		t.Fatalf("second Create: want ErrExists, got %v", err)
	}
}

func TestManager_CollectionLimit(t *testing.T) {
	m, _ := newTestManager(t, func(c *Config) { c.MaxCollections = 2 })
	if _, _, err := m.Create("c1", CreateOptions{}); err != nil {
		t.Fatalf("Create c1: %v", err)
	}
	if _, _, err := m.Create("c2", CreateOptions{}); err != nil {
		t.Fatalf("Create c2: %v", err)
	}
	if _, _, err := m.Create("c3", CreateOptions{}); !errors.Is(err, ErrTooManyCollections) {
		t.Fatalf("Create c3: want ErrTooManyCollections, got %v", err)
	}
}

func TestManager_IdleCloseThenReopen(t *testing.T) {
	clock := newFakeClock()
	m, _ := newTestManager(t, func(c *Config) { c.Now = clock.now; c.IdleClose = time.Minute })
	ctx := context.Background()

	if _, _, err := m.Create("c1", CreateOptions{}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := m.Upsert(ctx, "c1", map[string]string{"a": "aa"}, nil); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	clock.advance(2 * time.Minute)
	m.Sweep()

	info, err := m.Stat("c1")
	if err != nil || info.Open {
		t.Fatalf("expected closed after sweep: %+v, %v", info, err)
	}

	if _, err := m.Search(ctx, "c1", "aa"); err != nil {
		t.Fatalf("Search after idle close: %v", err)
	}
	info, err = m.Stat("c1")
	if err != nil || !info.Open {
		t.Fatalf("expected open after Search: %+v, %v", info, err)
	}
}

func TestManager_LRUCap(t *testing.T) {
	m, _ := newTestManager(t, func(c *Config) { c.MaxOpen = 2 })
	ctx := context.Background()
	for _, id := range []string{"c1", "c2", "c3"} {
		if _, _, err := m.Create(id, CreateOptions{}); err != nil {
			t.Fatalf("Create %s: %v", id, err)
		}
	}
	if _, err := m.Search(ctx, "c1", "x"); err != nil {
		t.Fatalf("search c1: %v", err)
	}
	if _, err := m.Search(ctx, "c2", "x"); err != nil {
		t.Fatalf("search c2: %v", err)
	}
	if _, err := m.Search(ctx, "c3", "x"); err != nil {
		t.Fatalf("search c3: %v", err)
	}

	deadline := time.Now().Add(2 * time.Second)
	for {
		_, open := m.Counts()
		if open <= 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("open count never dropped to MaxOpen, still %d", open)
		}
		time.Sleep(10 * time.Millisecond)
	}

	info, err := m.Stat("c1")
	if err != nil || info.Open {
		t.Fatalf("expected c1 closed (LRU victim), got %+v, %v", info, err)
	}
	if _, _, err := m.Create("c1dup", CreateOptions{}); err != nil {
		t.Fatalf("manager still usable: %v", err)
	}
}

func TestManager_EmbedderMismatch(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	m1, err := New(Config{Root: dir, Embedder: embedding.NewDeterministicEmbedder(384)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, _, err := m1.Create("c1", CreateOptions{}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := m1.Upsert(ctx, "c1", map[string]string{"a": "aa"}, nil); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := m1.CloseAll(); err != nil {
		t.Fatalf("CloseAll: %v", err)
	}

	m2, err := New(Config{Root: dir, Embedder: nil})
	if err != nil {
		t.Fatalf("New (mismatched embedder): %v", err)
	}
	t.Cleanup(func() { _ = m2.CloseAll() })

	if _, err := m2.Search(ctx, "c1", "aa"); !errors.Is(err, zenith.ErrEmbedderMismatch) {
		t.Fatalf("Search: want ErrEmbedderMismatch, got %v", err)
	}
	list := m2.List()
	if len(list) != 1 || list[0].ID != "c1" {
		t.Fatalf("List should still show c1: %+v", list)
	}
}

func TestManager_TrashRecovery(t *testing.T) {
	dir := t.TempDir()
	trashDir := filepath.Join(dir, ".trash-x-1")
	if err := os.MkdirAll(trashDir, 0o700); err != nil {
		t.Fatalf("mkdir trash: %v", err)
	}
	if err := os.WriteFile(filepath.Join(trashDir, "file"), []byte("x"), 0o600); err != nil {
		t.Fatalf("write trash file: %v", err)
	}

	m, err := New(Config{Root: dir, Embedder: embedding.NewDeterministicEmbedder(384)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = m.CloseAll() })

	if _, err := os.Stat(trashDir); !os.IsNotExist(err) {
		t.Fatalf("expected trash dir gone, stat err = %v", err)
	}
}

func TestManager_SamePIDStaleLock(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	emb := embedding.NewDeterministicEmbedder(384)

	m1, err := New(Config{Root: dir, Embedder: emb})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, _, err := m1.Create("c1", CreateOptions{}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := m1.Upsert(ctx, "c1", map[string]string{"a": "aa"}, nil); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := m1.CloseAll(); err != nil {
		t.Fatalf("CloseAll: %v", err)
	}

	lockPath := filepath.Join(dir, "c1", "index.db.lock")
	if err := os.WriteFile(lockPath, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o600); err != nil {
		t.Fatalf("write fake lock: %v", err)
	}

	m2, err := New(Config{Root: dir, Embedder: emb})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = m2.CloseAll() })

	if _, err := m2.Search(ctx, "c1", "aa"); err != nil {
		t.Fatalf("Search: want success (stale lock removed), got %v", err)
	}
}

func TestManager_RootInUse(t *testing.T) {
	dir := t.TempDir()
	m1, err := New(Config{Root: dir, Embedder: embedding.NewDeterministicEmbedder(384)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := New(Config{Root: dir, Embedder: embedding.NewDeterministicEmbedder(384)}); !errors.Is(err, ErrRootInUse) {
		t.Fatalf("second New: want ErrRootInUse, got %v", err)
	}
	if err := m1.CloseAll(); err != nil {
		t.Fatalf("CloseAll: %v", err)
	}
	m2, err := New(Config{Root: dir, Embedder: embedding.NewDeterministicEmbedder(384)})
	if err != nil {
		t.Fatalf("New after CloseAll: %v", err)
	}
	_ = m2.CloseAll()
}

func TestManager_Shutdown(t *testing.T) {
	m, _ := newTestManager(t, nil)
	ctx := context.Background()
	if _, _, err := m.Create("c1", CreateOptions{}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := m.CloseAll(); err != nil {
		t.Fatalf("CloseAll: %v", err)
	}
	if _, err := m.Search(ctx, "c1", "x"); !errors.Is(err, ErrShuttingDown) {
		t.Fatalf("Search after shutdown: want ErrShuttingDown, got %v", err)
	}
}

func TestManager_DirtyMeta(t *testing.T) {
	dir := t.TempDir()
	m, err := New(Config{Root: dir, Embedder: embedding.NewDeterministicEmbedder(384)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, _, err := m.Create("c1", CreateOptions{}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := m.CloseAll(); err != nil {
		t.Fatalf("CloseAll: %v", err)
	}

	cdir := filepath.Join(dir, "c1")
	meta, err := readMeta(cdir)
	if err != nil {
		t.Fatalf("readMeta: %v", err)
	}
	meta.Dirty = true
	meta.DocCount = 7
	if err := writeMeta(cdir, meta); err != nil {
		t.Fatalf("writeMeta: %v", err)
	}

	m2, err := New(Config{Root: dir, Embedder: embedding.NewDeterministicEmbedder(384)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = m2.CloseAll() })

	info, err := m2.Stat("c1")
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if info.DocCount != 7 || info.DocCountExact {
		t.Fatalf("expected DocCount=7, DocCountExact=false, got %+v", info)
	}
}

func TestManager_ConcurrentAccess(t *testing.T) {
	clock := newFakeClock()
	m, _ := newTestManager(t, func(c *Config) { c.MaxOpen = 1; c.Now = clock.now; c.IdleClose = 50 * time.Millisecond })
	ctx := context.Background()
	if _, _, err := m.Create("c1", CreateOptions{}); err != nil {
		t.Fatalf("Create c1: %v", err)
	}
	if _, _, err := m.Create("c2", CreateOptions{}); err != nil {
		t.Fatalf("Create c2: %v", err)
	}

	stop := make(chan struct{})
	var sweepWG sync.WaitGroup
	sweepWG.Add(1)
	go func() {
		defer sweepWG.Done()
		for {
			select {
			case <-stop:
				return
			default:
				clock.advance(100 * time.Millisecond)
				m.Sweep()
				time.Sleep(time.Millisecond)
			}
		}
	}()

	var wg sync.WaitGroup
	errCh := make(chan error, 64)
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				id := fmt.Sprintf("g%d-%d", g, i)
				if _, err := m.Upsert(ctx, "c1", map[string]string{id: "body " + id}, nil); err != nil {
					errCh <- fmt.Errorf("upsert %s: %w", id, err)
					return
				}
				if _, err := m.Search(ctx, "c1", "body"); err != nil {
					errCh <- fmt.Errorf("search after %s: %w", id, err)
					return
				}
			}
		}(g)
	}
	for g := 0; g < 2; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				if _, err := m.Search(ctx, "c2", "x"); err != nil {
					errCh <- fmt.Errorf("search c2: %w", err)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(stop)
	sweepWG.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}

	info, err := m.Stat("c1")
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if info.DocCount != 800 {
		t.Fatalf("final DocCount = %d, want 800", info.DocCount)
	}
	for g := 0; g < 16; g++ {
		for i := 0; i < 50; i++ {
			id := fmt.Sprintf("g%d-%d", g, i)
			if _, err := m.GetDoc(ctx, "c1", id); err != nil {
				t.Errorf("GetDoc(%s): %v", id, err)
			}
		}
	}
}

func TestManager_ConcurrentDeleteVsUse(t *testing.T) {
	m, _ := newTestManager(t, nil)
	ctx := context.Background()
	if _, _, err := m.Create("c1", CreateOptions{}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := m.Upsert(ctx, "c1", map[string]string{"a": "aa"}, nil); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				_, err := m.Search(ctx, "c1", "aa")
				if err != nil && !errors.Is(err, ErrNotFound) {
					t.Errorf("Search: unexpected error %v", err)
				}
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = m.Delete("c1")
	}()
	wg.Wait()
}
