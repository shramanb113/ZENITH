// Package collections manages persistent, multi-tenant ZENITH collections:
// each collection is its own durable *zenith.DB, lazily opened on first use
// and idle-closed or LRU-evicted to bound memory and file descriptors. It has
// no HTTP code; see internal/sidecar/collections_http.go for the wire layer.
package collections

import (
	"container/list"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/metrics"
	"github.com/shramanb113/ZENITH/pkg/zenith"
)

// registeredRoots prevents two Managers in this process from managing the
// same root directory at once (mirrors pkg/zenith's own inProcReg).
var registeredRoots sync.Map // map[string]struct{}, keyed by filepath.Abs(root)

// Config configures a Manager. Zero values take the defaults noted.
type Config struct {
	Root           string          // required; created with MkdirAll(0o700) if missing
	Embedder       zenith.Embedder // nil = BM25 only (zenith.WithBM25Only())
	MaxCollections int             // default 1000
	DefaultMaxDocs int             // default 1_000_000
	DefaultMaxBody int64           // default 8 << 20 (8 MiB)
	MaxOpen        int             // default 64
	IdleClose      time.Duration   // default 10 * time.Minute
	Now            func() time.Time
	Log            *slog.Logger
	// QueryCacheRedisAddr, when set, enables a shared Redis L2 query-result
	// cache tier across every collection this Manager opens. Each
	// collection automatically gets its own ID as the cache namespace (see
	// openOpts), so collections never see each other's cached results on
	// the shared Redis. "" (default) keeps every collection's cache
	// in-process (L1) only.
	QueryCacheRedisAddr string
	// QueryCacheSize overrides the L1 entry count every collection's
	// zenith.DB uses (default: zenith's own default, 1000). <= 0 (the Go
	// zero value included) leaves that default in place; there is
	// currently no way to force-disable every collection's query cache
	// through this field specifically (0 means "unconfigured" here, the
	// same convention every other numeric field on this Config already
	// uses) — open a direct zenith.DB with zenith.WithQueryCacheSize(0)
	// for that one collection if you need it.
	QueryCacheSize int
	// QueryCacheTTL overrides the L2 (Redis) entry TTL (default: 5m). Has
	// no effect unless QueryCacheRedisAddr is also set. <= 0 leaves the
	// default in place.
	QueryCacheTTL time.Duration
	// QueryCacheSemanticThreshold enables near-duplicate query-cache
	// matching for every collection above this cosine similarity (default:
	// 0, disabled). See zenith.WithQueryCacheSemanticThreshold.
	QueryCacheSemanticThreshold float64
	// ANNThresholdBandPct enables latency-adaptive ANN-vs-exact banding for
	// every collection (default: 0, disabled). See
	// zenith.WithANNThresholdBand.
	ANNThresholdBandPct float64
}

// CreateOptions customizes a single collection's quotas at Create time.
type CreateOptions struct {
	MaxDocs      int   // 0 = DefaultMaxDocs; else must be 1..10_000_000
	MaxBodyBytes int64 // 0 = DefaultMaxBody; else must be 1024..64<<20
}

// Info is a point-in-time snapshot of one collection's state.
type Info struct {
	ID            string    `json:"id"`
	Embedder      string    `json:"embedder"`
	CreatedAt     time.Time `json:"created_at"`
	LastUsed      time.Time `json:"last_used,omitempty"` // zero until first use since process start
	DocCount      int       `json:"doc_count"`
	DocCountExact bool      `json:"doc_count_exact"`
	MaxDocs       int       `json:"max_docs"`
	MaxBodyBytes  int64     `json:"max_body_bytes"`
	Open          bool      `json:"open"`
	DiskBytes     int64     `json:"disk_bytes,omitempty"` // Stat only
	WALBytes      int64     `json:"wal_bytes,omitempty"`  // Stat only
}

// UpsertResult reports the outcome of a Manager.Upsert call.
type UpsertResult struct{ Upserted, New, DocCount int }

// entry is one collection's bookkeeping. life guards open/close/delete
// against concurrent data-plane use; writeMu serialises writes to meta (both
// the in-memory copy and the on-disk file) and to the underlying db for
// Upsert/DeleteDoc. lastUsed and elem are Manager-level LRU bookkeeping and
// are only ever touched under Manager.mu.
type entry struct {
	id      string
	dir     string
	life    sync.RWMutex
	db      *zenith.DB // nil when closed; guarded by life
	deleted bool       // guarded by life
	writeMu sync.Mutex
	meta    Meta // guarded by writeMu

	lastUsed time.Time     // guarded by Manager.mu
	elem     *list.Element // guarded by Manager.mu; nil when not in openLRU
}

// Manager owns a directory of persistent collections, each a lazily opened
// *zenith.DB. Safe for concurrent use.
type Manager struct {
	cfg     Config
	embName string // EmbedderName(cfg.Embedder), computed once
	rootKey string // filepath.Abs(cfg.Root), for registeredRoots

	mu      sync.Mutex
	entries map[string]*entry
	openLRU *list.List // front = most recently used; values are *entry
	closed  bool
}

// EmbedderName identifies e for storage in Meta.Embedder. nil means
// BM25-only; an embedder that doesn't implement embedding.Named is recorded
// as "unknown" rather than checked for a mismatch on reopen.
func EmbedderName(e zenith.Embedder) string {
	if e == nil {
		return "none:bm25-only"
	}
	if n, ok := e.(embedding.Named); ok {
		return n.Name()
	}
	return "unknown"
}

// New opens (or creates) the collections root and scans it for existing
// collections. Opening a Manager on a root another live Manager in this
// process already manages returns ErrRootInUse.
func New(cfg Config) (*Manager, error) {
	if cfg.Root == "" {
		return nil, fmt.Errorf("collections: Config.Root is required")
	}
	if cfg.MaxCollections <= 0 {
		cfg.MaxCollections = 1000
	}
	if cfg.DefaultMaxDocs <= 0 {
		cfg.DefaultMaxDocs = 1_000_000
	}
	if cfg.DefaultMaxBody <= 0 {
		cfg.DefaultMaxBody = 8 << 20
	}
	if cfg.MaxOpen <= 0 {
		cfg.MaxOpen = 64
	}
	if cfg.IdleClose <= 0 {
		cfg.IdleClose = 10 * time.Minute
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.Log == nil {
		cfg.Log = slog.Default()
	}

	if err := os.MkdirAll(cfg.Root, 0o700); err != nil {
		return nil, fmt.Errorf("collections: root: %w", err)
	}
	absRoot, err := filepath.Abs(cfg.Root)
	if err != nil {
		return nil, fmt.Errorf("collections: root: %w", err)
	}
	if _, loaded := registeredRoots.LoadOrStore(absRoot, struct{}{}); loaded {
		return nil, ErrRootInUse
	}

	m := &Manager{
		cfg:     cfg,
		embName: EmbedderName(cfg.Embedder),
		rootKey: absRoot,
		entries: make(map[string]*entry),
		openLRU: list.New(),
	}
	if err := m.scan(); err != nil {
		registeredRoots.Delete(absRoot)
		return nil, err
	}
	return m, nil
}

// scan clears interrupted deletes, then loads metadata for every remaining
// subdirectory. Nothing is opened; a directory whose metadata can't be read
// is logged and skipped, never deleted.
func (m *Manager) scan() error {
	dirEntries, err := os.ReadDir(m.cfg.Root)
	if err != nil {
		return fmt.Errorf("collections: scan root: %w", err)
	}

	for _, d := range dirEntries {
		if d.IsDir() && strings.HasPrefix(d.Name(), ".trash-") {
			_ = os.RemoveAll(filepath.Join(m.cfg.Root, d.Name()))
		}
	}

	for _, d := range dirEntries {
		name := d.Name()
		if !d.IsDir() || strings.HasPrefix(name, ".") {
			continue
		}
		dir := filepath.Join(m.cfg.Root, name)
		meta, err := readMeta(dir)
		if err != nil {
			m.cfg.Log.Error("collections: skipping unreadable collection", "dir", name, "error", err)
			continue
		}

		// Stale-lock workaround: pkg/zenith's lock is only considered stale if
		// its PID is dead. In a container the server restarts as PID 1 every
		// time, so a lock left by a SIGKILLed previous instance would read "1",
		// which looks alive forever. Removing it here is safe: this root is
		// registered to this Manager alone (above) and nothing has been opened
		// yet, so there is no other owner of the lock to race with.
		lockPath := filepath.Join(dir, "index.db.lock")
		if data, err := os.ReadFile(lockPath); err == nil {
			if strings.TrimSpace(string(data)) == strconv.Itoa(os.Getpid()) {
				_ = os.Remove(lockPath)
			}
		}

		m.entries[name] = &entry{id: name, dir: dir, meta: meta}
		metrics.CollectionDocuments.WithLabelValues(name).Set(float64(meta.DocCount))
	}
	m.refreshCollectionsGauge()
	return nil
}

// refreshCollectionsGauge recomputes zenith_collections{state} from the
// manager's actual entry/open-LRU state, rather than incrementing/
// decrementing it at every call site (which would drift silently if any
// error path forgot to undo an increment). Called after every state change
// below, plus once per Sweep.
func (m *Manager) refreshCollectionsGauge() {
	total, open := m.Counts()
	metrics.Collections.WithLabelValues("open").Set(float64(open))
	metrics.Collections.WithLabelValues("closed").Set(float64(total - open))
}

// openOpts returns the zenith.Open options shared by every collection,
// specialized per collection id so a shared Redis (if configured) never lets
// one collection's cached results collide with another's.
func (m *Manager) openOpts(id string) []zenith.Option {
	opts := make([]zenith.Option, 0, 10)
	opts = append(opts, zenith.WithoutWordVectors(), zenith.WithLimit(100), zenith.WithQueryCacheNamespace(id),
		zenith.WithQueryCacheObserver(metrics.NewQueryCacheObserver()))
	if m.cfg.QueryCacheRedisAddr != "" {
		opts = append(opts, zenith.WithQueryCacheRedisAddr(m.cfg.QueryCacheRedisAddr))
	}
	if m.cfg.QueryCacheSize > 0 {
		opts = append(opts, zenith.WithQueryCacheSize(m.cfg.QueryCacheSize))
	}
	if m.cfg.QueryCacheTTL > 0 {
		opts = append(opts, zenith.WithQueryCacheTTL(m.cfg.QueryCacheTTL))
	}
	if m.cfg.QueryCacheSemanticThreshold > 0 {
		opts = append(opts, zenith.WithQueryCacheSemanticThreshold(m.cfg.QueryCacheSemanticThreshold))
	}
	if m.cfg.ANNThresholdBandPct > 0 {
		opts = append(opts, zenith.WithANNThresholdBand(m.cfg.ANNThresholdBandPct))
	}
	if m.cfg.Embedder != nil {
		opts = append(opts, zenith.WithEmbedder(m.cfg.Embedder))
	} else {
		opts = append(opts, zenith.WithBM25Only())
	}
	return opts
}

// acquire returns e with e.life held for read, opening its db lazily if
// necessary. The caller must call e.life.RUnlock() when done.
func (m *Manager) acquire(id string) (*entry, error) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil, ErrShuttingDown
	}
	e, ok := m.entries[id]
	m.mu.Unlock()
	if !ok {
		return nil, ErrNotFound
	}

	for {
		e.life.RLock()
		if e.deleted {
			e.life.RUnlock()
			return nil, ErrNotFound
		}
		if e.db != nil {
			m.mu.Lock()
			e.lastUsed = m.cfg.Now()
			if e.elem != nil {
				m.openLRU.MoveToFront(e.elem)
			}
			m.mu.Unlock()
			return e, nil
		}
		e.life.RUnlock()

		e.life.Lock()
		if !e.deleted && e.db == nil {
			db, err := zenith.Open(filepath.Join(e.dir, "index.db"), m.openOpts(id)...)
			if err != nil {
				e.life.Unlock()
				return nil, err
			}
			e.db = db
			m.mu.Lock()
			e.elem = m.openLRU.PushFront(e)
			e.lastUsed = m.cfg.Now()
			m.mu.Unlock()
			metrics.CollectionLifecycleTotal.WithLabelValues("opened").Inc()
			m.refreshCollectionsGauge()
		}
		e.life.Unlock()
		go m.enforceMaxOpen()
	}
}

// lookup returns e with e.life held for read, without opening its db. Used
// by the metadata-only operations (RotateKey, CheckKey, MaxBody, Stat) so
// reading a collection's quota or key never forces it open.
func (m *Manager) lookup(id string) (*entry, error) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil, ErrShuttingDown
	}
	e, ok := m.entries[id]
	m.mu.Unlock()
	if !ok {
		return nil, ErrNotFound
	}
	e.life.RLock()
	if e.deleted {
		e.life.RUnlock()
		return nil, ErrNotFound
	}
	return e, nil
}

func (e *entry) metaSnapshot() Meta {
	e.writeMu.Lock()
	defer e.writeMu.Unlock()
	return e.meta
}

// With runs fn with id's db open, holding it open for fn's duration.
func (m *Manager) With(ctx context.Context, id string, fn func(db *zenith.DB) error) error {
	e, err := m.acquire(id)
	if err != nil {
		return err
	}
	defer e.life.RUnlock()
	return fn(e.db)
}

// Create provisions a new collection and returns its info plus its
// plaintext API key (shown only here and by RotateKey).
func (m *Manager) Create(id string, o CreateOptions) (Info, string, error) {
	if !ValidID(id) {
		return Info{}, "", fmt.Errorf("%w: %q", ErrInvalidID, id)
	}

	maxDocs := o.MaxDocs
	switch {
	case maxDocs == 0:
		maxDocs = m.cfg.DefaultMaxDocs
	case maxDocs < 1 || maxDocs > 10_000_000:
		return Info{}, "", fmt.Errorf("collections: invalid option: max_docs must be 1..10000000")
	}
	maxBody := o.MaxBodyBytes
	switch {
	case maxBody == 0:
		maxBody = m.cfg.DefaultMaxBody
	case maxBody < 1024 || maxBody > 64<<20:
		return Info{}, "", fmt.Errorf("collections: invalid option: max_body_bytes must be 1024..%d", 64<<20)
	}

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return Info{}, "", ErrShuttingDown
	}
	if len(m.entries) >= m.cfg.MaxCollections {
		m.mu.Unlock()
		return Info{}, "", ErrTooManyCollections
	}
	if _, exists := m.entries[id]; exists {
		m.mu.Unlock()
		return Info{}, "", ErrExists
	}
	m.mu.Unlock()

	dir := filepath.Join(m.cfg.Root, id)
	if err := os.Mkdir(dir, 0o700); err != nil {
		if os.IsExist(err) {
			return Info{}, "", ErrExists
		}
		return Info{}, "", err
	}

	plain, keyHash, err := newKey()
	if err != nil {
		os.RemoveAll(dir)
		return Info{}, "", err
	}

	now := m.cfg.Now()
	meta := Meta{
		Version: 1, ID: id, CreatedAt: now, Embedder: m.embName,
		KeySHA256: keyHash, MaxDocs: maxDocs, MaxBodyBytes: maxBody,
	}
	if err := writeMeta(dir, meta); err != nil {
		os.RemoveAll(dir)
		return Info{}, "", err
	}

	db, err := zenith.Open(filepath.Join(dir, "index.db"), m.openOpts(id)...)
	if err != nil {
		os.RemoveAll(dir)
		return Info{}, "", err
	}

	e := &entry{id: id, dir: dir, db: db, meta: meta, lastUsed: now}
	m.mu.Lock()
	e.elem = m.openLRU.PushFront(e)
	m.entries[id] = e
	m.mu.Unlock()
	go m.enforceMaxOpen()

	metrics.CollectionDocuments.WithLabelValues(id).Set(0)
	metrics.CollectionLifecycleTotal.WithLabelValues("created").Inc()
	metrics.CollectionLifecycleTotal.WithLabelValues("opened").Inc()
	m.refreshCollectionsGauge()

	info := Info{
		ID: id, Embedder: meta.Embedder, CreatedAt: now, LastUsed: now,
		DocCount: 0, DocCountExact: true, MaxDocs: maxDocs, MaxBodyBytes: maxBody, Open: true,
	}
	return info, plain, nil
}

// Delete removes a collection permanently: it is closed, its directory moved
// aside and removed, and it is dropped from the manager's bookkeeping.
func (m *Manager) Delete(id string) error {
	m.mu.Lock()
	e, ok := m.entries[id]
	m.mu.Unlock()
	if !ok {
		return ErrNotFound
	}

	e.life.Lock()
	defer e.life.Unlock()
	if e.deleted {
		return ErrNotFound
	}
	e.deleted = true
	if e.db != nil {
		if err := e.db.Close(); err != nil {
			m.cfg.Log.Error("collections: close before delete failed", "id", id, "error", err)
		}
		e.db = nil
	}

	trash := filepath.Join(m.cfg.Root, fmt.Sprintf(".trash-%s-%d", id, m.cfg.Now().UnixNano()))
	if err := os.Rename(e.dir, trash); err != nil {
		m.cfg.Log.Error("collections: rename to trash failed", "id", id, "error", err)
	} else if err := os.RemoveAll(trash); err != nil {
		m.cfg.Log.Error("collections: trash cleanup failed", "id", id, "error", err)
	}

	m.mu.Lock()
	delete(m.entries, id)
	if e.elem != nil {
		m.openLRU.Remove(e.elem)
		e.elem = nil
	}
	m.mu.Unlock()

	metrics.CollectionLifecycleTotal.WithLabelValues("deleted").Inc()
	m.refreshCollectionsGauge()
	metrics.ForgetCollection(id)
	return nil
}

// List returns every collection's Info, sorted by id. It never walks disk
// (no DiskBytes/WALBytes); use Stat for a single collection's sizes.
func (m *Manager) List() []Info {
	m.mu.Lock()
	ids := make([]string, 0, len(m.entries))
	ents := make([]*entry, 0, len(m.entries))
	for id, e := range m.entries {
		ids = append(ids, id)
		ents = append(ents, e)
	}
	m.mu.Unlock()

	sort.Sort(&entriesByID{ids: ids, ents: ents})

	out := make([]Info, len(ents))
	for i, e := range ents {
		out[i] = m.infoFromEntry(e)
	}
	return out
}

// entriesByID sorts ids and ents together by id.
type entriesByID struct {
	ids  []string
	ents []*entry
}

func (s *entriesByID) Len() int           { return len(s.ids) }
func (s *entriesByID) Less(i, j int) bool { return s.ids[i] < s.ids[j] }
func (s *entriesByID) Swap(i, j int) {
	s.ids[i], s.ids[j] = s.ids[j], s.ids[i]
	s.ents[i], s.ents[j] = s.ents[j], s.ents[i]
}

func (m *Manager) infoFromEntry(e *entry) Info {
	meta := e.metaSnapshot()

	e.life.RLock()
	open := e.db != nil
	e.life.RUnlock()

	m.mu.Lock()
	lastUsed := e.lastUsed
	m.mu.Unlock()

	return Info{
		ID: meta.ID, Embedder: meta.Embedder, CreatedAt: meta.CreatedAt, LastUsed: lastUsed,
		DocCount: meta.DocCount, DocCountExact: !meta.Dirty,
		MaxDocs: meta.MaxDocs, MaxBodyBytes: meta.MaxBodyBytes, Open: open,
	}
}

// Stat returns id's Info including a live walk of its on-disk size.
func (m *Manager) Stat(id string) (Info, error) {
	e, err := m.lookup(id)
	if err != nil {
		return Info{}, err
	}
	info := m.infoFromEntry(e)
	e.life.RUnlock()

	info.DiskBytes = dirSize(e.dir)
	info.WALBytes = walSize(e.dir)
	return info, nil
}

func dirSize(dir string) int64 {
	var total int64
	_ = filepath.WalkDir(dir, func(_ string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if info, err := d.Info(); err == nil {
			total += info.Size()
		}
		return nil
	})
	return total
}

func walSize(dir string) int64 {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	var total int64
	for _, ent := range ents {
		name := ent.Name()
		if name == "index.db.wal" || strings.HasPrefix(name, "index.db.wal.old-") {
			if info, err := ent.Info(); err == nil {
				total += info.Size()
			}
		}
	}
	return total
}

// RotateKey replaces id's API key and returns the new plaintext key. The old
// key stops matching CheckKey immediately.
func (m *Manager) RotateKey(id string) (string, error) {
	e, err := m.lookup(id)
	if err != nil {
		return "", err
	}
	defer e.life.RUnlock()

	plain, hash, err := newKey()
	if err != nil {
		return "", err
	}

	e.writeMu.Lock()
	defer e.writeMu.Unlock()
	old := e.meta.KeySHA256
	e.meta.KeySHA256 = hash
	if err := writeMeta(e.dir, e.meta); err != nil {
		e.meta.KeySHA256 = old
		return "", err
	}
	return plain, nil
}

// CheckKey reports whether presented is id's current API key. Returns false
// (never an error) for a missing or deleted collection.
func (m *Manager) CheckKey(id, presented string) bool {
	e, err := m.lookup(id)
	if err != nil {
		return false
	}
	meta := e.metaSnapshot()
	e.life.RUnlock()
	return keyMatches(presented, meta.KeySHA256)
}

// MaxBody returns id's configured request-body cap.
func (m *Manager) MaxBody(id string) (int64, error) {
	e, err := m.lookup(id)
	if err != nil {
		return 0, err
	}
	meta := e.metaSnapshot()
	e.life.RUnlock()
	return meta.MaxBodyBytes, nil
}

// Upsert indexes docs into id, enforcing its document quota. On quota
// overflow nothing is written. attrs may be nil.
func (m *Manager) Upsert(ctx context.Context, id string, docs map[string]string, attrs map[string]zenith.Attrs) (UpsertResult, error) {
	e, err := m.acquire(id)
	if err != nil {
		return UpsertResult{}, err
	}
	defer e.life.RUnlock()

	e.writeMu.Lock()
	defer e.writeMu.Unlock()

	newIDs := 0
	for docID := range docs {
		if _, found, _ := e.db.Get(docID); !found {
			newIDs++
		}
	}

	oldCount := e.meta.DocCount
	if oldCount+newIDs > e.meta.MaxDocs {
		return UpsertResult{}, ErrQuota
	}

	e.meta.DocCount = oldCount + newIDs
	e.meta.Dirty = true
	if err := writeMeta(e.dir, e.meta); err != nil {
		e.meta.DocCount = oldCount
		e.meta.Dirty = false
		return UpsertResult{}, err
	}

	if err := e.db.AddBatchWithAttrs(ctx, docs, attrs); err != nil {
		e.meta.DocCount = oldCount
		e.meta.Dirty = false
		_ = writeMeta(e.dir, e.meta)
		return UpsertResult{}, err
	}

	e.meta.Dirty = false
	if err := writeMeta(e.dir, e.meta); err != nil {
		return UpsertResult{}, err
	}

	metrics.CollectionDocuments.WithLabelValues(id).Set(float64(e.meta.DocCount))
	metrics.CollectionDocsUpsertedTotal.WithLabelValues(id).Add(float64(len(docs)))

	return UpsertResult{Upserted: len(docs), New: newIDs, DocCount: e.meta.DocCount}, nil
}

// DeleteDoc removes one document from id.
func (m *Manager) DeleteDoc(ctx context.Context, id, docID string) error {
	e, err := m.acquire(id)
	if err != nil {
		return err
	}
	defer e.life.RUnlock()

	e.writeMu.Lock()
	defer e.writeMu.Unlock()

	if _, found, _ := e.db.Get(docID); !found {
		return ErrDocNotFound
	}
	if err := e.db.Delete(ctx, docID); err != nil {
		return err
	}
	e.meta.DocCount--
	if err := writeMeta(e.dir, e.meta); err != nil {
		return err
	}
	metrics.CollectionDocuments.WithLabelValues(id).Set(float64(e.meta.DocCount))
	metrics.CollectionDocsDeletedTotal.WithLabelValues(id).Inc()
	return nil
}

// GetDoc returns the original text of docID in collection id.
func (m *Manager) GetDoc(ctx context.Context, id, docID string) (string, error) {
	e, err := m.acquire(id)
	if err != nil {
		return "", err
	}
	defer e.life.RUnlock()

	text, found, err := e.db.Get(docID)
	if err != nil {
		return "", err
	}
	if !found {
		return "", ErrDocNotFound
	}
	return text, nil
}

// GetDocAttrs returns the metadata attrs stored with docID in collection
// id (nil, nil if it has none). Pairs with GetDoc, which returns only the
// text; split into its own method rather than widening GetDoc's signature
// because most callers (the native /v1/collections HTTP route included)
// never need attrs back out, only in for filtering.
func (m *Manager) GetDocAttrs(ctx context.Context, id, docID string) (zenith.Attrs, error) {
	e, err := m.acquire(id)
	if err != nil {
		return nil, err
	}
	defer e.life.RUnlock()

	_, found, err := e.db.Get(docID)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, ErrDocNotFound
	}
	return e.db.GetAttrs(docID)
}

// Search runs a query against collection id. Latency is measured against
// the real wall clock, not cfg.Now (which exists so tests can fast-forward
// idle/LRU timing, not to control what a real request actually took).
func (m *Manager) Search(ctx context.Context, id, query string, opts ...zenith.SearchOption) ([]zenith.Result, error) {
	start := time.Now()
	var res []zenith.Result
	err := m.With(ctx, id, func(db *zenith.DB) error {
		var err error
		res, err = db.Search(ctx, query, opts...)
		return err
	})
	metrics.CollectionQueryDuration.WithLabelValues(id).Observe(time.Since(start).Seconds())
	outcome := "ok"
	if err != nil {
		outcome = "error"
	}
	metrics.CollectionQueriesTotal.WithLabelValues(id, outcome).Inc()
	return res, err
}

// evict closes every entry selector returns, recording reason
// ("closed_idle" or "closed_lru") against each one actually closed. selector
// runs under m.mu and must only read openLRU/lastUsed; the Close itself runs
// outside that lock since it checkpoints to disk. An entry currently in use
// (life.TryLock fails) is skipped and retried on the next sweep.
func (m *Manager) evict(reason string, selector func() []*entry) {
	m.mu.Lock()
	victims := selector()
	m.mu.Unlock()

	closed := 0
	for _, e := range victims {
		if !e.life.TryLock() {
			continue
		}
		if e.db != nil {
			if err := e.db.Close(); err != nil {
				m.cfg.Log.Error("collections: close failed", "id", e.id, "error", err)
			}
			e.db = nil
			closed++
		}
		e.life.Unlock()

		m.mu.Lock()
		if e.elem != nil {
			m.openLRU.Remove(e.elem)
			e.elem = nil
		}
		m.mu.Unlock()
	}
	if closed > 0 {
		metrics.CollectionLifecycleTotal.WithLabelValues(reason).Add(float64(closed))
		m.refreshCollectionsGauge()
	}
}

// Sweep closes every open collection idle for longer than cfg.IdleClose, and
// refreshes the per-collection disk/WAL size gauges (a live directory walk,
// same as Stat, run here instead of on every request).
func (m *Manager) Sweep() {
	m.evict("closed_idle", func() []*entry {
		cutoff := m.cfg.Now().Add(-m.cfg.IdleClose)
		var victims []*entry
		for el := m.openLRU.Back(); el != nil; el = el.Prev() {
			e := el.Value.(*entry)
			if e.lastUsed.Before(cutoff) {
				victims = append(victims, e)
			}
		}
		return victims
	})

	m.mu.Lock()
	type idDir struct{ id, dir string }
	snapshot := make([]idDir, 0, len(m.entries))
	for id, e := range m.entries {
		snapshot = append(snapshot, idDir{id, e.dir})
	}
	m.mu.Unlock()
	for _, s := range snapshot {
		metrics.CollectionDiskBytes.WithLabelValues(s.id).Set(float64(dirSize(s.dir)))
		metrics.CollectionWALBytes.WithLabelValues(s.id).Set(float64(walSize(s.dir)))
	}
}

// enforceMaxOpen closes the least-recently-used open collections until the
// open count is back at or below cfg.MaxOpen.
func (m *Manager) enforceMaxOpen() {
	m.evict("closed_lru", func() []*entry {
		over := m.openLRU.Len() - m.cfg.MaxOpen
		if over <= 0 {
			return nil
		}
		victims := make([]*entry, 0, over)
		for el := m.openLRU.Back(); el != nil && len(victims) < over; el = el.Prev() {
			victims = append(victims, el.Value.(*entry))
		}
		return victims
	})
}

// Run sweeps idle collections every interval until ctx is done.
func (m *Manager) Run(ctx context.Context, interval time.Duration) {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			m.Sweep()
		}
	}
}

// Counts returns the total number of collections and how many are open.
func (m *Manager) Counts() (total, open int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.entries), m.openLRU.Len()
}

// CloseAll closes every open collection and releases the root registration.
// Safe to call once; a second call is a no-op.
func (m *Manager) CloseAll() error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil
	}
	m.closed = true
	ents := make([]*entry, 0, len(m.entries))
	for _, e := range m.entries {
		ents = append(ents, e)
	}
	m.mu.Unlock()

	var errs []error
	closed := 0
	for _, e := range ents {
		e.life.Lock()
		if e.db != nil {
			if err := e.db.Close(); err != nil {
				errs = append(errs, fmt.Errorf("collections: close %s: %w", e.id, err))
			}
			e.db = nil
			closed++
		}
		e.life.Unlock()
	}
	if closed > 0 {
		metrics.CollectionLifecycleTotal.WithLabelValues("closed_shutdown").Add(float64(closed))
	}
	metrics.Collections.WithLabelValues("open").Set(0)
	metrics.Collections.WithLabelValues("closed").Set(float64(len(ents)))

	registeredRoots.Delete(m.rootKey)
	return errors.Join(errs...)
}
