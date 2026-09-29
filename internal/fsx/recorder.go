package fsx

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
)

// OpKind names a recorded filesystem operation.
type OpKind int

const (
	OpCreate OpKind = iota
	OpWrite
	OpSync
	OpTruncate
	OpRename
	OpRemove
	OpSyncDir
	OpMark // a label a test inserts to record "this was acknowledged"
)

// Op is one recorded operation. Paths are relative to the recorder's root.
type Op struct {
	Kind  OpKind
	Path  string
	Path2 string // rename target
	Off   int64
	Data  []byte
	Size  int64 // truncate size, or 1 for a truncating create
	Label string
}

// Recorder is a Hooks that logs every mutation under root, in order, without
// changing behaviour. CrashState then rebuilds the directory as it could look
// after a power failure at any point in the log.
type Recorder struct {
	root string
	mu   sync.Mutex
	ops  []Op
}

// NewRecorder records operations on files under root.
func NewRecorder(root string) *Recorder {
	return &Recorder{root: filepath.Clean(root)}
}

func (r *Recorder) rel(p string) (string, bool) {
	rel, err := filepath.Rel(r.root, filepath.Clean(p))
	if err != nil || strings.HasPrefix(rel, "..") {
		return "", false
	}
	return filepath.ToSlash(rel), true
}

func (r *Recorder) add(op Op) {
	r.mu.Lock()
	r.ops = append(r.ops, op)
	r.mu.Unlock()
}

// Mark inserts a label into the log; a test calls it right after an operation
// was acknowledged to the caller (Save returned, Add returned).
func (r *Recorder) Mark(label string) { r.add(Op{Kind: OpMark, Label: label}) }

// Len is the number of operations recorded so far.
func (r *Recorder) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.ops)
}

// Ops returns a copy of the log.
func (r *Recorder) Ops() []Op {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]Op(nil), r.ops...)
}

func (r *Recorder) Write(path string, off int64, p []byte) (int, error) {
	if rel, ok := r.rel(path); ok {
		r.add(Op{Kind: OpWrite, Path: rel, Off: off, Data: append([]byte(nil), p...)})
	}
	return len(p), nil
}
func (r *Recorder) Sync(path string) error {
	if rel, ok := r.rel(path); ok {
		r.add(Op{Kind: OpSync, Path: rel})
	}
	return nil
}
func (r *Recorder) Truncate(path string, size int64) error {
	if rel, ok := r.rel(path); ok {
		r.add(Op{Kind: OpTruncate, Path: rel, Size: size})
	}
	return nil
}
func (r *Recorder) Create(path string, trunc bool) {
	if rel, ok := r.rel(path); ok {
		sz := int64(0)
		if trunc {
			sz = 1
		}
		r.add(Op{Kind: OpCreate, Path: rel, Size: sz})
	}
}
func (r *Recorder) Rename(oldpath, newpath string) error {
	o, ok1 := r.rel(oldpath)
	n, ok2 := r.rel(newpath)
	if ok1 && ok2 {
		r.add(Op{Kind: OpRename, Path: o, Path2: n})
	}
	return nil
}
func (r *Recorder) Remove(path string) {
	if rel, ok := r.rel(path); ok {
		r.add(Op{Kind: OpRemove, Path: rel})
	}
}
func (r *Recorder) SyncDir(dir string) error {
	if rel, ok := r.rel(dir); ok {
		r.add(Op{Kind: OpSyncDir, Path: rel})
	}
	return nil
}

// ---- crash simulation ----

// Persistence says what a crash may leave of state that was never fsynced.
type Persistence int

const (
	// Nothing: only fsynced data and fsynced directory entries survive.
	Nothing Persistence = iota
	// Everything: all writes and directory operations reached the disk (the
	// crash lost nothing; still a valid state to recover from).
	Everything
	// Torn: a random subset of unsynced writes survives, each possibly cut short,
	// and a random prefix of the unsynced directory operations of each directory.
	Torn
)

// inode is a file's content in the simulation.
type inode struct {
	vol     []byte // what the running process saw
	dur     []byte // what an fsync made durable
	pending []Op   // writes/truncates since the last fsync, in order
}

func applyWrite(b []byte, off int64, data []byte) []byte {
	end := int(off) + len(data)
	if end > len(b) {
		b = append(b, make([]byte, end-len(b))...)
	}
	copy(b[off:], data)
	return b
}

func applyTruncate(b []byte, size int64) []byte {
	if int(size) <= len(b) {
		return b[:size]
	}
	return append(b, make([]byte, int(size)-len(b))...)
}

type dirOp struct {
	kind    OpKind // OpCreate, OpRename, OpRemove
	path    string
	path2   string
	newNode *inode // for a create
}

// CrashState materialises, into dst, the directory as it could be after a power
// failure right after the first k recorded operations. seed drives Torn.
//
// Model. Data reaches the disk only through fsync of that file; a directory
// operation (create, rename, remove) only through fsync of its directory. What
// was not synced may or may not survive (see Persistence). Directory operations
// persist as a prefix of the ones issued since the directory was last synced —
// journalled filesystems apply metadata in order — and each rename is atomic:
// after a crash the old or the new name exists, never neither.
func (r *Recorder) CrashState(k int, p Persistence, seed int64, dst string) error {
	ops := r.Ops()
	if k > len(ops) {
		k = len(ops)
	}
	rng := rand.New(rand.NewSource(seed))

	visible := map[string]*inode{}  // namespace as the process sees it
	durable := map[string]*inode{}  // namespace as of the last directory syncs
	pending := map[string][]dirOp{} // dir -> directory ops since its last sync
	dirOf := func(path string) string { return filepath.ToSlash(filepath.Dir(filepath.FromSlash(path))) }

	for _, op := range ops[:k] {
		switch op.Kind {
		case OpCreate:
			ino := visible[op.Path]
			if ino == nil {
				ino = &inode{}
				visible[op.Path] = ino
				pending[dirOf(op.Path)] = append(pending[dirOf(op.Path)], dirOp{kind: OpCreate, path: op.Path, newNode: ino})
			} else { // O_TRUNC of an existing file
				ino.vol = nil
				ino.pending = append(ino.pending, Op{Kind: OpTruncate, Size: 0})
			}
		case OpWrite:
			if ino := visible[op.Path]; ino != nil {
				ino.vol = applyWrite(ino.vol, op.Off, op.Data)
				ino.pending = append(ino.pending, op)
			}
		case OpTruncate:
			if ino := visible[op.Path]; ino != nil {
				ino.vol = applyTruncate(ino.vol, op.Size)
				ino.pending = append(ino.pending, op)
			}
		case OpSync:
			if ino := visible[op.Path]; ino != nil {
				ino.dur = append([]byte(nil), ino.vol...)
				ino.pending = nil
			}
		case OpRename:
			if ino := visible[op.Path]; ino != nil {
				delete(visible, op.Path)
				visible[op.Path2] = ino
				pending[dirOf(op.Path2)] = append(pending[dirOf(op.Path2)], dirOp{kind: OpRename, path: op.Path, path2: op.Path2})
				if dirOf(op.Path) != dirOf(op.Path2) {
					pending[dirOf(op.Path)] = append(pending[dirOf(op.Path)], dirOp{kind: OpRename, path: op.Path, path2: op.Path2})
				}
			}
		case OpRemove:
			if visible[op.Path] != nil {
				delete(visible, op.Path)
				pending[dirOf(op.Path)] = append(pending[dirOf(op.Path)], dirOp{kind: OpRemove, path: op.Path})
			}
		case OpSyncDir:
			// Everything pending in this directory becomes durable: the durable
			// namespace of the directory now equals the visible one.
			for path := range durable {
				if dirOf(path) == op.Path {
					delete(durable, path)
				}
			}
			for path, ino := range visible {
				if dirOf(path) == op.Path {
					durable[path] = ino
				}
			}
			delete(pending, op.Path)
		}
	}

	// Choose the namespace that survives.
	final := map[string]*inode{}
	switch p {
	case Everything:
		for path, ino := range visible {
			final[path] = ino
		}
	default:
		for path, ino := range durable {
			final[path] = ino
		}
		dirs := make([]string, 0, len(pending))
		for d := range pending {
			dirs = append(dirs, d)
		}
		sort.Strings(dirs)
		if p == Torn {
			for _, d := range dirs {
				list := pending[d]
				n := rng.Intn(len(list) + 1)
				for _, dop := range list[:n] {
					switch dop.kind {
					case OpCreate:
						final[dop.path] = dop.newNode
					case OpRename:
						if ino := final[dop.path]; ino != nil {
							delete(final, dop.path)
							final[dop.path2] = ino
						}
					case OpRemove:
						delete(final, dop.path)
					}
				}
			}
		}
	}

	if err := os.MkdirAll(dst, 0o755); err != nil {
		return err
	}
	for path, ino := range final {
		var content []byte
		switch p {
		case Everything:
			content = ino.vol
		case Nothing:
			content = ino.dur
		case Torn:
			content = append([]byte(nil), ino.dur...)
			for _, w := range ino.pending {
				switch w.Kind {
				case OpWrite:
					// Pages reach the disk independently and in any order: keep a
					// random subset of the 4 KiB blocks the write covers.
					const page = 4096
					for off := w.Off; off < w.Off+int64(len(w.Data)); {
						end := (off/page + 1) * page
						if end > w.Off+int64(len(w.Data)) {
							end = w.Off + int64(len(w.Data))
						}
						if rng.Intn(2) == 0 {
							content = applyWrite(content, off, w.Data[off-w.Off:end-w.Off])
						}
						off = end
					}
				case OpTruncate:
					if rng.Intn(2) == 0 {
						content = applyTruncate(content, w.Size)
					}
				}
			}
		}
		full := filepath.Join(dst, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(full, content, 0o644); err != nil {
			return err
		}
	}
	return nil
}

// Marks returns the labels recorded among the first k operations, in order.
func (r *Recorder) Marks(k int) []string {
	var out []string
	for i, op := range r.Ops() {
		if i >= k {
			break
		}
		if op.Kind == OpMark {
			out = append(out, op.Label)
		}
	}
	return out
}

// ---- disk full ----

// DiskFull is a Hooks that lets a fixed number of bytes be written and then
// fails every further write with ENOSPC (a write that straddles the limit is cut
// short, as a real one is). Creates, renames and syncs still work: a full disk
// still has room for directory entries, and a test that needs those to fail can
// wrap it.
type DiskFull struct {
	mu     sync.Mutex
	budget int64 // bytes still allowed; < 0 = unlimited
	Failed int   // writes refused so far
}

// NewDiskFull allows budget more bytes to be written.
func NewDiskFull(budget int64) *DiskFull { return &DiskFull{budget: budget} }

// SetBudget changes how many more bytes may be written (negative = unlimited).
func (d *DiskFull) SetBudget(n int64) {
	d.mu.Lock()
	d.budget = n
	d.mu.Unlock()
}

// ErrNoSpace is what a refused write returns.
var ErrNoSpace = fmt.Errorf("fsx: %w", syscall.ENOSPC)

func (d *DiskFull) Write(path string, off int64, p []byte) (int, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.budget < 0 || int64(len(p)) <= d.budget {
		if d.budget >= 0 {
			d.budget -= int64(len(p))
		}
		return len(p), nil
	}
	n := int(d.budget)
	d.budget = 0
	d.Failed++
	return n, ErrNoSpace
}
func (d *DiskFull) Sync(string) error            { return nil }
func (d *DiskFull) Truncate(string, int64) error { return nil }
func (d *DiskFull) Create(string, bool)          {}
func (d *DiskFull) Rename(_, _ string) error     { return nil }
func (d *DiskFull) Remove(string)                {}
func (d *DiskFull) SyncDir(string) error         { return nil }
