// Package fsx is the thin layer through which the durable parts of ZENITH (the
// segment writer, the manifest, the write-ahead log) touch the filesystem.
//
// In production it is a direct call to package os plus one atomic pointer load.
// Its purpose is testing: a test installs Hooks to observe every mutating
// operation (see Recorder, which records them so a test can rebuild the disk as
// a power failure would have left it) or to make one fail (see DiskFull).
// Reads are not routed through it; only operations that change what is on disk.
package fsx

import (
	"io"
	"os"
	"path/filepath"
	"sync/atomic"
)

// Hooks observes and may veto filesystem mutations. All methods may be called
// concurrently. A nil error from a veto-capable method lets the operation proceed.
type Hooks interface {
	// Write is called before p is written at offset off of path. It returns how
	// many bytes may be written; if that is less than len(p), the write is cut
	// there and err is returned to the caller.
	Write(path string, off int64, p []byte) (n int, err error)
	// Sync is called before path's data is made durable.
	Sync(path string) error
	// Truncate is called before path is truncated to size.
	Truncate(path string, size int64) error
	// Create is called after path was created (trunc=false) or opened with O_TRUNC.
	Create(path string, trunc bool)
	// Rename is called before oldpath is renamed to newpath.
	Rename(oldpath, newpath string) error
	// Remove is called after path was removed.
	Remove(path string)
	// SyncDir is called before dir's entries are made durable.
	SyncDir(dir string) error
}

type box struct{ h Hooks }

var active atomic.Pointer[box]

// SetHooks installs h (nil removes any hooks) and returns a function that
// restores the previous ones. Tests that install hooks must not run in parallel
// with anything else that writes through fsx.
func SetHooks(h Hooks) (restore func()) {
	var b *box
	if h != nil {
		b = &box{h}
	}
	prev := active.Swap(b)
	return func() { active.Store(prev) }
}

func hooks() Hooks {
	if b := active.Load(); b != nil {
		return b.h
	}
	return nil
}

// File is an *os.File whose mutations are reported to the installed Hooks.
type File struct {
	f    *os.File
	path string
	pos  int64
}

// OpenFile is os.OpenFile.
func OpenFile(path string, flag int, perm os.FileMode) (*File, error) {
	_, statErr := os.Stat(path)
	existed := statErr == nil
	f, err := os.OpenFile(path, flag, perm)
	if err != nil {
		return nil, err
	}
	if h := hooks(); h != nil {
		if !existed {
			h.Create(path, false)
		} else if flag&os.O_TRUNC != 0 {
			h.Create(path, true)
		}
	}
	return &File{f: f, path: path}, nil
}

// Create is os.Create.
func Create(path string) (*File, error) {
	return OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o666)
}

func (f *File) Name() string { return f.path }

func (f *File) Read(p []byte) (int, error) {
	n, err := f.f.Read(p)
	f.pos += int64(n)
	return n, err
}

func (f *File) Seek(offset int64, whence int) (int64, error) {
	n, err := f.f.Seek(offset, whence)
	if err == nil {
		f.pos = n
	}
	return n, err
}

func (f *File) Stat() (os.FileInfo, error) { return f.f.Stat() }

func (f *File) Write(p []byte) (int, error) {
	n, herr := len(p), error(nil)
	if h := hooks(); h != nil {
		n, herr = h.Write(f.path, f.pos, p)
	}
	w, err := f.f.Write(p[:n])
	f.pos += int64(w)
	if err == nil {
		err = herr
	}
	return w, err
}

func (f *File) WriteAt(p []byte, off int64) (int, error) {
	n, herr := len(p), error(nil)
	if h := hooks(); h != nil {
		n, herr = h.Write(f.path, off, p)
	}
	w, err := f.f.WriteAt(p[:n], off)
	if err == nil {
		err = herr
	}
	return w, err
}

func (f *File) Sync() error {
	if h := hooks(); h != nil {
		if err := h.Sync(f.path); err != nil {
			return err
		}
	}
	return f.f.Sync()
}

func (f *File) Truncate(size int64) error {
	if h := hooks(); h != nil {
		if err := h.Truncate(f.path, size); err != nil {
			return err
		}
	}
	return f.f.Truncate(size)
}

func (f *File) Close() error { return f.f.Close() }

var _ io.ReadWriteSeeker = (*File)(nil)

// Rename is os.Rename.
func Rename(oldpath, newpath string) error {
	if h := hooks(); h != nil {
		if err := h.Rename(oldpath, newpath); err != nil {
			return err
		}
	}
	return os.Rename(oldpath, newpath)
}

// Remove is os.Remove.
func Remove(path string) error {
	err := os.Remove(path)
	if err == nil {
		if h := hooks(); h != nil {
			h.Remove(path)
		}
	}
	return err
}

// SyncDir makes the directory entries of dir (creations, renames, removals)
// durable. On Windows a directory cannot be fsynced and the real call is skipped
// (NTFS journals metadata); the hook is still told, so the simulated durability
// model is the same on every platform.
func SyncDir(dir string) error {
	if h := hooks(); h != nil {
		if err := h.SyncDir(dir); err != nil {
			return err
		}
	}
	d, err := os.Open(filepath.Clean(dir))
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
