//go:build windows

package segment

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

// mapping is a read-only memory map of a whole file. The bytes stay valid
// until Close; anything aliasing them must not be used afterwards. While a
// file is mapped Windows refuses to delete or replace it, so callers must
// Close a segment before removing or renaming over its file.
type mapping struct {
	data []byte
	addr uintptr
	h    syscall.Handle
}

func mapFile(path string) (*mapping, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := fi.Size()
	if size == 0 {
		return nil, fmt.Errorf("segment: %s is empty", path)
	}
	h, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READONLY, uint32(size>>32), uint32(size), nil)
	if err != nil {
		return nil, fmt.Errorf("segment: CreateFileMapping %s: %w", path, err)
	}
	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_READ, 0, 0, uintptr(size))
	if err != nil {
		syscall.CloseHandle(h)
		return nil, fmt.Errorf("segment: MapViewOfFile %s: %w", path, err)
	}
	// addr is an OS mapping, not Go memory, so converting it to a slice is the
	// documented way to expose it.
	data := unsafe.Slice((*byte)(*(*unsafe.Pointer)(unsafe.Pointer(&addr))), int(size))
	return &mapping{data: data, addr: addr, h: h}, nil
}

func (m *mapping) close() error {
	if m == nil || m.data == nil {
		return nil
	}
	m.data = nil
	err := syscall.UnmapViewOfFile(m.addr)
	if cerr := syscall.CloseHandle(m.h); err == nil {
		err = cerr
	}
	return err
}
