//go:build !windows

package segment

import (
	"fmt"
	"os"
	"syscall"
)

// mapping is a read-only memory map of a whole file. The bytes stay valid
// until Close; anything aliasing them must not be used afterwards.
type mapping struct {
	data []byte
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
	if int64(int(size)) != size {
		return nil, fmt.Errorf("segment: %s is too large to map on this platform", path)
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("segment: mmap %s: %w", path, err)
	}
	return &mapping{data: data}, nil
}

func (m *mapping) close() error {
	if m == nil || m.data == nil {
		return nil
	}
	err := syscall.Munmap(m.data)
	m.data = nil
	return err
}
