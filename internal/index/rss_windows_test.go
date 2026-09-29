//go:build windows

package index

import (
	"syscall"
	"unsafe"
)

// procMem returns the process's working set (all resident pages, including
// mapped file pages) and private bytes (memory only this process owns: heap,
// stacks) — their difference is roughly the resident share of the mapped index.
func procMem() (workingSet, private uint64) {
	type counters struct {
		cb                         uint32
		PageFaultCount             uint32
		PeakWorkingSetSize         uintptr
		WorkingSetSize             uintptr
		QuotaPeakPagedPoolUsage    uintptr
		QuotaPagedPoolUsage        uintptr
		QuotaPeakNonPagedPoolUsage uintptr
		QuotaNonPagedPoolUsage     uintptr
		PagefileUsage              uintptr
		PeakPagefileUsage          uintptr
		PrivateUsage               uintptr
	}
	var c counters
	c.cb = uint32(unsafe.Sizeof(c))
	h, _ := syscall.GetCurrentProcess()
	proc := syscall.NewLazyDLL("psapi.dll").NewProc("GetProcessMemoryInfo")
	if r, _, _ := proc.Call(uintptr(h), uintptr(unsafe.Pointer(&c)), uintptr(c.cb)); r == 0 {
		return 0, 0
	}
	return uint64(c.WorkingSetSize), uint64(c.PrivateUsage)
}
