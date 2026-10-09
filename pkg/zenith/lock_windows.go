//go:build windows

package zenith

import (
	"fmt"
	"syscall"
)

const processQueryInformation = 0x0400

// processAlive returns true if the process with the given PID is still running.
// On Windows, we attempt to open the process handle; failure means it's gone.
func processAlive(pid int) bool {
	h, err := syscall.OpenProcess(processQueryInformation, false, uint32(pid))
	if err != nil {
		return false
	}
	syscall.CloseHandle(h)
	return true
}

// processStartToken identifies *which* process currently holds pid, so a
// recycled PID (a dead lock-holder's PID reassigned by the OS to an
// unrelated, currently-alive process) can be told apart from the original
// holder. The process creation time is unique per PID lifetime on Windows,
// so it serves as that disambiguator. Returns ("", false) if the process
// can't be opened or its times can't be read — callers must fall back to
// plain PID-liveness in that case, not treat it as stale.
func processStartToken(pid int) (string, bool) {
	h, err := syscall.OpenProcess(processQueryInformation, false, uint32(pid))
	if err != nil {
		return "", false
	}
	defer syscall.CloseHandle(h)

	var creation, exit, kernel, user syscall.Filetime
	if err := syscall.GetProcessTimes(h, &creation, &exit, &kernel, &user); err != nil {
		return "", false
	}
	return fmt.Sprintf("%x-%x", creation.HighDateTime, creation.LowDateTime), true
}
