//go:build !windows

package zenith

import "syscall"

// processAlive returns true if the process with the given PID is still running.
// On Unix, sending signal 0 probes the process without disturbing it.
func processAlive(pid int) bool {
	err := syscall.Kill(pid, 0)
	return err == nil
}

// processStartToken has no portable implementation on Unix (retrieving a
// process's start time without /proc or platform-specific APIs isn't
// available via the standard syscall package here); callers fall back to
// plain PID-liveness, unchanged from before this disambiguator existed.
func processStartToken(pid int) (string, bool) {
	return "", false
}
