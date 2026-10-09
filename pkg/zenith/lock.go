package zenith

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
)

// inProcReg prevents two Open calls in the same process from targeting the
// same database file. Keyed by absolute path.
var inProcReg sync.Map // map[string]struct{}

type fileLock struct {
	absPath  string
	lockPath string
}

// acquireLock grabs both the in-process registry slot and a cross-process
// lock file at absPath+".lock". Returns ErrLocked immediately if either is
// already held. On success the caller must call release() when done.
//
// Stale lock files (left by a crashed process) are detected by reading the
// PID stored inside and checking whether that process is still alive.
func acquireLock(absPath string) (*fileLock, error) {
	// In-process check first — fast path.
	if _, loaded := inProcReg.LoadOrStore(absPath, struct{}{}); loaded {
		return nil, ErrLocked
	}

	lockPath := absPath + ".lock"
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		if !os.IsExist(err) {
			inProcReg.Delete(absPath)
			return nil, fmt.Errorf("zenith: lock file: %w", err)
		}

		// Lock file exists — check for stale lock (dead process).
		if stale := isStale(lockPath); !stale {
			inProcReg.Delete(absPath)
			return nil, ErrLocked
		}

		// Stale lock — remove and retry once.
		os.Remove(lockPath)
		f, err = os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			inProcReg.Delete(absPath)
			return nil, ErrLocked
		}
	}

	f.WriteString(lockToken())
	f.Close()

	return &fileLock{absPath: absPath, lockPath: lockPath}, nil
}

// lockToken formats this process's lock-file contents: PID, plus a
// process-start disambiguator when the platform can supply one (see
// processStartToken). The token lets isStale tell a live process that
// genuinely holds the lock apart from an unrelated process that merely
// inherited the same, recycled PID after the original holder died.
func lockToken() string {
	pid := os.Getpid()
	if tok, ok := processStartToken(pid); ok {
		return fmt.Sprintf("%d %s\n", pid, tok)
	}
	return fmt.Sprintf("%d\n", pid)
}

// release removes the lock file and frees the in-process registry slot.
func (l *fileLock) release() {
	os.Remove(l.lockPath)
	inProcReg.Delete(l.absPath)
}

// isStale reads the PID (and, if present, the process-start token) from the
// lock file and returns true if the original holder is gone. A lock file
// predates this token (or was written on a platform with no
// processStartToken support) is just "<pid>\n" — that format, and a
// still-alive PID whose current start token can't be read either, falls
// back to plain PID-liveness, unchanged from before this disambiguator
// existed.
//
// When a stored token IS present and the PID is alive, the stored token is
// compared against that PID's *current* start token: a mismatch means the
// OS recycled the PID for an unrelated process after the true holder died,
// so the lock is stale even though some process answers to that PID right
// now. This closes the race where rapid process churn lets a dead holder's
// PID be reused before the staleness check runs (see
// TestCollections_SurvivesHardKill).
//
// Returns false (not stale) on any read error so we err on the side of
// caution.
func isStale(lockPath string) bool {
	data, err := os.ReadFile(lockPath)
	if err != nil {
		return false
	}
	fields := strings.Fields(string(data))
	if len(fields) == 0 {
		return false
	}
	pid, err := strconv.Atoi(fields[0])
	if err != nil {
		return false
	}
	if !processAlive(pid) {
		return true
	}
	if len(fields) < 2 {
		// Legacy lock file, or no token was available when it was written.
		return false
	}
	storedToken := fields[1]
	currentToken, ok := processStartToken(pid)
	if !ok {
		// Can't read the current token to compare — don't guess stale.
		return false
	}
	return currentToken != storedToken
}
