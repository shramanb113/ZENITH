package index

import (
	"os"
)

// failpoint terminates the process at a named point when the environment asks
// for it (ZENITH_FAILPOINT=<name>). It exists so crash-recovery tests can kill
// the process at exact moments of a flush or compaction — after the segment is
// written but before the manifest commits, and so on — rather than relying on
// a random kill to land there. With the variable unset it costs one map-free
// string comparison.
func failpoint(name string) {
	if fp := os.Getenv("ZENITH_FAILPOINT"); fp != "" && fp == name {
		os.Exit(137)
	}
}
