//go:build !windows

package index

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

// procMem returns resident set size and anonymous (private) resident memory.
func procMem() (workingSet, private uint64) {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return 0, 0
	}
	defer f.Close()
	var rss, anon uint64
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 2 {
			continue
		}
		kb, _ := strconv.ParseUint(fields[1], 10, 64)
		switch fields[0] {
		case "VmRSS:":
			rss = kb << 10
		case "RssAnon:":
			anon = kb << 10
		}
	}
	return rss, anon
}
