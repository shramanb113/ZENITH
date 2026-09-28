//go:build race

package index

// raceEnabled reports whether the race detector is compiled in (timing-sensitive
// tests skip themselves: -race slows the engine 5-10x).
const raceEnabled = true
