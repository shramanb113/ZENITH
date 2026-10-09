package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"
)

// ─── terminal detection ───────────────────────────────────────────────────────

var colorEnabled = func() bool {
	stat, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return (stat.Mode() & os.ModeCharDevice) != 0
}()

// ─── 256-color palette ────────────────────────────────────────────────────────

const (
	ansiReset = "\033[0m"
	ansiBold  = "\033[1m"
)

func col(n int) string { return fmt.Sprintf("\033[38;5;%dm", n) }

// Neon palette — matches the approved design
var (
	cBrand   = col(213) // hot pink      — ZENITH wordmark, command verbs
	cAccent  = col(141) // soft violet   — ⟡ glyph, arrows, step labels
	cFile    = col(255) // near-white    — file names (+ bold)
	cPath    = col(60)  // muted indigo  — full paths
	cRule    = col(59)  // dark blue-gray — dividers, box lines
	cMuted   = col(60)  // muted indigo  — footer, secondary text
	cSuccess = col(84)  // mint green    — ✓
	cWarn    = col(215) // amber         — !
	cErrCol  = col(203) // coral red     — ✗
	cDim     = col(237) // very dark gray — dim text inside boxes
	cVer     = col(59)  // dark blue-gray — version string
	cHi      = col(84)  // mint green    — score bar ≥ 0.7 of top
	cMid     = col(215) // amber         — score bar 0.4–0.7 of top
	cLo      = col(60)  // muted indigo  — score bar < 0.4 of top
	cProg    = col(213) // hot pink      — live index counter
)

func cc(s, code string) string {
	if !colorEnabled {
		return s
	}
	return code + s + ansiReset
}

// ─── backward-compat aliases used by other files ─────────────────────────────

func bold(s string) string   { return cc(s, ansiBold) }
func dim(s string) string    { return cc(s, cDim) }
func green(s string) string  { return cc(s, cSuccess) }
func yellow(s string) string { return cc(s, cWarn) }
func muted(s string) string  { return cc(s, cMuted) }

// ─── banner ───────────────────────────────────────────────────────────────────

func printBanner(ver string) {
	const interior = 48 // visible chars between the two │ borders

	side := cc("│", cRule)
	top := "  " + cc("╭"+strings.Repeat("─", interior)+"╮", cRule)
	bot := "  " + cc("╰"+strings.Repeat("─", interior)+"╯", cRule)
	blank := "  " + side + strings.Repeat(" ", interior) + side

	// Compute padding using rune counts (not byte lengths) so multi-byte
	// Unicode characters (e.g. ·) don't break box alignment.
	runes := utf8.RuneCountInString
	wordmarkPlain := "Z E N I T H"
	verPlain := "v" + ver
	midPad := interior - 4 - runes(wordmarkPlain) - runes(verPlain) - 2
	if midPad < 1 {
		midPad = 1
	}
	lineWordmark := "  " + side + "    " +
		cc(wordmarkPlain, ansiBold+cBrand) +
		strings.Repeat(" ", midPad) +
		cc(verPlain, cVer) +
		"  " + side

	subPlain := "    local-first hybrid semantic search"
	lineSub := "  " + side + cc(subPlain, cMuted) +
		strings.Repeat(" ", interior-runes(subPlain)) + side

	techPlain := "    lex  ·  bk-tree  ·  vector  ·  rrf"
	lineTech := "  " + side + cc(techPlain, cDim) +
		strings.Repeat(" ", interior-runes(techPlain)) + side

	fmt.Println()
	fmt.Println(top)
	fmt.Println(blank)
	fmt.Println(lineWordmark)
	fmt.Println(lineSub)
	fmt.Println(lineTech)
	fmt.Println(blank)
	fmt.Println(bot)
	fmt.Println()
}

// ─── header ───────────────────────────────────────────────────────────────────

// printHeader prints the per-command header line.
//
//	⟡ search  "kubernetes pods"
func printHeader(command, subject string) {
	glyph := cc("⟡", cAccent)
	cmd := cc(command, ansiBold+cBrand)
	subj := cc(subject, ansiBold+cFile)
	fmt.Printf("\n  %s  %s  %s\n\n", glyph, cmd, subj)
}

// ─── divider ─────────────────────────────────────────────────────────────────

func printDivider() {
	fmt.Println(cc("  "+strings.Repeat("━", 66), cRule))
}

// ─── footer ───────────────────────────────────────────────────────────────────

// printFooter prints a muted summary line with · separators.
func printFooter(parts ...string) {
	sep := cc(" · ", cDim)
	joined := strings.Join(parts, sep)
	fmt.Printf("\n  %s\n\n", cc(joined, cMuted))
}

// ─── search results ───────────────────────────────────────────────────────────

// printResult prints one search result row with an inline score bar.
// topScore is used to scale the bar relative to the best result.
func printResult(rank int, id string, score, topScore float64) {
	name := filepath.Base(id)
	display := shortenPath(id)

	// Truncate before formatting so column widths are stable.
	if len(name) > 20 {
		name = name[:17] + "..."
	}
	if len(display) > 44 {
		display = "…" + display[len(display)-43:]
	}

	rankFmt := fmt.Sprintf("%2d", rank)
	nameFmt := fmt.Sprintf("%-20s", name)
	pathFmt := fmt.Sprintf("%-44s", display)
	scoreFmt := fmt.Sprintf("%.3f", score)

	ratio := 0.0
	if topScore > 0 {
		ratio = score / topScore
	}
	bar := buildScoreBar(ratio)
	barColor := scoreColor(ratio)

	fmt.Printf("  %s  %s  %s  %s  %s\n",
		cc(rankFmt, ansiBold+cFile),
		cc(nameFmt, ansiBold+cFile),
		cc(pathFmt, cPath),
		cc(bar, barColor),
		cc(scoreFmt, barColor),
	)
}

func buildScoreBar(ratio float64) string {
	const n = 5
	filled := int(math.Round(ratio * n))
	if filled < 0 {
		filled = 0
	}
	if filled > n {
		filled = n
	}
	return strings.Repeat("█", filled) + strings.Repeat("░", n-filled)
}

func scoreColor(ratio float64) string {
	if ratio >= 0.7 {
		return cHi
	}
	if ratio >= 0.4 {
		return cMid
	}
	return cLo
}


// ─── live index progress ─────────────────────────────────────────────────────

// printProgress overwrites the current line with a live file counter.
// Call clearProgress() before printing anything else.
func printProgress(n int64, filename string) {
	if !colorEnabled {
		return
	}
	if len(filename) > 28 {
		filename = "…" + filename[len(filename)-27:]
	}
	line := fmt.Sprintf("  %s  %s  %s",
		cc(fmt.Sprintf("%6d", n), ansiBold+cProg),
		cc("files", cMuted),
		cc(fmt.Sprintf("%-30s", filename), cPath),
	)
	fmt.Printf("%s\r", line)
}

// clearProgress erases the progress line so normal output can follow.
func clearProgress() {
	if !colorEnabled {
		return
	}
	fmt.Printf("\r%s\r", strings.Repeat(" ", 72))
}

// ─── utilities ────────────────────────────────────────────────────────────────

func shortenPath(p string) string {
	if home, err := os.UserHomeDir(); err == nil {
		if rel, err := filepath.Rel(home, p); err == nil && !strings.HasPrefix(rel, "..") {
			p = "~/" + rel
		}
	}
	return filepath.ToSlash(p)
}
