package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	rcBeginMarker = "# >>> zenith >>>"
	rcEndMarker   = "# <<< zenith <<<"
)

// pathContains reports whether the OS PATH-style list contains dir
// (case-insensitive on Windows, trailing separators ignored).
func pathContains(list, dir string) bool {
	sep := string(os.PathListSeparator)
	want := normPathEntry(dir)
	for _, e := range strings.Split(list, sep) {
		if e != "" && normPathEntry(e) == want {
			return true
		}
	}
	return false
}

func normPathEntry(p string) string {
	p = filepath.Clean(strings.TrimSpace(p))
	if runtime.GOOS == "windows" {
		p = strings.ToLower(p)
	}
	return p
}

// pathWithout returns list with every entry equal to dir removed.
func pathWithout(list, dir string) string {
	sep := string(os.PathListSeparator)
	want := normPathEntry(dir)
	var keep []string
	for _, e := range strings.Split(list, sep) {
		if e == "" || normPathEntry(e) == want {
			continue
		}
		keep = append(keep, e)
	}
	return strings.Join(keep, sep)
}

// rcBlock is the marked block written to a shell rc file.
func rcBlock(dir string) string {
	return fmt.Sprintf("%s\nexport PATH=\"%s:$PATH\"\n%s\n", rcBeginMarker, dir, rcEndMarker)
}

// addRCBlock returns content with the zenith PATH block present exactly once,
// replacing any previous block (so an install into a new directory updates it).
func addRCBlock(content, dir string) string {
	content = removeRCBlock(content)
	if content != "" && !strings.HasSuffix(content, "\n") {
		content += "\n"
	}
	return content + rcBlock(dir)
}

// removeRCBlock strips the zenith-marked block, leaving everything else intact.
func removeRCBlock(content string) string {
	start := strings.Index(content, rcBeginMarker)
	if start < 0 {
		return content
	}
	end := strings.Index(content[start:], rcEndMarker)
	if end < 0 {
		return content
	}
	end += start + len(rcEndMarker)
	if end < len(content) && content[end] == '\n' {
		end++
	}
	return content[:start] + content[end:]
}

// shellRCFile picks the rc file for the user's login shell.
func shellRCFile() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	switch filepath.Base(os.Getenv("SHELL")) {
	case "zsh":
		return filepath.Join(home, ".zshrc"), nil
	case "bash":
		if runtime.GOOS == "darwin" {
			return filepath.Join(home, ".bash_profile"), nil
		}
		return filepath.Join(home, ".bashrc"), nil
	default:
		return filepath.Join(home, ".profile"), nil
	}
}

// powershellUserPath runs a PowerShell snippet with dir exposed as
// $env:ZENITH_DIR (no quoting/escaping of the path is ever needed).
func powershellUserPath(script, dir string) (string, error) {
	cmd := exec.Command("powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	cmd.Env = append(os.Environ(), "ZENITH_DIR="+dir)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("powershell: %w: %s", err, strings.TrimSpace(string(out)))
	}
	return strings.TrimSpace(string(out)), nil
}

const psAddPath = `$d=$env:ZENITH_DIR;$p=[Environment]::GetEnvironmentVariable('Path','User');` +
	`$parts=@(); if($p){$parts=$p.Split(';')|Where-Object{$_ -ne ''}};` +
	`if($parts -contains $d){'unchanged'}else{` +
	`[Environment]::SetEnvironmentVariable('Path',(($parts+$d) -join ';'),'User');'added'}`

const psRemovePath = `$d=$env:ZENITH_DIR;$p=[Environment]::GetEnvironmentVariable('Path','User');` +
	`if(-not $p){'unchanged'}else{$parts=$p.Split(';');` +
	`if(-not ($parts -contains $d)){'unchanged'}else{` +
	`$new=(($parts|Where-Object{$_ -ne '' -and $_ -ne $d}) -join ';');` +
	`[Environment]::SetEnvironmentVariable('Path',$new,'User');'removed'}}`

// ensureOnPath makes dir available on future shells' PATH. It reports whether
// anything changed and a short instruction to show the user.
func ensureOnPath(dir string) (changed bool, note string, err error) {
	if runtime.GOOS == "windows" {
		res, err := powershellUserPath(psAddPath, dir)
		if err != nil {
			return false, "", err
		}
		if res == "added" {
			return true, "PATH updated for your user account — open a NEW terminal for it to take effect.", nil
		}
		return false, "", nil
	}

	if pathContains(os.Getenv("PATH"), dir) {
		return false, "", nil
	}
	rc, err := shellRCFile()
	if err != nil {
		return false, "", err
	}
	existing, _ := os.ReadFile(rc)
	updated := addRCBlock(string(existing), dir)
	if updated == string(existing) {
		return false, "", nil
	}
	if err := os.WriteFile(rc, []byte(updated), 0o644); err != nil {
		return false, "", err
	}
	return true, fmt.Sprintf("Added a PATH line to %s — run 'source %s' or open a new terminal.", rc, rc), nil
}

// removeFromPath undoes ensureOnPath. Errors are returned for reporting but
// are never fatal to an uninstall.
func removeFromPath(dir string) (changed bool, err error) {
	if runtime.GOOS == "windows" {
		res, err := powershellUserPath(psRemovePath, dir)
		if err != nil {
			return false, err
		}
		return res == "removed", nil
	}
	rc, err := shellRCFile()
	if err != nil {
		return false, err
	}
	existing, err := os.ReadFile(rc)
	if err != nil {
		return false, nil
	}
	updated := removeRCBlock(string(existing))
	if updated == string(existing) {
		return false, nil
	}
	return true, os.WriteFile(rc, []byte(updated), 0o644)
}

func runVersion(bin string) (string, error) {
	out, err := exec.Command(bin, "version").CombinedOutput()
	return string(out), err
}
