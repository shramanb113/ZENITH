package collections

import "regexp"

// idRE: lowercase only, to avoid case-insensitive filesystem collisions
// (Windows/macOS) between two ids that a case-sensitive one would keep apart.
var idRE = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,62}$`)

// ValidID reports whether id is safe to use as a collection directory name:
// it must match idRE and must not be a Windows reserved device name (an
// exact match is reserved even with an extension elsewhere, but here the id
// is used bare as a directory name, so a plain equality check is enough).
func ValidID(id string) bool {
	if !idRE.MatchString(id) {
		return false
	}
	switch id {
	case "con", "prn", "aux", "nul",
		"com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
		"lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9":
		return false
	}
	return true
}
