package zenith

import "errors"

var (
	ErrClosed              = errors.New("zenith: DB is closed")
	ErrLocked              = errors.New("zenith: database is locked by another process or instance")
	ErrInvalidID           = errors.New("zenith: invalid document ID")
	ErrEmptyDocument       = errors.New("zenith: document text is empty")
	ErrIDTooLong           = errors.New("zenith: document ID exceeds maximum length (512 bytes)")
	ErrInvalidOption       = errors.New("zenith: invalid option value")
	ErrIndexFull           = errors.New("zenith: index has reached the configured memory limit")
	ErrIncompatibleVersion = errors.New("zenith: index file uses an older on-disk format — convert it with `zenith migrate <file>` (or zenith.Migrate), or delete it and re-index")
	ErrEmbedderMismatch    = errors.New("zenith: index file was written with a different embedder — delete it and re-index, or reopen with the original embedder")
)
