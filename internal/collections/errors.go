package collections

import "errors"

var (
	ErrNotFound           = errors.New("collections: collection not found")
	ErrExists             = errors.New("collections: collection already exists")
	ErrInvalidID          = errors.New("collections: invalid collection id")
	ErrQuota              = errors.New("collections: document quota exceeded")
	ErrTooManyCollections = errors.New("collections: collection limit reached")
	ErrShuttingDown       = errors.New("collections: server is shutting down")
	ErrRootInUse          = errors.New("collections: root already managed in this process")
	ErrDocNotFound        = errors.New("collections: document not found")
)
