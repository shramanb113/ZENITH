package crawler

import (
	"context"

	"github.com/fsnotify/fsnotify"
)

// SimulateEvent injects a synthetic fsnotify event for testing, from the
// external crawler_test package.
func (w *Watcher) SimulateEvent(ctx context.Context, event fsnotify.Event) {
	w.handleEvent(ctx, event)
}
