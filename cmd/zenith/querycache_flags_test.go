package main

import (
	"testing"

	"github.com/spf13/cobra"
)

func newTestCommandForFlags(t *testing.T) *cobra.Command {
	t.Helper()
	cmd := &cobra.Command{Use: "test"}
	addEngineFlags(cmd)
	return cmd
}

func TestAddEngineFlags_RegistersQueryCacheFlags(t *testing.T) {
	cmd := newTestCommandForFlags(t)
	for _, name := range []string{
		"query-cache-size",
		"query-cache-ttl",
		"query-cache-redis-addr",
		"query-cache-semantic-threshold",
		"ann-threshold-band-pct",
	} {
		if cmd.Flags().Lookup(name) == nil {
			t.Errorf("flag %q not registered by addEngineFlags", name)
		}
	}
}
