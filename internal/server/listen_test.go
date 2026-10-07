package server

import "testing"

func TestResolveListenAddr(t *testing.T) {
	tests := []struct {
		addr, bind, want string
	}{
		{":7700", "127.0.0.1", "127.0.0.1:7700"},
		{"0.0.0.0:7700", "127.0.0.1", "0.0.0.0:7700"},
		{"[::1]:1", "x", "[::1]:1"},
	}
	for _, tt := range tests {
		got, err := ResolveListenAddr(tt.addr, tt.bind)
		if err != nil {
			t.Fatalf("ResolveListenAddr(%q, %q): unexpected error: %v", tt.addr, tt.bind, err)
		}
		if got != tt.want {
			t.Fatalf("ResolveListenAddr(%q, %q) = %q, want %q", tt.addr, tt.bind, got, tt.want)
		}
	}
}

func TestCheckExposure(t *testing.T) {
	tests := []struct {
		name        string
		hostport    string
		key         string
		allowUnauth bool
		wantErr     bool
	}{
		{"loopback, no key", "127.0.0.1:7700", "", false, false},
		{"localhost, no key", "localhost:7700", "", false, false},
		{"0.0.0.0, no key, not allowed", "0.0.0.0:7700", "", false, true},
		{"0.0.0.0, with key", "0.0.0.0:7700", "k", false, false},
		{"0.0.0.0, no key, allowed", "0.0.0.0:7700", "", true, false},
		{"empty host counts as non-loopback", ":7700", "", false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := CheckExposure(tt.hostport, tt.key, tt.allowUnauth)
			if tt.wantErr && err == nil {
				t.Fatal("expected an error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
