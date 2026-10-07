package server

import (
	"fmt"
	"net"
)

// IsLoopbackHost reports whether host is a loopback address or "localhost".
// An empty host (e.g. ":7700") is NOT loopback — it means "all interfaces".
func IsLoopbackHost(host string) bool {
	if host == "" {
		return false
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// ResolveListenAddr fills in a missing host in addr (e.g. ":7700") with bind,
// leaving an addr that already specifies a host untouched.
func ResolveListenAddr(addr, bind string) (string, error) {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return "", fmt.Errorf("invalid listen address %q: %w", addr, err)
	}
	if host == "" {
		host = bind
	}
	return net.JoinHostPort(host, port), nil
}

// CheckExposure refuses to listen on a non-loopback address with no shared
// key and no explicit opt-in, so a server is never silently exposed
// unauthenticated on a network interface.
func CheckExposure(hostport, key string, allowUnauth bool) error {
	host, _, err := net.SplitHostPort(hostport)
	if err != nil {
		return fmt.Errorf("invalid listen address %q: %w", hostport, err)
	}
	if IsLoopbackHost(host) || key != "" || allowUnauth {
		return nil
	}
	return fmt.Errorf("refusing to listen on %s without a key: set --key / ZENITH_KEY, or pass --allow-unauthenticated", hostport)
}
