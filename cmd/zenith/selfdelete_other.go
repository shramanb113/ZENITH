//go:build !windows

package main

import "os"

func removeBinary(path string) error { return os.Remove(path) }
