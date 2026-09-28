//go:build windows

package main

import (
	"fmt"
	"os"
	"os/exec"
	"syscall"
)

// removeBinary deletes the binary. Windows refuses to delete a running .exe,
// so if the direct delete fails we hand the job to a detached cmd.exe that
// waits for this process to exit and then deletes it — the user never has to
// remove the file by hand.
func removeBinary(path string) error {
	if err := os.Remove(path); err == nil {
		return nil
	}
	script := fmt.Sprintf(`ping -n 3 127.0.0.1 >nul & del /f /q "%s" & del /f /q "%s.old"`, path, path)
	cmd := exec.Command("cmd", "/C", script)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x00000008 | 0x00000200} // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
	if err := cmd.Start(); err != nil {
		return err
	}
	_ = cmd.Process.Release()
	return nil
}
