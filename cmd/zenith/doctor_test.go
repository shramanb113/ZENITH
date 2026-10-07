package main

import "testing"

func TestDoctorCmd_HasDBFlag(t *testing.T) {
	if doctorCmd.Flags().Lookup("db") == nil {
		t.Fatal(`expected "db" flag to be registered on doctorCmd`)
	}
}
