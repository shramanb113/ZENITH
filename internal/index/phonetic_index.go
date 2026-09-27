package index

import "sync"

type PhoneticIndex struct {
	mu           sync.RWMutex
	phoneticData map[string][]uint64
}

func NewPhoneticIndex() *PhoneticIndex {
	return &PhoneticIndex{
		phoneticData: make(map[string][]uint64),
	}
}

func (pi *PhoneticIndex) RLock()   { pi.mu.RLock() }
func (pi *PhoneticIndex) RUnlock() { pi.mu.RUnlock() }
func (pi *PhoneticIndex) Lock()    { pi.mu.Lock() }
func (pi *PhoneticIndex) Unlock()  { pi.mu.Unlock() }

func (pi *PhoneticIndex) GetData() map[string][]uint64 { return pi.phoneticData }

// ReplaceAll atomically swaps the backing map for a freshly decoded one.
// Callers must hold pi.Lock() for the duration of the swap.
func (pi *PhoneticIndex) ReplaceAll(phoneticData map[string][]uint64) {
	pi.phoneticData = phoneticData
}
