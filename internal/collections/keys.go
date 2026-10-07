package collections

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
)

// newKey generates a fresh collection API key: 32 random bytes, base64url
// encoded and prefixed, plus the hex-encoded SHA-256 hash that gets stored
// (the plaintext itself is never persisted).
func newKey() (plain string, sha256hex string, err error) {
	var raw [32]byte
	if _, err = rand.Read(raw[:]); err != nil {
		return "", "", err
	}
	plain = "zk_" + base64.RawURLEncoding.EncodeToString(raw[:])
	sum := sha256.Sum256([]byte(plain))
	sha256hex = hex.EncodeToString(sum[:])
	return plain, sha256hex, nil
}

// keyMatches reports whether presented hashes to storedHex, in constant time.
func keyMatches(presented, storedHex string) bool {
	sum := sha256.Sum256([]byte(presented))
	want, err := hex.DecodeString(storedHex)
	if err != nil || len(want) != len(sum) {
		return false
	}
	return subtle.ConstantTimeCompare(sum[:], want) == 1
}
