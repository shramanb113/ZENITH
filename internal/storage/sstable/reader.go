package sstable

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"

	"github.com/shramanb113/ZENITH/internal/storage/memtable"
)

var (
	ErrFooter = errors.New("sstable corrupt : Footer gone")
	ErrBlock  = errors.New("sstable corrupt : block CRC mismatch")
	ErrBloom  = errors.New("sstable: corrupt bloom filter")
	ErrHeader = errors.New("sstable : Header mismatch or corrupt")
	ErrIndex  = errors.New("sstable : corrupt Index")
)

type BloomReader struct {
	bits     []uint64
	numsBits uint32
	numsHash uint8
}

// [numBits(4)][numHash(1)][bitset bytes...]
func decode(data []byte) (*BloomReader, error) {
	if len(data) < 5 {
		return nil, ErrBloom
	}

	numBits := binary.LittleEndian.Uint32(data[0:4])
	numHash := data[4]
	bitsetBytes := data[5:]

	expectedWords := int((numBits + 63) / 64)
	if len(bitsetBytes) != expectedWords*8 {
		return nil, ErrBloom
	}

	bits := make([]uint64, expectedWords)
	for i := 0; i < expectedWords; i++ {
		bits[i] = binary.LittleEndian.Uint64(bitsetBytes[i*8:])
	}

	return &BloomReader{
		bits:     bits,
		numsBits: numBits,
		numsHash: numHash,
	}, nil
}

func (b *BloomReader) mayContain(key []byte) bool {
	h1, h2 := bloomHash(key)
	for i := uint8(0); i < b.numsHash; i++ {
		pos_i := (h1 + uint64(i)*h2) % uint64(b.numsBits)
		if b.bits[pos_i/64]&(1<<(pos_i%64)) == 0 {
			return false
		}
	}
	return true
}

type Reader struct {
	file   *os.File
	footer Footer
	index  []IndexEntry
	bloom  *BloomReader
}

// referring from the function from writeBloom in writer.go
func (r *Reader) readBloom() error {

	if r.footer.BloomSize == 0 {
		return ErrBloom
	}

	buf := make([]byte, r.footer.BloomSize)
	if _, err := r.file.ReadAt(buf, int64(r.footer.BloomOffset)); err != nil {
		return fmt.Errorf("sstable: read bloom: %w", err)
	}

	br, err := decode(buf)

	if err != nil {
		return err
	}

	r.bloom = br

	return nil

}

// referring from the function writeIndex in writer.go
func (r *Reader) readIndex() error {

	if r.footer.IndexSize == 0 {
		return ErrIndex
	}

	buf := make([]byte, r.footer.IndexSize)
	if _, err := r.file.ReadAt(buf, int64(r.footer.IndexOffset)); err != nil {
		return fmt.Errorf("sstable: read index: %w", err)
	}

	r.index = make([]IndexEntry, 0)
	pos := 0

	for pos < len(buf) {

		if pos+2 > len(buf) {
			return fmt.Errorf("Keylen corrupted in index")
		}

		keyLen := binary.LittleEndian.Uint16(buf[pos : pos+2])
		pos += 2

		// keylen(2) , key array(keylen which we extracted) , offset(8) + size (4)
		if pos+int(keyLen)+12 > len(buf) {
			return ErrIndex
		}

		lastKey := make([]byte, keyLen)
		copy(lastKey, buf[pos:pos+int(keyLen)])
		pos += int(keyLen)

		offset := binary.LittleEndian.Uint64(buf[pos : pos+8])
		pos += 8

		size := binary.LittleEndian.Uint32(buf[pos : pos+4])
		pos += 4

		r.index = append(r.index, IndexEntry{
			LastKey: lastKey,
			Offset:  offset,
			Size:    size,
		})

	}

	return nil

}

// similarly referring from writeFooter in writer.go
func (r *Reader) readFooter() error {

	info, err := r.file.Stat()

	if err != nil {
		return fmt.Errorf("sstable: stat: %w", err)
	}

	if info.Size() < FooterSize {
		return ErrFooter
	}

	buf := make([]byte, FooterSize)
	if _, err := r.file.ReadAt(buf, info.Size()-int64(FooterSize)); err != nil {
		return fmt.Errorf("sstable: read footer: %w", err)
	}

	if string(buf[32:36]) != MagicBytes {
		return ErrFooter
	}

	r.footer = Footer{
		IndexOffset: binary.LittleEndian.Uint64(buf[0:8]),
		IndexSize:   binary.LittleEndian.Uint32(buf[8:12]),
		BloomOffset: binary.LittleEndian.Uint64(buf[12:20]),
		BloomSize:   binary.LittleEndian.Uint32(buf[20:24]),
		EntryCount:  binary.LittleEndian.Uint64(buf[24:32]),
	}
	copy(r.footer.Magic[:], buf[32:36])

	return nil

}

func OpenReader(path string) (*Reader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("sstable: open: %w", err)
	}

	r := &Reader{file: f}

	if err := r.readFooter(); err != nil {
		f.Close()
		return nil, err
	}

	if err := r.readBloom(); err != nil {
		f.Close()
		return nil, err
	}

	if err := r.readIndex(); err != nil {
		f.Close()
		return nil, err
	}
	return r, nil
}

// ─── Get ──────────────────────────────────────────────────────────────────────

// Get returns the value for key and true if found and not a tombstone.
// Returns (nil, false) if the key is not present or has been deleted.

// binary searching the block (sorted is the advantage here)
func (r *Reader) findBlock(key []byte) int {

	lo, hi := 0, len(r.index)-1

	for lo <= hi {
		mid := int(lo + (hi-lo)/2)

		cmp := bytes.Compare(r.index[mid].LastKey, key)

		if cmp < 0 {
			lo = mid + 1
		} else {
			hi = mid - 1
		}

	}

	if lo >= len(r.index) {
		return -1 // key is beyond the largest key in the SSTable
	}
	return lo
}

type blockEntry struct {
	key     []byte
	value   []byte
	deleted bool
}

func (r *Reader) readBlock(entry IndexEntry) ([]blockEntry, error) {
	buf := make([]byte, entry.Size)

	// reading from the indexed
	if _, err := r.file.ReadAt(buf, int64(entry.Offset)); err != nil {
		if err == io.EOF {
			return nil, ErrBlock
		}

		return nil, fmt.Errorf("sstable: read block at %d: %w", entry.Offset, err)
	}

	if len(buf) < 6 {
		return nil, ErrBlock
	}

	storedCRC := binary.LittleEndian.Uint32(buf[0:4])
	entryCount := int(binary.LittleEndian.Uint16(buf[4:6]))
	payload := buf[6:]

	if crc32.ChecksumIEEE(payload) != storedCRC {
		return nil, ErrBlock
	}

	entries := make([]blockEntry, 0, entryCount)
	pos := 0

	for i := 0; i < entryCount; i++ {

		//  pos + 4 for storedCRC + 2 for block length
		if pos+7 > len(payload) {
			return nil, ErrBlock
		}

		keyLen := int(binary.LittleEndian.Uint16(payload[pos : pos+2]))
		valLen := int(binary.LittleEndian.Uint32(payload[pos+2 : pos+6]))
		deleted := payload[pos+6] == 1
		pos += 7

		if (pos + keyLen + valLen) > len(payload) {
			return nil, ErrBlock
		}

		key := make([]byte, keyLen)
		copy(key, payload[pos:pos+keyLen])
		pos += keyLen

		var value []byte
		if valLen > 0 {
			value = make([]byte, valLen)
			copy(value, payload[pos:pos+valLen])
		}
		pos += valLen

		entries = append(entries, blockEntry{
			key:     key,
			value:   value,
			deleted: deleted,
		})

	}
	return entries, nil
}

// Get returns the value for key and true if found and not a tombstone.
// Returns (nil, false, nil) if the key is genuinely absent. Returns a non-nil
// error if the block containing the key could not be read or failed its CRC
// check — a corrupt block must never be reported to the caller as "not
// found", since that silently resurrects a stale or already-deleted value
// from an older SSTable further down the read path.
func (r *Reader) Get(key []byte) ([]byte, bool, error) {

	// bloom filter search (if not found then we dont even need to scratch the disk)
	if !r.bloom.mayContain(key) {
		return nil, false, nil
	}

	blockIdx := r.findBlock(key)
	if blockIdx < 0 {
		return nil, false, nil
	}

	entries, err := r.readBlock(r.index[blockIdx])
	if err != nil {
		return nil, false, fmt.Errorf("sstable: get %q: %w", key, err)
	}

	for _, e := range entries {
		if bytes.Equal(e.key, key) {
			if e.deleted {
				// Tombstone — key was deleted, do not fall through
				return nil, false, nil
			}
			// Defensive copy — caller must not mutate stored data
			val := make([]byte, len(e.value))
			copy(val, e.value)
			return val, true, nil
		}
	}

	return nil, false, nil

}

func (r *Reader) Close() error {
	return r.file.Close()
}

// IterateAll returns every entry in the SSTable in lexicographic key order.
// Called by the compactor to perform k-way merges across multiple SSTables.
func (r *Reader) IterateAll() ([]memtable.Entry, error) {
	var all []memtable.Entry
	for _, idx := range r.index {
		blocks, err := r.readBlock(idx)
		if err != nil {
			return nil, fmt.Errorf("sstable: iterate block at offset %d: %w", idx.Offset, err)
		}
		for _, b := range blocks {
			key := make([]byte, len(b.key))
			copy(key, b.key)
			var val []byte
			if len(b.value) > 0 {
				val = make([]byte, len(b.value))
				copy(val, b.value)
			}
			all = append(all, memtable.Entry{
				Key:     key,
				Value:   val,
				Deleted: b.deleted,
			})
		}
	}
	return all, nil
}

// MaxKey returns the largest key in the SSTable (last key of the last index block).
func (r *Reader) MaxKey() []byte {
	if len(r.index) == 0 {
		return nil
	}
	src := r.index[len(r.index)-1].LastKey
	k := make([]byte, len(src))
	copy(k, src)
	return k
}

// MinKey returns the smallest key in the SSTable by reading the first entry of
// the first data block.
func (r *Reader) MinKey() ([]byte, error) {
	if len(r.index) == 0 {
		return nil, nil
	}
	blocks, err := r.readBlock(r.index[0])
	if err != nil {
		return nil, fmt.Errorf("sstable: read first block for min key: %w", err)
	}
	if len(blocks) == 0 {
		return nil, nil
	}
	k := make([]byte, len(blocks[0].key))
	copy(k, blocks[0].key)
	return k, nil
}

// Iterator streams entries from an SSTable one data block at a time, instead
// of materialising the whole file in memory the way IterateAll does. Used by
// the compactor's k-way merge so peak memory during compaction is bounded by
// (number of input files × one block) rather than the total size of every
// input file combined.
type Iterator struct {
	r        *Reader
	idxPos   int
	entries  []blockEntry
	entryPos int
}

// NewIterator returns a streaming iterator over every live and tombstoned
// entry in the SSTable, in key order.
func (r *Reader) NewIterator() *Iterator {
	return &Iterator{r: r}
}

// Next returns the next entry, or ok=false once the SSTable is exhausted.
func (it *Iterator) Next() (memtable.Entry, bool, error) {
	for it.entryPos >= len(it.entries) {
		if it.idxPos >= len(it.r.index) {
			return memtable.Entry{}, false, nil
		}
		entries, err := it.r.readBlock(it.r.index[it.idxPos])
		if err != nil {
			return memtable.Entry{}, false, fmt.Errorf("sstable: iterate block %d: %w", it.idxPos, err)
		}
		it.idxPos++
		it.entries = entries
		it.entryPos = 0
	}

	be := it.entries[it.entryPos]
	it.entryPos++

	key := make([]byte, len(be.key))
	copy(key, be.key)
	entry := memtable.Entry{Key: key, Deleted: be.deleted}
	if !be.deleted {
		entry.Value = make([]byte, len(be.value))
		copy(entry.Value, be.value)
	}
	return entry, true, nil
}
