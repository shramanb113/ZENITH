package wal

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

// Write Ahead Log

type SyncMode int

const (
	SyncAlways SyncMode = iota
	SyncPeriodic
	SyncGroupCommit
)

type CompressionCodec int

const (
	CodecNone CompressionCodec = iota
	CodecSnappy
	CodecZstd
)

type WALConfig struct {
	SyncMode          SyncMode
	SyncInterval      time.Duration
	GroupCommitWindow time.Duration
	MaxSegmentSize    int64
	Dir               string
	Codec             CompressionCodec
	ZstdLevel         int
}

type WAL struct {
	cfg         WALConfig
	mu          sync.Mutex
	seq         atomic.Uint64
	closed      atomic.Bool
	file        *os.File
	buf         *bufio.Writer
	syncCh      chan struct{}
	syncDone    chan struct{}
	byteWritten uint64
}

type OpType byte

const (
	OpTypePut OpType = iota + 1
	OpTypeDelete
)

func (op OpType) Valid() bool {
	switch op {
	case OpTypePut, OpTypeDelete:
		return true
	default:
		return false
	}
}

type Record struct {
	Seq   uint64
	Op    OpType
	Key   []byte
	Value []byte
}

func (r *Record) Validate() error {
	if !r.Op.Valid() {
		return errors.New("invalid op type")
	}
	if r.Op == OpTypeDelete && len(r.Value) != 0 {
		return errors.New("delete should not have value")
	}
	return nil
}

func encodeRecord(r *Record) ([]byte, error) {
	if r == nil {
		return nil, errors.New("nil record")
	}

	if err := r.Validate(); err != nil {
		return nil, err
	}

	if len(r.Key) > math.MaxUint16 {
		return nil, errors.New("key too large: max 65535 bytes")
	}

	if len(r.Value) > math.MaxUint32 {
		return nil, errors.New("value too large")
	}

	const hdrSize = 8 + 1 + 2 + 4 // seq + op + keyLen + valLen
	totalSize := hdrSize + len(r.Key) + len(r.Value)

	buf := make([]byte, totalSize)
	offset := 0

	binary.LittleEndian.PutUint64(buf[offset:], r.Seq)
	offset += 8

	buf[offset] = byte(r.Op)
	offset += 1

	binary.LittleEndian.PutUint16(buf[offset:], uint16(len(r.Key)))
	offset += 2

	binary.LittleEndian.PutUint32(buf[offset:], uint32(len(r.Value)))
	offset += 4

	copy(buf[offset:], r.Key)
	offset += len(r.Key)

	copy(buf[offset:], r.Value)

	return buf, nil
}

// Entry wire format: [CRC32 (4)] [LENGTH (4)] [BODY (LENGTH)]
// CRC covers [LENGTH + BODY] so a corrupt length is also detected.

const (
	crcSize    = 4
	lenSize    = 4
	headerSize = crcSize + lenSize
)

func buildEntry(body []byte) ([]byte, error) {
	if len(body) == 0 {
		return nil, errors.New("empty body")
	}

	if len(body) > math.MaxUint32 {
		return nil, errors.New("body too large")
	}

	totalSize := headerSize + len(body)
	buf := make([]byte, totalSize)

	binary.LittleEndian.PutUint32(buf[crcSize:], uint32(len(body)))
	copy(buf[headerSize:], body)

	// CRC covers [length field + body]
	payload := buf[crcSize:]
	crc := crc32.ChecksumIEEE(payload)
	binary.LittleEndian.PutUint32(buf[0:], crc)

	return buf, nil
}

func decodeRecord(body []byte) (Record, error) {
	var r Record
	bo := 0

	if len(body) < 15 {
		return r, errors.New("body too small")
	}

	r.Seq = binary.LittleEndian.Uint64(body[bo:])
	bo += 8

	r.Op = OpType(body[bo])
	bo += 1

	keyLen := binary.LittleEndian.Uint16(body[bo:])
	bo += 2

	valLen := binary.LittleEndian.Uint32(body[bo:])
	bo += 4

	if int(keyLen)+int(valLen) != len(body)-bo {
		return r, errors.New("invalid key/value length")
	}

	r.Key = make([]byte, keyLen)
	copy(r.Key, body[bo:bo+int(keyLen)])
	bo += int(keyLen)

	r.Value = make([]byte, valLen)
	copy(r.Value, body[bo:bo+int(valLen)])

	if err := r.Validate(); err != nil {
		return r, err
	}

	return r, nil
}

func (w *WAL) Append(ctx context.Context, r *Record) (uint64, error) {
	if w.closed.Load() {
		return 0, errors.New("wal is closed")
	}

	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	default:
	}

	if err := r.Validate(); err != nil {
		return 0, err
	}

	body, err := encodeRecord(r)
	if err != nil {

		// Rollback if error is found
		w.seq.Add(^uint64(0))
		return 0, err
	}

	entry, err := buildEntry(body)
	if err != nil {
		w.seq.Add(^uint64(0))
		return 0, err
	}

	w.mu.Lock()

	r.Seq = w.seq.Add(1)

	n, err := w.buf.Write(entry)
	if err != nil {
		w.mu.Unlock()
		return 0, err
	}
	if n != len(entry) {
		w.mu.Unlock()
		return 0, errors.New("partial write")
	}

	w.byteWritten += uint64(len(entry))

	if w.cfg.SyncMode == SyncAlways {
		if err := w.buf.Flush(); err != nil {
			w.mu.Unlock()
			return 0, err
		}
		if err := w.file.Sync(); err != nil {
			w.mu.Unlock()
			return 0, err
		}
	}

	w.mu.Unlock()

	return r.Seq, nil
}

// Recover reads all valid records from file, returning:
//   - the slice of decoded records in the order they were written
//   - the byte offset of the last valid record (use to truncate a corrupt tail)
//   - any hard I/O error (CRC mismatches and torn tails are NOT errors — they
//     just stop iteration and truncate; a genuine I/O error is propagated so
//     the caller fails loudly instead of silently losing acknowledged writes)
//
// A "torn tail" — an incomplete final record left by a crash mid-write — is
// distinguished from a large-but-valid record by comparing the record's
// declared length against the bytes physically remaining in the file, not
// against a fixed byte ceiling. A record legitimately larger than any fixed
// ceiling (e.g. a big document) must never be treated as corruption.
func Recover(file *os.File) ([]Record, int64, error) {
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return nil, 0, err
	}

	info, err := file.Stat()
	if err != nil {
		return nil, 0, err
	}
	fileSize := info.Size()

	var records []Record
	var offset int64

	reader := bufio.NewReader(file)

	for {
		remaining := fileSize - offset
		if remaining <= 0 {
			break
		}

		// --- CRC ---
		crcBuf := make([]byte, crcSize)
		_, err := io.ReadFull(reader, crcBuf)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			if errors.Is(err, io.ErrUnexpectedEOF) {
				// Torn tail: fewer bytes than a header remain.
				return records, offset, nil
			}
			return records, offset, fmt.Errorf("wal: read entry header: %w", err)
		}
		storedCRC := binary.LittleEndian.Uint32(crcBuf)

		// --- LENGTH ---
		lenBuf := make([]byte, lenSize)
		if _, err := io.ReadFull(reader, lenBuf); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return records, offset, nil
			}
			return records, offset, fmt.Errorf("wal: read entry length: %w", err)
		}
		length := binary.LittleEndian.Uint32(lenBuf)

		if length == 0 {
			// A zero-length record can never be valid — torn/corrupt tail.
			return records, offset, nil
		}

		// A record whose declared length claims more bytes than physically
		// remain in the file cannot possibly be complete: this is a torn
		// tail, regardless of how large the declared length is.
		if int64(headerSize)+int64(length) > remaining {
			return records, offset, nil
		}

		// --- BODY ---
		body := make([]byte, length)
		if _, err := io.ReadFull(reader, body); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return records, offset, nil
			}
			return records, offset, fmt.Errorf("wal: read entry body: %w", err)
		}

		// --- CRC CHECK ---
		h := crc32.NewIEEE()
		h.Write(lenBuf)
		h.Write(body)
		if h.Sum32() != storedCRC {
			// CRC mismatch on an otherwise complete-looking record: treat as
			// a torn/corrupted tail and stop here, same as a physical torn
			// write — only the tail of an append-only log can be corrupt.
			return records, offset, nil
		}

		// --- DECODE ---
		rec, err := decodeRecord(body)
		if err != nil {
			return records, offset, nil
		}

		records = append(records, rec)
		offset += int64(crcSize + lenSize + length)
	}

	return records, offset, nil
}

func OpenWAL(path string, cfg WALConfig) (*WAL, []Record, error) {
	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return nil, nil, err
	}

	if cfg.SyncMode == SyncPeriodic || cfg.SyncMode == SyncGroupCommit {
		file.Close()
		return nil, nil, errors.New("SyncPeriodic and SyncGroupCommit not yet implemented")
	}

	// Recover seeks to 0 internally (Bug 6 fix), so no explicit seek needed here.
	records, offset, err := Recover(file)
	if err != nil {
		file.Close()
		return nil, nil, err
	}

	if err := file.Truncate(offset); err != nil {
		file.Close()
		return nil, nil, err
	}

	if _, err := file.Seek(offset, io.SeekStart); err != nil {
		file.Close()
		return nil, nil, err
	}

	if err := file.Sync(); err != nil {
		file.Close()
		return nil, nil, err
	}

	var maxSeq uint64
	for _, r := range records {
		if r.Seq > maxSeq {
			maxSeq = r.Seq
		}
	}

	var seq atomic.Uint64
	seq.Store(maxSeq)

	wal := &WAL{
		file:        file,
		buf:         bufio.NewWriterSize(file, 64*1024),
		byteWritten: uint64(offset),
		cfg: WALConfig{
			Dir: filepath.Dir(path),
		},
	}

	// Store directly on the heap-allocated struct — no copy involved
	wal.seq.Store(maxSeq)

	return wal, records, nil
}

// AppendBatch writes all records to the WAL and performs a single fsync at the
// end, regardless of batch size. This is significantly faster than calling
// Append in a loop when SyncAlways is set, because N records pay for exactly
// one fsync instead of N. Returns the assigned sequence number for each record
// in the same order. An empty or nil batch is a no-op returning nil, nil.
func (w *WAL) AppendBatch(ctx context.Context, records []*Record) ([]uint64, error) {
	if len(records) == 0 {
		return nil, nil
	}
	if w.closed.Load() {
		return nil, errors.New("wal is closed")
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Encode all records before acquiring the lock so we hold it as briefly as possible.
	entries := make([][]byte, 0, len(records))
	for _, r := range records {
		if err := r.Validate(); err != nil {
			return nil, err
		}
		body, err := encodeRecord(r)
		if err != nil {
			return nil, err
		}
		entry, err := buildEntry(body)
		if err != nil {
			return nil, err
		}
		entries = append(entries, entry)
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	seqs := make([]uint64, len(records))
	for i, entry := range entries {
		records[i].Seq = w.seq.Add(1)
		seqs[i] = records[i].Seq

		n, err := w.buf.Write(entry)
		if err != nil {
			return nil, err
		}
		if n != len(entry) {
			return nil, errors.New("partial write")
		}
		w.byteWritten += uint64(len(entry))
	}

	// Single fsync for the entire batch.
	if w.cfg.SyncMode == SyncAlways {
		if err := w.buf.Flush(); err != nil {
			return nil, err
		}
		if err := w.file.Sync(); err != nil {
			return nil, err
		}
	}

	return seqs, nil
}

// Size returns the number of bytes written to the WAL since it was opened
// or last Reset, including buffered-but-not-yet-flushed data. Callers use it
// to decide when to checkpoint and bound how much a crash needs to replay.
func (w *WAL) Size() uint64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.byteWritten
}

// Reset flushes, syncs, and truncates the WAL file to zero, then resets
// internal state so the WAL can accept new records. Called after a
// successful gob checkpoint — the delta journal is no longer needed.
func (w *WAL) Reset() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.buf.Flush(); err != nil {
		return err
	}
	if err := w.file.Truncate(0); err != nil {
		return err
	}
	if _, err := w.file.Seek(0, io.SeekStart); err != nil {
		return err
	}
	// Sync AFTER truncating (not before): what must be durable is the
	// truncation itself, so a crash right after Reset can't come back to a
	// half-truncated WAL sitting alongside an already-durable snapshot.
	if err := w.file.Sync(); err != nil {
		return err
	}
	w.buf.Reset(w.file)
	w.byteWritten = 0
	w.seq.Store(0)
	return nil
}

func (w *WAL) Close() error {
	if !w.closed.CompareAndSwap(false, true) {
		return nil
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.buf.Flush(); err != nil {
		return err
	}

	if err := w.file.Sync(); err != nil {
		return err
	}

	return w.file.Close()
}
