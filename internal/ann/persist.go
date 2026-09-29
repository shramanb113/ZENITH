package ann

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
)

// Persistence. Rebuilding the graph on every open costs minutes at a million
// vectors, so the graph (links only — the vectors already live in the segments)
// is written to a sidecar file and read back at open.
//
// The file is a *hint*, never the truth. The engine's segments are the truth,
// and the graph may be older than they are: documents were added, replaced or
// deleted since it was written. Each node therefore carries a tag chosen by the
// engine (the segment generation its vector lived in when the graph was
// written). At load the engine binds every live vector to its node by ID and
// compares tags: a match means the node's links were built for exactly that
// vector; a mismatch (the document was replaced) or a missing node (added since)
// re-inserts that document; a node no live vector claims is tombstoned. So a
// stale file yields a slightly less tidy graph, never a wrong answer.
//
// Writing does not stop the engine: the graph is copied in short chunks, each
// under the engine's read lock (WriteOptions.Lock), and nodes added while
// writing are simply left out — they are picked up as "missing" at load.

const (
	fileMagic   = "ZANN"
	fileVersion = 1
)

var crcTable = crc32.MakeTable(crc32.Castagnoli)

// WriteOptions control how a live graph is snapshotted.
type WriteOptions struct {
	// Chunk is how many nodes are copied per lock hold (default 1024).
	Chunk int
	// Lock acquires whatever protects the graph from mutation (the engine's read
	// lock) and returns the release function. nil means the caller guarantees
	// the graph is not being mutated.
	Lock func() (unlock func())
	// Tag returns the tag of a node's vector and whether the node may be
	// written; it is called under Lock. A node it rejects (its vector is not in
	// a segment yet) is left out.
	Tag func(id uint64) (tag uint64, ok bool)
	// Identity is opaque bytes stored in the header and returned by Read, so the
	// engine can refuse a file built for another embedder or dimension.
	Identity []byte
}

// WriteTo writes a snapshot of x to w. Dead nodes and rejected nodes are
// dropped, links to them are filtered out and the rest renumbered, so the file
// holds a compact graph. Returns the number of nodes written.
func (x *Index) WriteTo(w io.Writer, opt WriteOptions) (int, error) {
	chunk := opt.Chunk
	if chunk <= 0 {
		chunk = 1024
	}
	lock := opt.Lock
	if lock == nil {
		lock = func() func() { return func() {} }
	}

	// Pass 1: decide which nodes are written and their new numbers.
	unlock := lock()
	n0 := len(x.ids)
	unlock()
	newIdx := make([]int32, n0)
	tags := make([]uint64, 0, n0)
	kept := 0
	for lo := 0; lo < n0; lo += chunk {
		unlock := lock()
		hi := min(lo+chunk, n0)
		for i := lo; i < hi; i++ {
			newIdx[i] = -1
			if x.dead[i] {
				continue
			}
			if tag, ok := opt.Tag(x.ids[i]); ok {
				newIdx[i] = int32(kept)
				tags = append(tags, tag)
				kept++
			}
		}
		unlock()
	}

	crc := crc32.New(crcTable)
	bw := bufio.NewWriterSize(io.MultiWriter(w, crc), 1<<20)
	var scratch [binary.MaxVarintLen64]byte
	putU := func(v uint64) {
		n := binary.PutUvarint(scratch[:], v)
		bw.Write(scratch[:n])
	}

	// The header carries the entry point (the survivor with the most levels), so
	// it is found before the nodes are streamed out.
	entry, maxLevel := int32(-1), -1
	for lo := 0; lo < n0; lo += chunk {
		unlock := lock()
		hi := min(lo+chunk, n0)
		for i := lo; i < hi; i++ {
			if newIdx[i] < 0 {
				continue
			}
			lv := len(x.links[i]) - 1
			if lv > maxLevel {
				entry, maxLevel = newIdx[i], lv
			}
		}
		unlock()
	}

	bw.WriteString(fileMagic)
	putU(fileVersion)
	putU(uint64(x.m))
	putU(uint64(x.efConstruction))
	putU(uint64(kept))
	putU(uint64(entry + 1)) // 0 = empty graph
	putU(uint64(maxLevel + 1))
	putU(uint64(len(opt.Identity)))
	bw.Write(opt.Identity)

	// Pass 2: each node's id, tag and (filtered, renumbered) links.
	buf := make([][][]int32, 0, chunk)
	ids := make([]uint64, 0, chunk)
	for lo := 0; lo < n0; lo += chunk {
		buf, ids = buf[:0], ids[:0]
		unlock := lock()
		hi := min(lo+chunk, n0)
		for i := lo; i < hi; i++ {
			if newIdx[i] < 0 {
				continue
			}
			ids = append(ids, x.ids[i])
			lv := make([][]int32, len(x.links[i]))
			for l, nbs := range x.links[i] {
				out := make([]int32, 0, len(nbs))
				for _, nb := range nbs {
					if int(nb) < n0 && newIdx[nb] >= 0 {
						out = append(out, newIdx[nb])
					}
				}
				lv[l] = out
			}
			buf = append(buf, lv)
		}
		unlock()
		// Node numbers are assigned in ascending original order, so the k-th node
		// of this chunk has new number (first new number of the chunk)+k, and the
		// tags slice is indexed by new number.
		first := -1
		for i := lo; i < hi; i++ {
			if newIdx[i] >= 0 {
				first = int(newIdx[i])
				break
			}
		}
		for k, lv := range buf {
			nodeNo := first + k
			putU(ids[k])
			putU(tags[nodeNo])
			putU(uint64(len(lv)))
			for _, nbs := range lv {
				putU(uint64(len(nbs)))
				for _, nb := range nbs {
					putU(uint64(nb))
				}
			}
		}
	}
	if err := bw.Flush(); err != nil {
		return 0, err
	}
	var foot [4]byte
	binary.LittleEndian.PutUint32(foot[:], crc.Sum32())
	if _, err := w.Write(foot[:]); err != nil {
		return 0, err
	}
	return kept, nil
}

// ErrBadFile is returned by Read for a file that is truncated, corrupt or was
// written by an incompatible version.
var ErrBadFile = errors.New("ann: unreadable graph file")

// Read parses a graph written by WriteTo. The returned index has no vectors:
// the caller must bind each with SetVec (checking NodeTag) and then call
// Finish. identity is what the writer stored in WriteOptions.Identity.
func Read(data []byte) (x *Index, identity []byte, err error) {
	if len(data) < len(fileMagic)+4 {
		return nil, nil, fmt.Errorf("%w: too short", ErrBadFile)
	}
	body, foot := data[:len(data)-4], data[len(data)-4:]
	if crc32.Checksum(body, crcTable) != binary.LittleEndian.Uint32(foot) {
		return nil, nil, fmt.Errorf("%w: checksum mismatch", ErrBadFile)
	}
	if string(body[:4]) != fileMagic {
		return nil, nil, fmt.Errorf("%w: bad magic", ErrBadFile)
	}
	p := body[4:]
	next := func() (uint64, bool) {
		v, n := binary.Uvarint(p)
		if n <= 0 {
			return 0, false
		}
		p = p[n:]
		return v, true
	}
	var hdr [7]uint64
	for i := range hdr {
		v, ok := next()
		if !ok {
			return nil, nil, fmt.Errorf("%w: truncated header", ErrBadFile)
		}
		hdr[i] = v
	}
	if hdr[0] != fileVersion {
		return nil, nil, fmt.Errorf("%w: version %d", ErrBadFile, hdr[0])
	}
	m, efc, count, entry1, maxLevel1, idLen := int(hdr[1]), int(hdr[2]), int(hdr[3]), int(hdr[4]), int(hdr[5]), int(hdr[6])
	if idLen < 0 || idLen > len(p) || count < 0 || count > len(p) {
		return nil, nil, fmt.Errorf("%w: implausible header", ErrBadFile)
	}
	identity = append([]byte(nil), p[:idLen]...)
	p = p[idLen:]

	x = New(m, efc)
	x.ids = make([]uint64, count)
	x.vecs = make([][]uint16, count)
	x.dead = make([]bool, count)
	x.links = make([][][]int32, count)
	x.tags = make([]uint64, count)
	x.idx = make(map[uint64]int32, count)
	// Links are carved out of shared blocks: a million nodes must not mean
	// millions of tiny allocations.
	var arena []int32
	carve := func(n int) []int32 {
		if n > len(arena) {
			arena = make([]int32, max(n, 1<<16))
		}
		s := arena[:n:n]
		arena = arena[n:]
		return s
	}
	for i := 0; i < count; i++ {
		id, ok1 := next()
		tag, ok2 := next()
		nl, ok3 := next()
		if !ok1 || !ok2 || !ok3 || nl == 0 || nl > 64 {
			return nil, nil, fmt.Errorf("%w: node %d", ErrBadFile, i)
		}
		x.ids[i], x.tags[i] = id, tag
		x.idx[id] = int32(i)
		lv := make([][]int32, nl)
		for l := range lv {
			c, ok := next()
			if !ok || int(c) > len(p) {
				return nil, nil, fmt.Errorf("%w: node %d links", ErrBadFile, i)
			}
			nbs := carve(int(c))
			for j := range nbs {
				v, ok := next()
				if !ok || int(v) >= count {
					return nil, nil, fmt.Errorf("%w: node %d neighbour", ErrBadFile, i)
				}
				nbs[j] = int32(v)
			}
			lv[l] = nbs
		}
		x.links[i] = lv
	}
	x.live = count
	x.entry = int32(entry1 - 1)
	x.maxLevel = maxLevel1 - 1
	x.bound = make([]bool, count)
	return x, identity, nil
}

// NodeTag returns the tag recorded for id in a graph returned by Read.
func (x *Index) NodeTag(id uint64) (tag uint64, ok bool) {
	n, ok := x.idx[id]
	if !ok || int(n) >= len(x.tags) {
		return 0, false
	}
	return x.tags[n], true
}

// SetVec binds a loaded node to its vector.
func (x *Index) SetVec(id uint64, v []uint16) {
	if n, ok := x.idx[id]; ok {
		x.vecs[n] = v
		if int(n) < len(x.bound) {
			x.bound[n] = true
		}
	}
}

// Finish completes a graph returned by Read once every live vector has been
// bound: nodes nobody claimed are tombstoned (they were deleted or replaced
// since the file was written) and the entry point is repaired if it was one of
// them. It returns how many nodes were tombstoned.
func (x *Index) Finish() int {
	gone := 0
	for n := range x.ids {
		if x.dead[n] || (n < len(x.bound) && x.bound[n]) {
			continue
		}
		x.dead[n] = true
		delete(x.idx, x.ids[n])
		x.live--
		gone++
	}
	x.bound, x.tags = nil, nil
	if x.entry >= 0 && int(x.entry) < len(x.dead) && x.dead[x.entry] {
		best, bestLevel := int32(-1), -1
		for n := range x.ids {
			if !x.dead[n] && len(x.links[n])-1 > bestLevel {
				best, bestLevel = int32(n), len(x.links[n])-1
			}
		}
		x.entry, x.maxLevel = best, bestLevel
	}
	return gone
}
