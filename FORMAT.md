# ZENITH on-disk format and compatibility policy

This document is the contract for the files ZENITH writes. It answers two
questions a user of an embedded engine should be able to ask before adopting it:
*what exactly is on disk*, and *what happens to my index when I upgrade*.

## What is on disk

An index named `zenith.db` is a small **manifest** plus one or more immutable
**segment** files beside it:

```
zenith.db                 manifest: which segments make up the index (a few hundred bytes)
zenith.db.seg-000001      immutable segment (documents, postings, vectors …)
zenith.db.seg-000004      a later flush; segment numbers only grow
zenith.db.ann             the HNSW graph, so opening a large index does not rebuild it (optional)
zenith.db.wal             write-ahead journal of changes not yet flushed (pkg/zenith / server)
zenith.db.wal.old-000007  a journal a checkpoint has cut but not yet made redundant (transient)
zenith.db.lock            single-writer lock (pkg/zenith)
```

Everything except the `.wal`, `.wal.old-*` and `.lock` is described here; delete
nothing by hand — `zenith compact` and open-time cleanup manage the files.

### Graph file (`zenith.db.ann`)

A *hint*, never the truth: the segments are authoritative, and a missing, damaged,
foreign or stale graph file only costs a rebuild at open — it can never change an
answer. It holds the HNSW links (not the vectors, which already live in the
segments): a small header (graph parameters, and an identity string naming the
embedder and dimension it was built for), then per node its document ID, the
generation of the segment its vector lived in when the file was written, and its
links per layer; a CRC-32C closes the file. At open every live vector is bound to
its node by document ID and generation: a match reuses the node's links, a
document added or replaced since (different or missing generation) is inserted
into the graph afterwards, and a node no live vector claims is dropped. If more
than half of the documents would need inserting, the file is ignored and the graph
is rebuilt. The file is rewritten after a compaction, on a clean close, and in the
background once 50,000 graph changes have accumulated since the last write — never
on every flush, whose cost stays proportional to the delta.

### Journal (`zenith.db.wal`, `zenith.db.wal.old-*`)

Every add and delete is appended to the journal and `fsync`ed before it is
acknowledged. A checkpoint (a flush of the in-memory delta to a new segment) cuts
the journal at one instant: everything acknowledged so far becomes the archive
`zenith.db.wal.old-NNNNNN` and a fresh journal takes over, while the delta is
frozen and written as a segment *without* blocking searches or writes. Only after
the manifest commit are the archives deleted. Recovery replays archives oldest
first, then the live journal, on top of the last committed manifest; replaying a
document that already is in a segment is harmless (adding is idempotent, and a
later delete in the journal still wins).

### Manifest (format version 6)

Multi-byte integers are big-endian (so any version's header can be read the same
way):

| Field | Size | Meaning |
|-------|------|---------|
| magic | 4 | `ZNTH` |
| version | u16 | `6` |
| embedder name | u32 length + bytes | model that produced the vectors, e.g. `onnx:all-MiniLM-L6-v2` |
| dims | u32 | vector dimensionality |
| body length | u32 | |
| body | JSON | `generation`, `next_gen`, and the ordered list of segments (`file`, `gen`, `docs`, `bytes`) |
| CRC-32C | u32 | of the body |

### Segment (layout version 1)

Little-endian. Opened by memory-mapping the file read-only, so opening costs
O(sections) regardless of size and the data lives in the OS page cache rather
than the Go heap. Every section is 8-byte aligned so fixed-width arrays are read
in place.

```
[ header 64 B ][ section ]…[ section ][ directory ][ footer 16 B ]
```

Header: magic `ZNSG`, layout version, vector dims, and the row counts. The footer
locates the directory; each directory entry gives a section's kind, offset,
length and CRC-32C; the footer also carries a CRC over header + directory.

Sections (kinds are part of the format and are never renumbered): sorted document
IDs; document original-ID and text blobs; BM25 document lengths; float16 document
vectors; the BM25 term dictionary with delta-varint postings and per-document
forward index; edge-n-gram fragment and phonetic-code postings; float16 word
vectors; the IDs this segment *deletes* from older segments; document attributes;
opaque metadata.

### Why it is safe to crash

A flush or compaction is committed in two steps:

1. Write the new segment file completely and `fsync` it.
2. Atomically replace the manifest (write a temp file, `fsync`, rename, sync the
   directory).

The manifest is the only thing that decides what the index *is*. A crash before
step 2 leaves the previous manifest — and therefore the previous, complete
index — untouched; the half-finished segment is an orphan that is deleted the
next time the index is opened. A crash after step 2 leaves the new index. There
is no state in between. Deletions are recorded in the *next* segment (a
`deletes` section) and applied to older segments when the index is opened.

`zenith doctor` re-verifies every section checksum; a torn or bit-rotted segment
is reported instead of being searched.

This is tested, not just argued:

- `internal/index/crash_test.go` kills a child process at named points inside
  flush and compaction (before and after the manifest commit) and, separately,
  hard-kills a busy writer at random moments (`kill -9`), then checks that the
  index opens, every acknowledged document is present with its exact text, and no
  acknowledged deletion reappears.
- `kill -9` cannot lose data the operating system already holds, a power failure
  can. `internal/fsx` routes every write, fsync, rename, remove and directory
  fsync of the segment writer, the manifest and the journal through hooks that
  record them; `internal/index/powerloss_test.go` and `pkg/zenith/powerloss_test.go`
  then rebuild the directory as it could look after a power cut **after every
  recorded operation** — with only fsynced data and fsynced directory entries
  (nothing else survives), with everything surviving, and with random torn subsets
  (unsynced pages persisting independently, in any order, 4 KiB at a time; directory
  operations persisting as a prefix; renames atomic) — and require the index to
  open, pass checksum verification, and hold exactly the state after the last
  acknowledged operation or the one in flight. Removing the segment fsync or the
  manifest fsync makes these tests fail. They found one real bug: a journal file's
  directory entry was never fsynced when it was first created, so a power cut right
  after the first acknowledged write could lose the whole journal.
- A full disk is injected at byte granularity (`fsx.DiskFull`): a failed flush,
  compaction or journal append returns the out-of-space error, leaves no partial
  file behind, never acknowledges the write, keeps the previous on-disk state
  intact and the engine serving, and succeeds once space returns.

What is not modelled: hardware that lies about `fsync`, torn writes below 4 KiB
(sector-atomicity is assumed), and filesystems that reorder directory operations.

## Compatibility policy

**Formats are versioned, and a version is never silently reinterpreted.**

- **Older format → newer release.** A release always ships a conversion path for
  the formats it replaces. `zenith migrate` (or `zenith.Migrate` from Go) converts
  in place: it copies the original aside as `<file>.v<N>.bak`, writes the new
  index next to it, swaps it in atomically, then re-opens and checksums the result.
  No embedding model is needed; vectors are copied, not recomputed. Until you
  migrate, an older-format file is *refused with a clear error* — never opened
  wrongly, and never overwritten by an empty index.
- **Newer format → older release.** Refused with an explicit "unsupported version"
  error. Downgrades are not supported; keep the `.bak` if you might need one.
- **Embedding model.** The model's identity is in the manifest header. Opening an
  index with a different model is refused (`ErrEmbedderMismatch`) because mixing
  vector spaces gives plausible-looking but wrong results. Switching models means
  re-indexing; your source documents are untouched.
- **What counts as a format change.** Any change to the manifest layout, a segment
  section's encoding, or the meaning of a stored value. Search-time behaviour
  (ranking, analysis) is *not* part of the format: it can change between releases
  without a migration, and is documented in release notes when it does.
- **Pre-1.0.** Format versions may still advance between minor releases, but each
  advance ships with a migration from the immediately preceding format and a
  golden-file test (below). From 1.0, files written by any 1.x release open in
  every later 1.x release without migration.

| Format | Written by | Status |
|--------|-----------|--------|
| 3 | early releases | not supported — re-index |
| 4, 5 | up to the release before segments (single gob file) | `zenith migrate` |
| 6 | current (manifest + segments) | native |

### How the policy is enforced

`internal/index/testdata/` holds **real files written by earlier releases** — a
version-5 index produced by the previous release's own library, with the search
results that release returned for it — and a version-6 index written by the
current one. `TestGolden_*` loads or migrates each and asserts the same results, so
an accidental format break, or a migration that changes what a search returns,
fails CI.
