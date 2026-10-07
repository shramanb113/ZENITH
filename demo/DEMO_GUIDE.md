# ZENITH Docker Demo — Pitch Guide

**Status: pre-flight fully done, 2026-10-05 morning, for a 4pm meeting the same day.**
Every number below is from a real run (`docker compose build` + full `demo/run.sh`, all 8
sections, exit code 0) — not projected, not from an old log. Both models are already
cached in the Docker volume as of this writing; **you do not need to run anything before
the meeting** unless you want the optional data reset in §3. If a number here ever
disagrees with what you see live, trust the live run; this file is a script, not a
guarantee.

---

## 1. The 10-second pitch

**Define the scope before anything else — this is not "replace your database":**
Postgres (or whatever RDBMS you already run) stays your system of record for
transactional, relational, day-to-day application data. ZENITH doesn't touch that job
and isn't trying to. What it replaces is the *search* stack you'd otherwise have to
assemble on top of that database — a vector DB subscription, plus full-text search, plus
your own code to fuse the two into one ranking.

> "ZENITH isn't trying to replace Postgres — it's the embedded hybrid search engine you
> add next to it. Your relational database stays the system of record; ZENITH is the
> specialized engine that makes your content — documents, tickets, logs, scanned files —
> actually searchable: typos, meaning, filters, multiple languages, without a vector DB
> subscription or a hand-wired pgvector setup. I can show you all of that running in a
> container right now."

**The five things that back that up, if asked to go deeper:**

1. Postgres keeps owning the data model — users, orders, relationships, transactions.
   ZENITH owns making the *text/document content* your app has findable; it's additive,
   not a rip-and-replace.
2. What it removes is the separate vector DB (Pinecone/Qdrant/Weaviate) you'd otherwise
   provision, pay for, and keep available over a network — ZENITH *is* the storage engine
   (real WAL, memtables, SSTables, Bloom filters, leveled compaction) running in-process.
3. It also removes the DIY work pgvector leaves you with: you'd still bolt an extension
   onto Postgres, hand-tune its ANN index, and write your own score-fusion SQL. ZENITH's
   storage layer was purpose-built for search from the ground up, with BM25 + fuzzy +
   phonetic + neural vectors fused via Reciprocal Rank Fusion out of the box.
4. Because it owns its own storage format end to end, it owns crash safety too — WAL
   replay, non-blocking checkpoints, segment compaction. The Go library is tested against
   hard-kill loops and simulated power-loss/disk-full fault injection at every cut point;
   real power loss on hardware has not been tested.
5. It's embedded: compiles into your Go binary or runs as a CLI, so there's no extra
   service, API key, or hosting bill just to get good search over what Postgres already
   references — or over things Postgres never stores well, like scanned PDFs or images.

---

## 2. What ZENITH can do — in plain terms (the non-technical version)

Use this if the room isn't technical, or to warm up before the technical backup points
land. Lead with this, not with "LSM-tree" — save the architecture talk for whoever asks
a follow-up question.

**Picture a research assistant who has read every document you own, speaks 109
languages, never sleeps, and keeps what it has saved even if you kill it.**

- **It reads your mind, not just your keywords.** Type "heart attack symptoms" and it
  finds the right medical note even if those exact words never appear anywhere in it —
  it understood what you meant, the same way a smart colleague would, not a Ctrl+F.
- **It can't be fooled by typos.** Type "recieve" instead of "receive," or "Transfomer"
  instead of "Transformer" — it still finds the right document instantly. It's like it
  already knows what you meant to type before you finish typing it.
- **It turns pictures into searchable text, like magic.** Hand it a scanned, slightly
  crooked photo of a printed report — something a human would have to squint at — and a
  few seconds later you can search for a phrase buried inside that image as if it were a
  Word document all along.
- **It speaks languages it was never explicitly told to translate.** Ask a question in
  plain English and it will reach into a document written entirely in Hindi, in a
  completely different alphabet, and hand you back the right one — with zero translation
  step in between. It's reading *meaning*, across languages, not matching words.
- **What it has saved, it keeps.** Kill the server outright — no warning, no graceful
  shutdown — and the index files it already wrote come back intact and give the same
  answers. (The Go library goes further: every write it confirms is logged first, and
  it's tested against simulated power cuts. The demo's command-line server is simpler —
  it saves on shutdown — and we say so.)
- **It fits in your pocket, metaphorically.** All of this — the mind-reading, the
  typo-correction, the OCR, the 109 languages, the crash-safe index files — runs quietly inside
  the app itself. No cloud subscription, no "search service" bill arriving every month,
  no waiting on someone else's server halfway across the world. It's as fast and private
  as searching the documents sitting right next to you.

Every one of these maps to a real, demoed moment: typo tolerance → §5 Section 2;
mind-reading → §5 Section 3; OCR → §5 Section 5; multilingual → §5 Section 7;
on-disk durability → §5 Section 8. Each bullet restates what the live run actually
does, including the limit spelled out in the durability bullet.

---

## 3. Pre-flight — already done, two things worth knowing

The full script downloads two models the first time it runs: the reranker (~23 MB) and
LaBSE (~450 MB). On a normal connection that's several minutes you do **not** want to
burn live in front of someone — so this was already run to completion this morning and
re-verified end-to-end (exit code 0, all 8 sections correct). **You don't need to repeat
this before the meeting.** Two real issues came up while doing that, both now handled:

1. **`zenith models pull` never checked if a model was already there — it always
   re-downloaded.** That meant every rehearsal of the old `demo/run.sh` silently re-fetched
   LaBSE's 450 MB again, even though it was already cached. **Fixed**: `demo/run.sh` now
   has a `pull_if_missing` helper that checks the named Docker volume first and skips the
   download when the model's already present. Verified — the fixed script now prints
   `"labse already cached in Docker volume ... skipping re-download"` and finishes
   sections 6–7 in seconds instead of minutes.
2. **A separate dedup cache (`~/.zenith/file_hashes.json`) lives inside the same Docker
   volume as the index data, but tracks something different** — which files have already
   been indexed, independent of whether the actual index still has them. If you (or
   anyone) ever manually clears the index data inside the volume without also clearing
   this file, every subsequent `index` call will silently skip re-indexing (it thinks the
   files are unchanged) and searches will return nothing. **You will not hit this** unless
   you go poking around inside the Docker volume by hand — `demo/run.sh` itself never
   triggers it. Mentioned here only so it isn't a mystery if it ever comes up.

**Current state, as of this morning:** both models are cached, and a full demo corpus is
already indexed in the Docker volume from the rehearsal run. That means if you run
`demo/run.sh` again as-is, section 1 will print `0 files indexed` instead of
`5 files indexed` etc. — because the documents are already there, not because anything's
broken. **Searches still return the exact same correct results either way.** See §3 if
you'd rather see the "N files indexed" numbers live instead.

**Do not run `docker compose down -v` at any point.** The `-v` flag deletes the named
volume and wipes the cached models — you'd be back to a 450 MB download live in the
meeting. Plain `docker compose down` (no `-v`, which is what `demo/run.sh` itself runs at
the end) is fine and keeps the volume.

---

## 4. Meeting-day checklist — exactly what to do at 4pm

1. **A few minutes before:** open Docker Desktop and confirm the whale icon shows it's
   running (starting the engine cold adds 10–20 seconds you don't want mid-pitch). Open a
   terminal, `cd D:/golang-projects/ZENITH`.
2. **(Optional, cosmetic only) If you want fresh "N files indexed" numbers** instead of
   "0 files indexed" (both are correct — this only changes what's printed, not what the
   searches find), run this first. It resets only the index data, never the cached models:
   ```bash
   docker run --rm -v zenith_zenith-home:/data debian:bookworm-slim sh -c "rm -rf /data/data/* /data/.zenith/data /data/.zenith/file_hashes.json"
   ```
3. **Run the demo:**
   ```bash
   export MSYS_NO_PATHCONV=1          # Git Bash/MSYS only — stops path-mangling into Docker
   bash demo/run.sh
   ```
   That's the entire command. It walks through all 8 sections in order below, printing a
   boxed header before each one. With both models cached, total runtime is **under a
   minute** for sections 1–7; section 8 (SIGKILL durability check) adds a few seconds for
   the container kill/restart cycle.
4. **Narrate over it — don't wait for it to finish before talking.** Use the
   section-by-section script in §5 for exactly what to say at each point, including the
   two things to disclose proactively (§6) rather than wait to be asked.
5. **If something unexpected appears on screen:** don't improvise explanations for raw
   Docker output (image pulls, container lifecycle lines) — that's normal `compose`
   chatter, not an error. Only the boxed section headers and the result tables matter for
   the pitch.
6. **If you want to jump to one section instead of the whole script** (e.g. to re-run just
   the multilingual query during Q&A), copy the matching `compose run --rm zenith ...` line
   out of `demo/run.sh` — every section is a single self-contained command block.
7. **After the meeting:** `bash demo/run.sh` already ends with a plain `docker compose
   down` (no `-v`) — nothing further to clean up, and the models stay cached for next time.

---

## 5. Section-by-section script — what to say, what they'll see

### Section 1 — Mixed-format ingestion
**Command:** indexes `demo/corpus/clinical`, `demo/corpus/misc`, `demo/corpus/images`
(`.txt .md .csv .json .log .png`), each tagged with a `domain` attribute.

**What appears:** `5 files indexed`, `4 files indexed`, `1 file indexed`, each in well
under half a second.

**Say:** "This is a real mixed-format corpus — clinical notes, JSON configs, CSVs, log
files, a scanned PNG — going in with one command each, tagged with metadata as they're
indexed. No separate OCR step, no separate 'load into Elasticsearch' step."

### Section 2 — Typo tolerance
**Command:** searches `"receive invoice"` against a document that actually contains the
misspelling `"recieve"`.

**What appears:** `support-tickets.csv` ranks #1 at score 0.143, in 5ms.

**Say:** "The query is spelled correctly, the document isn't — real support tickets are
full of typos like this. ZENITH's fuzzy/phonetic layer catches it without any fuzzy
syntax in the query. 5 milliseconds."

### Section 3 — Out-of-domain semantic search
**Command:** searches `"heart attack symptoms"` — a phrase that appears nowhere, verbatim,
in the corpus.

**What appears:** `cardiac_3.txt` ranks #1 (0.143), ahead of the other cardiac notes and
everything else — in 5ms.

**Say:** "This phrase isn't in any document. The clinical notes talk about chest pain,
ECG findings, troponin levels — clinically related, zero shared words. This is the
neural embedding model doing real semantic matching, not a keyword trick."

### Section 4 — Metadata filtering
**Command:** the same kind of query (`"server restart during import"`), scoped with
`--where domain=misc`.

**What appears:** 4 results, all from the `misc/` set only.

**Say:** "Same hybrid search, but now scoped by metadata — exactly what you'd need for
multi-tenant search, or 'only search this customer's documents.' The filter is a real
indexed structure, not a post-filter that throws away ranking quality."

### Section 5 — OCR'd image hit
**Command:** searches `"crash recovery documents survived"` — text that exists **only**
inside a scanned PNG, nowhere as a filename or in any other document.

**What appears:** `scanned-report.png||ocr||c0` (the OCR'd text chunk) ranks #1 at 0.143.

**Say:** "That image was never transcribed by a human. Tesseract OCR ran at index time,
and the extracted text is now a fully searchable document — same ranking pipeline as
everything else. Point this at a folder of scanned reports, screenshots, or receipts and
they become searchable."

### Section 6 — Reranking (the sharpest visual moment)
**Commands:** pulls the `ms-marco-MiniLM-L-6-v2` cross-encoder, then runs the same query
twice — with and without `--rerank`.

**What appears (verified this run):**

| | without `--rerank` | with `--rerank` |
|---|---|---|
| #1 `support-tickets.csv` | 0.143 | **0.998** |
| #2 | 0.123 | 0.000 |
| #3 | 0.120 | 0.000 |

**Say:** "Watch the scores. Without reranking, the top few results are close together —
0.143, 0.123, 0.120 — a soft hybrid ranking. Flip on `--rerank`, and the cross-encoder
jointly reads the query and each candidate and says: 0.998 for the real answer, 0.000 for
everything else. That's a cross-encoder decisively separating the one relevant document
from the rest — the clearest 'this actually works' moment in the whole demo."

### Section 7 — Multilingual (LaBSE) — the technical depth moment
**Commands:** pulls LaBSE (768-dim, 109 languages, ~450 MB — already cached from your
warm-up run), indexes a separate 8-document multilingual corpus, runs three queries.

**What appears (verified this run):**

1. `"capital of France"` → `french.txt` #1 (0.143). *(Honest caveat to say out loud: this
   query shares the literal word "France" with the lexical pass — it's not proof of pure
   semantic matching by itself.)*
2. `"white marble mausoleum built by a Mughal emperor in Agra"` → `taj_mahal.txt` #1
   (0.095), ahead of `hindi.txt` and `himalaya.txt`. **Zero shared vocabulary** — the
   query is English, the matched document is a real Hindi Wikipedia passage in
   Devanagari script. This is the one to slow down on.
3. `"bat and ball game played between two teams"` → `cricket.txt` lands at **#4**, not
   top-3. This is a disclosed, known limitation — see §6.

**Say:** "This second query is the real test: an English description of the Taj Mahal,
against a Hindi Wikipedia article, with no overlapping words at all. It still ranks
first. That's cross-lingual semantic understanding, not keyword luck."

Then, proactively, before anyone asks: "The third query — a cricket description — misses
the top 3. I'll come back to that in a second, because I'd rather tell you about it than
have you find it."

### Section 8 — Committed data survives SIGKILL
**Command:** searches the saved index, starts the server (which opens those files),
`SIGKILL`s it, searches the same files again and diffs the two outputs, then runs a
CRC-32C pass over every segment.

**What appears:** the same top-5 list before and after,
`IDENTICAL: the committed segments came through the SIGKILL unchanged.`, and
`every segment passes its CRC-32C checks`.

**Say:** "I just killed the server with SIGKILL while it had the index open — no graceful
shutdown. The files it had already committed are untouched: same results, every checksum
passes. To be precise about the limit: this command-line server saves new writes on
shutdown, so writes sent to it after it starts aren't covered by this test. The Go library
logs every write before acknowledging it."

---

## 6. What to disclose before they ask (builds more trust than it costs)

Say these proactively, briefly, framed as "here's where the edges are" — not apologetically.

1. **The cricket query (Section 7).** On this specific 8-document, deliberately
   adversarial demo corpus, the hardest cross-lingual query doesn't make top-3. Root
   cause is documented and investigated (not hand-waved): the real trained
   sentence-transformers LaBSE projection was reconstructed and verified — three
   independent pooling/projection strategies all converge on the identical result, which
   points at the frozen, int8-quantized model's inherent cross-lingual ceiling on a small
   corpus, not an implementation bug. Full writeup: `README.md` → "Real-world Hindi
   accuracy check" and `ROADMAP.md` item 6.
2. **Multilingual is smoke-tested, not BEIR-benchmarked.** Five languages pass a
   hand-built sanity check (`TestMultilingualSmoke`); it hasn't been run against a
   published multilingual retrieval benchmark. Say "solid for same-language and curated
   retrieval; arbitrary zero-shared-vocabulary retrieval at scale shouldn't be oversold"
   — your own README's words, and accurate.
3. **OCR requires Linux/Docker (Tesseract via CGo).** It's not yet verified on a bare
   Windows dev box — this demo container is literally where it gets proven, which is a
   fine thing to say out loud ("you're watching the real verification, not a recording").

---

## 7. Anticipated questions

**"How is this different from Elasticsearch / Meilisearch / pgvector?"**
Those are servers you run and talk to over a network, or extensions bolted onto a
general-purpose database. ZENITH is its own storage engine — a real LSM-tree (WAL,
memtables, SSTables, compaction) built from scratch, with hybrid search as a native
query layer on top, compiled straight into your binary. No ops overhead, no second
process to monitor, no network hop on the hot path.

**"So does this replace our Postgres database?"** — get ahead of this one, it's the most
likely misread of the pitch. No. Postgres stays the system of record for your
transactional/relational data; ZENITH doesn't do joins, transactions, or ACID multi-table
writes, and isn't trying to. It replaces the *search* stack you'd otherwise bolt onto
Postgres — a vector DB, full-text search, and your own fusion code — with one embedded
engine. Think "the specialized search layer next to your database," not "a database
replacement."

**"What happens if the embedding model can't load?"**
Non-fatal, automatic fallback: lexical/fuzzy/phonetic search keeps working at full
quality via a deterministic hash-based embedder. Semantic search degrades gracefully
instead of the whole system failing.

**"Has this been used for anything real?"**
Not in production by an outside team yet — it's open source and pre-1.0. What I can point
to instead is the evidence anyone can re-run: the MS MARCO numbers on all 6,980 dev
queries, the BEIR runs, and the crash-safety suite (hard-kill loops plus simulated
power-loss and disk-full sweeps over every cut point).

**"What's the latency at scale?"**
Hybrid search runs single-digit milliseconds on this demo corpus; it's been benchmarked
against MS MARCO and SciFact (see `ROADMAP.md`) with an HNSW ANN index kicking in above
20k documents so it doesn't degrade to a linear scan at scale.

**"Can I filter by metadata / scope to a tenant?"**
Yes — Section 4. Filters (`eq`, `in`, `range`, `exists`, boolean combinators) are a real
indexed structure (`attrIndex`), not a post-filter — same JSON filter grammar works
across the CLI, gRPC, and HTTP surfaces.

**"What's left unfinished?"**
Be honest and specific rather than vague: Office document ingestion (.docx/.pptx/.xlsx)
and the 1M-document scale test are deferred backlog items, not silently dropped — see
`ROADMAP.md`'s "Deferred backlog" section if asked for detail.

---

## 8. Closing line

> "Everything you just watched — the typo tolerance, the semantic search, the OCR'd
> image, the reranking, the cross-lingual match, the index surviving a SIGKILL — ran in a plain
> Docker container, no cloud service, no GPU, no external API call. That's the whole
> pitch: production-grade hybrid search that ships inside your app instead of next to
> it."

---

## Appendix: file map for this demo

| Path | Purpose |
|---|---|
| `demo/run.sh` | The scripted walkthrough — single source of truth for section order |
| `demo/corpus/clinical/`, `misc/`, `images/` | Section 1–6 corpus (mixed formats + 1 scanned PNG) |
| `demo/corpus/multilingual/` | Section 7 corpus — 8 files, 5 hand-written + 3 real Hindi Wikipedia extracts |
| `demo/corpus/MULTILINGUAL_SOURCES.md` | Attribution for the Wikipedia extracts — deliberately **outside** the indexed `multilingual/` dir so it can't contaminate the zero-shared-vocabulary test |
| `docker-compose.yml` | Single `zenith` service wrapping the `Dockerfile` |
| `README.md` § "Real-world Hindi accuracy check" | The honest long-form writeup of the Section 7 investigation |
| `ROADMAP.md` item 6 | Same investigation, roadmap framing + follow-up status |
