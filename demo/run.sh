#!/usr/bin/env bash
# ZENITH Docker demo walkthrough: mixed-format ingestion, typo tolerance,
# out-of-domain semantic search, metadata filtering, OCR'd images, a
# multilingual model, reranking, and crash recovery — all against the real
# CLI/gRPC engine (never the in-memory HTTP namespace sidecar, which has no
# persistence by design: see CLAUDE.md / internal/sidecar).
#
# Usage: ./demo/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Prevent Git Bash/MSYS from rewriting container paths like /home/zenith/...
# into Windows paths before they reach the docker CLI.
export MSYS_NO_PATHCONV=1

compose() { docker compose -f docker-compose.yml "$@"; }

# `zenith models pull` always re-downloads unconditionally (no skip-if-present
# check) — fine for a one-off pull, but this script runs it every time
# demo/run.sh runs, which would re-fetch LaBSE's 450 MB from HuggingFace on
# every single rehearsal. Check the persistent named volume directly first and
# skip the pull when the model (identified by its last-written file, so a
# partial prior pull still re-runs) is already there.
pull_if_missing() {
	local id="$1" marker="$2"
	local vol
	vol=$(docker volume ls -q | grep -E '_zenith-home$' | head -1)
	if [ -n "$vol" ] && docker run --rm -v "$vol":/home/zenith debian:bookworm-slim \
			test -f "/home/zenith/.zenith/models/$id/$marker" 2>/dev/null; then
		echo "-- $id already cached in Docker volume '$vol', skipping re-download --"
	else
		compose run --rm zenith models pull "$id"
	fi
}

section() {
	echo
	echo "════════════════════════════════════════════════════════════════"
	echo "  $1"
	echo "════════════════════════════════════════════════════════════════"
}

DB=/home/zenith/data/zenith.db
MLDB=/home/zenith/data/multilingual.db

section "Build"
compose build

section "1. Mixed-format ingestion (.txt .md .csv .json .log .png, with metadata attrs)"
compose run --rm zenith index /home/zenith/corpus/clinical --db "$DB" --attr domain=clinical
compose run --rm zenith index /home/zenith/corpus/misc     --db "$DB" --attr domain=misc
compose run --rm zenith index /home/zenith/corpus/images   --db "$DB" --attr domain=image

section "2. Typo tolerance — query 'receive invoice', document says 'recieve'"
compose run --rm zenith search "receive invoice" --db "$DB" --max 3

section "3. Out-of-domain semantic search — query 'heart attack symptoms', not a literal match anywhere in the corpus"
compose run --rm zenith search "heart attack symptoms" --db "$DB" --max 5

section "4. Metadata filtering — same kind of query, scoped to domain=misc only"
compose run --rm zenith search "server restart during import" --db "$DB" --where domain=misc --max 5

section "5. OCR'd image hit — query text only exists inside the scanned PNG"
compose run --rm zenith search "crash recovery documents survived" --db "$DB" --max 3

section "6. Reranking — cross-encoder reorders the hybrid shortlist"
pull_if_missing ms-marco-MiniLM-L-6-v2 model.onnx
echo "-- without --rerank --"
compose run --rm zenith search "server restart during import" --db "$DB" --max 5
echo "-- with --rerank --"
compose run --rm zenith search "server restart during import" --db "$DB" --max 5 --rerank

section "7. Multilingual model (LaBSE) — separate index, English query against non-English passages"
pull_if_missing labse dense.safetensors   # also fetches the real trained dense+tanh projection (2_Dense)
compose run --rm zenith index /home/zenith/corpus/multilingual --db "$MLDB" --model labse
echo "-- 'capital of France': shares the literal word 'France' with the lexical pass --"
compose run --rm zenith search "capital of France" --db "$MLDB" --model labse --max 3
echo "-- true cross-lingual: English query, zero shared vocabulary with a real Hindi Wikipedia passage --"
compose run --rm zenith search "white marble mausoleum built by a Mughal emperor in Agra" --db "$MLDB" --model labse --max 3
echo "-- known limitation (ROADMAP item 6): this 8-doc corpus's hardest cross-lingual query still misses top-3 --"
compose run --rm zenith search "bat and ball game played between two teams" --db "$MLDB" --model labse --max 5

section "8. Committed data survives SIGKILL — kill the live server, re-check the saved index"
TMPD=$(mktemp -d)
echo "-- baseline: search the index that sections 1-5 saved to disk --"
compose run --rm -T zenith search "heart attack symptoms" --db "$DB" --max 5 2>/dev/null | grep -v ' · ' > "$TMPD/before.txt"
cat "$TMPD/before.txt"
compose up -d zenith
sleep 3
CID=$(compose ps -q zenith)
echo "server $CID has the index open; killing it with SIGKILL (no graceful shutdown, no save)..."
docker kill --signal=SIGKILL "$CID"
sleep 1
echo "-- the same search against the same files, after the kill --"
compose run --rm -T zenith search "heart attack symptoms" --db "$DB" --max 5 2>/dev/null | grep -v ' · ' > "$TMPD/after.txt"
cat "$TMPD/after.txt"
if diff -u "$TMPD/before.txt" "$TMPD/after.txt"; then
	echo "IDENTICAL: the committed segments came through the SIGKILL unchanged."
else
	echo "MISMATCH after SIGKILL — investigate before presenting."; exit 1
fi
echo "-- CRC-32C pass over every segment --"
compose run --rm -T zenith doctor --db "$DB" 2>/dev/null | grep -E 'CRC-32C|failed verification' || true
echo "Note: 'zenith serve' (gRPC) keeps new writes in memory until a graceful shutdown; a SIGKILL drops them."
echo "      Write-ahead-logged durability is the Go library's (pkg/zenith); see README 'Docker demo'."
rm -rf "$TMPD"

section "Done"
compose down
