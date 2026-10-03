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
compose run --rm zenith models pull ms-marco-MiniLM-L-6-v2
echo "-- without --rerank --"
compose run --rm zenith search "server restart during import" --db "$DB" --max 5
echo "-- with --rerank --"
compose run --rm zenith search "server restart during import" --db "$DB" --max 5 --rerank

section "7. Multilingual model (LaBSE) — separate index, English query against non-English passages"
compose run --rm zenith models pull labse   # also fetches the real trained dense+tanh projection (2_Dense)
compose run --rm zenith index /home/zenith/corpus/multilingual --db "$MLDB" --model labse
echo "-- 'capital of France': shares the literal word 'France' with the lexical pass --"
compose run --rm zenith search "capital of France" --db "$MLDB" --model labse --max 3
echo "-- true cross-lingual: English query, zero shared vocabulary with a real Hindi Wikipedia passage --"
compose run --rm zenith search "white marble mausoleum built by a Mughal emperor in Agra" --db "$MLDB" --model labse --max 3
echo "-- known limitation (ROADMAP item 6): this 8-doc corpus's hardest cross-lingual query still misses top-3 --"
compose run --rm zenith search "bat and ball game played between two teams" --db "$MLDB" --model labse --max 5

section "8. Crash recovery — SIGKILL the live gRPC server, restart, confirm no corruption"
compose up -d zenith
sleep 2
echo "-- before kill --"
CID=$(compose ps -q zenith)
docker exec "$CID" /usr/local/bin/zenith-client localhost:8080 || true
echo
echo "killing the server container with SIGKILL ($CID)..."
docker kill --signal=SIGKILL "$CID"
sleep 1
compose up -d zenith
sleep 2
echo "-- after kill + restart, same checks again --"
CID2=$(compose ps -q zenith)
docker exec "$CID2" /usr/local/bin/zenith-client localhost:8080 || true

section "Done"
compose down
