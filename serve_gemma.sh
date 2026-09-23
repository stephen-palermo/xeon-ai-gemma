#!/usr/bin/env bash
set -euo pipefail

# Start the model once as an HTTP server so the ~5-8s model load is paid a
# single time; each POST /generate request is then decode-bound.
#
# Usage:
#   ./serve_gemma.sh                 # e2b on 127.0.0.1:8000
#   ./serve_gemma.sh --model e4b     # extra args forwarded to run_gemma.py
#   PORT=9000 ./serve_gemma.sh       # change port
#   HOST=0.0.0.0 ./serve_gemma.sh    # LAN (no auth: trusted networks only)
#
# Query it:
#   curl -s localhost:8000/generate -d '{"prompt":"how many people in the image?"}'

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

# One thread per physical core (avoids hyperthread oversubscription).
CORES="$(lscpu -p=core | grep -v '^#' | sort -u | wc -l)"

exec python3 ./run_gemma.py --model e2b --threads "$CORES" \
    --http "$PORT" --host "$HOST" "$@"
