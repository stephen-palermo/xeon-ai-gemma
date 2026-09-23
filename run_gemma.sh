#!/usr/bin/env bash
set -euo pipefail

# Low-latency single run: small int4 model, no benchmark repeats, and one
# thread per physical core (avoids the hyperthread oversubscription that
# inflates sys time). Extra args are forwarded to run_gemma.py.
CORES="$(lscpu -p=core | grep -v '^#' | sort -u | wc -l)"

time python3 ./run_gemma.py --model e2b --threads "$CORES" "$@"
