#!/usr/bin/env bash
set -euo pipefail

# Use all available CPU cores for inference.
CORES="$(nproc)"

time python3 ./run_gemma.py --repeat 5 --model e4b --threads "$CORES"
