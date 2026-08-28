#!/usr/bin/env bash
#
# Set up the virtual environment needed to run run_gemma.py.
#
# Usage:
#   ./simple_setup.sh
#   source openvino_env/bin/activate
#   python3 ./run_gemma.py

set -euo pipefail

python3 -m venv openvino_env
source openvino_env/bin/activate

pip install --upgrade pip
pip install openvino-genai==2026.3.1
pip install openvino==2026.3.1
pip install huggingface_hub pillow numpy

echo
echo "Done. Activate the environment with:"
echo "    source openvino_env/bin/activate"
echo "Then run:"
echo "    python3 ./run_gemma.py"
