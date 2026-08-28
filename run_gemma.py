#!/usr/bin/env python3
"""
Minimal script: download and run OpenVINO/gemma-4-E4B-it-int8-ov with OpenVINO GenAI.

Setup
-----
    python3 -m venv openvino_env
    source openvino_env/bin/activate
    pip install openvino-genai==2026.3.1
    pip install openvino==2026.3.1
    pip install huggingface_hub pillow numpy py-cpuinfo

Model: https://huggingface.co/OpenVINO/gemma-4-E4B-it-int8-ov

Usage
-----
    time python3 ./run_gemma.py                         # default prompt + image.png
    time python3 ./run_gemma.py --prompt "Describe the image."
    time python3 ./run_gemma.py --image photo.jpg
"""

import argparse
from importlib.metadata import version
import os
import cpuinfo

# Enable Intel AMX only when the CPU reports the required AMX capabilities.
cpu_flags = set(cpuinfo.get_cpu_info().get("flags", []))
AMX_DETECTED = {"amx_tile", "amx_int8", "amx_bf16"}.issubset(cpu_flags)
if AMX_DETECTED:
    os.environ["ONEDNN_MAX_CPU_ISA"] = "avx512_core_amx"

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from huggingface_hub import snapshot_download
from PIL import Image

MODEL_ID = "OpenVINO/gemma-4-E4B-it-int8-ov"


def load_image(path):
    """Load an image as an OpenVINO tensor (HWC, uint8)."""
    img = Image.open(path).convert("RGB")
    return ov.Tensor(np.array(img)[None])


def main():
    parser = argparse.ArgumentParser(description="Run gemma-4 with OpenVINO GenAI.")
    parser.add_argument("--prompt", default="How many people in the image?",
                        help="Text prompt for the model.")
    parser.add_argument("--image", default="image.png",
                        help="Image file to ask about.")
    parser.add_argument("--device", default="CPU", help="OpenVINO device.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    print(f"OpenVINO base: {ov.__version__}")
    print(f"OpenVINO GenAI: {version('openvino-genai')}")
    print(f"AMX detected: {AMX_DETECTED}")
    print(f"AMX used: {AMX_DETECTED}")
    print(f"Prompt: {args.prompt}")
    print(f"Image source: {args.image}")

    # Download the model (cached after the first run).
    model_path = snapshot_download(repo_id=MODEL_ID)

    pipe = ov_genai.VLMPipeline(model_path, args.device)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens

    result = pipe.generate(args.prompt, images=[load_image(args.image)],
                           generation_config=config)

    print(result)


if __name__ == "__main__":
    main()
