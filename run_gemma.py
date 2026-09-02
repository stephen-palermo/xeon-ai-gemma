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
    time python3 ./run_gemma.py --kv-cache-precision f16   # disable int8 KV cache
    python3 ./run_gemma.py --repeat 5 --max-new-tokens 512 # benchmark tokens/s
"""

import argparse
from importlib.metadata import version
import os
import time
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
    parser.add_argument("--kv-cache-precision", default="u8",
                        choices=["u8", "f16", "f32"],
                        help="KV cache precision. 'u8' quantizes the cache to "
                             "int8 for faster decoding and lower memory.")
    parser.add_argument("--cache-dir", default="ov_cache",
                        help="Directory for OpenVINO's compiled-model cache. "
                             "Speeds up startup on repeat runs.")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of timed generations (model is loaded "
                             "once). Reports per-run time and tokens/sec.")
    args = parser.parse_args()

    print(f"OpenVINO base: {ov.__version__}")
    print(f"OpenVINO GenAI: {version('openvino-genai')}")
    print(f"AMX detected: {AMX_DETECTED}")
    print(f"AMX used: {AMX_DETECTED}")
    print(f"KV cache precision: {args.kv_cache_precision}")
    print(f"Prompt: {args.prompt}")
    print(f"Image source: {args.image}")

    # Download the model (cached after the first run).
    model_path = snapshot_download(repo_id=MODEL_ID)

    # KV cache acceleration: quantizing the runtime KV cache to int8 (u8)
    # lowers memory bandwidth and speeds up token generation. This is a
    # plugin-side setting only; no model re-download or rebuild is required.
    plugin_config = {
        "KV_CACHE_PRECISION": args.kv_cache_precision,
        # Cache the compiled model on disk so later process starts skip the
        # (multi-second) graph compilation step.
        "CACHE_DIR": args.cache_dir,
        # Optimize for single-request response time rather than throughput.
        "PERFORMANCE_HINT": "LATENCY",
    }

    pipe = ov_genai.VLMPipeline(model_path, args.device, **plugin_config)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens

    images = [load_image(args.image)]

    # Warm-up run (buffer allocation, first-token setup) excluded from timing.
    if args.repeat > 1:
        pipe.generate(args.prompt, images=images, generation_config=config)

    times = []
    for i in range(args.repeat):
        start = time.perf_counter()
        result = pipe.generate(args.prompt, images=images,
                               generation_config=config)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        try:
            n_tokens = result.perf_metrics.get_num_generated_tokens()
        except Exception:
            n_tokens = None
        tok_per_s = f"{n_tokens / elapsed:.1f}" if n_tokens else "n/a"
        print(f"[run {i + 1}/{args.repeat}] {elapsed:.2f}s  "
              f"tokens={n_tokens}  tokens/s={tok_per_s}")

    print(result)

    if args.repeat > 1:
        best = min(times)
        avg = sum(times) / len(times)
        print(f"\nTiming over {args.repeat} runs: "
              f"best={best:.2f}s  avg={avg:.2f}s")


if __name__ == "__main__":
    main()
