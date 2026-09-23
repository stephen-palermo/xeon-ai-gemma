#!/usr/bin/env python3
"""
Minimal script: download and run OpenVINO/gemma-4-E4B-it-int8-ov with OpenVINO GenAI.

Setup
-----
    python3 -m venv openvino_env
    source openvino_env/bin/activate
    pip install openvino-genai==2026.4.0
    pip install openvino==2026.4.0
    pip install huggingface_hub pillow numpy py-cpuinfo

Model: https://huggingface.co/OpenVINO/gemma-4-E4B-it-int8-ov

Usage
-----
    time python3 ./run_gemma.py                         # default prompt + image.png
    time python3 ./run_gemma.py --prompt "Describe the image."
    time python3 ./run_gemma.py --image photo.jpg
    time python3 ./run_gemma.py --kv-cache-precision f16   # disable int8 KV cache
    python3 ./run_gemma.py --repeat 5 --max-new-tokens 512 # benchmark tokens/s
    python3 ./run_gemma.py --threads 32                    # cap inference threads
    python3 ./run_gemma.py --serve                         # load once, prompt loop
    python3 ./run_gemma.py --http 8000                     # load once, HTTP server
    python3 ./run_gemma.py --model 31b                     # larger 31B model

Models: OpenVINO/gemma-4-E4B-it-int8-ov (default, alias "e4b")
        OpenVINO/gemma-4-31B-it-int8-ov (alias "31b")
"""

import argparse
import os
import time

_IMPORT_START = time.perf_counter()

from importlib.metadata import version
import cpuinfo

# Skip Hugging Face network chatter (telemetry, progress bars, Xet
# reconstruction messages). The model is loaded from the local cache below.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Enable Intel AMX only when the CPU reports the required AMX capabilities.
# A pre-set ONEDNN_MAX_CPU_ISA is respected, so AMX can be disabled from the
# environment, e.g. ONEDNN_MAX_CPU_ISA=avx512_core_bf16 python3 ./run_gemma.py
cpu_flags = set(cpuinfo.get_cpu_info().get("flags", []))
AMX_DETECTED = {"amx_tile", "amx_int8", "amx_bf16"}.issubset(cpu_flags)
ISA_OVERRIDE = os.environ.get("ONEDNN_MAX_CPU_ISA")
if AMX_DETECTED and ISA_OVERRIDE is None:
    os.environ["ONEDNN_MAX_CPU_ISA"] = "avx512_core_amx"
AMX_USED = AMX_DETECTED and "amx" in os.environ.get("ONEDNN_MAX_CPU_ISA", "").lower()

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from huggingface_hub import snapshot_download
from PIL import Image

# Wall time spent importing heavy deps (openvino, genai, hf) + CPU probe.
IMPORT_ELAPSED = time.perf_counter() - _IMPORT_START

# Selectable models. Keys are convenience aliases for --model; any full
# Hugging Face repo id is also accepted.
MODELS = {
    "e4b": "OpenVINO/gemma-4-E4B-it-int8-ov",
    "31b": "OpenVINO/gemma-4-31B-it-int8-ov",
}
MODEL_ID = MODELS["e4b"]


def load_image(path):
    """Load an image as an OpenVINO tensor (HWC, uint8)."""
    img = Image.open(path).convert("RGB")
    return ov.Tensor(np.array(img)[None])


def run_http_server(pipe, default_images, config, host, port):
    """Serve the warm pipeline over HTTP so callers skip the per-process
    import + 8 s compiled-model load. POST JSON {"prompt", optional "image"}
    to /generate. Binds to localhost by default; this is a local dev tool
    with no auth, so do not expose it on an untrusted network."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    lock = threading.Lock()  # VLMPipeline.generate is not re-entrant.

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/generate":
                self._send(404, {"error": "use POST /generate"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": f"bad JSON: {exc}"})
                return
            prompt = data.get("prompt")
            if not prompt:
                self._send(400, {"error": "missing 'prompt'"})
                return
            images = default_images
            if data.get("image"):
                try:
                    images = [load_image(data["image"])]
                except Exception as exc:  # noqa: BLE001 - report to client
                    self._send(400, {"error": f"image load failed: {exc}"})
                    return
            start = time.perf_counter()
            with lock:
                result = pipe.generate(prompt, images=images,
                                       generation_config=config)
            elapsed = time.perf_counter() - start
            try:
                n_tokens = result.perf_metrics.get_num_generated_tokens()
            except Exception:  # noqa: BLE001 - metrics are best-effort
                n_tokens = None
            self._send(200, {
                "text": str(result),
                "tokens": n_tokens,
                "seconds": round(elapsed, 3),
                "tokens_per_s": round(n_tokens / elapsed, 1) if n_tokens else None,
            })

        def log_message(self, *args):
            pass  # keep the console clean

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}  (POST /generate). Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Run gemma-4 with OpenVINO GenAI.")
    parser.add_argument("--model", default=MODEL_ID,
                        help="Model to run: an alias (" +
                             ", ".join(MODELS) + ") or any Hugging Face repo "
                             f"id. Default: {MODEL_ID}")
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
    parser.add_argument("--threads", type=int, default=None,
                        help="Cap OpenVINO inference threads (physical-core "
                             "count often lowers latency and CPU thrash).")
    parser.add_argument("--serve", action="store_true",
                        help="Keep the model loaded and read prompts from "
                             "stdin so import/compile cost is paid only once.")
    parser.add_argument("--http", type=int, default=None, metavar="PORT",
                        help="Keep the model loaded and serve POST /generate "
                             "on this port (localhost). Paid once, then each "
                             "request is decode-bound.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address for --http. Defaults to localhost; "
                             "there is no auth, so avoid untrusted networks.")
    args = parser.parse_args()

    model_id = MODELS.get(args.model, args.model)

    print(f"OpenVINO base: {ov.__version__}")
    print(f"OpenVINO GenAI: {version('openvino-genai')}")
    print(f"AMX detected: {AMX_DETECTED}")
    print(f"AMX used: {AMX_USED}")
    print(f"KV cache precision: {args.kv_cache_precision}")
    print(f"Model: {model_id}")
    print(f"Prompt: {args.prompt}")
    print(f"Image source: {args.image}")

    # Load from the local cache first (offline, no hub round-trip). Only hit
    # the network on the first run when the model is not yet present.
    t0 = time.perf_counter()
    try:
        model_path = snapshot_download(repo_id=model_id, local_files_only=True)
    except Exception:
        model_path = snapshot_download(repo_id=model_id)
    t_download = time.perf_counter() - t0

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
    if args.threads is not None:
        plugin_config["INFERENCE_NUM_THREADS"] = args.threads

    t0 = time.perf_counter()
    pipe = ov_genai.VLMPipeline(model_path, args.device, **plugin_config)
    t_build = time.perf_counter() - t0

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens

    images = [load_image(args.image)]

    print(f"\n[startup] imports={IMPORT_ELAPSED:.2f}s  "
          f"model-load={t_download:.2f}s  pipeline-build={t_build:.2f}s")

    # HTTP mode: pay import/compile once, then serve requests over the network.
    if args.http:
        run_http_server(pipe, images, config, args.host, args.http)
        return

    # Serve mode: pay import/compile once, then answer prompts from stdin.
    if args.serve:
        print("Ready. Enter a prompt (blank line or Ctrl-D to exit).")
        while True:
            try:
                prompt = input("prompt> ").strip()
            except EOFError:
                break
            if not prompt:
                break
            start = time.perf_counter()
            result = pipe.generate(prompt, images=images,
                                   generation_config=config)
            elapsed = time.perf_counter() - start
            try:
                n_tokens = result.perf_metrics.get_num_generated_tokens()
            except Exception:
                n_tokens = None
            tok_per_s = f"{n_tokens / elapsed:.1f}" if n_tokens else "n/a"
            print(f"{result}\n[{elapsed:.2f}s  tokens={n_tokens}  "
                  f"tokens/s={tok_per_s}]")
        return

    # Warm-up run (buffer allocation, first-token setup) excluded from timing.
    # Only a couple of tokens are needed to trigger the one-time setup cost.
    if args.repeat > 1:
        warmup_config = ov_genai.GenerationConfig()
        warmup_config.max_new_tokens = 2
        pipe.generate(args.prompt, images=images, generation_config=warmup_config)

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
