# xeon-ai-gemma

Run the [OpenVINO Gemma model](https://huggingface.co/OpenVINO/gemma-4-E4B-it-int8-ov)
for image + text inference on Intel Xeon CPUs (with optional AMX acceleration)
using OpenVINO GenAI.

## Prerequisites

1. Get access to the [OpenVINO Gemma model](https://huggingface.co/OpenVINO/gemma-4-E4B-it-int8-ov)
   on Hugging Face and create a user access token.
2. Have an image named `image.png` in this directory (the default), or supply
   your own with `--image`.

## Quick start

### Option A — Without Docker (Python virtual environment)

From this directory, run the following. Replace `your_token_here` with your
Hugging Face token:

```bash
export HF_TOKEN="your_token_here"
bash setup_gemma.sh
source openvino_env/bin/activate
python3 ./run_gemma.py
```

Use a different image or prompt:

```bash
python3 ./run_gemma.py --image photo.jpg --prompt "Describe this image."
```

Benchmark tokens/sec (the model is loaded once and generation is timed):

```bash
python3 ./run_gemma.py --repeat 5 --max-new-tokens 512
```

The model is downloaded from Hugging Face on the first run and cached for
later runs. On subsequent runs, skip the Hugging Face update check (and the
"Fetching 22 files" line) by enabling offline mode:

```bash
# Per run
HF_HUB_OFFLINE=1 python3 ./run_gemma.py

# Or for the whole shell session
export HF_HUB_OFFLINE=1
python3 ./run_gemma.py
```

### Option B — With Docker

The `Dockerfile` copies `image.png` into the image and sets
`--cache-dir /cache/ov_cache` as part of the entrypoint. Mount a `cache`
volume so the downloaded model and compiled graph persist between runs.

```bash
# 1. From this directory, make sure an image.png exists (the Dockerfile COPYs it)
#    or copy your own:  cp /path/to/photo.jpg image.png

# 2. Build the image
docker build -t xeon-ai-gemma .

# 3. Run it (pass your Hugging Face token and mount the cache volume)
docker run --rm -it \
  -e HF_TOKEN="your_token_here" \
  -v "$PWD/cache:/cache" \
  xeon-ai-gemma
```

The first run downloads the model into `./cache`; later runs reuse it.

**Passing arguments** — anything after the image name is forwarded to
`run_gemma.py`:

```bash
# Custom prompt / image
docker run --rm -it -e HF_TOKEN="your_token_here" -v "$PWD/cache:/cache" \
  xeon-ai-gemma --prompt "Describe this image." --image image.png

# Benchmark tokens/sec (model loaded once, timed 5x)
docker run --rm -it -e HF_TOKEN="your_token_here" -v "$PWD/cache:/cache" \
  xeon-ai-gemma --repeat 5 --max-new-tokens 512
```

To use a different image at runtime, mount it in and reference the mounted
path:

```bash
docker run --rm -it -e HF_TOKEN="your_token_here" \
  -v "$PWD/cache:/cache" \
  -v "$PWD/photo.jpg:/app/photo.jpg" \
  xeon-ai-gemma --image photo.jpg
```

Enable offline mode inside the container (after the model is cached) by adding
another `-e` flag. `HF_HUB_OFFLINE=1` skips the Hugging Face update check, so
the model must already be in your mounted `./cache` from a previous online run:

```bash
docker run --rm -it \
  -e HF_TOKEN="your_token_here" \
  -e HF_HUB_OFFLINE="1" \
  -v "$PWD/cache:/cache" \
  -v "$PWD/photo.jpg:/app/photo.jpg" \
  xeon-ai-gemma --image photo.jpg
```

Or run offline against the image baked into the container (`image.png`), with
no extra file mount:

```bash
docker run --rm -it \
  -e HF_TOKEN="your_token_here" \
  -e HF_HUB_OFFLINE=1 \
  -v "$PWD/cache:/cache" \
  xeon-ai-gemma
```

## Notes

- **INT8 quantized + OpenVINO IR:** ships pre-quantized to int8 in OpenVINO
  format, giving lower memory use and faster inference than an FP16/FP32 build,
  while the OpenVINO runtime adds AMX/AVX-512 acceleration and the int8 KV-cache
  option your script uses.
- **AMX acceleration:** the container and the venv both inherit the host CPU
  flags automatically. The script auto-detects AMX and prints
  `AMX used: True` on capable Xeon CPUs — no extra flags needed.
- **Disable AMX:** set `ONEDNN_MAX_CPU_ISA` to a non-AMX instruction set before
  running. The script respects a value you provide and will print
  `AMX used: False`:

  ```bash
  # Drop to AVX-512 BF16 (no AMX)
  ONEDNN_MAX_CPU_ISA=avx512_core_bf16 python3 ./run_gemma.py

  # Combine with offline mode / timing
  time ONEDNN_MAX_CPU_ISA=avx512_core_bf16 HF_HUB_OFFLINE=1 python3 ./run_gemma.py
  ```

  Use `avx512_core_vnni` or `avx512_core` to go lower. With Docker, pass it in
  with `-e ONEDNN_MAX_CPU_ISA=avx512_core_bf16`.
- **KV cache precision:** defaults to int8 (`u8`) for faster decoding and lower
  memory. Disable with `--kv-cache-precision f16`.