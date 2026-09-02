# Container image for run_gemma.py (OpenVINO GenAI, CPU/AMX inference).
FROM python:3.11-slim

# HF + OpenVINO caches live here; mount a volume at /cache to persist them.
ENV HF_HOME=/cache/hf \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install \
        openvino-genai==2026.3.1 \
        openvino==2026.3.1 \
        huggingface_hub pillow numpy py-cpuinfo

COPY run_gemma.py image.png ./

# Write the OpenVINO compiled-model cache into the mounted volume too.
ENTRYPOINT ["python3", "./run_gemma.py", "--cache-dir", "/cache/ov_cache"]
