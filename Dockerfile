# Chatterbox-Turbo (350M, MIT) behind Wyoming, in one process. NVIDIA (CUDA).
# chatterbox-tts 0.1.7 pins torch==2.6.0, which has no sm_120 (Blackwell) build;
# an RTX 50-series card needs cu128, first shipped in torch 2.7. So: torch
# 2.9.1+cu128 from the base image, chatterbox installed --no-deps.
# AMD and Intel GPUs: Dockerfile.wheels.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt && pip install --no-deps chatterbox-tts==0.1.7
COPY shim.py /app/shim.py
ENTRYPOINT ["python", "/app/shim.py"]
