# Chatterbox-Turbo (350M, MIT) behind Wyoming, in one process.
# chatterbox-tts 0.1.7 pins torch==2.6.0, which has no sm_120 (Blackwell) build;
# an RTX 50-series card needs cu128, first shipped in torch 2.7. So: torch
# 2.9.1+cu128 from the base image, chatterbox installed --no-deps,
# its other requirements listed here. gradio left out: only its demo UI uses it.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install numpy==1.26.4 librosa==0.11.0 s3tokenizer==0.3.0 transformers==5.2.0 diffusers==0.29.0 \
        resemble-perth==1.0.1 conformer==0.3.2 safetensors==0.5.3 spacy-pkuseg==1.0.1 pykakasi==2.3.0 \
        pyloudnorm==0.2.0 omegaconf==2.3.1 wyoming==1.8.0 \
    && pip install --no-deps chatterbox-tts==0.1.7
COPY shim.py /app/shim.py
ENTRYPOINT ["python", "/app/shim.py"]
