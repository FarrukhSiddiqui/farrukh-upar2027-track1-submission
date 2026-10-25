# UPAR 2027 Track 1 — reproducible inference and training.
# Same base image as the Codabench evaluation container.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-train.txt .
# --no-deps for open_clip keeps the image's torch 2.4.1 untouched
RUN pip install --no-cache-dir tqdm gdown requests ftfy regex huggingface_hub \
    && pip install --no-cache-dir --no-deps open_clip_torch==3.3.0
COPY . .
RUN chmod +x run_pipeline.sh get_clip_weights.sh

# Default: inference with the released submission mounted at /weights
# (see README.md for the full commands).
CMD ["python", "predict.py", "--help"]
