#!/bin/bash
# Setup script for RunPod training pod.
# Run this once after the pod starts: bash setup_runpod.sh
# Assumes RunPod PyTorch 2.1+ template (CUDA 12.1).

set -e

echo "=== Schema Model — RunPod Environment Setup ==="

# 1. System packages
apt-get update -qq && apt-get install -y -qq git wget unzip

# 2. Python packages — ML stack
pip install --upgrade pip -q
pip install \
  "transformers>=4.53.0" \
  "peft>=0.12.0" \
  "trl>=0.11.0" \
  "bitsandbytes>=0.43.0" \
  "accelerate>=0.34.0" \
  "datasets>=3.0.0" \
  qwen-vl-utils \
  -q
# Note: flash-attn intentionally excluded — takes 1h+ to compile and hangs SSH.
# Notebook falls back to eager attention + MAX_SEQ=12288 clip (OOM-safe on A100).

# 3. vLLM (for inference validation on pod)
pip install "vllm>=0.6.0" -q

# 4. Project dependencies
pip install \
  "anthropic>=0.34.0" \
  runpod \
  jupyter jupyterlab \
  tqdm pandas pyarrow \
  matplotlib seaborn plotly \
  requests warcio beautifulsoup4 lxml duckdb \
  jsonschema rdflib \
  playwright \
  -q

playwright install chromium

# 5. Clone/sync project code to /workspace
if [ ! -d "/workspace/schema" ]; then
  echo "Cloning project repo..."
  git clone https://github.com/Volcanex/schema.git /workspace/schema
else
  echo "Repo already present, pulling latest..."
  git -C /workspace/schema pull
fi

# 6. Set up environment file
if [ ! -f /workspace/schema/.env ]; then
  cp /workspace/schema/.env.example /workspace/schema/.env 2>/dev/null || \
    echo "HF_TOKEN=\nANTHROPIC_API_KEY=" > /workspace/schema/.env
  echo "Created .env — fill in your API keys at /workspace/schema/.env"
fi

# 7. Download base model (cached to network volume)
MODEL_DIR="/workspace/models/models--Qwen--Qwen3-VL-8B-Instruct"
if [ ! -d "$MODEL_DIR" ]; then
  echo "Downloading Qwen3-VL-8B-Instruct (~16GB — expect 5-10 min on fast pod)..."
  python -c "
from huggingface_hub import snapshot_download
import os
model_id = 'Qwen/Qwen3-VL-8B-Instruct'
token = os.getenv('HF_TOKEN')
print('Downloading model weights + processor...')
path = snapshot_download(model_id, token=token, cache_dir='/workspace/models')
print('Done. Cached at:', path)
"
else
  echo "Base model already present at $MODEL_DIR"
fi

# 8. Download training dataset from HuggingFace Hub
# HF repo layout: data/processed/train.jsonl, data/processed/eval.jsonl, data/screenshots_v2/*.png
# Downloaded to /workspace/schema so relative paths in JSONL resolve correctly
DATA_DIR="/workspace/schema/data/processed"
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
  echo "Downloading training dataset from HuggingFace Hub (~3.5GB)..."
  python -c "
import os
from huggingface_hub import snapshot_download
local = snapshot_download(
    repo_id='Volcanex/schema-ie-training',
    repo_type='dataset',
    token=os.getenv('HF_TOKEN'),
    local_dir='/workspace/schema',
)
print('Dataset downloaded to:', local)
print('Train data at: /workspace/schema/data/processed/train.jsonl')
"
  # Untar screenshots if they came as a tarball
  if [ -f "/workspace/schema/data/screenshots_v2.tar.gz" ] && [ ! -d "/workspace/schema/data/screenshots_v2" ]; then
    echo "Extracting screenshots tarball..."
    cd /workspace/schema/data && tar xzf screenshots_v2.tar.gz && rm screenshots_v2.tar.gz
    echo "Screenshots extracted"
  fi
else
  echo "Training data already present at $DATA_DIR/train.jsonl"
fi

echo ""
echo "=== Setup complete ==="
echo "Jupyter Lab URL: Check RunPod dashboard for the proxy URL"
echo "Start Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=schema-model"
