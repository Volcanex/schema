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
  transformers>=4.45.0 \
  peft>=0.12.0 \
  trl>=0.11.0 \
  bitsandbytes>=0.43.0 \
  accelerate>=0.34.0 \
  datasets>=3.0.0 \
  qwen-vl-utils \
  flash-attn --no-build-isolation \
  -q

# 3. vLLM (for inference validation on pod)
pip install vllm>=0.6.0 -q

# 4. Project dependencies
pip install \
  anthropic>=0.34.0 \
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
# If code is on a network volume, it's already at /workspace.
# Otherwise, clone from GitHub:
# git clone https://github.com/YOUR_ORG/schema-model.git /workspace/schema-model

# 6. Set up environment file
if [ ! -f /workspace/.env ]; then
  cp /workspace/schema-model/.env.example /workspace/schema-model/.env
  echo "Created .env from example — fill in your API keys"
fi

# 7. Download base model (cached to network volume)
MODEL_DIR="/workspace/models/Qwen2.5-VL-7B-Instruct"
if [ ! -d "$MODEL_DIR" ]; then
  echo "Downloading base model..."
  python -c "
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import os
model_id = 'Qwen/Qwen2.5-VL-7B-Instruct'
token = os.getenv('HF_TOKEN')
print('Downloading processor...')
AutoProcessor.from_pretrained(model_id, token=token, cache_dir='/workspace/models')
print('Downloading model weights...')
Qwen2VLForConditionalGeneration.from_pretrained(model_id, token=token, cache_dir='/workspace/models')
print('Done.')
"
else
  echo "Base model already present at $MODEL_DIR"
fi

echo ""
echo "=== Setup complete ==="
echo "Jupyter Lab URL: Check RunPod dashboard for the proxy URL"
echo "Start Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=schema-model"
