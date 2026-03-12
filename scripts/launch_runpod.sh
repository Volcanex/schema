#!/bin/bash
# launch_runpod.sh — spin up a RunPod A100 pod and run setup
# Usage: bash scripts/launch_runpod.sh [--gpu "NVIDIA A100 80GB PCIe"]
#
# Prereqs:
#   ~/bin/runpodctl v2.1.6+
#   RUNPOD_API_KEY set (loaded from .env if present)
#   HF_TOKEN set (loaded from .env if present)

set -e

# ── Load .env ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
  export $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs)
  echo "Loaded .env"
fi

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set — add it to .env or export it}"
: "${HF_TOKEN:?HF_TOKEN not set — add it to .env or export it}"

RUNPODCTL="$HOME/bin/runpodctl"
[ -x "$RUNPODCTL" ] || { echo "runpodctl not found at $RUNPODCTL"; exit 1; }

# ── Config ─────────────────────────────────────────────────────────────────────
GPU_ID="${1:-NVIDIA A100 80GB PCIe}"
POD_NAME="schema-training-$(date +%Y%m%d-%H%M)"
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK=50   # GB — OS + packages
VOLUME_DISK=100     # GB — model weights + dataset + checkpoints
PORTS="8888/http,22/tcp"

echo ""
echo "=== Launching RunPod pod ==="
echo "  GPU:   $GPU_ID"
echo "  Name:  $POD_NAME"
echo "  Image: $IMAGE"
echo "  Disk:  ${CONTAINER_DISK}GB container / ${VOLUME_DISK}GB volume"
echo ""

# ── Create pod ─────────────────────────────────────────────────────────────────
POD_JSON=$($RUNPODCTL pod create \
  --name "$POD_NAME" \
  --gpu-id "$GPU_ID" \
  --image "$IMAGE" \
  --container-disk-in-gb $CONTAINER_DISK \
  --volume-in-gb $VOLUME_DISK \
  --volume-mount-path /workspace \
  --ports "$PORTS" \
  --env "{\"HF_TOKEN\":\"$HF_TOKEN\",\"RUNPOD_API_KEY\":\"$RUNPOD_API_KEY\"}" \
  --cloud-type SECURE \
  2>&1)

echo "$POD_JSON"

POD_ID=$(echo "$POD_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null || \
         echo "$POD_JSON" | grep -oP '"id"\s*:\s*"\K[^"]+' | head -1)

if [ -z "$POD_ID" ]; then
  echo "ERROR: Could not parse pod ID from response."
  exit 1
fi

echo ""
echo "Pod created: $POD_ID"
echo "Waiting for pod to reach RUNNING state..."

# ── Wait for RUNNING ────────────────────────────────────────────────────────────
for i in $(seq 1 30); do
  sleep 10
  STATUS=$($RUNPODCTL pod get "$POD_ID" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    pods = d if isinstance(d, list) else [d]
    for p in pods:
        if p.get('id') == '$POD_ID':
            print(p.get('desiredStatus', p.get('status', 'UNKNOWN')))
            break
except: print('UNKNOWN')
" 2>/dev/null || echo "UNKNOWN")

  echo "  [$i/30] Status: $STATUS"
  if [[ "$STATUS" == "RUNNING" ]]; then
    break
  fi
  if [[ "$STATUS" == "EXITED" || "$STATUS" == "FAILED" ]]; then
    echo "Pod failed to start."
    exit 1
  fi
done

echo ""
echo "=== Pod is RUNNING ==="
echo ""

# ── Print SSH command ───────────────────────────────────────────────────────────
SSH_CMD=$($RUNPODCTL ssh connect "$POD_ID" 2>/dev/null || echo "")

echo "Pod ID:    $POD_ID"
echo "Pod Name:  $POD_NAME"
echo ""
echo "To SSH in:"
echo "  $RUNPODCTL ssh connect $POD_ID"
echo ""
echo "Once connected, run setup:"
echo "  git clone https://github.com/Volcanex/schema.git /workspace/schema"
echo "  bash /workspace/schema/scripts/setup_runpod.sh"
echo ""
echo "Jupyter Lab (after setup):"
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=schema-model"
echo "  Then open: RunPod dashboard → Connect → Port 8888"
echo ""
echo "To stop the pod when done:"
echo "  $RUNPODCTL pod stop $POD_ID"
echo "  $RUNPODCTL pod delete $POD_ID"
echo ""
echo "Pod saved to: /tmp/runpod_active_pod.txt"
echo "$POD_ID $POD_NAME" > /tmp/runpod_active_pod.txt
