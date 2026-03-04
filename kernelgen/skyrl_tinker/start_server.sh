#!/usr/bin/env bash
# Start the SkyRL Tinker-compatible local server.
# 2×H100 setup: GPU 0 = trainer (FSDP), GPU 1 = generator (vLLM).
# The training script connects to this via base_url=http://localhost:8000.

set -euo pipefail

SKYRL_DIR="${SKYRL_DIR:-/tmp/SkyRL}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"

# 2×H100 backend config:
#   colocate_all=false     → separate GPUs for training and inference
# Override BACKEND_CONFIG for other setups (e.g., single-GPU colocated mode).
BACKEND_CONFIG="${BACKEND_CONFIG:-{\"trainer.placement.colocate_all\": false}}"

# Clone SkyRL if not present
if [ ! -d "$SKYRL_DIR" ]; then
  echo "Cloning SkyRL into $SKYRL_DIR ..."
  git clone https://github.com/NovaSky-AI/SkyRL.git "$SKYRL_DIR"
fi

cd "$SKYRL_DIR"

# Kill any stale Ray processes from previous runs
ray stop --force 2>/dev/null || true

# Install deps with uv (tinker + fsdp extras give us the local Tinker server + trainer)
echo "Installing SkyRL dependencies with uv ..."
uv venv --python 3.12
uv sync --extra tinker --extra fsdp

echo "Starting SkyRL Tinker server on port $PORT with model $MODEL ..."
echo "Backend config: $BACKEND_CONFIG"
# GPU 0 = FSDP trainer, GPU 1 = vLLM generator
uv run --extra tinker --extra fsdp \
  -m skyrl.tinker.api \
  --base-model "$MODEL" \
  --backend fsdp \
  --backend-config "$BACKEND_CONFIG" \
  --port "$PORT"
