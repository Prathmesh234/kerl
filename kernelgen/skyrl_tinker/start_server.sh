#!/usr/bin/env bash
# Start the SkyRL Tinker-compatible local server.
# 1 trainer + 1 generator on Qwen3-8B.
# The training script connects to this via base_url=http://localhost:8000.

set -euo pipefail

SKYRL_DIR="${SKYRL_DIR:-/tmp/SkyRL}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"

# Clone SkyRL if not present
if [ ! -d "$SKYRL_DIR" ]; then
  echo "Cloning SkyRL into $SKYRL_DIR ..."
  uv tool run git clone https://github.com/NovaSky-AI/SkyRL.git "$SKYRL_DIR"
fi

cd "$SKYRL_DIR"

# Install deps with uv (tinker + fsdp extras give us the local Tinker server + trainer)
echo "Installing SkyRL dependencies with uv ..."
uv venv --python 3.12
uv sync --extra tinker --extra fsdp

echo "Starting SkyRL Tinker server on port $PORT with model $MODEL ..."
# 1 trainer (FSDP) + 1 generator (vLLM) — default for a single-node run
uv run --extra tinker --extra fsdp \
  -m skyrl.tinker.api \
  --base-model "$MODEL" \
  --backend fsdp \
  --port "$PORT"
