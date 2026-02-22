#!/bin/bash
set -e
cd /Users/prathmeshbhatt/Desktop/kernelgen/SVD

# Load variables from .env
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -z "$CHECKPOINT_PATH" ]; then
  echo "Error: CHECKPOINT_PATH is not set in .env"
  exit 1
fi

echo "Downloading $CHECKPOINT_PATH ..."
uv run tinker checkpoint download "$CHECKPOINT_PATH" --output ./

echo "Running analysis..."
uv run python3 delta_W_Matrix_Heatmap.py

echo "Cleaning up extracted folders..."
RUN_ID=$(echo "$CHECKPOINT_PATH" | awk -F'//' '{print $2}' | awk -F':' '{print $1}')
if [ -n "$RUN_ID" ]; then
  rm -rf ${RUN_ID}*
fi

open heatmap_deltaW.png
