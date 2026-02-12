#!/bin/bash
# =============================================================================
# KernelBench Environment - Deploy Modal Container
# =============================================================================
# Checks if Modal is configured, syncs deps with uv, and deploys the
# Modal app (builds the H100 container image if it doesn't exist).
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "KernelBench Environment - Modal Deployment"
echo "=============================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# Sync project dependencies
echo "Syncing project dependencies..."
cd "$SCRIPT_DIR"
uv sync

# Check if Modal is configured
if ! uv run --no-sync modal token info &> /dev/null; then
    echo ""
    echo "ERROR: Modal not configured!"
    echo "Run: uv run modal token set --token-id <ID> --token-secret <SECRET>"
    echo "Get your token from: https://modal.com/settings"
    exit 1
fi

echo "Modal token verified."
echo ""

# Deploy the Modal app (builds container image if not cached)
echo "Deploying Modal app (kernelbench-triton)..."
echo "This will build the H100 container image if it doesn't exist."
echo ""

uv run --no-sync modal deploy "$SCRIPT_DIR/modal_app.py"

echo ""
echo "=============================================="
echo "Deployment complete!"
echo "=============================================="
echo ""
echo "Available remote functions:"
echo "  - benchmark_triton_kernel  (generic input_shapes)"
echo "  - benchmark_kernelbench    (nn.Module pattern)"
echo "  - benchmark_batch          (sequential batch)"
echo ""
echo "Usage from Python:"
echo "  from orchestrator import KernelBenchOrchestrator"
echo "  with KernelBenchOrchestrator() as orch:"
echo "      result = orch.run(triton_code=..., pytorch_code=...)"
