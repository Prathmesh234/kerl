# skyrl_tinker

Drop-in replacement for the Tinker cloud sampler using **SkyRL tx** — an open-source local implementation of the Tinker REST API by NovaSky-AI (Anyscale).

## What is SkyRL tx?

[SkyRL](https://github.com/NovaSky-AI/SkyRL) is a modular full-stack RL library for LLMs. The `skyrl.tinker.api` module implements the exact same REST API that Thinking Machines' Tinker cloud service exposes, so the **`tinker` Python client package works unchanged** — you just point `base_url` at localhost instead of the cloud.

Architecture (1 trainer + 1 generator):
- **Generator** — vLLM serves Qwen3-8B for fast token sampling
- **Trainer** — FSDP trains the model with GRPO importance-sampling loss
- **Client** — `tinker.ServiceClient(base_url="http://localhost:8000")` talks to both via the Tinker REST API

Everything else in `kernelgentinker-multiturn.py` is identical to the parent folder. The only changes from the original are:

| Field | Original | Here |
|-------|----------|------|
| `base_url` | `None` (Tinker cloud) | `"http://localhost:8000"` |
| `model_name` | `Qwen/Qwen3-32B` | `Qwen/Qwen3-8B` |
| `log_path` | `...-qwen3-32b` | `...-qwen3-8b` |

## GPU Requirements

Qwen3-8B in bfloat16 needs:

| Setup | VRAM | Backend Config | Notes |
|-------|------|----------------|-------|
| 2× H100 80 GB | 80 GB each | `colocate_all=false`, `cpu_offload=true` | **Recommended.** GPU 0 = trainer, GPU 1 = generator |
| 1× H100 80 GB | 80 GB | `colocate_all=true` (default) | Trainer + generator share one GPU via sleep/wake |
| 2× A100 40 GB | 40 GB each | `colocate_all=false`, `cpu_offload=true` | One GPU per component |
| 1× A100 40 GB | 40 GB | — | Will OOM — too small |

CUDA 12.8+ required.

### Memory breakdown (2×H100, `colocate_all=false`, `cpu_offload=true`)

| Component | Device | Memory |
|-----------|--------|--------|
| Base model (bf16) | GPU 0 | ~16 GB |
| LoRA adapter (rank 16) | GPU 0 | ~200 MB |
| Gradients | GPU 0 | ~200 MB (LoRA only) |
| Activations (grad ckpt) | GPU 0 | ~2–4 GB |
| Optimizer states (Adam) | **CPU** | ~400 MB (LoRA only) |
| **GPU 0 total** | | **~20 GB / 80 GB** |
| vLLM inference engine | GPU 1 | ~18 GB |
| **GPU 1 total** | | **~18 GB / 80 GB** |

## Setup

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install training script dependencies

```bash
cd kernelgen/skyrl_tinker
uv sync
```

### 3. Clone and set up SkyRL (one-time)

```bash
git clone https://github.com/NovaSky-AI/SkyRL.git /tmp/SkyRL
cd /tmp/SkyRL
uv venv --python 3.12
uv sync --extra tinker --extra fsdp
```

### 4. Set environment variables

```bash
# HuggingFace token (required for gated models like Qwen3-8B)
export HF_TOKEN="hf_your_token_here"

# Tinker API key (dummy value for local server)
export TINKER_API_KEY="tml-dummy"

# Pin both GPUs
export CUDA_VISIBLE_DEVICES=0,1
```

## Running (2×H100)

### Terminal 1 — Start the SkyRL server

```bash
cd kernelgen/skyrl_tinker
export HF_TOKEN="hf_your_token_here"
export CUDA_VISIBLE_DEVICES=0,1
./start_server.sh
```

This clones SkyRL (if needed), installs deps with uv, kills stale Ray processes,
and starts `skyrl.tinker.api` on port 8000 with Qwen3-8B using the FSDP backend.

The default backend config for 2×H100 is:
```json
{
  "trainer.placement.colocate_all": false,
  "trainer.policy.fsdp_config.cpu_offload": true
}
```

- `colocate_all=false` — GPU 0 runs the FSDP trainer, GPU 1 runs vLLM inference
- `cpu_offload=true` — optimizer states live on CPU, preventing OOM on the training GPU

To override defaults:

```bash
MODEL=Qwen/Qwen3-8B PORT=8000 SKYRL_DIR=/tmp/SkyRL ./start_server.sh
```

For **single-GPU** mode (1× H100, colocated):

```bash
CUDA_VISIBLE_DEVICES=0 \
  BACKEND_CONFIG='{"trainer.placement.colocate_all": true}' \
  ./start_server.sh
```

Wait for the log line confirming the server is ready before starting the training script.

### Terminal 2 — Run training

```bash
cd kernelgen/skyrl_tinker
export HF_TOKEN="hf_your_token_here"
export TINKER_API_KEY="tml-dummy"
export CUDA_VISIBLE_DEVICES=0,1
uv run python kernelgentinker-multiturn.py
```

Override any config field via CLI (uses `chz`):

```bash
uv run python kernelgentinker-multiturn.py max_turns=4 batch_size=4 group_size=8
```

## Troubleshooting

### CUDA OOM on training GPU

If you see `torch.OutOfMemoryError` during model loading, ensure:
1. `cpu_offload=true` is in the backend config (offloads optimizer states to CPU)
2. No stale Ray processes are holding GPU memory: `ray stop --force`
3. No other processes are using the GPUs: `nvidia-smi`

### `ModuleNotFoundError: No module named 'tinker_cookbook'`

Run `uv sync` in the `skyrl_tinker/` directory — the dependency was added to `pyproject.toml`.

### `TINKER_API_KEY` error

Set `export TINKER_API_KEY="tml-dummy"` — the local server accepts any value.

### Server stuck at "Starting background engine..."

This is normal. The engine enters a polling loop waiting for client requests.
The model weights are downloaded lazily when the training script sends its
first `create_model` request. Check server logs after starting the training script.

### Stale Ray processes

If the server fails on restart, kill lingering Ray workers:

```bash
ray stop --force
pkill -9 -f "ray::"
```

## Files

```
skyrl_tinker/
├── AGENTS.md                      # this file
├── pyproject.toml                 # uv-managed deps (tinker client + training deps)
├── start_server.sh                # clones SkyRL, installs with uv, starts server
├── kernelgentinker-multiturn.py   # training script (identical to parent, 4 lines changed)
└── multi_turn_queue.py            # exact copy from parent
```

## References

- [SkyRL GitHub](https://github.com/NovaSky-AI/SkyRL)
- [SkyRL tx README](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-tx)
- [SkyRL Docs](https://docs.skyrl.ai/docs)
- [Tinker API (Thinking Machines)](https://github.com/thinking-machines-lab/tinker)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
