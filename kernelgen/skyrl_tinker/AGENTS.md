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

| Setup | VRAM | Notes |
|-------|------|-------|
| 1× H100 80 GB | 80 GB | Trainer + generator share one GPU |
| 2× A100 40 GB | 40 GB each | One GPU per component, comfortable |
| 1× A100 40 GB | 40 GB | Will OOM — too small |

CUDA 12.8+ required.

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

## Running

### Terminal 1 — Start the SkyRL server

```bash
cd kernelgen/skyrl_tinker
./start_server.sh
```

This clones SkyRL (if needed), installs deps with uv, and starts `skyrl.tinker.api` on port 8000 with Qwen3-8B using the FSDP backend.

To override defaults:

```bash
MODEL=Qwen/Qwen3-8B PORT=8000 SKYRL_DIR=/tmp/SkyRL ./start_server.sh
```

Wait for the log line confirming the server is ready before starting the training script.

### Terminal 2 — Run training

```bash
cd kernelgen/skyrl_tinker
uv run python kernelgentinker-multiturn.py
```

Override any config field via CLI (uses `chz`):

```bash
uv run python kernelgentinker-multiturn.py max_turns=4 batch_size=4 group_size=8
```

To pin specific GPUs:

```bash
# 2-GPU: GPU 0 for generator, GPU 1 for trainer
CUDA_VISIBLE_DEVICES=0,1 ./start_server.sh   # Terminal 1
CUDA_VISIBLE_DEVICES=0,1 uv run python kernelgentinker-multiturn.py  # Terminal 2

# Single 80 GB GPU
CUDA_VISIBLE_DEVICES=0 ./start_server.sh
CUDA_VISIBLE_DEVICES=0 uv run python kernelgentinker-multiturn.py
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
- [SkyRL Docs](https://docs.skyrl.ai/docs)
- [SkyRL tx announcement](https://x.com/tyler_griggs_/status/1975245326463475860)
- [Tinker API (Thinking Machines)](https://github.com/thinking-machines-lab/tinker)
