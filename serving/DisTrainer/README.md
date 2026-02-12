# DisTrainer

Distributed GRPO Trainer using TorchTitan primitives (FSDP2, DCP).

## Overview

DisTrainer is a training-only component for async RL pipelines. It consumes pre-generated completions from JSONL files and trains the model using GRPO (Group Relative Policy Optimization).

**Model**: Arcee Trinity-Mini (26B parameters total, 3B active per token)
**Architecture**: Mixture of Experts (MoE)
**Training Method**: LoRA on attention layers only (q_proj, k_proj, v_proj, o_proj)

## Quick Start

### 4 GPU Setup (2 Generator + 2 Trainer)

```bash
# Use GPUs 2 and 3 for training (GPUs 0-1 for DisGenerator)
export CUDA_VISIBLE_DEVICES=2,3

# Launch with torchrun (2 GPUs)
cd serving/DisTrainer
torchrun --nproc_per_node=2 -m DisTrainer.train --config config/train_config.toml
```

### 8 GPU Setup (4 Generator + 4 Trainer)

```bash
# Use GPUs 4-7 for training (GPUs 0-3 for DisGenerator)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Launch with torchrun (4 GPUs)
cd serving/DisTrainer
torchrun --nproc_per_node=4 -m DisTrainer.train --config config/train_config.toml
```

### Trigger Training

The trainer exposes HTTP endpoints. Once running, you can trigger training via:

```bash
# Train for 5 steps
curl -X POST "http://localhost:8000/train?num_steps=5" \
  -H "Content-Type: application/json" \
  -d '{"num_steps": 5}'

# Check status
curl http://localhost:8000/status

# Save checkpoint
curl -X POST http://localhost:8000/checkpoint
```

## Configuration

Edit `config/train_config.toml`:

```toml
[model]
name = "arcee-ai/Trinity-Mini"  # Downloads from HuggingFace
adapter_path = ""  # Leave empty to start fresh
use_lora = true
lora_r = 8
lora_alpha = 16
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
model_type = "moe"  # Trinity-Mini is MoE
freeze_experts = true  # Freeze expert/MLP layers (vLLM compatibility)

[training]
learning_rate = 5e-6
beta = 0.01  # KL penalty coefficient
max_grad_norm = 1.0
num_generations_per_prompt = 4

[parallel_dims]
dp = 4  # Data parallel (adjust to match GPU count: 2 or 4)
tp = 1  # No tensor parallelism
pp = 1  # No pipeline parallelism

[data]
generations_dir = "./data/generations"
max_sequence_length = 8192

[checkpoint]
save_dir = "./models"
save_interval = 1  # Save checkpoint after every batch
keep_latest_k = 3
```

**Note**: For 2 GPU setup, change `dp = 2` in the config.

## Architecture

Trinity-Mini is a 26B parameter MoE model with 3B active parameters per token. DisTrainer uses:

- **LoRA Training**: Adapters on attention layers only (not expert layers)
- **Expert Freezing**: MLP/expert layers frozen (vLLM doesn't support LoRA on experts yet)
- **FSDP2**: Fully Sharded Data Parallelism for memory efficiency
- **DCP**: Distributed Checkpointing for fast saves

This configuration trains approximately **10-15% of total parameters** while maintaining model quality.

## Data Format

DisTrainer consumes JSONL batches from DisGenerator (`./data/generations/batch_*.jsonl`):

```json
{
  "group_id": "prompt-uuid",
  "prompt": "Why is the sky blue?",
  "prompt_ids": [12800, 15, 245],
  "completions": [
    {
      "text": "The sky appears blue because...",
      "completion_ids": [45, 12, 98],
      "reward": 0.85,
      "old_logprobs": [-0.12, -0.45],
      "action_mask": [1, 1, 1, 0, 0]
    }
  ]
}
```

## Checkpoint Format

Saved policies are named: `policy-{N}-{YYYYMMDD_HHMMSS}`

Each checkpoint contains:
- **DCP format**: Distributed state dict shards (for resuming training)
- **PEFT adapter**: HuggingFace LoRA adapter (for vLLM inference)
- **Signal file**: `.policy_ready` (triggers hot-swap in DisGenerator)

## Hot-Swap Policy Updates

DisTrainer automatically exports vLLM-compatible adapters after each checkpoint:

1. Saves checkpoint → `models/policy-N-timestamp/`
2. Creates `.policy_ready` signal file
3. DisGenerator detects new policy via `PolicyManager`
4. New trajectories use updated policy (zero downtime)

## Monitoring

### Console Logging

```
Step 1/100 | Loss: 0.234 | KL: 0.012 | Avg Reward: 0.85
Step 2/100 | Loss: 0.198 | KL: 0.009 | Avg Reward: 0.88
```

### Weights & Biases (Optional)

```bash
# Enable W&B logging
export WANDB_PROJECT="asyncrl-trinity"
export WANDB_RUN_NAME="trinity-grpo-run1"

# Disable W&B
export WANDB_DISABLED="true"
```

## Requirements

- **PyTorch**: 2.3+ (for FSDP2)
- **Transformers**: Latest (for Trinity-Mini support)
- **PEFT**: For LoRA adapters
- **GPUs**:
  - 4 GPU setup: 2x GPUs with 40GB+ VRAM (A100/H100)
  - 8 GPU setup: 4x GPUs with 40GB+ VRAM (A100/H100)
