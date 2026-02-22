# -*- coding: utf-8 -*-
"""
KernelBench SFT with Tinker
============================
Same thing as the SFTTrainer notebook, but on Tinker.

Loads ppbhatt500/kernelbook-triton-reasoning-traces,
formats as chat messages (user: pytorch_code, assistant: <think>...<triton>...),
trains Qwen3-8B with cross_entropy loss on the assistant turn via LoRA.

Usage:
  python kernelbench_sft_tinker.py
  python kernelbench_sft_tinker.py learning_rate=1e-4 n_epochs=2
"""

import os
import logging
import time
import random

import chz
import datasets
import torch

import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from kernelgentinker import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARN)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "./tmp/kernelbench-sft-qwen3-8b"
    model_name: str = "Qwen/Qwen3-8B"
    dataset_name: str = "ppbhatt500/kernelbook-triton-reasoning-traces"
    batch_size: int = 4
    learning_rate: float = 2e-5
    lora_rank: int = 128
    max_length: int = 16384
    n_epochs: int = 3
    save_every: int = 40
    eval_every: int = 20
    eval_size: int = 17  # ~10% of 170
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE


# ---------------------------------------------------------------------------
# Dataset -> chat messages (mirrors notebook's format_trace_for_sft)
# ---------------------------------------------------------------------------

def _print_example(example_idx: int, pytorch_code: str, user_content: str, full_completion: str):
    print("=" * 80)
    print(f"EXAMPLE {example_idx} BEFORE SENDING TO TICKER DATUM PREP")
    print("=" * 80)
    print("--- PYTORCH CODE ---")
    print(pytorch_code)
    print("\n--- USER CONTENT ---")
    print(user_content)
    print("\n--- FULL COMPLETION ---")
    print(full_completion)
    print("=" * 80)

def load_and_format_dataset(config: Config) -> list[list[dict]]:
    """Load HF dataset and convert each row to [user, assistant] messages."""
    ds = datasets.load_dataset(config.dataset_name, split="train")
    logger.info(f"Loaded {len(ds)} rows from {config.dataset_name}")

    conversations = []
    for i, row in enumerate(ds):
        pytorch_code = row["pytorch_code"]

        reasoning = row.get("model_reasoning", "")
        if reasoning:
            triton_code = row.get("triton_code", "")
            think_part = f"<think>\n{reasoning}\n</think>"
            triton_part = f"<triton>\n{triton_code}\n</triton>" if triton_code else "<triton>\n</triton>"
            full_completion = f"{think_part}\n\n{triton_part}"
        else:
            full_completion = row.get("full_completion", "")

        user_content = (
            f"Convert the following PyTorch code to an optimized Triton kernel:\n\n"
            f"```python\n{pytorch_code}\n```\n\n"
            f"Generate a complete Triton implementation that produces the same output as the PyTorch code."
        )

        if i == 1:
            _print_example(i, pytorch_code, user_content, full_completion)

        conversations.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": full_completion},
        ])

    return conversations


# ---------------------------------------------------------------------------
# Chat messages -> Tinker Datums
# ---------------------------------------------------------------------------

def make_datums(
    conversations: list[list[dict]],
    renderer: renderers.Renderer,
    max_length: int,
    train_on_what: renderers.TrainOnWhat,
) -> list[types.Datum]:
    """
    renderer.build_supervised_example() converts messages -> (model_input, weights)
    where weights=1 for tokens we train on (assistant turn), weights=0 for context.
    Then we wrap into a Datum for cross_entropy loss.
    """
    datums = []
    skipped = 0

    for messages in conversations:
        try:
            model_input, weights = renderer.build_supervised_example(
                messages, train_on_what=train_on_what
            )
            tokens = model_input.to_ints()

            # Truncate
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
                weights = weights[:max_length]

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(tokens[1:])),
                    "weights": TensorData.from_torch(
                        weights[1:].float() if isinstance(weights, torch.Tensor)
                        else torch.tensor(weights[1:], dtype=torch.float32)
                    ),
                },
            )
            datums.append(datum)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                logger.warning(f"Skipped example: {e}")

    if skipped:
        logger.info(f"Skipped {skipped}/{skipped + len(datums)} examples")
    return datums


import numpy as np

# ---------------------------------------------------------------------------
# Helper: compute loss from Tinker's loss_fn_outputs
# ---------------------------------------------------------------------------

def compute_loss(fwd_bwd_result, batch):
    """
    Per the Tinker docs, loss_fn_outputs is a list of dicts (one per datum),
    each containing 'logprobs' (per-token log-probs). We compute:
      loss = -dot(logprobs, weights) / sum(weights)
    """
    logprobs = np.concatenate([
        np.array(output['logprobs'].tolist() if hasattr(output['logprobs'], 'tolist') else output['logprobs'])
        for output in fwd_bwd_result.loss_fn_outputs
    ])
    weights = np.concatenate([
        np.array(d.loss_fn_inputs['weights'].tolist() if hasattr(d.loss_fn_inputs['weights'], 'tolist') else d.loss_fn_inputs['weights'])
        for d in batch
    ])
    total_weight = weights.sum()
    if total_weight == 0:
        return None
    return float(-np.dot(logprobs, weights) / total_weight)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main(config: Config):
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path, wandb_project="kernelbench-sft", wandb_name="qwen-8b-sft-lora256",
        config=config, do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Model: {config.model_name} | Renderer: {renderer_name}")

    # Load data
    all_convos = load_and_format_dataset(config)
    random.seed(42)
    random.shuffle(all_convos)
    eval_convos = all_convos[:config.eval_size]
    train_convos = all_convos[config.eval_size:]
    logger.info(f"Train: {len(train_convos)} | Eval: {len(eval_convos)}")

    # Convert to datums
    train_datums = make_datums(train_convos, renderer, config.max_length, config.train_on_what)
    eval_datums = make_datums(eval_convos, renderer, config.max_length, config.train_on_what)
    logger.info(f"Train datums: {len(train_datums)} | Eval datums: {len(eval_datums)}")

    # Tinker setup
    service_client = tinker.ServiceClient(base_url=config.base_url)
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)

    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        start_step = resume_info.get("batch", 0)
        logger.info(f"Resuming from step {start_step}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_step = 0

    adam_params = types.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    # Training
    steps_per_epoch = max(1, len(train_datums) // config.batch_size)
    total_steps = steps_per_epoch * config.n_epochs
    logger.info(f"{steps_per_epoch} steps/epoch x {config.n_epochs} epochs = {total_steps} total steps")

    global_step = start_step
    for epoch in range(config.n_epochs):
        random.shuffle(train_datums)

        for batch_idx in range(steps_per_epoch):
            if global_step < start_step:
                global_step += 1
                continue

            t0 = time.time()

            # Checkpoint
            if config.save_every > 0 and global_step > 0 and global_step % config.save_every == 0:
                checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"sft-{global_step:06d}",
                    log_path=config.log_path, kind="both",
                    loop_state={"batch": global_step},
                )

            # Batch
            start = batch_idx * config.batch_size
            batch = train_datums[start : start + config.batch_size]

            # Train step (pipelined: submit both before waiting)
            fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_future = training_client.optim_step(adam_params)
            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            # Metrics
            metrics = {
                "progress/step": global_step,
                "progress/epoch": epoch,
                "progress/done_frac": (global_step + 1) / total_steps,
            }

            # Extract train loss: -dot(logprobs, weights) / sum(weights)
            train_loss = compute_loss(fwd_bwd_result, batch)
            if train_loss is not None:
                metrics["train/loss"] = train_loss

            # Eval (use forward() — no gradients, no weight updates)
            if config.eval_every > 0 and global_step % config.eval_every == 0 and eval_datums:
                eval_batch = eval_datums[:config.batch_size]
                eval_result = training_client.forward(
                    eval_batch, loss_fn="cross_entropy"
                ).result()
                eval_loss = compute_loss(eval_result, eval_batch)
                if eval_loss is not None:
                    metrics["eval/loss"] = eval_loss

            metrics["time/step"] = time.time() - t0
            ml_logger.log_metrics(metrics, step=global_step)

            parts = [f"Step {global_step}/{total_steps} (epoch {epoch})"]
            if "train/loss" in metrics:
                parts.append(f"train={metrics['train/loss']:.4f}")
            if "eval/loss" in metrics:
                parts.append(f"eval={metrics['eval/loss']:.4f}")
            parts.append(f"{metrics['time/step']:.1f}s")
            logger.info(" | ".join(parts))

            global_step += 1

    # Final save
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"sft-final-{global_step}",
        log_path=config.log_path, kind="both",
        loop_state={"batch": global_step},
    )
    sampler_path = training_client.save_weights_for_sampler(name="sft-final").result().path
    logger.info(f"SFT done! Sampler weights: {sampler_path}")
    logger.info(f"Use for GRPO: python kernelbench_grpo_tinker.py sft_checkpoint_path={sampler_path}")

    ml_logger.close()
    return sampler_path


if __name__ == "__main__":
    chz.nested_entrypoint(main, argv=[])