# -*- coding: utf-8 -*-
"""
KernelBench GRPO with Tinker
============================
Reinforcement Learning (GRPO) pipeline for training Qwen3-8B to generate
optimized Triton kernels from PyTorch code.

Uses KernelBench Level 1 + Level 2 problems (200 tasks):
  - 20 held out for eval
  - 180 used for training

Reward:
  - 0.0  if no <triton> tag
  - -0.5 if missing triton_kernel_wrapper
  - -0.5 if execution crash or incorrect output
  -  1.0 + log(speedup) if correct (capped at 5.0)

Usage:
  python kernelgentinker.py
  python kernelgentinker.py learning_rate=5e-5 batch_size=8
  python kernelgentinker.py sft_checkpoint_path=tinker://...
"""

import logging
import math
import os
import random
import re
import time
from collections import Counter, deque

import chz
import datasets
import modal
import torch
from dotenv import load_dotenv
from tqdm import tqdm

import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARN)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "./tmp/kernelbench-grpo-qwen3-8b"
    model_name: str = "Qwen/Qwen3-8B"
    sft_checkpoint_path: str | None = None   # pass in sampler weights from SFT to warm-start
    batch_size: int = 4          # problems per GRPO batch
    group_size: int = 8           # completions per problem
    learning_rate: float = 2e-5
    lora_rank: int = 32           # reduced from 128 to prevent overfitting
    save_every: int = 40
    max_tokens: int = 16384
    keep_last_sampler_checkpoints: int = 2
    eval_size: int = 20           # held-out problems for eval
    eval_every: int = 10          # eval frequency in batches


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer. Your task is to convert PyTorch code into optimized Triton kernels.

=== TRITON PRIMER: CORE CONCEPTS ===

1. SPMD PROGRAMMING MODEL
   Triton uses Single Program, Multiple Data: the SAME kernel code runs on many GPU threads simultaneously.
   Each thread (program instance) processes a different portion of data.

   Key insight: Use tl.program_id() to determine which data THIS instance should process.

   Example: To process array of size N with BLOCK_SIZE per program:
   - Program 0 processes elements [0, BLOCK_SIZE)
   - Program 1 processes elements [BLOCK_SIZE, 2*BLOCK_SIZE)
   - etc.

2. COMPILE-TIME vs RUNTIME
   Triton kernels are COMPILED before execution. Some values must be known at compile-time.

   COMPILE-TIME (tl.constexpr):
   - BLOCK_SIZE, num_warps, num_stages
   - Arguments to tl.arange(start, end) - both must be constants
   - Tensor shape parameters marked with : tl.constexpr

   RUNTIME:
   - Actual data values
   - Loop bounds (range(0, N, BLOCK_SIZE) is fine - N can be runtime)
   - Loaded tensor elements

   CRITICAL: tl.arange(0, BLOCK_SIZE) is valid but tl.arange(0, n) where n is runtime is NOT.
   Solution: Use fixed BLOCK_SIZE with masking for boundaries.

3. MEMORY SAFETY
   GPU memory is accessed via pointers. Out-of-bounds access causes crashes.

   Always use MASKING:
   offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
   mask = offsets < N
   data = tl.load(ptr + offsets, mask=mask, other=0.0)

4. TRITON LANGUAGE SCOPE
   Inside @triton.jit functions, you are in Triton-land:
   - Use tl.* operations ONLY
   - No torch.* functions
   - No Python control flow on tensor data (use tl.where instead)

   Outside (in wrapper), you are in Python-land:
   - Use torch.* freely
   - Allocate tensors, compute grid sizes
   - Launch kernels

5. MATRIX OPERATIONS
   tl.dot(A, B) performs matrix multiplication:
   - Requires A.shape = (M, K) and B.shape = (K, N)
   - Results in shape (M, N)
   - Use tl.trans(B) if B is (N, K) to get (K, N)

=== YOUR TASK ===

For each PyTorch operation, you should:
1. Analyze the operation and memory access patterns
2. Think step-by-step about how to parallelize it
3. Choose appropriate BLOCK_SIZE and num_warps
4. Write the complete Triton kernel implementation

Your response MUST follow this format:

<think>
[Your step-by-step reasoning about the conversion]
- What operation is being performed?
- What are the input/output shapes?
- How should this be parallelized?
- What memory access pattern should be used?
- What BLOCK_SIZE and num_warps are optimal?
</think>

<triton>
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_name(...):
    # Your Triton kernel implementation
    ...

def triton_kernel_wrapper(input_tensors):
    # Wrapper that calls the kernel and returns output
    ...
</triton>

=== CRITICAL REQUIREMENTS ===

1. The wrapper function MUST be named `triton_kernel_wrapper`
2. The wrapper takes the SAME inputs as Model.forward() - just the input tensors, NOT model weights
3. If the model has weights (nn.Linear, nn.Conv2d, etc.), the wrapper should create random weights or accept them as additional parameters
4. IMPORTANT: If get_init_inputs() returns parameters, the wrapper MUST accept these as keyword arguments with defaults matching those values
5. Triton API Limitations: tl.tanh, tl.pow, tl.unsqueeze do NOT exist - use tl.exp for tanh, ** operator for pow, reshape for unsqueeze

=== TRITON KERNEL RULES - MUST FOLLOW ===

IMPORTS - Only use these:
  import triton
  import triton.language as tl

INSIDE @triton.jit KERNELS - Use ONLY triton.language (tl) operations:
- tl.load(), tl.store() - memory access
- tl.arange(), tl.zeros(), tl.full() - tensor creation
- tl.sum(), tl.max(), tl.min() - reductions
- tl.exp(), tl.log(), tl.sqrt(), tl.abs() - math ops
- tl.maximum(), tl.minimum() - element-wise min/max
- tl.where() - conditional selection
- tl.program_id(), tl.num_programs() - grid info
- Standard operators: +, -, *, /, %, <, >, ==, &, |

NEVER use inside @triton.jit:
- torch.* functions (torch.sum, torch.mean, torch.relu, etc.)
- .pow(), .sqrt(), .exp() methods on tensors - use ** operator for pow, tl.sqrt(), tl.exp()
- Python classes or objects
- nn.* modules

CONSTEXPR RULES:
- tl.arange(start, end) - both start and end MUST be constants or tl.constexpr
- BLOCK_SIZE: tl.constexpr in kernel signature
- Use powers of 2: 64, 128, 256, 512, 1024

WRAPPER FUNCTION PATTERN:
  def triton_kernel_wrapper(x):
      output = torch.empty_like(x)
      BLOCK_SIZE = 1024
      grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
      my_kernel[grid](x, output, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
      return output

COMMON OPERATIONS:
- ReLU: tl.maximum(x, 0.0)
- Sigmoid: 1.0 / (1.0 + tl.exp(-x))
- Tanh: (tl.exp(2*x) - 1) / (tl.exp(2*x) + 1)
- Softmax: exp_x = tl.exp(x - tl.max(x)); exp_x / tl.sum(exp_x)
- Mean: tl.sum(x) / n_elements
"""


# ---------------------------------------------------------------------------
# Tag extraction utilities
# ---------------------------------------------------------------------------

def extract_think(content: str) -> str | None:
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_triton(content: str) -> str | None:
    m = re.search(r"<triton>(.*?)</triton>", content, re.DOTALL)
    return m.group(1).strip() if m else None


def format_reward(content: str) -> float:
    has_think = bool(re.search(r"<think>.*?</think>", content, re.DOTALL))
    has_triton = bool(re.search(r"<triton>.*?</triton>", content, re.DOTALL))
    if has_think and has_triton:
        return 1.0
    elif has_triton:
        return 0.5
    return 0.0


def save_rollout_to_jsonl(log_path: str, data: dict):
    """Append a rollout dictionary as a JSON line for later analysis."""
    import json
    os.makedirs(log_path, exist_ok=True)
    filepath = os.path.join(log_path, "rollouts.jsonl")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")




# ---------------------------------------------------------------------------
# Modal benchmark client
# ---------------------------------------------------------------------------

benchmark_kernelbench = modal.Function.from_name("kernelbench-triton", "benchmark_kernelbench")


def get_reward(content: str, pytorch_code: str) -> float:
    """
    Compute reward for a generated completion.

    Reward decomposition:
      - No <triton> tags              -> 0.0
      - Missing triton_kernel_wrapper -> -0.5
      - Correct kernel                ->  1.0 + log(speedup), capped at 5.0
      - Incorrect output              -> -0.5
      - Execution crash               -> -0.5
    """
    triton_code = extract_triton(content)
    if triton_code is None:
        return 0.0

    if "def triton_kernel_wrapper" not in triton_code:
        logger.debug("  ⚠ Missing triton_kernel_wrapper")
        return -0.5

    try:
        result = benchmark_kernelbench.remote(
            triton_code=triton_code,
            pytorch_code=pytorch_code,
            n_correctness=5,
            n_trials=20,
            kernel_name="tinker_grpo_kernel",
            rtol=1e-4,
            atol=1e-4,
        )

        if result.get("error"):
            logger.debug(f"  ❌ Execution error: {result['error'][:100]}")
            return -0.5

        if not result["correctness"]:
            logger.debug("  ✗ Incorrect output")
            return -0.5

        speedup = result.get("speedup", 1.0)
        speedup_bonus = max(0.0, min(4.0, math.log(max(speedup, 1e-6))))
        reward = 1.0 + speedup_bonus
        logger.debug(f"  ✅ Correct | {speedup:.2f}x | reward={reward:.3f}")
        return reward

    except Exception as e:
        logger.debug(f"  ❌ Modal call failed: {e}")
        return -0.5


# ---------------------------------------------------------------------------
# Dataset loading — Level 1 + Level 2 only (200 problems)
# ---------------------------------------------------------------------------

def load_kernelbench_problems() -> list[dict]:
    """Load Level 1 and Level 2 KernelBench problems only (~200 tasks)."""
    all_problems = []
    for level_split in ["level_1", "level_2"]:
        ds = datasets.load_dataset("ScalingIntelligence/KernelBench", split=level_split)
        for item in ds:
            all_problems.append({
                "code": item["code"],
                "level": item["level"],
                "name": item["name"],
                "problem_id": item["problem_id"],
            })
    return all_problems


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main(config: Config):
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project="kernelgen-grpo-qwen3-8b",
        wandb_name="kernelgen-grpo-run",
        config=config,
        do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Model: {config.model_name} | Renderer: {renderer_name}")

    # Load and split problems
    all_problems = load_kernelbench_problems()
    random.seed(42)
    random.shuffle(all_problems)
    eval_problems = all_problems[:config.eval_size]
    train_problems = all_problems[config.eval_size:]
    logger.info(f"Loaded {len(all_problems)} problems (L1+L2) | Train: {len(train_problems)} | Eval: {len(eval_problems)}")

    n_train_batches = len(train_problems) // config.batch_size
    logger.info(f"Training: {n_train_batches} batches of {config.batch_size} problems")

    # -------------------------------------------------------------------------
    # Tinker client setup
    # -------------------------------------------------------------------------
    service_client = tinker.ServiceClient(base_url=config.base_url)
    rest_client = service_client.create_rest_client()
    sampler_ckpt_queue: deque[str] = deque()
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)

    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    elif config.sft_checkpoint_path:
        training_client = service_client.create_training_client_from_state(
            config.sft_checkpoint_path
        )
        start_batch = 0
        logger.info(f"Warm-starting from SFT checkpoint: {config.sft_checkpoint_path}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0
        logger.info("Starting from base model (no SFT checkpoint)")

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=0.8,
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Checkpoint
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"grpo-{batch_idx:06d}",
                log_path=config.log_path,
                kind="both",
                loop_state={"batch": batch_idx},
            )

        # Frozen sampler weights for this batch
        sampling_path = (
            training_client.save_weights_for_sampler(name=f"{batch_idx:06d}").result().path
        )
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)
        ##this is basically saving our weights into a deque and updating it. As soon as we are done with a checkpoint we pop it 
        ##if it has > 2 then we popeleft from ti and delete the checkpoint from tinker
        sampler_ckpt_queue.append(sampling_path)
        if config.keep_last_sampler_checkpoints > 0:
            while len(sampler_ckpt_queue) > config.keep_last_sampler_checkpoints:
                old_path = sampler_ckpt_queue.popleft()
                rest_client.delete_checkpoint_from_tinker_path(old_path).result()

        # Get batch of problems
        batch_start = batch_idx * config.batch_size
        batch_problems = train_problems[batch_start : batch_start + config.batch_size]

        # -----------------------------------------------------------------
        # Step 1: Build prompts and dispatch sampling
        # -----------------------------------------------------------------
        futures_P: list = []
        codes_P: list[str] = []
        prompts_P: list[list[int]] = []
        problems_P: list[dict] = []  # track which problem each future belongs to

        for problem in batch_problems:
            pytorch_code = problem["code"]
            user_content = (
                f"Convert the following PyTorch code to an optimized Triton kernel:\n\n"
                f"```python\n{pytorch_code}\n```\n\n"
                f"Generate a complete Triton implementation that produces the same output as the PyTorch code."
            )
            convo = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            model_input = renderer.build_generation_prompt(convo)
            prompt_tokens = model_input.to_ints()

            future = sampling_client.sample(
                prompt=model_input,
                num_samples=config.group_size,
                sampling_params=sampling_params,
            )
            futures_P.append(future)
            codes_P.append(pytorch_code)
            prompts_P.append(prompt_tokens)
            problems_P.append(problem)

        # -----------------------------------------------------------------
        # Step 2: Collect completions, compute rewards
        # -----------------------------------------------------------------
        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        n_correct = 0
        n_total = 0

        for future, prompt_tokens, pytorch_code, prob in tqdm(
            zip(futures_P, prompts_P, codes_P, problems_P),
            total=len(futures_P),
            desc=f"Batch {batch_idx}",
        ):
            sample_result = future.result()
            rewards_G: list[float] = []
            tokens_G_T: list[list[int]] = []
            logprobs_G_T: list[list[float]] = []
            ob_lens_G: list[int] = []

            for sequence in sample_result.sequences:
                sampled_tokens = sequence.tokens
                sampled_logprobs = sequence.logprobs
                assert sampled_logprobs is not None

                all_tokens = prompt_tokens + sampled_tokens
                tokens_G_T.append(all_tokens)
                ob_lens_G.append(len(prompt_tokens) - 1)
                logprobs_G_T.append(sampled_logprobs)

                parsed_message, _ = renderer.parse_response(sampled_tokens)
                content = renderers.get_text_content(parsed_message)

                fmt_r = format_reward(content)
                if fmt_r < 0.5:
                    reward = 0.0
                else:
                    reward = get_reward(content, pytorch_code)

                # Save rollout for analysis
                save_rollout_to_jsonl(config.log_path, {
                    "batch_idx": batch_idx,
                    "problem_name": prob.get("name", ""),
                    "problem_level": prob.get("level", ""),
                    "completion": content,
                    "reward": reward,
                })

                rewards_G.append(reward)
                n_total += 1
                if reward >= 1.0:
                    n_correct += 1

            # GRPO: advantages = reward - group_mean
            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            rewards_P.append(mean_reward)

            if all(a == 0.0 for a in advantages_G):
                continue

            for tokens, logprobs, advantage, ob_len in zip(
                tokens_G_T, logprobs_G_T, advantages_G, ob_lens_G
            ):
                input_tokens = [int(t) for t in tokens[:-1]]
                target_tokens = tokens[1:]
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                )
                datums_D.append(datum)

        # -----------------------------------------------------------------
        # Step 3: Forward-backward + optimizer step
        # -----------------------------------------------------------------
        if datums_D:
            # Pipelined: dispatch both before waiting (Tinker sequences internally)
            fwd_bwd_future = training_client.forward_backward(datums_D, loss_fn="importance_sampling")
            optim_future = training_client.optim_step(adam_params)
            fwd_bwd_future.result()
            optim_future.result()

        # Log
        batch_reward = sum(rewards_P) / len(rewards_P) if rewards_P else 0.0
        metrics["time/total"] = time.time() - t_start
        metrics["reward/mean"] = batch_reward
        metrics["reward/correct_frac"] = n_correct / max(n_total, 1)
        metrics["data/n_datums"] = len(datums_D)
        ml_logger.log_metrics(metrics, step=batch_idx)

        logger.info(
            f"Batch {batch_idx}/{n_train_batches} | "
            f"reward={batch_reward:.3f} | correct={n_correct}/{n_total} | "
            f"datums={len(datums_D)} | time={metrics['time/total']:.1f}s"
        )

    # -------------------------------------------------------------------------
    # Final checkpoint
    # -------------------------------------------------------------------------
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"grpo-final-{n_train_batches}",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    sampler_path = training_client.save_weights_for_sampler(name="grpo-final").result().path
    logger.info(f"GRPO training done! Sampler weights: {sampler_path}")

    # -------------------------------------------------------------------------
    # Final eval (run once at the end)
    # -------------------------------------------------------------------------
    if eval_problems:
        logger.info(f"Running final eval on {len(eval_problems)} held-out problems...")
        sampling_client_eval = service_client.create_sampling_client(model_path=sampler_path)
        eval_futures = [
            sampling_client_eval.sample(
                prompt=renderer.build_generation_prompt([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Convert the following PyTorch code to an optimized Triton kernel:\n\n"
                        f"```python\n{ep['code']}\n```\n\n"
                        f"Generate a complete Triton implementation that produces the same output as the PyTorch code."
                    )},
                ]),
                num_samples=1,
                sampling_params=sampling_params,
            )
            for ep in eval_problems
        ]
        eval_rewards = []
        for ep, ef in tqdm(zip(eval_problems, eval_futures), total=len(eval_problems), desc="Eval"):
            eval_result = ef.result()
            parsed_message, _ = renderer.parse_response(eval_result.sequences[0].tokens)
            eval_content = renderers.get_text_content(parsed_message)
            eval_rewards.append(get_reward(eval_content, ep["code"]))
        eval_mean = sum(eval_rewards) / len(eval_rewards)
        eval_correct = sum(1 for r in eval_rewards if r >= 1.0) / len(eval_rewards)
        ml_logger.log_metrics({
            "eval/reward_mean": eval_mean,
            "eval/correct_frac": eval_correct,
        }, step=n_train_batches)
        logger.info(f"Final eval: reward={eval_mean:.3f} | correct={eval_correct:.1%}")

    ml_logger.close()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
