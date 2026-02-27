# -*- coding: utf-8 -*-
"""
KernelBench Multi-Turn GRPO with Tinker
========================================
Reinforcement Learning (GRPO) pipeline with multi-turn iterative refinement.
Each completion gets up to max_turns attempts: sample -> Modal validate ->
feedback -> re-sample -> ... with the final reward used for GRPO training.

The model learns from ALL its generated tokens across turns. Environment
feedback (Modal validation results) is treated as observation tokens
(advantage=0, logprobs=0) so the model is not penalized for them.

Uses KernelBench Level 1 + Level 2 problems (200 tasks):
  - 20 held out for eval
  - 180 used for training

Reward (based on final turn's result):
  -  0.0  if no <triton> tag
  - -0.5 if missing triton_kernel_wrapper
  - -0.5 if execution crash or incorrect output
  -  1.0 + log(speedup) if correct (capped at 5.0)

Usage:
  python kernelgentinker-multiturn.py
  python kernelgentinker-multiturn.py max_turns=6 batch_size=8
  python kernelgentinker-multiturn.py sft_checkpoint_path=tinker://...
"""

import json
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

from multi_turn_queue import MultiTurnQueue

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
    log_path: str = "./tmp/kernelbench-grpo-multiturn-qwen3-32b"
    model_name: str = "Qwen/Qwen3-32B"
    sft_checkpoint_path: str | None = None   # pass in sampler weights from SFT to warm-start
    batch_size: int = 4          # problems per GRPO batch
    group_size: int = 8           # completions per problem (multi-turn trajectories)
    learning_rate: float = 2e-5
    lora_rank: int = 32           # reduced from 128 to prevent overfitting
    save_every: int = 40
    max_tokens: int = 16384
    keep_last_sampler_checkpoints: int = 2
    eval_size: int = 20           # held-out problems for eval
    eval_every: int = 10          # eval frequency in batches
    max_turns: int = 4            # max refinement turns per trajectory
    turn_decay_factor: float = 0.9      # DAPO-inspired: exponential decay per turn for correct solutions
    turn_penalty_scale: float = 0.5     # DAPO-inspired: max extra penalty added across turns for failures


# ---------------------------------------------------------------------------
# DAPO-inspired turn-based reward shaping
# ---------------------------------------------------------------------------

def apply_turn_decay(
    base_reward: float,
    turn: int,
    max_turns: int,
    decay_factor: float = 0.9,
    penalty_scale: float = 0.5,
) -> float:
    """
    Soft turn-based reward shaping inspired by DAPO's overlong reward shaping.

    DAPO uses a graduated soft penalty for overlong responses instead of a hard
    cliff. We apply the same principle across turns:

    Correct solutions (reward >= 1.0):
        Exponential decay incentivizes solving in fewer turns.
        R_final = R_base * decay_factor^(turn - 1)
        e.g. decay=0.9: turn1=1.0x, turn2=0.9x, turn3=0.81x, turn4=0.73x

    Failed solutions (reward < 0):
        Graduated extra penalty that increases linearly from 0 (turn 1) to
        -penalty_scale (final turn), mirroring DAPO's soft overlong formula:
            R_final = R_base - (turn - 1) / (max_turns - 1) * penalty_scale
        e.g. scale=0.5, max_turns=4:
            turn1: no extra, turn2: -0.17, turn3: -0.33, turn4: -0.5

    Turn 1 is never additionally penalized - the model only sees the base
    correctness reward on its first attempt.
    """
    if turn <= 1:
        return base_reward

    if base_reward >= 1.0:
        # Correct: exponential decay rewards faster solutions
        return base_reward * (decay_factor ** (turn - 1))
    else:
        # Failed: graduated DAPO-style penalty grows with each wasted turn
        extra_penalty = (turn - 1) / max(max_turns - 1, 1) * penalty_scale
        return base_reward - extra_penalty


# ---------------------------------------------------------------------------
# System prompt (identical to kernelgentinker.py + multi-turn addendum)
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
   
   CRITICAL: tl.arange(0, BLOCK_SIZE) ✓  but  tl.arange(0, n) where n is runtime ✗
   Solution: Use fixed BLOCK_SIZE with masking for boundaries

3. MEMORY SAFETY
   GPU memory is accessed via pointers. Out-of-bounds access causes crashes.
   
   Always use MASKING:
   ```python
   offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
   mask = offsets < N  # Check boundaries
   data = tl.load(ptr + offsets, mask=mask, other=0.0)  # Safe
   ```
   
   The mask ensures we only touch valid memory locations.

4. MATRIX OPERATIONS
   tl.dot(A, B) performs matrix multiplication:
   - Requires A.shape = (M, K) and B.shape = (K, N)
   - Results in shape (M, N)
   - Use tl.trans(B) if B is (N, K) to get (K, N)
   
   Common pattern for GEMM:
   ```python
   # Load tiles
   a = tl.load(...)  # Shape: (BLOCK_M, BLOCK_K)
   b = tl.load(...)  # Shape: (BLOCK_N, BLOCK_K)
   # Transpose b to match dimensions
   b_t = tl.trans(b)  # Now: (BLOCK_K, BLOCK_N)
   # Multiply
   c = tl.dot(a, b_t)  # Result: (BLOCK_M, BLOCK_N)
   ```

=== YOUR TASK ===

For each PyTorch operation, you should:
1. Analyze the operation and memory access patterns
2. Think step-by-step about how to parallelize it
3. Choose appropriate BLOCK_SIZE and num_warps (num_warps controls thread parallelism per block, typically 4 or 8)
4. Write the complete Triton kernel implementation

Think step-by-step about the conversion before writing code. Then provide the complete
Triton implementation inside <triton>...</triton> tags:

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
3. If the model has weights (nn.Linear, nn.Conv2d, etc.), accept them as additional parameters in the wrapper - the benchmark harness will pass the reference model's weights automatically
4. **IMPORTANT**: If get_init_inputs() returns parameters (e.g., {'quantiles': 4, 'hidden_size': 128}), the wrapper MUST accept these as keyword arguments with defaults matching those values
5. **Triton API Limitations**: tl.tanh, tl.pow, tl.unsqueeze do NOT exist - use tl.exp for tanh, ** operator for pow, reshape for unsqueeze

=== TRITON KERNEL RULES - MUST FOLLOW ===

IMPORTS:
```python
import triton
import triton.language as tl
import torch  # wrapper only - for tensor allocation, NEVER inside @triton.jit
```

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
- .pow(), .sqrt(), .exp() methods on tensors - use ** operator for pow, tl.sqrt(), tl.exp() for others
- Python classes or objects
- nn.* modules

CONSTEXPR RULES:
- tl.arange(start, end) - both start and end MUST be constants or tl.constexpr
- BLOCK_SIZE: tl.constexpr in kernel signature
- Use powers of 2: 64, 128, 256, 512, 1024

REDUCTION PATTERN:
```python
# CORRECT - accumulate in block-sized buffer, reduce once at end
acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
for i in range(0, N, BLOCK_SIZE):
    offsets = i + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(ptr + offsets, mask=mask, other=0.0)
    acc += x
result = tl.sum(acc)

# WRONG - shape mismatch in loop
acc = tl.zeros([1], dtype=tl.float32)
acc += tl.sum(x)  # ERROR: shape changes!
```

WRAPPER FUNCTION PATTERN:
```python
def triton_kernel_wrapper(x):
    B, C = x.shape
    output = torch.empty((B, C), dtype=x.dtype, device=x.device)

    # Dimension args to torch.empty/randn/zeros must be Python ints:
    # CORRECT: torch.empty((B, C), ...)       -- B, C are ints from shape unpacking
    # CORRECT: torch.randn(int(N), int(M))    -- explicit int() conversion
    # WRONG:   torch.randn(some_tensor, ...)   -- Tensor as shape arg crashes

    # Never use a multi-element Tensor in a Python if/while:
    # WRONG:   if some_tensor:
    # CORRECT: if some_tensor.item() > 0:

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    my_kernel[grid](x, output, x.numel(), BLOCK_SIZE=BLOCK_SIZE)

    return output
```

COMMON OPERATIONS:
- ReLU: tl.maximum(x, 0.0)
- Sigmoid: 1.0 / (1.0 + tl.exp(-x))
- Tanh: (tl.exp(2*x) - 1) / (tl.exp(2*x) + 1)
- Softmax: exp_x = tl.exp(x - tl.max(x)); exp_x / tl.sum(exp_x)
- Mean: tl.sum(x) / n_elements

USE ASCII ONLY - no unicode characters like – or —, use - instead.


=== MULTI-TURN FEEDBACK ===

You may receive feedback on your generated kernel if it fails validation
or is slower than PyTorch. When you receive feedback, analyze the error
or performance issue, then generate an improved version. Always provide
the corrected code inside <triton>...</triton> tags.
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
    """Format bonus: rewards the model for using proper <think> + <triton> structure."""
    has_think = bool(re.search(r"<think>.*?</think>", content, re.DOTALL))
    has_triton = bool(re.search(r"<triton>.*?</triton>", content, re.DOTALL))
    if has_think and has_triton:
        return 1.0
    elif has_triton:
        return 0.5
    return 0.0


def save_rollout_to_jsonl(log_path: str, data: dict):
    """Append a rollout dictionary as a JSON line for later analysis."""
    os.makedirs(log_path, exist_ok=True)
    filepath = os.path.join(log_path, "rollouts.jsonl")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, default=str) + "\n")


# ---------------------------------------------------------------------------
# Modal benchmark client
# ---------------------------------------------------------------------------

benchmark_kernelbench = modal.Function.from_name("kernelbench-triton", "benchmark_kernelbench")


def get_reward_with_result(
    content: str,
    pytorch_code: str,
    turn: int = 1,
    max_turns: int = 1,
    decay_factor: float = 0.9,
    penalty_scale: float = 0.5,
) -> tuple[float, dict]:
    """
    Single reward function combining all reward signals:
      1. Format reward  - bonus for using <think> + <triton> structure
      2. Correctness    - benchmark result from Modal
      3. Speedup bonus  - log(speedup) for correct & fast kernels
      4. Turn decay     - DAPO-inspired shaping applied to the total

    Defaults (turn=1, max_turns=1) mean no turn decay - safe for eval/single-turn.
    """
    fmt_bonus = format_reward(content)

    triton_code = extract_triton(content)
    if triton_code is None:
        base = fmt_bonus  # only format reward (0.0 since no triton tag)
        result = {"correctness": False, "error": "No <triton> tags found"}
        return apply_turn_decay(base, turn, max_turns, decay_factor, penalty_scale), result

    if "def triton_kernel_wrapper" not in triton_code:
        base = fmt_bonus - 0.5  # format bonus + structure penalty
        result = {"correctness": False, "error": "Missing triton_kernel_wrapper"}
        return apply_turn_decay(base, turn, max_turns, decay_factor, penalty_scale), result

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
            base = fmt_bonus - 0.5
            return apply_turn_decay(base, turn, max_turns, decay_factor, penalty_scale), result

        if not result["correctness"]:
            base = fmt_bonus - 0.5
            return apply_turn_decay(base, turn, max_turns, decay_factor, penalty_scale), result

        speedup = result.get("speedup", 1.0)
        speedup_bonus = max(0.0, min(4.0, math.log(max(speedup, 1e-6))))
        base = fmt_bonus + 1.0 + speedup_bonus
        return apply_turn_decay(base, turn, max_turns, decay_factor, penalty_scale), result

    except Exception as e:
        base = fmt_bonus - 0.5
        return apply_turn_decay(base, turn, max_turns, decay_factor, penalty_scale), {"correctness": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Dataset loading - Level 1 + Level 2 only (200 problems)
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
# Multi-turn trajectory runner
# ---------------------------------------------------------------------------

def run_multiturn_trajectory(
    *,
    sampling_client,
    renderer,
    sampling_params,
    pytorch_code: str,
    problem: dict,
    max_turns: int,
    queue: MultiTurnQueue,
    tokenizer,
    turn_decay_factor: float = 0.9,
    turn_penalty_scale: float = 0.5,
) -> tuple[float, list[int], list[float], list[float], int, dict]:
    """
    Run a single multi-turn trajectory for one sequence.

    Returns:
        (final_reward, all_tokens, all_logprobs, all_advantages_mask,
         ob_len_initial, turn_history)

    Token accounting across turns:
      [initial_prompt] [turn1_completion] [feedback_tokens] [turn2_completion] ...

    - initial_prompt tokens: observation (logprobs=0, advantage_mask=0)
    - turnN_completion tokens: model-generated (has logprobs, advantage_mask=1)
    - feedback tokens: observation injected by env (logprobs=0, advantage_mask=0)

    The final reward is applied to ALL model-generated segments via advantage_mask.
    """
    user_content = (
        f"Convert the following PyTorch code to an optimized Triton kernel:\n\n"
        f"```python\n{pytorch_code}\n```\n\n"
        f"Generate a complete Triton implementation that produces the same output as the PyTorch code."
    )
    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Accumulators for the full trajectory
    all_tokens: list[int] = []
    all_logprobs: list[float] = []
    all_advantage_mask: list[float] = []  # 1.0 = model-generated, 0.0 = observation

    turn_history = []
    final_reward = 0.0
    final_result = {}

    for turn in range(1, max_turns + 1):
        # Build prompt from current conversation
        model_input = renderer.build_generation_prompt(convo)
        prompt_tokens = model_input.to_ints()

        if turn == 1:
            # First turn: prompt tokens are the initial observation
            all_tokens.extend(prompt_tokens)
            all_logprobs.extend([0.0] * len(prompt_tokens))
            all_advantage_mask.extend([0.0] * len(prompt_tokens))
        # For turn > 1, feedback tokens were already appended below

        # Sample 1 completion for this turn
        future = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        sample_result = future.result()
        sequence = sample_result.sequences[0]

        sampled_tokens = sequence.tokens
        sampled_logprobs = sequence.logprobs
        assert sampled_logprobs is not None

        # Append model-generated tokens
        #.extend basically concatenates the lists (so it becomes all tokens + all logprobs + all advantage)
        all_tokens.extend(sampled_tokens)
        all_logprobs.extend(sampled_logprobs)
        all_advantage_mask.extend([1.0] * len(sampled_tokens))

        # Parse and evaluate
        parsed_message, _ = renderer.parse_response(sampled_tokens)
        content = renderers.get_text_content(parsed_message)

        reward, result = get_reward_with_result(
            content, pytorch_code,
            turn=turn, max_turns=max_turns,
            decay_factor=turn_decay_factor, penalty_scale=turn_penalty_scale,
        )

        turn_record = {
            "turn": turn,
            "content": content,
            "reward": reward,
            "result": result,
        }
        turn_history.append(turn_record)
        final_reward = reward
        final_result = result

        # Check stopping condition
        stop, reason = queue.should_stop(turn, result)
        if stop:
            turn_record["stop_reason"] = reason
            break

        # Build feedback and extend conversation for next turn
        feedback = queue.build_feedback(result)
        turn_record["feedback_given"] = feedback

        convo.append({"role": "assistant", "content": content})
        convo.append({"role": "user", "content": feedback})

        # Tokenize the feedback turn (assistant reply + user feedback) as observation
        # We re-render the full conversation and diff to get the new tokens
        next_model_input = renderer.build_generation_prompt(convo)
        next_prompt_tokens = next_model_input.to_ints()

        # The new tokens = next_prompt_tokens that extend beyond what we already have
        # i.e. the assistant reply tokens + feedback tokens as rendered by the template
        prev_len = len(prompt_tokens) + len(sampled_tokens)
        feedback_tokens = next_prompt_tokens[prev_len:]

        all_tokens.extend(feedback_tokens)
        all_logprobs.extend([0.0] * len(feedback_tokens))
        all_advantage_mask.extend([0.0] * len(feedback_tokens))

    return final_reward, all_tokens, all_logprobs, all_advantage_mask, turn_history


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main(config: Config):
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project="kernelgen-grpo-multiturn-qwen3-32b",
        wandb_name="kernelgen-grpo-multiturn-run",
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
    logger.info(f"Training: {n_train_batches} batches of {config.batch_size} problems, max_turns={config.max_turns}")

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

    # Shared queue instance for stop-condition checks + feedback construction
    queue = MultiTurnQueue(max_turns=config.max_turns)

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
                name=f"grpo-mt-{batch_idx:06d}",
                log_path=config.log_path,
                kind="both",
                loop_state={"batch": batch_idx},
            )

        # Frozen sampler weights for this batch
        sampling_path = (
            training_client.save_weights_for_sampler(name=f"{batch_idx:06d}").result().path
        )
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)
        # Manage sampler checkpoint deque
        sampler_ckpt_queue.append(sampling_path)
        if config.keep_last_sampler_checkpoints > 0:
            while len(sampler_ckpt_queue) > config.keep_last_sampler_checkpoints:
                old_path = sampler_ckpt_queue.popleft()
                rest_client.delete_checkpoint_from_tinker_path(old_path).result()

        # Get batch of problems
        batch_start = batch_idx * config.batch_size
        batch_problems = train_problems[batch_start : batch_start + config.batch_size]

        # -----------------------------------------------------------------
        # Step 1+2: Multi-turn sampling + reward collection
        # -----------------------------------------------------------------
        # For each problem, run group_size independent multi-turn trajectories.
        # Each trajectory: sample -> validate -> feedback -> resample -> ...
        # Final reward from last turn used for GRPO advantage.
        # -----------------------------------------------------------------
        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        n_correct = 0
        n_total = 0
        total_turns = 0

        for problem in tqdm(batch_problems, desc=f"Batch {batch_idx}"):
            pytorch_code = problem["code"]
            rewards_G: list[float] = []
            trajectories_G: list[tuple] = []  # (all_tokens, all_logprobs, all_advantage_mask)

            for g in range(config.group_size):
                final_reward, all_tokens, all_logprobs, all_advantage_mask, turn_history = \
                    run_multiturn_trajectory(
                        sampling_client=sampling_client,
                        renderer=renderer,
                        sampling_params=sampling_params,
                        pytorch_code=pytorch_code,
                        problem=problem,
                        max_turns=config.max_turns,
                        queue=queue,
                        tokenizer=tokenizer,
                        turn_decay_factor=config.turn_decay_factor,
                        turn_penalty_scale=config.turn_penalty_scale,
                    )

                rewards_G.append(final_reward)
                trajectories_G.append((all_tokens, all_logprobs, all_advantage_mask))
                n_total += 1
                total_turns += len(turn_history)
                if final_reward >= 1.0:
                    n_correct += 1

                # Save rollout for analysis
                save_rollout_to_jsonl(config.log_path, {
                    "batch_idx": batch_idx,
                    "group_member": g,
                    "problem_name": problem.get("name", ""),
                    "problem_level": problem.get("level", ""),
                    "num_turns": len(turn_history),
                    "final_reward": final_reward,
                    "turns": turn_history,
                })

            # GRPO: advantages = reward - group_mean
            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            rewards_P.append(mean_reward)

            if all(a == 0.0 for a in advantages_G):
                continue

            # Build training datums from each trajectory
            for (all_tokens, all_logprobs, all_advantage_mask), advantage in zip(
                trajectories_G, advantages_G
            ):
                # input_tokens = all but last, target_tokens = all but first (standard LM shift)
                input_tokens = [int(t) for t in all_tokens[:-1]]
                target_tokens = all_tokens[1:]

                # Logprobs: shift to align with targets
                # all_logprobs[i] corresponds to generating all_tokens[i] given prior context
                # For observation tokens, logprobs are already 0.0
                # We drop the first token's logprob (no prediction for it) and
                # the prompt starts with logprob=0 anyway
                padded_logprobs = all_logprobs[1:]  # align with target_tokens

                # Advantages: apply group advantage only to model-generated tokens
                # advantage_mask[i] = 1.0 for model tokens, 0.0 for observation
                # Shift by 1 to align with target_tokens
                padded_advantages = [
                    advantage * mask for mask in all_advantage_mask[1:]
                ]

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
        avg_turns = total_turns / max(n_total, 1)
        metrics["time/total"] = time.time() - t_start
        metrics["reward/mean"] = batch_reward
        metrics["reward/correct_frac"] = n_correct / max(n_total, 1)
        metrics["data/n_datums"] = len(datums_D)
        metrics["multiturn/avg_turns"] = avg_turns
        ml_logger.log_metrics(metrics, step=batch_idx)

        logger.info(
            f"Batch {batch_idx}/{n_train_batches} | "
            f"reward={batch_reward:.3f} | correct={n_correct}/{n_total} | "
            f"avg_turns={avg_turns:.1f} | datums={len(datums_D)} | "
            f"time={metrics['time/total']:.1f}s"
        )

    # -------------------------------------------------------------------------
    # Final checkpoint
    # -------------------------------------------------------------------------
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"grpo-mt-final-{n_train_batches}",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    sampler_path = training_client.save_weights_for_sampler(name="grpo-mt-final").result().path
    logger.info(f"Multi-turn GRPO training done! Sampler weights: {sampler_path}")

    # -------------------------------------------------------------------------
    # Final eval (single-turn, same as kernelgentinker.py for comparability)
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
            reward, _ = get_reward_with_result(eval_content, ep["code"])
            eval_rewards.append(reward)
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
