# GRPO + SFT Training Findings

## 1. The "Missing `<think>` Tags" Issue
During our initial GRPO run, the model failed to generate `<think>` tokens 1058 out of 1064 times, despite the system prompt explicitly instructing it to do so. 

### Root Causes
1.  **Special Token IDs vs Text:** In Qwen3, `<think>` is not a literal string; it is a special token ID (`151667`). The model only generates this token when prompted with the specific context it was trained on.
2.  **Prompt Mismatch (Distribution Shift):**
    *   **SFT Phase:** The model was fine-tuned using a very simple user prompt (`Convert the following PyTorch code...`) **without** any system prompt. It mapped this exact sequence to the generation of the `<think>` token.
    *   **GRPO Phase:** We sent a massive 50-line system prompt followed by a different user prompt format. 
    *   Because the GRPO context completely violated the distribution the LoRA adapter was tuned on during SFT, the model's prediction confidence for token `151667` collapsed to zero. It fell back to its base pre-training behavior (generating raw Triton code).
**Fix:** We explicitly added the massive `SYSTEM_PROMPT` to the SFT training script dataloader. Now the adapter learns that the massive system prompt is the trigger for the `<think>` token.

## 2. High LoRA Ranks on Small Datasets
In the SFT configuration, the LoRA rank was set to `128` while fine-tuning on a dataset of only **170 examples**.

### Negative Impacts of $Rank = 128$
1.  **Massive Overfitting:** An 8B parameter model with a rank of 128 has tens of millions of trainable parameters. With only 170 examples, it acts like a sponge, memorizing the exact phrasing of the dataset rather than learning the generalized logic of Triton kernel compilation.
2.  **Repetition Spirals (Hallucination):** When analyzing failed GRPO rollouts (e.g., `reward = 0.0`), we observed the model falling into repetition loops, regurgitating the exact same memorized reasoning block over 100 times in a row (e.g., `"Actually, we need to load a scalar value..."`). Because the prompt shifted, and the model was massively overfit, its logits collapsed and triggered a memorized string over and over.
3.  **Killing RL Exploration:** GRPO works best when the model has high "entropy" — meaning it is flexible enough to explore different reasoning and coding pathways to discover optimized speeds. A massive LoRA rank solidifies the model's confidence into a rigid state, stopping it from attempting new, creative kernel block optimizations.

### Recommendation
**Decrease SFT LoRA Rank:** Change `lora_rank` from `128` down to **`32` or `16`**. This provides enough capacity to learn the `[System Prompt] -> <think> -> <triton>` XML structure, while acting as heavy regularization to prevent verbatim dataset memorization, leaving the model flexible for the GRPO reinforcement phase.
