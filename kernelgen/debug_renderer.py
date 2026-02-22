import os
import datasets
from dotenv import load_dotenv
from kernelgentinker import SYSTEM_PROMPT
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

load_dotenv()

def debug_render():
    model_name = "Qwen/Qwen3-8B"
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Load a real problem for the debug view
    ds = datasets.load_dataset("ScalingIntelligence/KernelBench", split="level_1")
    problem = ds[10] # ReLU
    pytorch_code = problem["code"]

    user_content = (
        f"Optimize this PyTorch module into a Triton kernel.\n\n"
        f"<pytorch>\n{pytorch_code}\n</pytorch>\n\n"
        f"First reason about the computation in <think>...</think> tags, "
        f"then provide your Triton implementation in <triton>...</triton> tags."
    )

    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # This is what you asked about:
    model_input = renderer.build_generation_prompt(convo)

    print("--- 1. RENDERED TEXT (What the model 'reads') ---")
    # We can decode the tokens back to text to see the markers
    print(tokenizer.decode(model_input.to_ints()))

    print("\n--- 2. RAW TOKENS (What the GPU actually processes) ---")
    print(model_input.to_ints()[:20], "... (truncated)")
    print(f"Total tokens: {len(model_input.to_ints())}")

if __name__ == "__main__":
    debug_render()
