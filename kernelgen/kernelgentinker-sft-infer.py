import os
import tinker
import datasets
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from kernelgentinker import SYSTEM_PROMPT

from dotenv import load_dotenv

def main():
    # Load credentials from .env
    load_dotenv()

    # Load a sample from the dataset used during SFT
    ds = datasets.load_dataset("ppbhatt500/kernelbook-triton-reasoning-traces", split="train")
    
    # We will grab the first sample to test
    pytorch_code = ds[2]["pytorch_code"]

    user_content = (
        f"Convert the following PyTorch code to an optimized Triton kernel:\n\n"
        f"```python\n{pytorch_code}\n```\n\n"
        f"Generate a complete Triton implementation that produces the same output as the PyTorch code.\n"
        f"Make sure to output your step-by-step reasoning inside <think>...</think> tags before the <triton>...</triton> code."
    )

    # Adding the system prompt to guide the output format correctly
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    model_name = "Qwen/Qwen3-8B"
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Convert the messages dict into Tinker's required format using the renderer
    model_input = renderer.build_generation_prompt(messages)

    # Initialize the Tinker client and attach to the saved sampler weights
    client = tinker.ServiceClient()
    model_path = "tinker://9d7deef6-ffd5-5ef1-9fad-8881d36ee616:train:0/sampler_weights/sft-final"
    
    print(f"Loading sampling client with model path: {model_path}")
    print("This might take a moment to provision...\n")
    sampling_client = client.create_sampling_client(model_path=model_path)
    
    sampling_params = tinker.types.SamplingParams(
        max_tokens=16384,
        stop=renderer.get_stop_sequences(),
        temperature=0.7# or 0.0 for greedy
    )
    
    print("Generating completion...")
    future = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params
    )
    
    # Wait for the generation to finish
    result = future.result()
    sampled_tokens = result.sequences[0].tokens
    parsed_message, _ = renderer.parse_response(sampled_tokens)
    content = renderers.format_content_as_string(parsed_message["content"])

    print("="*80)
    print("PYTORCH CODE (INPUT)")
    print("="*80)
    print(pytorch_code)
    
    print("\n" + "="*80)
    print("FULL COMPLETION (OUTPUT)")
    print("="*80)
    print(content)

if __name__ == "__main__":
    main()
