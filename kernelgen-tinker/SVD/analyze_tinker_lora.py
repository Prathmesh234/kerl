import os
import glob
import torch
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from dotenv import load_dotenv

load_dotenv()

def main():
    # 1. Directory where Tinker extracts the checkpoint
    # Based on the tinker CLI behavior for 'tinker://8a311f3b-8da0-5948-9fda-0462f34d1d24:train:0/weights/sft-final-114'
    ckpt_dir = "./8a311f3b-8da0-5948-9fda-0462f34d1d24_train_0_weights_sft-final-114"
    if not os.path.exists(ckpt_dir):
        # Maybe it extracted a bit differently, let's auto-detect
        dirs = [d for d in os.listdir(".") if os.path.isdir(d) and "weights" in d]
        if dirs:
            ckpt_dir = dirs[0]
        else:
            print("Error: Could not find extracted checkpoint directory.")
            return

    print(f"Loading checkpoint from: {ckpt_dir}")

    # 2. Locate the PyTorch state_dict (safetensors or bin)
    state_dict = {}
    safetensor_files = glob.glob(os.path.join(ckpt_dir, "*.safetensors"))
    bin_files = glob.glob(os.path.join(ckpt_dir, "*.bin"))

    if safetensor_files:
        print(f"Found safetensors: {safetensor_files[0]}")
        state_dict = load_file(safetensor_files[0])
    elif bin_files:
        print(f"Found bin file: {bin_files[0]}")
        state_dict = torch.load(bin_files[0], map_location=torch.device('cpu'), weights_only=True)
    else:
        print("Error: Could not find adapter model weights.")
        return

    # 3. Iterate to find the lora_A and lora_B for the first self-attention q_proj layer
    lora_A_key = None
    lora_B_key = None
    
    for key in state_dict.keys():
        if "layers.0.self_attn.q_proj" in key:
            if "lora_A" in key:
                lora_A_key = key
            elif "lora_B" in key:
                lora_B_key = key

    if not lora_A_key or not lora_B_key:
        print("Could not find layers.0.self_attn.q_proj lora tensors!")
        return

    print(f"Found LoRA A: {lora_A_key}")
    print(f"Found LoRA B: {lora_B_key}")

    # Load and force to float32 for stable CPU math
    A = state_dict[lora_A_key].to(dtype=torch.float32, device='cpu')
    B = state_dict[lora_B_key].to(dtype=torch.float32, device='cpu')

    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")

    # 4. Compute the full weight update (Delta W = B x A)
    # LoRA math: output = (W + B @ A) @ x  =>  Delta W = B @ A
    delta_W = torch.matmul(B, A)
    print(f"Delta W shape: {delta_W.shape}")

    # 5. Perform SVD
    print("Computing SVD...")
    U, S, V = torch.linalg.svd(delta_W, full_matrices=False)
    
    print(f"Singular values found: {len(S)}")

    # 6. Plot the singular values
    plt.figure(figsize=(10, 6))
    plt.plot(S.numpy(), marker='o', linestyle='-', markersize=4)
    plt.yscale('log')
    plt.title('Scree Plot of Singular Values: $\\Delta W$ (q_proj, layer 0)')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Magnitude (Log Scale)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    output_png = "svd_scree_plot.png"
    plt.savefig(output_png, dpi=300)
    print(f"Scree plot saved to {output_png}")

if __name__ == "__main__":
    main()
