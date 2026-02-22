import os
import glob
import torch
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from dotenv import load_dotenv

load_dotenv()

def main():
    # 1. Directory where Tinker extracts the checkpoint
    ckpt_dir = None
    dirs = [d for d in os.listdir(".") if os.path.isdir(d) and "weights" in d]
    if dirs:
        ckpt_dir = dirs[0]
    else:
        print("Error: Could not find extracted checkpoint directory.")
        return

    print(f"Loading checkpoint from: {ckpt_dir}")

    # 2. Locate the PyTorch state_dict
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

    # 3. Iterate to find the lora_A and lora_B for q_proj
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

    # Convert to float32 on CPU
    A = state_dict[lora_A_key].to(dtype=torch.float32, device='cpu')
    B = state_dict[lora_B_key].to(dtype=torch.float32, device='cpu')

    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")

    # 4. Compute full weight update delta_W
    delta_W = torch.matmul(B, A)
    print(f"Delta W shape: {delta_W.shape}")

    # 5. Perform SVD
    print("Computing SVD...")
    U, S, V = torch.linalg.svd(delta_W, full_matrices=False)
    
    # 6. Calculate Cumulative Explained Variance
    variances = S ** 2
    total_variance = torch.sum(variances)
    cumulative_variance = torch.cumsum(variances, dim=0)
    explained_variance_ratio = cumulative_variance / total_variance

    # 7. Plot the variance
    plt.figure(figsize=(10, 6))
    plt.plot(explained_variance_ratio.numpy(), marker='o', linestyle='-', markersize=4, color='blue', label='Cumulative Variance')
    
    # Red lines for 95% and 99%
    plt.axhline(y=0.95, color='red', linestyle='--', label='95% Threshold')
    plt.axhline(y=0.99, color='darkred', linestyle='--', label='99% Threshold')
    
    # Restrict X axis to first 150 dims
    plt.xlim(0, 150)
    plt.ylim(0, 1.05)
    
    plt.title('Cumulative Explained Variance: $\\Delta W$ (q_proj, layer 0)')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.5)
    
    output_png = "cumulative_explained_variance.png"
    plt.savefig(output_png, dpi=300)
    print(f"Plot saved to {output_png}")

if __name__ == "__main__":
    main()
