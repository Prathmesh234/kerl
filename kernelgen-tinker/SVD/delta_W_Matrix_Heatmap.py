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

    # 3. Iterate to find the lora_A and lora_B for MLP down_proj
    lora_A_key = None
    lora_B_key = None
    
    for key in state_dict.keys():
        if "layers.0.mlp.down_proj" in key:
            if "lora_A" in key:
                lora_A_key = key
            elif "lora_B" in key:
                lora_B_key = key

    if not lora_A_key or not lora_B_key:
        print("Could not find layers.0.mlp.down_proj lora tensors!")
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

    # 5. Extract a subset for the heatmap
    print("Slicing top-left 500x500 subset...")
    delta_W_subset = delta_W[:500, :500]

    # 6. Plot the heatmap
    plt.figure(figsize=(10, 8))
    # RdBu_r gives Red for positive values, Blue for negative values, and white around zero
    plt.imshow(delta_W_subset.numpy(), cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Magnitude ($\Delta W$ Weight Values)')
    
    plt.title('$\\Delta W$ Matrix Heatmap (The "Hardwiring" Plot)\\nmlp.down_proj, layer 0, Top-Left 500x500')
    plt.xlabel('Input Dimension Index')
    plt.ylabel('Output Dimension Index')
    
    # Identify the maximum absolute value to center the colormap around 0
    max_val = torch.max(torch.abs(delta_W_subset)).item()
    plt.clim(-max_val, max_val)
    
    output_png = "heatmap_deltaW.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_png}")

if __name__ == "__main__":
    main()
