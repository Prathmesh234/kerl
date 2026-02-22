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

    # 3. Extract lora_A and lora_B for q_proj
    weights_list = []
    for key in state_dict.keys():
        if "layers.0.self_attn.q_proj" in key:
            if "lora_A" in key or "lora_B" in key:
                print(f"Extracting: {key}")
                w = state_dict[key].to(dtype=torch.float32, device='cpu').flatten()
                weights_list.append(w)

    if not weights_list:
        print("Could not find layers.0.self_attn.q_proj lora tensors!")
        return

    # 4. Concatenate all weights into a single 1D array
    all_weights = torch.cat(weights_list)
    print(f"Total parameters analyzed: {all_weights.numel()}")

    # 5. Plot Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_weights.numpy(), bins=100, color='purple', alpha=0.7, edgecolor='black')
    
    # Logarithmic scale for Y to see heavy tails
    plt.yscale('log')
    
    plt.title('Weight Distribution Histogram (The "Outlier" Plot)\nq_proj, layer 0')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency (Log Scale)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    output_png = "weight_distribution.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to {output_png}")

if __name__ == "__main__":
    main()
