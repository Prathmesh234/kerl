#!/usr/bin/env python3
"""
Download Arcee Trinity-Mini base model and initial policy adapter.

This script downloads:
1. Base model from HuggingFace (arcee-ai/Trinity-Mini) → models/arcee-trinity/
2. Initial LoRA adapter from Modal → policies/policy-0-initial/

Prerequisites:
    pip install huggingface_hub modal transformers
    huggingface-cli login  # authenticate with HuggingFace (optional for public models)
    modal setup  # authenticate with Modal

Usage:
    python scripts/download_assets.py
    python scripts/download_assets.py --skip-model  # Only download adapter
    python scripts/download_assets.py --skip-adapter  # Only download model
"""

import argparse
import os
import sys
from pathlib import Path

# Modal volume configuration
VOLUME_NAME = "arcee-vol"
ADAPTER_PATH = "models/trinity-triton-sft-vllm/final_model"

# HuggingFace model
HF_MODEL_ID = "arcee-ai/Trinity-Mini"

# Local paths (relative to DisTrainer root)
SCRIPT_DIR = Path(__file__).parent
DISTRAINER_ROOT = SCRIPT_DIR.parent
LOCAL_MODEL_DIR = DISTRAINER_ROOT / "models" / "arcee-trinity"
LOCAL_ADAPTER_DIR = DISTRAINER_ROOT / "policies" / "policy-0-initial"


def download_from_huggingface(model_id: str, local_dir: Path):
    """Download model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    print(f"\nDownloading {model_id} from HuggingFace...")
    print(f"Destination: {local_dir.absolute()}")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ Model downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False


def download_from_modal(volume_name: str, remote_path: str, local_dir: Path):
    """Download files from Modal volume to local directory."""
    try:
        import modal
    except ImportError:
        print("ERROR: modal not installed. Run: pip install modal")
        return False

    print(f"\nConnecting to Modal volume: {volume_name}")
    try:
        vol = modal.Volume.from_name(volume_name)
    except Exception as e:
        print(f"❌ Could not connect to volume '{volume_name}': {e}")
        print("Make sure you've run 'modal setup' to authenticate.")
        return False

    print(f"Listing files in {remote_path}/ ...")
    try:
        entries = list(vol.listdir(remote_path, recursive=True))
    except Exception as e:
        print(f"❌ Could not list files in '{remote_path}': {e}")
        return False

    if not entries:
        print(f"❌ No files found at '{remote_path}' in volume '{volume_name}'.")
        return False

    print(f"Found {len(entries)} entries. Downloading to {local_dir}...")
    local_dir.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        remote_file = entry.path
        rel_path = os.path.relpath(remote_file, remote_path)
        local_file = local_dir / rel_path

        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            local_file.mkdir(parents=True, exist_ok=True)
            continue

        local_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading: {rel_path}")
        try:
            with open(local_file, "wb") as f:
                for chunk in vol.read_file(remote_file):
                    f.write(chunk)
        except Exception as e:
            print(f"  ❌ Failed to download {rel_path}: {e}")
            return False

    print(f"✅ Adapter downloaded to: {local_dir.absolute()}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Arcee Trinity-Mini model and adapter"
    )
    parser.add_argument(
        "--volume-name",
        default=VOLUME_NAME,
        help=f"Modal volume name (default: {VOLUME_NAME})",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip downloading base model from HuggingFace",
    )
    parser.add_argument(
        "--skip-adapter",
        action="store_true",
        help="Skip downloading initial adapter from Modal",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Downloading Arcee Trinity-Mini Assets")
    print("=" * 70)

    success = True

    # Download base model from HuggingFace
    if not args.skip_model:
        print("\n[1/2] Downloading base model from HuggingFace...")
        print(f"      Model: {HF_MODEL_ID}")
        if LOCAL_MODEL_DIR.exists() and any(LOCAL_MODEL_DIR.iterdir()):
            print(f"⚠️  Model already exists at {LOCAL_MODEL_DIR}")
            response = input("    Delete and re-download? [y/N]: ").strip().lower()
            if response == 'y':
                import shutil
                shutil.rmtree(LOCAL_MODEL_DIR)
                success &= download_from_huggingface(HF_MODEL_ID, LOCAL_MODEL_DIR)
            else:
                print("    Skipping model download")
        else:
            success &= download_from_huggingface(HF_MODEL_ID, LOCAL_MODEL_DIR)
    else:
        print("\n[1/2] Skipping base model download")

    # Download initial adapter from Modal
    if not args.skip_adapter:
        print("\n[2/2] Downloading initial LoRA adapter from Modal...")
        print(f"      Adapter: {ADAPTER_PATH}")
        if LOCAL_ADAPTER_DIR.exists() and any(LOCAL_ADAPTER_DIR.iterdir()):
            print(f"⚠️  Adapter already exists at {LOCAL_ADAPTER_DIR}")
            response = input("    Delete and re-download? [y/N]: ").strip().lower()
            if response == 'y':
                import shutil
                shutil.rmtree(LOCAL_ADAPTER_DIR)
                success &= download_from_modal(args.volume_name, ADAPTER_PATH, LOCAL_ADAPTER_DIR)
            else:
                print("    Skipping adapter download")
        else:
            success &= download_from_modal(args.volume_name, ADAPTER_PATH, LOCAL_ADAPTER_DIR)
    else:
        print("\n[2/2] Skipping adapter download")

    print("\n" + "=" * 70)
    if success:
        print("✅ Download complete!")
        print("\nDirectory structure:")
        print(f"  {LOCAL_MODEL_DIR.relative_to(DISTRAINER_ROOT)}/  # Base model (shared)")
        print(f"  {LOCAL_ADAPTER_DIR.relative_to(DISTRAINER_ROOT)}/  # Initial adapter")
        print("\nNext steps:")
        print("  1. Configure: Edit config/train_config.toml if needed")
        print("  2. Start DisGenerator: cd DisGenerator && ./scripts/start_all.sh 1p1d")
        print("  3. Start DisTrainer: torchrun --nproc_per_node=2 -m DisTrainer.train")
    else:
        print("❌ Download failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
