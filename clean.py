#!/usr/bin/env python3
import torch, gc, os, shutil
from pathlib import Path

def rm(path):
    if path.exists():
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
            else:
                shutil.rmtree(path)
            print(f"✓ Removed {path}")
        except Exception as e:
            print(f"⚠️ Could not remove {path}: {e}")

def cleanup():
    print("🧹 Starting aggressive cleanup...")

    # Free GPU/CPU RAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    print("✓ Cleared GPU + RAM")

    # HuggingFace caches
    rm(Path.home() / ".cache" / "huggingface")

    # Pip cache
    rm(Path.home() / ".cache" / "pip")

    # Evaluation + logs
    rm(Path("evaluation_results"))
    rm(Path("logs"))

    # Old checkpoints (keep only latest)
    ckpts = sorted(Path("outputs").glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if len(ckpts) > 1:
        for c in ckpts[:-1]:
            rm(c)

    # Datasets cache
    rm(Path.home() / ".cache" / "datasets")

    # Temp files
    rm(Path("/tmp"))

    print("✨ Aggressive cleanup done!")

if __name__ == "__main__":
    cleanup()
