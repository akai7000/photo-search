#!/usr/bin/env python3
"""Set up CPU and GPU build environments."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_in_venv(venv_name, *args):
    """Run a command in a specific venv."""
    if sys.platform == "win32":
        venv_python = ROOT / venv_name / "Scripts" / "python.exe"
    else:
        venv_python = ROOT / venv_name / "bin" / "python"
    
    cmd = [str(venv_python), *args]
    print(f"\n{'='*60}")
    print(f"Running in {venv_name}: {' '.join(args)}")
    print(f"{'='*60}")
    return subprocess.run(cmd, cwd=ROOT).returncode


def setup_cpu_env():
    """Set up CPU-only environment."""
    print("\n" + "="*60)
    print("Setting up CPU-only environment (venv-cpu)...")
    print("="*60)
    
    # Install base requirements
    if run_in_venv("venv-cpu", "-m", "pip", "install", "--upgrade", "pip") != 0:
        return False
    
    # Install CPU-only PyTorch
    if run_in_venv("venv-cpu", "-m", "pip", "install", 
                   "torch", "torchvision", "torchaudio",
                   "--index-url", "https://download.pytorch.org/whl/cpu") != 0:
        return False
    
    # Install other dependencies from requirements.txt
    if run_in_venv("venv-cpu", "-m", "pip", "install", "-r", "requirements.txt") != 0:
        return False
    
    print("\n" + "="*60)
    print("[OK] CPU environment ready!")
    print("="*60)
    return True


def setup_gpu_env():
    """Set up GPU environment."""
    print("\n" + "="*60)
    print("Setting up GPU environment (venv-gpu)...")
    print("="*60)
    print("[INFO] This will download ~3GB of GPU libraries")
    print("[INFO] Make sure you have at least 15GB free disk space")
    print("="*60)
    
    # Install base requirements
    if run_in_venv("venv-gpu", "-m", "pip", "install", "--upgrade", "pip") != 0:
        return False
    
    # Install GPU PyTorch with CUDA 12.4
    if run_in_venv("venv-gpu", "-m", "pip", "install",
                   "torch", "torchvision", "torchaudio",
                   "--index-url", "https://download.pytorch.org/whl/cu124") != 0:
        return False
    
    # Install other dependencies from requirements.txt
    if run_in_venv("venv-gpu", "-m", "pip", "install", "-r", "requirements.txt") != 0:
        return False
    
    print("\n" + "="*60)
    print("[OK] GPU environment ready!")
    print("="*60)
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Set up CPU and GPU build environments")
    parser.add_argument(
        "target",
        nargs="?",
        choices=["cpu", "gpu", "both"],
        default="both",
        help="Which environment(s) to set up (default: both)"
    )
    
    args = parser.parse_args()
    
    success = True
    if args.target in ("cpu", "both"):
        if not setup_cpu_env():
            success = False
    
    if args.target in ("gpu", "both"):
        if not setup_gpu_env():
            success = False
    
    if success:
        print("\n" + "="*60)
        print("[OK] All environments set up successfully!")
        print("\nTo build installers:")
        print("  CPU:  venv-cpu\\Scripts\\activate && python build_installers.py cpu")
        print("  GPU:  venv-gpu\\Scripts\\activate && python build_installers.py gpu")
        print("="*60)
        return 0
    else:
        print("\n[ERROR] Some environments failed to set up")
        return 1


if __name__ == "__main__":
    sys.exit(main())
