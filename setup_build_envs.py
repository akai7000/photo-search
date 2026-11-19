#!/usr/bin/env python3
"""Set up the CPU build environment."""
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Set up the CPU build environment")
    parser.add_argument(
        "target",
        nargs="?",
        choices=["cpu"],
        default="cpu",
        help="Environment to set up (default: cpu)"
    )
    
    parser.parse_args()

    if setup_cpu_env():
        print("\n" + "="*60)
        print("[OK] Environment set up successfully!")
        print("\nTo build installers:")
        print("  CPU:  venv-cpu\\Scripts\\activate && python build_installers.py cpu")
        print("="*60)
        return 0
    else:
        print("\n[ERROR] Environment setup failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
