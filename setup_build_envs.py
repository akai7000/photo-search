#!/usr/bin/env python3
"""Set up the Photo Search build environment."""
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


def setup_build_env():
    """Set up the single installer build environment."""
    print("\n" + "="*60)
    print("Setting up the build environment (venv)...")
    print("="*60)
    
    # Install base requirements
    if run_in_venv("venv", "-m", "pip", "install", "--upgrade", "pip") != 0:
        return False
    
    # Replace any existing CUDA build with the pinned CPU-only wheels.
    if run_in_venv("venv", "-m", "pip", "install",
                   "--force-reinstall",
                   "torch==2.6.0+cpu", "torchvision==0.21.0+cpu",
                   "--index-url", "https://download.pytorch.org/whl/cpu") != 0:
        return False
    
    # Install the pinned application and packaging dependencies.
    if run_in_venv("venv", "-m", "pip", "install",
                   "-r", "packaging\\requirements-windows.txt") != 0:
        return False
    
    print("\n" + "="*60)
    print("[OK] Build environment ready!")
    print("="*60)
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Set up the Photo Search build environment")
    parser.parse_args()

    if setup_build_env():
        print("\n" + "="*60)
        print("[OK] Environment set up successfully!")
        print("\nRun build.bat to create the installer.")
        print("="*60)
        return 0
    else:
        print("\n[ERROR] Environment setup failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
