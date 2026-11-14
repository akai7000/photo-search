#!/usr/bin/env python3
"""Build installers locally (CPU and GPU)."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def find_iscc():
    """Find ISCC.exe in Program Files directories."""
    import glob
    
    for pattern in [
        r"C:\Program Files\Inno Setup*\ISCC.exe",
        r"C:\Program Files (x86)\Inno Setup*\ISCC.exe",
    ]:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    print("\n" + "="*60)
    print("[ERROR] ISCC.exe not found")
    print("="*60)
    print("\nTo install Inno Setup:")
    print("  1. Download from: https://www.jrsoftware.org/isdl.php")
    print("  2. Or use Chocolatey (requires admin):")
    print("     choco install innosetup -y")
    print("\nAfter installation, run this script again.")
    print("="*60 + "\n")
    
    raise FileNotFoundError("ISCC.exe not found. Please install Inno Setup.")


def build_variant(variant):
    """Build a single variant (CPU or GPU)."""
    iscc_path = find_iscc()
    print(f"\n{'='*60}")
    print(f"Building {variant} installer...")
    print(f"Using ISCC: {iscc_path}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable,
        str(ROOT / "packaging" / "build_windows.py"),
        variant,
        "--iscc", iscc_path,
    ]
    
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Build Photo Search installers locally"
    )
    parser.add_argument(
        "variants",
        nargs="*",
        choices=["cpu", "gpu", "both"],
        default=["cpu"],
        help="Which installer(s) to build: cpu, gpu, or both (default: cpu)",
    )
    
    args = parser.parse_args()
    
    variants_to_build = []
    if not args.variants or args.variants == ["cpu"]:
        variants_to_build = ["CPU"]
    elif args.variants == ["gpu"]:
        variants_to_build = ["GPU"]
    elif args.variants == ["both"]:
        variants_to_build = ["CPU", "GPU"]
    
    success = True
    for variant in variants_to_build:
        if not build_variant(variant):
            success = False
            print(f"\n[ERROR] Failed to build {variant} installer")
            if variant == "GPU":
                print("        Make sure GPU dependencies are installed:")
                print("        pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0")
                print("        pip install onnxruntime-gpu==1.20.2")
    
    if success:
        print(f"\n{'='*60}")
        print("[OK] All installers built successfully!")
        print(f"Location: {ROOT / 'dist' / 'installers'}")
        print(f"{'='*60}")
        return 0
    else:
        print(f"\n[ERROR] Some builds failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
