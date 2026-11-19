#!/usr/bin/env python3
"""Build the CPU installer locally."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_python_version():
    """Check if Python version is compatible."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 11):
        print("\n" + "="*60)
        print("[ERROR] Python 3.11 or later is required")
        print(f"[ERROR] You are using Python {major}.{minor}")
        print("="*60 + "\n")
        sys.exit(1)
    
    if major >= 3 and minor > 13:
        print("\n" + "="*60)
        print(f"[WARN] Python {major}.{minor} may not be fully supported")
        print("[WARN] Python 3.11-3.13 is recommended for best compatibility")
        print("[WARN] Continuing anyway...")
        print("="*60 + "\n")


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


def build_cpu():
    """Build the CPU installer."""
    iscc_path = find_iscc()
    print(f"\n{'='*60}")
    print("Building CPU installer...")
    print(f"Using ISCC: {iscc_path}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable,
        str(ROOT / "packaging" / "build_windows.py"),
        "CPU",
        "--iscc", iscc_path,
    ]
    
    result = subprocess.run(cmd, cwd=ROOT)
    
    if result.returncode != 0:
        print("\n[ERROR] CPU build failed. Check the error above.")
        print("CPU builds require CPU-only PyTorch and torchvision.")
    
    return result.returncode == 0


def main():
    check_python_version()
    
    parser = argparse.ArgumentParser(
        description="Build Photo Search installers locally"
    )
    parser.add_argument(
        "variant",
        nargs="?",
        choices=["cpu"],
        default="cpu",
        help="Installer to build (default: cpu)",
    )
    
    parser.parse_args()

    if build_cpu():
        print(f"\n{'='*60}")
        print("[OK] Installer built successfully!")
        print(f"Location: {ROOT / 'dist' / 'installers'}")
        print(f"{'='*60}")
        return 0
    else:
        print("\n[ERROR] Installer build failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
