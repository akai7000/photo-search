"""Build a Windows installer from an isolated, preconfigured environment."""

import argparse
import importlib.metadata
import importlib.util
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def collect_notices(destination):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(sys.base_prefix) / "LICENSE.txt", destination / "Python-LICENSE.txt")
    seen = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name or name.lower().replace("_", "-") in seen:
            continue
        seen.add(name.lower().replace("_", "-"))
        target = destination / f"{name}-{distribution.version}"
        for item in distribution.files or ():
            parts = item.parts
            if not parts or not parts[0].endswith(".dist-info"):
                continue
            if item.name != "METADATA" and not any(
                part.lower().startswith(("license", "copying", "notice")) for part in parts[1:]
            ):
                continue
            source = Path(distribution.locate_file(item))
            if source.is_file():
                output = target.joinpath(*parts[1:])
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("CPU", "GPU"))
    parser.add_argument("--iscc", type=Path, required=True, help="Path to Inno Setup's ISCC.exe")
    parser.add_argument("--work", type=Path, default=ROOT / "build" / "windows")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "installers")
    args = parser.parse_args()
    if sys.platform != "win32" or not args.iscc.is_file():
        parser.error("Run on Windows and provide an existing Inno Setup compiler with --iscc")

    import torch
    import onnxruntime

    providers = onnxruntime.get_available_providers()
    if args.variant == "CPU":
        if torch.version.cuda is not None or "CUDAExecutionProvider" in providers:
            parser.error("CPU builds require CPU-only PyTorch and ONNX Runtime")
    elif torch.version.cuda != "12.4" or onnxruntime.__version__ != "1.20.2" or "CUDAExecutionProvider" not in providers:
        parser.error("GPU builds require PyTorch CUDA 12.4 and onnxruntime-gpu 1.20.2")

    insightface = importlib.util.find_spec("insightface")
    if insightface is None or insightface.origin is None:
        parser.error("InsightFace must be installed in the build environment")
    meanshape = Path(insightface.origin).parent / "data" / "objects" / "meanshape_68.pkl"
    if not meanshape.is_file():
        parser.error(f"Required InsightFace asset not found: {meanshape}")

    work = args.work.resolve() / args.variant
    dist = work / "dist"
    spec = work / "spec"
    spec.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    exe_name = "PhotoSearch" + args.variant
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--windowed",
        "--name", exe_name,
        "--add-data", f"{ROOT / 'images' / 'photo_search.jpg'};images",
        "--add-data", f"{meanshape};objects",
        "--collect-data", "open_clip",
        "--copy-metadata", "nicegui",
        "--copy-metadata", "open_clip_torch",
        "--copy-metadata", "insightface",
        "--distpath", str(dist), "--workpath", str(work / "build"),
        "--specpath", str(spec), str(ROOT / "main.py"),
    ], check=True, cwd=ROOT)

    bundle = dist / exe_name
    forbidden_extensions = {".db", ".sqlite", ".sqlite3", ".onnx", ".pt", ".pth"}
    for path in bundle.rglob("*"):
        if (path.is_file() and path.suffix.lower() in forbidden_extensions) or (
            path.is_dir() and path.name.lower() in {"thumbnails", "face_thumbnails"}
        ):
            raise RuntimeError(f"Unexpected data or model file in bundle: {path}")
    collect_notices(bundle / "third_party_licenses")
    version = runpy.run_path(str(ROOT / "core" / "version.py"))["APP_VERSION"]
    subprocess.run([
        str(args.iscc.resolve()), f"/DVariant={args.variant}", f"/DAppVersion={version}",
        f"/DBundleDir={bundle}", f"/DOutputDir={args.output.resolve()}",
        str(ROOT / "packaging" / "windows-installer.iss"),
    ], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
