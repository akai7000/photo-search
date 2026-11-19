"""Build a Windows installer from an isolated, preconfigured environment."""

import argparse
import importlib.metadata
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.face_models import MODEL_SPECS, ensure_models, verify_model


def pyinstaller_assets(models):
    icon = ROOT / "images" / "snowfoxninjalogo.ico"
    return [
        "--icon", str(icon),
        "--add-data", f"{icon};images",
        "--add-data", f"{ROOT / 'images' / 'snowfoxninjalogo.png'};images",
        *[item for model in models for item in ("--add-data", f"{model};models")],
    ]


def torchvision_assets(directory):
    libraries = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".pyd", ".dll"})
    if not any(path.name.startswith("_C") and path.suffix.lower() == ".pyd" for path in libraries):
        raise RuntimeError(f"torchvision native extension not found in {directory}")
    return [item for path in libraries for item in ("--add-binary", f"{path};torchvision")]


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
    for name in ("YuNet-LICENSE.txt", "SFace-LICENSE.txt"):
        shutil.copy2(ROOT / "packaging" / "model_licenses" / name, destination / name)


def validate_bundle(bundle):
    models_dir = bundle / "_internal" / "models"
    expected = {models_dir / name for name in MODEL_SPECS}
    for path in bundle.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".onnx", ".pt", ".pth"}:
            if path not in expected:
                raise RuntimeError(f"Unexpected data or model file in bundle: {path}")
        if path.is_dir() and path.name.lower() in {"thumbnails", "face_thumbnails"}:
            raise RuntimeError(f"Unexpected data directory in bundle: {path}")
    for name, (_, digest) in MODEL_SPECS.items():
        verify_model(models_dir / name, digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iscc", type=Path, required=True, help="Path to Inno Setup's ISCC.exe")
    parser.add_argument("--work", type=Path, default=ROOT / "build" / "windows")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "installers")
    args = parser.parse_args()
    if sys.platform != "win32" or not args.iscc.is_file():
        parser.error("Run on Windows and provide an existing Inno Setup compiler with --iscc")

    import torch
    import torchvision

    if torch.version.cuda is not None:
        parser.error("CUDA-enabled PyTorch is not supported by this application")
    models = ensure_models(ROOT / "models", download=True)

    work = args.work.resolve()
    dist = work / "dist"
    spec = work / "spec"
    spec.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    exe_name = "PhotoSearch"
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--windowed",
        "--name", exe_name,
        *pyinstaller_assets(models),
        *torchvision_assets(Path(torchvision.__file__).parent),
        "--collect-data", "open_clip",
        "--copy-metadata", "nicegui",
        "--copy-metadata", "open_clip_torch",
        "--distpath", str(dist), "--workpath", str(work / "build"),
        "--specpath", str(spec), str(ROOT / "main.py"),
    ], check=True, cwd=ROOT)

    bundle = dist / exe_name
    validate_bundle(bundle)
    collect_notices(bundle / "third_party_licenses")
    version = runpy.run_path(str(ROOT / "core" / "version.py"))["APP_VERSION"]
    subprocess.run([
        str(args.iscc.resolve()), f"/DAppVersion={version}",
        f"/DBundleDir={bundle}", f"/DOutputDir={args.output.resolve()}",
        str(ROOT / "packaging" / "windows-installer.iss"),
    ], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
