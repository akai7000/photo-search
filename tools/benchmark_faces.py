"""Benchmark the app's face pipeline without modifying its photo index."""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import sqlite3
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".raw"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_images(directory, index, limit):
    if index is not None:
        uri = index.resolve().as_uri() + "?mode=ro"
        with contextlib.closing(sqlite3.connect(uri, uri=True)) as connection:
            rows = connection.execute(
                "SELECT path FROM photos ORDER BY id LIMIT ?", (limit,)
            ).fetchall()
        paths = [Path(row[0]) for row in rows]
    else:
        if not directory.is_dir():
            raise ValueError(f"Image directory does not exist: {directory}")
        paths = sorted(
            (path for path in directory.rglob("*")
             if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: str(path.relative_to(directory)).casefold(),
        )[:limit]
    if not paths:
        raise ValueError("No images were selected")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Selected image is unavailable: {path}")
    return paths


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def detect_checked(detect_faces, path):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        faces = detect_faces(path)
    message = output.getvalue()
    if "Face detection error for " in message:
        raise RuntimeError(message.strip())
    if message:
        print(message, end="")
    return faces


def fingerprint_images(paths):
    image_hashes = [sha256_file(path) for path in paths]
    dataset_hash = hashlib.sha256("\n".join(image_hashes).encode("ascii")).hexdigest()
    return image_hashes, dataset_hash


def measure_samples(paths, image_hashes, detect_faces, repeat, warmup):
    for path in paths[:warmup]:
        detect_checked(detect_faces, path)

    samples = []
    for iteration in range(repeat):
        for path, image_hash in zip(paths, image_hashes):
            start = time.perf_counter()
            faces = detect_checked(detect_faces, path)
            elapsed = time.perf_counter() - start
            samples.append({
                "iteration": iteration + 1,
                "image_sha256": image_hash,
                "seconds": elapsed,
                "faces": len(faces),
            })

    durations = [sample["seconds"] for sample in samples]
    total = sum(durations)
    summary = {
        "calls": len(samples),
        "faces": sum(sample["faces"] for sample in samples),
        "total_seconds": total,
        "images_per_second": len(samples) / total,
        "mean_seconds": statistics.mean(durations),
        "median_seconds": statistics.median(durations),
        "p95_seconds": percentile(durations, 0.95),
    }
    return samples, summary


def benchmark(paths, repeat, warmup):
    import cv2
    from core import face_engine
    from core.face_models import runtime_models

    image_hashes, dataset_hash = fingerprint_images(paths)

    start = time.perf_counter()
    face_engine._models = face_engine._create_models()
    model_load_seconds = time.perf_counter() - start
    model_hashes = {path.name: sha256_file(path) for path in runtime_models()}

    samples, summary = measure_samples(paths, image_hashes, face_engine.detect_faces, repeat, warmup)
    return {
        "schema_version": 1,
        "benchmark": "app-yunet-sface",
        "scope": "face_engine.detect_faces, including image decode; excludes other indexing work",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "opencv": cv2.__version__,
            "requested_backend": "DNN_BACKEND_OPENCV",
            "requested_target": "DNN_TARGET_CPU",
            "opencl_enabled": cv2.ocl.useOpenCL(),
            "opencv_threads": cv2.getNumThreads(),
        },
        "face_engine_sha256": sha256_file(Path(face_engine.__file__)),
        "model_sha256": model_hashes,
        "dataset": {"image_count": len(paths), "sha256": dataset_hash},
        "warmup_images": min(warmup, len(paths)),
        "repeat": repeat,
        "model_load_seconds": model_load_seconds,
        "summary": summary,
        "samples": samples,
    }


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", type=Path, help="Read photo paths from this SQLite index")
    source.add_argument("--directory", type=Path, help="Recursively select images from this folder")
    parser.add_argument("--limit", type=positive_int, default=30)
    parser.add_argument("--repeat", type=positive_int, default=2)
    parser.add_argument("--warmup", type=nonnegative_int, default=2)
    parser.add_argument("--output", type=Path, required=True, help="Write JSON results here")
    args = parser.parse_args(argv)

    try:
        paths = select_images(args.directory, args.index, args.limit)
        result = benchmark(paths, args.repeat, args.warmup)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
