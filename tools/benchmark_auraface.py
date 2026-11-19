"""Benchmark YuNet detection and AuraFace embeddings without changing the app."""

import argparse
import importlib.metadata
import json
import os
import platform
import sqlite3
import sys
import time
from functools import partial
from pathlib import Path

import cv2
import onnxruntime
from insightface.app.common import Face
from insightface.model_zoo import get_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.benchmark_faces import (
    fingerprint_images,
    measure_samples,
    nonnegative_int,
    positive_int,
    select_images,
    sha256_file,
)
from tools.benchmark_opencv_faces import detect_yunet_faces


def detect_faces(path, detector, recognizer, max_dimension):
    def recognize(image, detection):
        face = Face(kps=detection[4:14].reshape(5, 2))
        return recognizer.get(image, face)

    return detect_yunet_faces(path, detector, recognize, max_dimension, "AuraFace")


def benchmark(paths, detector_model, recognizer_model, repeat, warmup, max_dimension, provider):
    if provider != "CPUExecutionProvider":
        raise ValueError(f"Only the CPU execution provider is supported: {provider}")
    image_hashes, dataset_hash = fingerprint_images(paths)
    model_hashes = {path.name: sha256_file(path) for path in (detector_model, recognizer_model)}
    if provider not in onnxruntime.get_available_providers():
        raise RuntimeError(f"Requested ONNX Runtime provider is unavailable: {provider}")

    cv2.ocl.setUseOpenCL(False)
    start = time.perf_counter()
    detector = cv2.FaceDetectorYN_create(
        str(detector_model), "", (320, 320), 0.8, 0.3, 5000,
        cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
    )
    recognizer = get_model(str(recognizer_model), providers=[provider])
    if detector is None or recognizer is None or recognizer.taskname != "recognition":
        raise RuntimeError("Could not load YuNet and the AuraFace recognition model")
    recognizer.prepare(ctx_id=0)
    actual_providers = recognizer.session.get_providers()
    if not actual_providers or actual_providers[0] != provider:
        raise RuntimeError(f"Requested {provider}, but AuraFace session uses {actual_providers}")
    model_load_seconds = time.perf_counter() - start

    process = partial(detect_faces, detector=detector, recognizer=recognizer, max_dimension=max_dimension)
    samples, summary = measure_samples(paths, image_hashes, process, repeat, warmup)
    return {
        "schema_version": 1,
        "benchmark": "opencv-yunet-auraface",
        "scope": "image decode, CPU YuNet detection and AuraFace embeddings; excludes other indexing work",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "opencv": cv2.__version__,
            "opencv_requested_backend": "DNN_BACKEND_OPENCV",
            "opencv_requested_target": "DNN_TARGET_CPU",
            "opencv_opencl_enabled": cv2.ocl.useOpenCL(),
            "opencv_threads": cv2.getNumThreads(),
            "insightface": importlib.metadata.version("insightface"),
            "onnxruntime": onnxruntime.__version__,
            "available_providers": onnxruntime.get_available_providers(),
            "recognition_session_providers": actual_providers,
        },
        "benchmark_source_sha256": sha256_file(Path(__file__)),
        "yunet_source_sha256": sha256_file(Path(sys.modules[detect_yunet_faces.__module__].__file__)),
        "model_sha256": model_hashes,
        "parameters": {
            "detection_max_dimension": max_dimension,
            "score_threshold": 0.8,
            "nms_threshold": 0.3,
            "top_k": 5000,
        },
        "dataset": {"image_count": len(paths), "sha256": dataset_hash},
        "warmup_images": min(warmup, len(paths)),
        "repeat": repeat,
        "model_load_seconds": model_load_seconds,
        "summary": summary,
        "samples": samples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", type=Path)
    source.add_argument("--directory", type=Path)
    parser.add_argument("--yunet", type=Path, required=True, help="YuNet ONNX model")
    parser.add_argument("--auraface", type=Path, required=True, help="AuraFace recognition ONNX model")
    parser.add_argument("--provider", choices=("CPUExecutionProvider",), required=True)
    parser.add_argument("--limit", type=positive_int, default=30)
    parser.add_argument("--repeat", type=positive_int, default=2)
    parser.add_argument("--warmup", type=nonnegative_int, default=2)
    parser.add_argument("--max-dimension", type=nonnegative_int, default=0,
                        help="Resize for detection; 0 uses full resolution")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = select_images(args.directory, args.index, args.limit)
        result = benchmark(paths, args.yunet, args.auraface, args.repeat, args.warmup,
                           args.max_dimension, args.provider)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, sqlite3.Error, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
