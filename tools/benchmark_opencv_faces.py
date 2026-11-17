"""Benchmark YuNet detection and SFace embeddings without changing the app."""

import argparse
import json
import os
import platform
import sqlite3
import sys
import time
from functools import partial
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.face_engine import detect_yunet_faces
from tools.benchmark_faces import (
    fingerprint_images,
    measure_samples,
    nonnegative_int,
    positive_int,
    select_images,
    sha256_file,
)


def detect_faces(path, detector, recognizer, max_dimension):
    def recognize(image, detection):
        aligned = recognizer.alignCrop(image, detection.reshape(1, -1))
        return recognizer.feature(aligned)

    return detect_yunet_faces(path, detector, recognize, max_dimension, "SFace")


def benchmark(paths, detector_model, recognizer_model, repeat, warmup, max_dimension, score_threshold=0.8):
    image_hashes, dataset_hash = fingerprint_images(paths)
    model_hashes = {path.name: sha256_file(path) for path in (detector_model, recognizer_model)}

    cv2.ocl.setUseOpenCL(False)
    start = time.perf_counter()
    detector = cv2.FaceDetectorYN_create(
        str(detector_model), "", (320, 320), score_threshold, 0.3, 5000,
        cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
    )
    recognizer = cv2.FaceRecognizerSF_create(
        str(recognizer_model), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
    )
    if detector is None or recognizer is None:
        raise RuntimeError("OpenCV could not load YuNet and SFace")
    model_load_seconds = time.perf_counter() - start
    process = partial(detect_faces, detector=detector, recognizer=recognizer, max_dimension=max_dimension)
    samples, summary = measure_samples(paths, image_hashes, process, repeat, warmup)
    return {
        "schema_version": 1,
        "benchmark": "opencv-yunet-sface",
        "scope": "image decode, YuNet detection and SFace embeddings; excludes other indexing work",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "opencv": cv2.__version__,
            "requested_backend": "DNN_BACKEND_OPENCV",
            "requested_target": "DNN_TARGET_CPU",
            "cuda_devices": cv2.cuda.getCudaEnabledDeviceCount(),
            "opencl_enabled": cv2.ocl.useOpenCL(),
            "opencv_threads": cv2.getNumThreads(),
        },
        "benchmark_source_sha256": sha256_file(Path(__file__)),
        "model_sha256": model_hashes,
        "parameters": {
            "detection_max_dimension": max_dimension,
            "score_threshold": score_threshold,
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


def score_threshold_arg(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", type=Path)
    source.add_argument("--directory", type=Path)
    parser.add_argument("--yunet", type=Path, required=True, help="YuNet ONNX model")
    parser.add_argument("--sface", type=Path, required=True, help="SFace ONNX model")
    parser.add_argument("--limit", type=positive_int, default=30)
    parser.add_argument("--repeat", type=positive_int, default=2)
    parser.add_argument("--warmup", type=nonnegative_int, default=2)
    parser.add_argument("--max-dimension", type=nonnegative_int, default=0,
                        help="Resize for detection; 0 uses full resolution")
    parser.add_argument("--score-threshold", type=score_threshold_arg, default=0.8,
                        help="YuNet detection confidence threshold (default: 0.8)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = select_images(args.directory, args.index, args.limit)
        result = benchmark(paths, args.yunet, args.sface, args.repeat, args.warmup,
                           args.max_dimension, args.score_threshold)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, sqlite3.Error, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
