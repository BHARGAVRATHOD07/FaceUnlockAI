"""
download_models.py — Download and verify official OpenCV Zoo ONNX models
-------------------------------------------------------------------------
Downloads:
  1. face_detection_yunet_2023mar.onnx (YuNet face detector + 5 landmarks)
  2. face_recognition_sface_2021dec.onnx (SFace face recognition model)

Integrity is strictly verified using official SHA256 checksums from OpenCV Zoo.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent.resolve()

MODELS_METADATA = [
    {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": (
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "expected_bytes": 232589,
    },
    {
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": (
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "expected_bytes": 38696353,
    },
]


def verify_file(filepath: Path, expected_sha256: str) -> bool:
    if not filepath.exists():
        return False
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower() == expected_sha256.lower()


def download_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for meta in MODELS_METADATA:
        dest = MODELS_DIR / meta["filename"]
        if verify_file(dest, meta["sha256"]):
            print(f"[models] Verified {meta['filename']} (checksum matches)")
            continue

        print(f"[models] Downloading {meta['filename']} ({meta['expected_bytes'] / (1024*1024):.2f} MB) ...")
        req = urllib.request.Request(
            meta["url"],
            headers={"User-Agent": "FaceUnlockAI/1.0 (Windows 11)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()

        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256.lower() != meta["sha256"].lower():
            raise RuntimeError(
                f"Checksum mismatch for {meta['filename']}!\n"
                f"Expected: {meta['sha256']}\n"
                f"Got:      {actual_sha256}"
            )

        dest.write_bytes(data)
        print(f"[models] Successfully saved and verified {dest.name}")


if __name__ == "__main__":
    try:
        download_models()
    except Exception as exc:
        print(f"[models] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
