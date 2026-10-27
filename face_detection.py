"""
face_detection.py — FaceUnlockAI Phase 1
-----------------------------------------
Dual-backend webcam face detector:

  • PRIMARY   — MediaPipe BlazeFace (Tasks API, mediapipe >= 1.0)
                Model: models/blaze_face_short_range.tflite
                Fast, accurate, runs entirely on-device, no internet needed.

  • FALLBACK  — OpenCV Haar Cascade (haarcascade_frontalface_default.xml)
                Used only when the MediaPipe model file is absent.

SECURITY NOTES:
  - Face detection only — NO identity verification is performed.
  - This script does NOT unlock Windows or any credential provider.
  - Webcam is opened only while this script is running.
  - No frames are saved to disk.

Usage:
    .venv313\\Scripts\\Activate.ps1
    python face_detection.py [--index INDEX] [--backend {mediapipe,haar}]

Exit:
    Press Q or Escape to quit.

Model file (required for MediaPipe backend):
    models/blaze_face_short_range.tflite  (229 746 bytes)
    Source: https://storage.googleapis.com/mediapipe-models/face_detector/
            blaze_face_short_range/float16/1/blaze_face_short_range.tflite
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# ─── paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.resolve()
_MODELS_DIR = _PROJECT_ROOT / "models"
_BLAZEFACE_MODEL = _MODELS_DIR / "blaze_face_short_range.tflite"
_HAAR_XML = _PROJECT_ROOT / "haarcascade_frontalface_default.xml"

# ─── display constants ────────────────────────────────────────────────────────
WINDOW_TITLE = "FaceUnlockAI – Face Detection"
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_BOX = (0, 220, 80)       # green bounding box
COLOR_KP = (255, 180, 0)       # cyan-ish keypoints
COLOR_OK = (0, 220, 80)
COLOR_WARN = (0, 165, 255)     # amber
COLOR_ERR = (0, 0, 230)        # red (BGR)
COLOR_TEXT = (255, 255, 255)

_WIN_BACKENDS = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]


# ─── data model ───────────────────────────────────────────────────────────────

class FaceBox(NamedTuple):
    """Normalised-to-pixel bounding box (pixel coords, top-left origin)."""
    x: int
    y: int
    w: int
    h: int
    score: float              # detection confidence [0, 1]
    keypoints: list[tuple[int, int]]  # pixel coords of landmarks (may be empty)


# ─── backend base ─────────────────────────────────────────────────────────────

class DetectorBase:
    name: str = "base"

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ─── MediaPipe backend ────────────────────────────────────────────────────────

class MediaPipeDetector(DetectorBase):
    """
    Uses mediapipe.tasks.python.vision.FaceDetector in VIDEO running mode.

    VIDEO mode is correct for webcam streams: it accepts a monotonically
    increasing timestamp and does NOT require a result_callback.
    """

    name = "MediaPipe BlazeFace"

    def __init__(self, model_path: Path, min_confidence: float = 0.5) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}\n"
                f"Download it from:\n"
                f"  https://storage.googleapis.com/mediapipe-models/face_detector/"
                f"blaze_face_short_range/float16/1/blaze_face_short_range.tflite\n"
                f"and save it to: {_MODELS_DIR}/"
            )

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_detection_confidence=min_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)
        self._ts_ms: int = 0  # monotonic timestamp counter

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        # MediaPipe requires SRGB (RGB byte order)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._ts_ms += 33   # ~30 fps; must be strictly monotonically increasing
        result = self._detector.detect_for_video(mp_image, self._ts_ms)

        boxes: list[FaceBox] = []
        for det in result.detections:
            bb = det.bounding_box
            # Clamp to frame bounds
            x = max(0, bb.origin_x)
            y = max(0, bb.origin_y)
            bw = min(bb.width, w - x)
            bh = min(bb.height, h - y)

            score = det.categories[0].score if det.categories else 0.0

            kps: list[tuple[int, int]] = []
            if det.keypoints:
                for kp in det.keypoints:
                    kps.append((int(kp.x * w), int(kp.y * h)))

            boxes.append(FaceBox(x=x, y=y, w=bw, h=bh, score=score, keypoints=kps))

        return boxes

    def close(self) -> None:
        self._detector.close()


# ─── Haar Cascade fallback ────────────────────────────────────────────────────

class HaarDetector(DetectorBase):
    """OpenCV Haar Cascade face detector (fallback when TFLite model absent)."""

    name = "OpenCV Haar Cascade"

    def __init__(self, xml_path: Path) -> None:
        if not xml_path.exists():
            raise FileNotFoundError(
                f"Haar XML not found: {xml_path}\n"
                "Expected file: haarcascade_frontalface_default.xml in project root."
            )
        try:
            self._cascade = cv2.CascadeClassifier(str(xml_path))
        except Exception as err:
            raise RuntimeError(
                f"CascadeClassifier failed to parse XML: {xml_path}\n"
                "The file may be corrupt. Re-download from:\n"
                "  https://github.com/opencv/opencv/tree/master/data/haarcascades"
            ) from err

        if self._cascade.empty():
            raise RuntimeError(
                f"CascadeClassifier failed to load from: {xml_path}\n"
                "The file may be corrupt. Re-download from:\n"
                "  https://github.com/opencv/opencv/tree/master/data/haarcascades"
            )

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if not isinstance(faces, np.ndarray) or len(faces) == 0:
            return []

        return [
            FaceBox(x=int(x), y=int(y), w=int(w), h=int(h),
                    score=1.0, keypoints=[])
            for (x, y, w, h) in faces
        ]


# ─── factory ──────────────────────────────────────────────────────────────────

def build_detector(preferred: str = "mediapipe") -> DetectorBase:
    """
    Construct the best available detector.

    preferred: "mediapipe" | "haar"
    Falls back gracefully with a console warning.
    """
    if preferred == "haar":
        print(f"[face_detection] Using {HaarDetector.name} (user requested).")
        return HaarDetector(_HAAR_XML)

    # Try MediaPipe first
    if _BLAZEFACE_MODEL.exists():
        try:
            det = MediaPipeDetector(_BLAZEFACE_MODEL)
            print(f"[face_detection] Backend: {det.name}")
            return det
        except Exception as exc:
            print(f"[WARNING] MediaPipe initialisation failed: {exc}")
            print("[face_detection] Falling back to Haar Cascade.")
    else:
        print(
            f"[WARNING] MediaPipe model not found at {_BLAZEFACE_MODEL}\n"
            f"          Falling back to Haar Cascade.\n"
            f"          To use MediaPipe, download blaze_face_short_range.tflite\n"
            f"          into the models/ directory (see SETUP.md)."
        )

    return HaarDetector(_HAAR_XML)


# ─── drawing ──────────────────────────────────────────────────────────────────

def _draw_faces(frame: np.ndarray, faces: list[FaceBox]) -> None:
    """Render bounding boxes, keypoints, and confidence score onto *frame*."""
    for face in faces:
        x, y, w, h = face.x, face.y, face.w, face.h

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_BOX, 2, cv2.LINE_AA)

        # Confidence label
        label = f"{face.score:.0%}"
        label_y = max(y - 8, 18)
        cv2.putText(frame, label, (x + 2, label_y), FONT, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x + 2, label_y), FONT, 0.55,
                    COLOR_BOX, 1, cv2.LINE_AA)

        # Landmark keypoints (MediaPipe provides 6)
        for kx, ky in face.keypoints:
            cv2.circle(frame, (kx, ky), 3, COLOR_KP, -1, cv2.LINE_AA)


def _overlay_hud(
    frame: np.ndarray,
    face_count: int,
    fps: float,
    backend_name: str,
    status_line: str,
) -> None:
    """Draw the heads-up display in the top-left corner."""
    face_color = COLOR_OK if face_count > 0 else COLOR_WARN
    lines = [
        (f"Backend : {backend_name}", COLOR_TEXT),
        (f"FPS     : {fps:.1f}", COLOR_TEXT),
        (f"Faces   : {face_count}", face_color),
        (status_line, COLOR_WARN),
    ]
    y = 30
    for text, color in lines:
        cv2.putText(frame, text, (11, y + 1), FONT, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), FONT, 0.58, color, 1, cv2.LINE_AA)
        y += 26


# ─── camera helpers ───────────────────────────────────────────────────────────

def _try_open(index: int) -> cv2.VideoCapture | None:
    for backend in _WIN_BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


# ─── main loop ────────────────────────────────────────────────────────────────

def main(camera_index: int = 0, backend_name: str = "mediapipe") -> int:
    # ── detector ──
    try:
        detector = build_detector(backend_name)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        return 1
    except Exception as exc:
        print(f"\n[ERROR] Could not initialise detector: {exc}")
        return 1

    # ── camera ──
    print(f"[face_detection] Opening camera index {camera_index} …")
    cap = _try_open(camera_index)
    if cap is None:
        print(
            f"\n[ERROR] Cannot open camera index {camera_index}.\n"
            "Causes: camera not connected, another app holds it, or Windows\n"
            "privacy setting is blocking access.\n"
            "  → Settings > Privacy & Security > Camera\n"
            "Try: python face_detection.py --index 1"
        )
        detector.close()
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[face_detection] Camera {camera_index}: {w}x{h}. Press Q or Esc to exit.")

    frame_count = 0
    consecutive_failures = 0
    max_failures = 30

    t_start = time.perf_counter()
    fps = 0.0
    last_fps_time = t_start

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print("[ERROR] Camera stopped delivering frames (disconnected?).")
                    break
                time.sleep(0.03)
                continue

            consecutive_failures = 0
            frame_count += 1

            # Mirror for natural (selfie) orientation
            frame = cv2.flip(frame, 1)

            # ── detect ──
            try:
                faces = detector.detect(frame)
            except Exception as exc:
                print(f"[WARNING] Detection error on frame {frame_count}: {exc}")
                faces = []

            # ── draw ──
            _draw_faces(frame, faces)
            _overlay_hud(
                frame,
                face_count=len(faces),
                fps=fps,
                backend_name=detector.name,
                status_line="Press Q or Esc to exit",
            )

            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("\n[face_detection] Exit requested by user.")
                break

            # update FPS once per second
            now = time.perf_counter()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - t_start)
                last_fps_time = now

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    elapsed = time.perf_counter() - t_start
    print(
        f"[face_detection] Done. {frame_count} frames in {elapsed:.1f} s "
        f"({frame_count / elapsed:.1f} fps avg)."
    )
    return 0


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FaceUnlockAI – Webcam face detector (Phase 1)"
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--backend", choices=["mediapipe", "haar"], default="mediapipe",
        help="Detection backend: mediapipe (default) or haar"
    )
    args = parser.parse_args()
    sys.exit(main(camera_index=args.index, backend_name=args.backend))
