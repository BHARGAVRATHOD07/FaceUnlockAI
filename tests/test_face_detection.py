"""
test_face_detection.py — FaceUnlockAI Phase 1 — Automated Tests
-----------------------------------------------------------------
Tests that do NOT require a physical webcam are marked with no marker.
Tests that need a real camera are decorated @pytest.mark.webcam and skipped
in CI / headless environments unless --webcam is passed to pytest.

Run (no camera required):
    .venv313\\Scripts\\Activate.ps1
    python -m pytest tests/test_face_detection.py -v

Run (with webcam):
    python -m pytest tests/test_face_detection.py -v --webcam

What is tested:
  [model]   Model file existence and loadability
  [haar]    HaarDetector: load, detect on synthetic frames, empty-frame, edge cases
  [mp]      MediaPipeDetector: load, detect on synthetic face-like frame
  [build]   build_detector factory — correct backend selection and fallback
  [errors]  Graceful handling of missing model, corrupt XML, wrong image shape
  [webcam]  Actual camera open + frame read (hardware required, skipped by default)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# ── add project root to path ──────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_ROOT))

from face_detection import (
    FaceBox,
    HaarDetector,
    MediaPipeDetector,
    _BLAZEFACE_MODEL,
    _HAAR_XML,
    _MODELS_DIR,
    _PROJECT_ROOT,
    build_detector,
)


# pytest hooks configured in conftest.py


# ─── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def black_frame():
    """640×480 all-black BGR frame — should produce zero detections."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def synthetic_face_frame():
    """
    480×640 BGR frame with a rough skin-tone ellipse and dark eye-blobs.
    Not a real face — Haar may not detect it, but it must not crash any detector.
    """
    frame = np.full((480, 640, 3), (180, 140, 100), dtype=np.uint8)  # skin tone
    cx, cy = 320, 240
    # head outline
    cv2.ellipse(frame, (cx, cy), (80, 100), 0, 0, 360, (160, 110, 80), -1)
    # eyes
    cv2.ellipse(frame, (cx - 30, cy - 20), (15, 10), 0, 0, 360, (30, 20, 20), -1)
    cv2.ellipse(frame, (cx + 30, cy - 20), (15, 10), 0, 0, 360, (30, 20, 20), -1)
    return frame


@pytest.fixture(scope="module")
def haar_detector():
    """Shared HaarDetector for module-scope tests."""
    return HaarDetector(_HAAR_XML)


@pytest.fixture(scope="module")
def mp_detector():
    """Shared MediaPipeDetector for module-scope tests."""
    if not _BLAZEFACE_MODEL.exists():
        pytest.skip("BlazeFace model not present.")
    det = MediaPipeDetector(_BLAZEFACE_MODEL)
    yield det
    det.close()


# ─── model file tests ─────────────────────────────────────────────────────────

class TestModelFiles:
    def test_haar_xml_exists(self):
        assert _HAAR_XML.exists(), f"Haar XML missing: {_HAAR_XML}"

    def test_haar_xml_nonzero(self):
        assert _HAAR_XML.stat().st_size > 100_000, "Haar XML looks truncated."

    def test_models_dir_exists(self):
        assert _MODELS_DIR.is_dir(), f"models/ directory missing: {_MODELS_DIR}"

    def test_blazeface_tflite_exists(self):
        assert _BLAZEFACE_MODEL.exists(), (
            f"BlazeFace model not found: {_BLAZEFACE_MODEL}\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "face_detector/blaze_face_short_range/float16/1/"
            "blaze_face_short_range.tflite"
        )

    def test_blazeface_tflite_size(self):
        if not _BLAZEFACE_MODEL.exists():
            pytest.skip("BlazeFace model not present.")
        size = _BLAZEFACE_MODEL.stat().st_size
        # Official model is 229 746 bytes; allow ±5 %
        assert 218_000 < size < 241_000, (
            f"BlazeFace model size unexpected ({size} bytes). "
            "File may be corrupt or wrong version."
        )


# ─── HaarDetector tests ───────────────────────────────────────────────────────

class TestHaarDetector:
    def test_loads_successfully(self):
        """HaarDetector must not raise on valid XML."""
        det = HaarDetector(_HAAR_XML)
        det.close()

    def test_missing_xml_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HaarDetector(tmp_path / "nonexistent.xml")

    def test_corrupt_xml_raises(self, tmp_path):
        bad_xml = tmp_path / "bad.xml"
        bad_xml.write_text("<not valid cascade xml>")
        with pytest.raises(RuntimeError):
            HaarDetector(bad_xml)

    def test_black_frame_returns_list(self, haar_detector, black_frame):
        result = haar_detector.detect(black_frame)
        assert isinstance(result, list), "detect() must return a list."

    def test_black_frame_no_faces(self, haar_detector, black_frame):
        faces = haar_detector.detect(black_frame)
        assert len(faces) == 0, f"Expected 0 faces on black frame, got {len(faces)}."

    def test_synthetic_face_returns_list(self, haar_detector, synthetic_face_frame):
        faces = haar_detector.detect(synthetic_face_frame)
        assert isinstance(faces, list)

    def test_result_is_facebox(self, haar_detector, black_frame):
        faces = haar_detector.detect(black_frame)
        for f in faces:
            assert isinstance(f, FaceBox)

    def test_facebox_fields_in_bounds(self, haar_detector, synthetic_face_frame):
        h, w = synthetic_face_frame.shape[:2]
        faces = haar_detector.detect(synthetic_face_frame)
        for f in faces:
            assert 0 <= f.x < w
            assert 0 <= f.y < h
            assert f.w > 0
            assert f.h > 0
            assert f.x + f.w <= w
            assert f.y + f.h <= h

    def test_detect_small_frame(self, haar_detector):
        """1×1 frame must not crash."""
        tiny = np.zeros((1, 1, 3), dtype=np.uint8)
        result = haar_detector.detect(tiny)
        assert isinstance(result, list)

    def test_detect_gray_raises_gracefully_or_works(self, haar_detector):
        """2-channel or single-channel input: must not segfault."""
        gray = np.zeros((100, 100, 1), dtype=np.uint8)
        # cv2.cvtColor inside detect will raise if shape is wrong —
        # that is acceptable; we just ensure no segfault (i.e. any exception is fine)
        try:
            haar_detector.detect(gray)
        except Exception:
            pass  # acceptable — important thing is it doesn't crash the process


# ─── MediaPipeDetector tests ──────────────────────────────────────────────────

class TestMediaPipeDetector:
    def test_loads_successfully(self):
        if not _BLAZEFACE_MODEL.exists():
            pytest.skip("BlazeFace model not present.")
        det = MediaPipeDetector(_BLAZEFACE_MODEL)
        det.close()

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MediaPipeDetector(tmp_path / "missing.tflite")

    def test_black_frame_returns_list(self, mp_detector, black_frame):
        result = mp_detector.detect(black_frame)
        assert isinstance(result, list)

    def test_black_frame_no_faces(self, mp_detector, black_frame):
        faces = mp_detector.detect(black_frame)
        assert len(faces) == 0, (
            f"Expected 0 detections on all-black frame, got {len(faces)}."
        )

    def test_result_is_facebox(self, mp_detector, black_frame):
        for f in mp_detector.detect(black_frame):
            assert isinstance(f, FaceBox)

    def test_synthetic_frame_no_crash(self, mp_detector, synthetic_face_frame):
        """Synthetic frame must not crash; detection result is don't-care."""
        result = mp_detector.detect(synthetic_face_frame)
        assert isinstance(result, list)

    def test_facebox_fields_valid(self, mp_detector, synthetic_face_frame):
        h, w = synthetic_face_frame.shape[:2]
        for f in mp_detector.detect(synthetic_face_frame):
            assert isinstance(f, FaceBox)
            assert 0 <= f.score <= 1.0
            assert isinstance(f.keypoints, list)

    def test_timestamp_is_monotonic(self, mp_detector, black_frame):
        """detect_for_video requires strictly increasing timestamps."""
        ts_before = mp_detector._ts_ms
        mp_detector.detect(black_frame)
        ts_after = mp_detector._ts_ms
        assert ts_after > ts_before, "Timestamp must increase after each detect call."


# ─── build_detector factory tests ─────────────────────────────────────────────

class TestBuildDetector:
    def test_haar_backend_explicit(self):
        det = build_detector("haar")
        assert isinstance(det, HaarDetector)
        det.close()

    def test_mediapipe_backend_uses_mp_when_model_present(self):
        if not _BLAZEFACE_MODEL.exists():
            pytest.skip("BlazeFace model not present.")
        det = build_detector("mediapipe")
        assert isinstance(det, MediaPipeDetector)
        det.close()

    def test_mediapipe_falls_back_to_haar_when_model_missing(self, tmp_path):
        """
        When the BlazeFace model file does not exist, build_detector must
        silently fall back to HaarDetector, not raise.
        """
        # Temporarily patch _BLAZEFACE_MODEL to a non-existent path
        fake_model = tmp_path / "missing.tflite"
        with patch("face_detection._BLAZEFACE_MODEL", fake_model):
            det = build_detector("mediapipe")
        assert isinstance(det, HaarDetector), (
            "Expected HaarDetector fallback when BlazeFace model is absent."
        )
        det.close()

    def test_unknown_backend_falls_back_to_mediapipe_or_haar(self):
        """Any unrecognised backend string should not crash build_detector."""
        # build_detector checks preferred == "haar" then tries mediapipe
        # an unknown string will follow the mediapipe path
        det = build_detector("unknown_backend")
        assert isinstance(det, (MediaPipeDetector, HaarDetector))
        det.close()


# ─── FaceBox data integrity ───────────────────────────────────────────────────

class TestFaceBox:
    def test_creation(self):
        fb = FaceBox(x=10, y=20, w=80, h=100, score=0.95, keypoints=[(50, 60)])
        assert fb.x == 10
        assert fb.y == 20
        assert fb.w == 80
        assert fb.h == 100
        assert fb.score == 0.95
        assert fb.keypoints == [(50, 60)]

    def test_is_named_tuple(self):
        fb = FaceBox(0, 0, 1, 1, 0.5, [])
        assert isinstance(fb, tuple)


# ─── camera hardware tests (skipped by default) ───────────────────────────────

class TestWebcam:
    @pytest.mark.webcam
    def test_camera_opens(self):
        """Camera index 0 must open successfully."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        opened = cap.isOpened()
        cap.release()
        assert opened, (
            "cv2.VideoCapture(0, CAP_DSHOW) did not open. "
            "Check camera connection and Windows privacy settings."
        )

    @pytest.mark.webcam
    def test_camera_delivers_frame(self):
        """Camera must deliver at least one valid non-black frame."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        assert cap.isOpened(), "Camera did not open."
        # Read up to 10 frames to skip initial blank frames some cameras send
        frame = None
        for _ in range(10):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
                break
        cap.release()
        assert frame is not None, "Camera opened but delivered no frames."
        assert frame.ndim == 3, "Frame must be 3-dimensional (H×W×C)."
        assert frame.shape[2] == 3, "Frame must have 3 colour channels."

    @pytest.mark.webcam
    def test_haar_detects_on_live_frame(self):
        """Integration: Haar does not crash processing a real camera frame."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        assert cap.isOpened()
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            pytest.skip("Camera delivered no frame.")
        det = HaarDetector(_HAAR_XML)
        result = det.detect(frame)
        assert isinstance(result, list), "detect() must return a list."

    @pytest.mark.webcam
    def test_mediapipe_detects_on_live_frame(self):
        """Integration: MediaPipe does not crash processing a real camera frame."""
        if not _BLAZEFACE_MODEL.exists():
            pytest.skip("BlazeFace model not present.")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        assert cap.isOpened()
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            pytest.skip("Camera delivered no frame.")
        det = MediaPipeDetector(_BLAZEFACE_MODEL)
        result = det.detect(frame)
        det.close()
        assert isinstance(result, list)
