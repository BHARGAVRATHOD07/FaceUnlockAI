"""
test_face_recognition.py — FaceUnlockAI Phase 2 Automated Tests
-----------------------------------------------------------------
Comprehensive unit and integration tests for Face Registration and Identity Verification.

Test Coverage:
  1. [models]      YuNet detector & SFace recognizer model loading and integrity.
  2. [quality]     Face crop quality assessment (blur, brightness, contrast, size).
  3. [template]    Enrollment template generation, mean normalization, metadata integrity.
  4. [lifecycle]   Enrollment check, template loading, and explicit purge/deletion.
  5. [verifier]    Identity verification with mock/synthetic embeddings:
                     • Enrolled owner verification (high similarity >= 0.415)
                     • Unknown person rejection (low similarity <= 0.345)
                     • Ambiguous match detection (borderline similarity [0.345, 0.415])
                     • Multiple faces security lockout
                     • Missing enrollment handling
                     • Poor quality / blur rejection
  6. [privacy]     Privacy guarantees:
                     • Default zero retention of raw face images
                     • Restrictive permissions handling
                     • Absence of Windows passwords / credentials in templates
                     • Disclaimers: mock tests do NOT establish true biometric security!
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from face_registration import (
    FaceEnrollmentManager,
    QualityAssessment,
    assess_quality,
    secure_directory_acls,
    MIN_FACE_WIDTH,
    MIN_FACE_HEIGHT,
    MIN_SHARPNESS_LAPLACIAN,
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
)
from face_recognition import (
    FaceVerifier,
    VerificationResult,
    VerificationStatus,
    COSINE_THRESHOLD,
    AMBIGUITY_MARGIN,
    UPPER_VERIFIED_THRESHOLD,
    LOWER_REJECTED_THRESHOLD,
    YUNET_MODEL,
    SFACE_MODEL,
)

# ─── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_face_frame():
    """640x480 frame containing a synthetic high-contrast face pattern."""
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Add a patterned face region
    cx, cy = 320, 240
    cv2.circle(img, (cx, cy), 90, (180, 160, 140), -1)
    cv2.circle(img, (cx - 30, cy - 20), 12, (20, 20, 20), -1)
    cv2.circle(img, (cx + 30, cy - 20), 12, (20, 20, 20), -1)
    cv2.line(img, (cx - 25, cy + 40), (cx + 25, cy + 40), (40, 40, 180), 6)
    return img


@pytest.fixture
def normalized_embedding():
    """128-dimensional unit-norm random embedding."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal((1, 128)).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ─── 1. MODEL INTEGRITY TESTS ─────────────────────────────────────────────────

class TestModelFiles:
    def test_yunet_model_file_exists(self):
        assert YUNET_MODEL.exists(), f"YuNet model missing at {YUNET_MODEL}"
        assert YUNET_MODEL.stat().st_size > 100_000, "YuNet model file is truncated"

    def test_sface_model_file_exists(self):
        assert SFACE_MODEL.exists(), f"SFace model missing at {SFACE_MODEL}"
        assert SFACE_MODEL.stat().st_size > 30_000_000, "SFace model file is truncated"

    def test_yunet_detector_instantiates(self):
        detector = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320))
        assert detector is not None

    def test_sface_recognizer_instantiates(self):
        recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
        assert recognizer is not None


# ─── 2. QUALITY ASSESSMENT TESTS ──────────────────────────────────────────────

class TestQualityAssessment:
    def test_rejects_too_small_face(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Face box smaller than MIN_FACE_WIDTH (80px)
        small_face = np.array([100, 100, 40, 40] + [0] * 11, dtype=np.float32)
        res = assess_quality(frame, small_face)
        assert not res.is_valid
        assert "too small" in res.reason.lower()

    def test_rejects_too_dark_face(self):
        # Frame with mean brightness < 40
        dark_frame = np.full((480, 640, 3), 15, dtype=np.uint8)
        face_data = np.array([100, 100, 150, 150] + [0] * 11, dtype=np.float32)
        res = assess_quality(dark_frame, face_data)
        assert not res.is_valid
        assert "dark" in res.reason.lower()

    def test_rejects_overexposed_face(self):
        # Frame with mean brightness > 225
        bright_frame = np.full((480, 640, 3), 245, dtype=np.uint8)
        face_data = np.array([100, 100, 150, 150] + [0] * 11, dtype=np.float32)
        res = assess_quality(bright_frame, face_data)
        assert not res.is_valid
        assert "bright" in res.reason.lower() or "glare" in res.reason.lower()

    def test_rejects_blurry_face(self):
        # Generate uniform/smooth crop with very low Laplacian variance
        smooth_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        face_data = np.array([100, 100, 150, 150] + [0] * 11, dtype=np.float32)
        res = assess_quality(smooth_frame, face_data)
        assert not res.is_valid
        assert "blur" in res.reason.lower() or "contrast" in res.reason.lower()

    def test_accepts_good_quality_synthetic_face(self, synthetic_face_frame):
        # Bounding box covering the patterned synthetic face region
        face_data = np.array([230, 150, 180, 180] + [0] * 11, dtype=np.float32)
        res = assess_quality(synthetic_face_frame, face_data)
        assert res.is_valid, f"Expected valid quality, got: {res.reason}"
        assert res.sharpness >= MIN_SHARPNESS_LAPLACIAN
        assert MIN_BRIGHTNESS <= res.brightness <= MAX_BRIGHTNESS


# ─── 3. TEMPLATE STORAGE & LIFECYCLE TESTS ─────────────────────────────────────

class TestTemplateLifecycle:
    def test_template_save_and_load(self, tmp_path, monkeypatch, normalized_embedding):
        fake_template_file = tmp_path / "test_template.npz"
        fake_enrollment_dir = tmp_path / "registered_face"
        monkeypatch.setattr("face_registration.TEMPLATE_FILE", fake_template_file)
        monkeypatch.setattr("face_registration.ENROLLMENT_DIR", fake_enrollment_dir)
        monkeypatch.setattr("face_recognition.TEMPLATE_FILE", fake_template_file)

        manager = FaceEnrollmentManager()

        # Simulate 5 samples with high consistency
        samples = [normalized_embedding.copy() for _ in range(5)]
        quality_records = [{"step": i + 1, "sharpness": 80.0, "brightness": 120.0, "contrast": 45.0} for i in range(5)]

        saved_path = manager.save_template(
            owner_name="Alice Owner",
            sample_embeddings=samples,
            quality_records=quality_records,
            raw_images=None,
        )

        assert saved_path.exists()
        assert manager.is_enrolled()

        info = manager.get_enrollment_info()
        assert info is not None
        assert info["owner_name"] == "Alice Owner"
        assert info["sample_count"] == 5
        assert info["pairwise_min_similarity"] >= 0.99
        assert "privacy_notice" in info

        # Verify no raw images saved by default
        raw_dir = fake_enrollment_dir / "raw_samples"
        assert not raw_dir.exists()

    def test_delete_enrollment(self, tmp_path, monkeypatch, normalized_embedding):
        fake_template_file = tmp_path / "test_template.npz"
        fake_enrollment_dir = tmp_path / "registered_face"
        fake_enrollment_dir.mkdir(parents=True, exist_ok=True)
        fake_template_file.write_text("mock template data")

        monkeypatch.setattr("face_registration.TEMPLATE_FILE", fake_template_file)
        monkeypatch.setattr("face_registration.ENROLLMENT_DIR", fake_enrollment_dir)

        manager = FaceEnrollmentManager()
        assert manager.is_enrolled()

        deleted = manager.delete_enrollment()
        assert deleted is True
        assert not fake_template_file.exists()
        assert not manager.is_enrolled()


# ─── 4. VERIFICATION LOGIC TESTS (MOCK / SYNTHETIC) ───────────────────────────

class TestIdentityVerificationLogic:
    def test_not_enrolled_status(self, tmp_path, monkeypatch):
        # Point to empty template path
        fake_template_file = tmp_path / "non_existent.npz"
        monkeypatch.setattr("face_recognition.TEMPLATE_FILE", fake_template_file)

        verifier = FaceVerifier()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = verifier.verify_frame(dummy_frame)

        assert result.status == VerificationStatus.NOT_ENROLLED
        assert "No owner enrolled" in result.message

    def test_no_face_detected_status(self, tmp_path, monkeypatch, normalized_embedding):
        fake_template_file = tmp_path / "test_template.npz"
        monkeypatch.setattr("face_recognition.TEMPLATE_FILE", fake_template_file)

        # Save dummy template
        np.savez_compressed(
            fake_template_file,
            mean_embedding=normalized_embedding,
            sample_embeddings=normalized_embedding,
            metadata=json.dumps({"owner_name": "Test Owner"}),
        )

        verifier = FaceVerifier()
        # All black frame has zero faces
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = verifier.verify_frame(black_frame)

        assert result.status == VerificationStatus.NO_FACE

    def test_multiple_faces_lockout(self, tmp_path, monkeypatch, normalized_embedding):
        fake_template_file = tmp_path / "test_template.npz"
        monkeypatch.setattr("face_recognition.TEMPLATE_FILE", fake_template_file)
        np.savez_compressed(
            fake_template_file,
            mean_embedding=normalized_embedding,
            sample_embeddings=normalized_embedding,
            metadata=json.dumps({"owner_name": "Test Owner"}),
        )

        verifier = FaceVerifier()

        # Mock detector returning 2 faces
        mock_detector = MagicMock()
        mock_faces = np.array([
            [100, 100, 100, 100] + [0] * 11,
            [350, 100, 100, 100] + [0] * 11,
        ], dtype=np.float32)
        mock_detector.detect.return_value = (1, mock_faces)
        verifier._detector = mock_detector

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = verifier.verify_frame(frame)

        assert result.status == VerificationStatus.MULTIPLE_FACES
        assert "Security Lockout" in result.message

    def test_verified_owner_match(self, tmp_path, monkeypatch, normalized_embedding):
        """Simulates identical or near-identical embedding yielding >= 0.415 similarity."""
        fake_template_file = tmp_path / "test_template.npz"
        monkeypatch.setattr("face_recognition.TEMPLATE_FILE", fake_template_file)
        np.savez_compressed(
            fake_template_file,
            mean_embedding=normalized_embedding,
            sample_embeddings=normalized_embedding,
            metadata=json.dumps({"owner_name": "Authorized Owner"}),
        )

        verifier = FaceVerifier(threshold=COSINE_THRESHOLD, ambiguity_margin=AMBIGUITY_MARGIN)

        # Mock single detected face with good quality
        mock_detector = MagicMock()
        mock_face = np.array([[150, 100, 150, 150, 180, 130, 220, 130, 200, 160, 185, 200, 215, 200, 0.99]], dtype=np.float32)
        mock_detector.detect.return_value = (1, mock_face)
        verifier._detector = mock_detector

        # Mock recognizer feature extraction returning exact same embedding (sim = 1.0)
        mock_recognizer = MagicMock()
        mock_recognizer.alignCrop.return_value = np.full((112, 112, 3), 128, dtype=np.uint8)
        mock_recognizer.feature.return_value = normalized_embedding.copy()
        mock_recognizer.match.return_value = 0.95  # High similarity > 0.415
        verifier._recognizer = mock_recognizer

        # Frame with sufficient contrast/brightness to pass quality
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        result = verifier.verify_frame(frame)

        assert result.status == VerificationStatus.VERIFIED_OWNER
        assert result.similarity >= UPPER_VERIFIED_THRESHOLD
        assert "ACCESS GRANTED" in result.message

    def test_rejected_unknown_person(self, tmp_path, monkeypatch, normalized_embedding):
        """Simulates an orthogonal/unrelated face embedding yielding low similarity <= 0.345."""
        fake_template_file = tmp_path / "test_template.npz"
        monkeypatch.setattr("face_recognition.TEMPLATE_FILE", fake_template_file)
        np.savez_compressed(
            fake_template_file,
            mean_embedding=normalized_embedding,
            sample_embeddings=normalized_embedding,
            metadata=json.dumps({"owner_name": "Authorized Owner"}),
        )

        verifier = FaceVerifier(threshold=COSINE_THRESHOLD, ambiguity_margin=AMBIGUITY_MARGIN)

        mock_detector = MagicMock()
        mock_face = np.array([[150, 100, 150, 150, 180, 130, 220, 130, 200, 160, 185, 200, 215, 200, 0.99]], dtype=np.float32)
        mock_detector.detect.return_value = (1, mock_face)
        verifier._detector = mock_detector

        mock_recognizer = MagicMock()
        mock_recognizer.alignCrop.return_value = np.full((112, 112, 3), 128, dtype=np.uint8)
        mock_recognizer.feature.return_value = np.ones((1, 128), dtype=np.float32)
        mock_recognizer.match.return_value = 0.12  # Low similarity < 0.345
        verifier._recognizer = mock_recognizer

        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        result = verifier.verify_frame(frame)

        assert result.status == VerificationStatus.REJECTED_UNKNOWN
        assert result.similarity <= LOWER_REJECTED_THRESHOLD
        assert "ACCESS DENIED" in result.message

    def test_ambiguous_borderline_match(self, tmp_path, monkeypatch, normalized_embedding):
        """Simulates borderline score between 0.345 and 0.415 triggering AMBIGUOUS."""
        fake_template_file = tmp_path / "test_template.npz"
        monkeypatch.setattr("face_recognition.TEMPLATE_FILE", fake_template_file)
        np.savez_compressed(
            fake_template_file,
            mean_embedding=normalized_embedding,
            sample_embeddings=normalized_embedding,
            metadata=json.dumps({"owner_name": "Authorized Owner"}),
        )

        verifier = FaceVerifier(threshold=COSINE_THRESHOLD, ambiguity_margin=AMBIGUITY_MARGIN)

        mock_detector = MagicMock()
        mock_face = np.array([[150, 100, 150, 150, 180, 130, 220, 130, 200, 160, 185, 200, 215, 200, 0.99]], dtype=np.float32)
        mock_detector.detect.return_value = (1, mock_face)
        verifier._detector = mock_detector

        mock_recognizer = MagicMock()
        mock_recognizer.alignCrop.return_value = np.full((112, 112, 3), 128, dtype=np.uint8)
        mock_recognizer.feature.return_value = np.ones((1, 128), dtype=np.float32)
        # Score right at threshold: 0.380 (within [0.345, 0.415])
        mock_recognizer.match.return_value = 0.380
        verifier._recognizer = mock_recognizer

        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        result = verifier.verify_frame(frame)

        assert result.status == VerificationStatus.AMBIGUOUS
        assert LOWER_REJECTED_THRESHOLD < result.similarity < UPPER_VERIFIED_THRESHOLD
        assert "Borderline match" in result.message or "Hold steady" in result.message


# ─── 5. PRIVACY & SECURITY TESTS ──────────────────────────────────────────────

class TestPrivacyAndSecurityConstraints:
    def test_no_windows_credentials_in_template(self, tmp_path, monkeypatch, normalized_embedding):
        fake_template_file = tmp_path / "test_template.npz"
        fake_enrollment_dir = tmp_path / "registered_face"
        monkeypatch.setattr("face_registration.TEMPLATE_FILE", fake_template_file)
        monkeypatch.setattr("face_registration.ENROLLMENT_DIR", fake_enrollment_dir)

        manager = FaceEnrollmentManager()
        samples = [normalized_embedding.copy() for _ in range(5)]
        manager.save_template("SafeOwner", samples, [])

        with np.load(fake_template_file, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))

        forbidden_keys = ["password", "pin", "credential", "token", "ntlm", "hash", "secret"]
        for k in meta.keys():
            for forbidden in forbidden_keys:
                assert forbidden not in k.lower(), f"Forbidden security key detected: {k}"

    def test_secure_directory_acls_runs_safely(self, tmp_path):
        # Must execute without unhandled crash or exception
        success = secure_directory_acls(tmp_path)
        assert isinstance(success, bool)

    def test_mock_test_biometric_limitation_notice(self):
        """
        Explaining requirement: Mock tests verify pipeline behavior and threshold gating,
        but do NOT establish biometric security against real-world adversarial attacks
        (presentation attacks, identical twins, deepfakes).
        """
        disclaimer = (
            "IMPORTANT: Automated unit tests with synthetic and mocked embeddings verify "
            "algorithmic correctness, threshold gating, and boundary condition handling. "
            "They do NOT establish true biometric spoof-resistance or field security."
        )
        assert len(disclaimer) > 50
