"""
face_recognition.py — FaceUnlockAI Phase 2
---------------------------------------------
Local Face Identity Verification Module for Windows 11.

Features:
  1. Maintained deep model pipeline: OpenCV YuNet (detection) + SFace (recognition).
  2. Documented thresholding:
       • Baseline SFace cosine similarity: 0.363 (official OpenCV Zoo threshold)
       • Verified decision threshold:      0.380
       • Ambiguity margin:                 ±0.035 [0.345 to 0.415]
  3. Safety & security handling:
       • Unknown-person rejection
       • Ambiguous match detection (requires steady, clearer view rather than false-accept)
       • Multiple faces security lockout (prevents shoulder-surfing / presentation confusion)
       • Poor lighting and motion-blur rejection
       • Missing enrollment guidance
  4. Local computation: Zero data egress, no cloud APIs, no raw image uploads.
  5. Interactive controls:
       • 'Q' / ESC  — Exit
       • 'E'        — Open enrollment if missing or re-enroll
       • 'D'        — Delete enrollment data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ─── PATHS ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
ENROLLMENT_DIR = DATA_DIR / "registered_face"
TEMPLATE_FILE = ENROLLMENT_DIR / "owner_template.npz"

YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# ─── DECISION THRESHOLDS ───────────────────────────────────────────────────────
# Official SFace cosine similarity threshold is 0.363.
# A slightly tighter threshold (0.380) with an ambiguity band gives balanced verification.
COSINE_THRESHOLD = 0.380
AMBIGUITY_MARGIN = 0.035
UPPER_VERIFIED_THRESHOLD = COSINE_THRESHOLD + AMBIGUITY_MARGIN   # 0.415
LOWER_REJECTED_THRESHOLD = COSINE_THRESHOLD - AMBIGUITY_MARGIN   # 0.345

# Quality limits
MIN_SHARPNESS = 40.0
MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 230.0


class VerificationStatus(Enum):
    VERIFIED_OWNER = "VERIFIED_OWNER"
    REJECTED_UNKNOWN = "REJECTED_UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    POOR_QUALITY = "POOR_QUALITY"
    NOT_ENROLLED = "NOT_ENROLLED"


@dataclass
class VerificationResult:
    status: VerificationStatus
    similarity: float
    confidence_pct: float
    message: str
    face_box: Optional[Tuple[int, int, int, int]] = None
    landmarks: Optional[np.ndarray] = None
    owner_name: str = ""
    details: Optional[dict] = None


class FaceVerifier:
    """
    Evaluates camera frames against the enrolled owner template.
    """

    def __init__(
        self,
        threshold: float = COSINE_THRESHOLD,
        ambiguity_margin: float = AMBIGUITY_MARGIN,
    ) -> None:
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin
        self.upper_threshold = threshold + ambiguity_margin
        self.lower_threshold = threshold - ambiguity_margin

        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None

        self._owner_mean_embedding: Optional[np.ndarray] = None
        self._owner_sample_embeddings: Optional[np.ndarray] = None
        self._owner_name: str = "Device Owner"
        self._template_metadata: dict = {}

        self.load_models()
        self.reload_enrollment()

    def load_models(self) -> None:
        if not YUNET_MODEL.exists() or not SFACE_MODEL.exists():
            print("[verifier] Required ONNX models missing. Attempting download...", flush=True)
            from models.download_models import download_models
            download_models()

        if not YUNET_MODEL.exists():
            raise FileNotFoundError(f"YuNet model missing at {YUNET_MODEL}")
        if not SFACE_MODEL.exists():
            raise FileNotFoundError(f"SFace model missing at {SFACE_MODEL}")

        self._recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")

    def reload_enrollment(self) -> bool:
        """Loads or refreshes the enrolled owner template from disk."""
        if not TEMPLATE_FILE.exists():
            self._owner_mean_embedding = None
            self._owner_sample_embeddings = None
            self._owner_name = ""
            self._template_metadata = {}
            return False

        try:
            with np.load(TEMPLATE_FILE, allow_pickle=False) as data:
                self._owner_mean_embedding = np.array(data["mean_embedding"], dtype=np.float32)
                self._owner_sample_embeddings = np.array(data["sample_embeddings"], dtype=np.float32)
                self._template_metadata = json.loads(str(data["metadata"]))
                self._owner_name = self._template_metadata.get("owner_name", "Device Owner")
            return True
        except Exception as exc:
            print(f"[verifier] Error reading template file: {exc}", file=sys.stderr)
            self._owner_mean_embedding = None
            return False

    def is_enrolled(self) -> bool:
        return self._owner_mean_embedding is not None

    def _ensure_detector(self, width: int, height: int) -> cv2.FaceDetectorYN:
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(YUNET_MODEL),
                "",
                (width, height),
                score_threshold=0.75,
                nms_threshold=0.3,
                top_k=5,
            )
        else:
            self._detector.setInputSize((width, height))
        return self._detector

    def verify_frame(self, frame: np.ndarray) -> VerificationResult:
        """
        Processes a single BGR frame and returns a VerificationResult.
        """
        if not self.is_enrolled():
            return VerificationResult(
                status=VerificationStatus.NOT_ENROLLED,
                similarity=0.0,
                confidence_pct=0.0,
                message="No owner enrolled. Please run enrollment first.",
            )

        h_frame, w_frame = frame.shape[:2]
        detector = self._ensure_detector(w_frame, h_frame)

        retval, faces = detector.detect(frame)
        num_faces = len(faces) if (faces is not None and retval == 1) else 0

        if num_faces == 0:
            return VerificationResult(
                status=VerificationStatus.NO_FACE,
                similarity=0.0,
                confidence_pct=0.0,
                message="Searching for face...",
            )

        if num_faces > 1:
            # Multi-face safety lock
            return VerificationResult(
                status=VerificationStatus.MULTIPLE_FACES,
                similarity=0.0,
                confidence_pct=0.0,
                message=f"Multiple faces ({num_faces}) detected — Security Lockout",
                details={"faces_count": num_faces},
            )

        face_data = faces[0]
        x, y, w, h = [int(v) for v in face_data[:4]]
        landmarks = face_data[4:14].reshape((5, 2)).astype(int)

        # Quality filters
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_frame, x + w), min(h_frame, y + h)
        if (x2 - x1) < 60 or (y2 - y1) < 60:
            return VerificationResult(
                status=VerificationStatus.POOR_QUALITY,
                similarity=0.0,
                confidence_pct=0.0,
                message="Face too small or off-screen. Move closer.",
                face_box=(x, y, w, h),
                landmarks=landmarks,
            )

        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        if sharpness < MIN_SHARPNESS:
            return VerificationResult(
                status=VerificationStatus.POOR_QUALITY,
                similarity=0.0,
                confidence_pct=0.0,
                message=f"Image too blurry (sharpness: {sharpness:.1f}). Hold steady.",
                face_box=(x, y, w, h),
                landmarks=landmarks,
            )

        if brightness < MIN_BRIGHTNESS:
            return VerificationResult(
                status=VerificationStatus.POOR_QUALITY,
                similarity=0.0,
                confidence_pct=0.0,
                message="Lighting too dark. Increase illumination.",
                face_box=(x, y, w, h),
                landmarks=landmarks,
            )

        if brightness > MAX_BRIGHTNESS:
            return VerificationResult(
                status=VerificationStatus.POOR_QUALITY,
                similarity=0.0,
                confidence_pct=0.0,
                message="Glare or overexposure detected.",
                face_box=(x, y, w, h),
                landmarks=landmarks,
            )

        # Align & extract feature embedding using SFace
        aligned = self._recognizer.alignCrop(frame, face_data)
        query_feat = self._recognizer.feature(aligned)

        # Cosine similarity against enrolled template
        # 1. Similarity with mean embedding
        mean_sim = float(self._recognizer.match(query_feat, self._owner_mean_embedding, cv2.FaceRecognizerSF_FR_COSINE))

        # 2. Maximum similarity against individual enrolled exemplars (with slight penalty factor)
        exemplar_max = 0.0
        if self._owner_sample_embeddings is not None and len(self._owner_sample_embeddings) > 0:
            exemplar_sims = [
                float(self._recognizer.match(query_feat, e.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE))
                for e in self._owner_sample_embeddings
            ]
            exemplar_max = max(exemplar_sims)

        effective_sim = max(mean_sim, exemplar_max * 0.95)

        # Linear confidence percentage mapping:
        # 0.20 -> 0%, 0.38 (threshold) -> 75%, 0.60+ -> 100%
        if effective_sim < 0.20:
            conf_pct = max(0.0, effective_sim / 0.20 * 25.0)
        elif effective_sim < self.threshold:
            conf_pct = 25.0 + ((effective_sim - 0.20) / (self.threshold - 0.20)) * 50.0
        else:
            conf_pct = min(100.0, 75.0 + ((effective_sim - self.threshold) / 0.25) * 25.0)

        # Threshold decision logic
        if effective_sim >= self.upper_threshold:
            status = VerificationStatus.VERIFIED_OWNER
            msg = f"ACCESS GRANTED — Owner: {self._owner_name} (Sim: {effective_sim:.3f})"
        elif effective_sim <= self.lower_threshold:
            status = VerificationStatus.REJECTED_UNKNOWN
            msg = f"ACCESS DENIED — Unknown Person (Sim: {effective_sim:.3f})"
        else:
            status = VerificationStatus.AMBIGUOUS
            msg = f"VERIFYING — Borderline match ({effective_sim:.3f}). Hold steady."

        return VerificationResult(
            status=status,
            similarity=effective_sim,
            confidence_pct=round(conf_pct, 1),
            message=msg,
            face_box=(x, y, w, h),
            landmarks=landmarks,
            owner_name=self._owner_name,
            details={
                "mean_similarity": round(mean_sim, 4),
                "exemplar_max": round(exemplar_max, 4),
                "sharpness": round(sharpness, 1),
                "brightness": round(brightness, 1),
            },
        )

    def run_live_verification(
        self,
        camera_index: int = 0,
        target_width: int = 640,
        target_height: int = 480,
    ) -> None:
        """
        Interactive camera loop with real-time verification HUD.
        """
        print("\n" + "=" * 65)
        print("  FaceUnlockAI — Live Identity Verification")
        print("=" * 65)
        print(f"Enrolled Status:    {'ENROLLED (' + self._owner_name + ')' if self.is_enrolled() else 'NOT ENROLLED'}")
        print(f"Cosine Threshold:   {self.threshold:.3f} (Ambiguity band: [{self.lower_threshold:.3f}, {self.upper_threshold:.3f}])")
        print("Controls:           Press 'Q' or ESC to exit.")
        print("                    Press 'E' to open enrollment.")
        print("                    Press 'D' to delete enrolled template.")
        print("=" * 65 + "\n")

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[verifier] ERROR: Cannot open webcam index {camera_index}.", file=sys.stderr)
            return

        window_name = "FaceUnlockAI — Identity Verification"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        fps_history: list[float] = []
        last_time = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue

                # Mirror for natural presentation
                frame = cv2.flip(frame, 1)
                h_frame, w_frame = frame.shape[:2]

                # Measure FPS
                now = time.perf_counter()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    fps_history.append(1.0 / dt)
                    if len(fps_history) > 30:
                        fps_history.pop(0)
                fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

                result = self.verify_frame(frame)
                display = frame.copy()

                # Choose colors based on status
                if result.status == VerificationStatus.VERIFIED_OWNER:
                    theme_color = (0, 220, 0)     # Vibrant Green
                    badge_text = f"OWNER: {result.owner_name.upper()} [CONFIRMED]"
                elif result.status == VerificationStatus.REJECTED_UNKNOWN:
                    theme_color = (0, 0, 240)     # Vibrant Red
                    badge_text = "UNKNOWN PERSON [ACCESS DENIED]"
                elif result.status == VerificationStatus.AMBIGUOUS:
                    theme_color = (0, 190, 255)   # Amber / Yellow
                    badge_text = "AMBIGUOUS MATCH [HOLD STEADY]"
                elif result.status == VerificationStatus.MULTIPLE_FACES:
                    theme_color = (180, 0, 220)   # Purple / Warning
                    badge_text = "MULTIPLE FACES [SECURITY LOCK]"
                elif result.status == VerificationStatus.NOT_ENROLLED:
                    theme_color = (200, 200, 200) # Gray
                    badge_text = "NOT ENROLLED [PRESS 'E' TO ENROLL]"
                elif result.status == VerificationStatus.POOR_QUALITY:
                    theme_color = (0, 140, 255)   # Orange
                    badge_text = "POOR IMAGE QUALITY"
                else:
                    theme_color = (120, 120, 120) # Gray
                    badge_text = "SEARCHING FOR FACE..."

                # Draw bounding box and landmarks
                if result.face_box is not None:
                    x, y, w, h = result.face_box
                    cv2.rectangle(display, (x, y), (x + w, y + h), theme_color, 2)
                    if result.landmarks is not None:
                        for lx, ly in result.landmarks:
                            cv2.circle(display, (lx, ly), 3, (0, 255, 255), -1)

                    # Top label on face box
                    label = f"{badge_text} ({result.similarity:.2f})"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display, (x, y - 24), (x + label_size[0] + 10, y), theme_color, -1)
                    cv2.putText(display, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Top HUD Banner
                cv2.rectangle(display, (0, 0), (w_frame, 70), (15, 15, 15), -1)
                cv2.putText(display, badge_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, theme_color, 2)
                cv2.putText(display, result.message, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)

                # Top-Right FPS Counter
                cv2.putText(display, f"{fps:.1f} FPS", (w_frame - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

                # Bottom Score / Status Bar
                cv2.rectangle(display, (0, h_frame - 65), (w_frame, h_frame), (15, 15, 15), -1)

                # Visual similarity score bar
                bar_x, bar_y, bar_w, bar_h = 20, h_frame - 50, 260, 14
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
                fill_ratio = max(0.0, min(1.0, result.similarity))
                fill_w = int(bar_w * fill_ratio)
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), theme_color, -1)

                # Draw threshold indicator line on bar
                thresh_x = int(bar_x + bar_w * self.threshold)
                cv2.line(display, (thresh_x, bar_y - 3), (thresh_x, bar_y + bar_h + 3), (255, 255, 255), 2)

                score_text = f"Sim: {result.similarity:.3f} (Thresh: {self.threshold:.3f})"
                cv2.putText(display, score_text, (bar_x + bar_w + 15, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

                # Hotkey reminders
                cv2.putText(display, "[Q] Exit   [E] Enroll / Re-enroll   [D] Delete Template", (20, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # Q or Escape
                    print("\n[verifier] Exit requested by user.")
                    break

                elif key in (ord("e"), ord("E")):    # Enroll
                    print("\n[verifier] Switching to Owner Enrollment...")
                    cap.release()
                    cv2.destroyAllWindows()
                    from face_registration import FaceEnrollmentManager
                    enroll_mgr = FaceEnrollmentManager()
                    enroll_mgr.start_enrollment(camera_index=camera_index)
                    self.reload_enrollment()
                    cap = cv2.VideoCapture(camera_index, backend)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(camera_index)
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

                elif key in (ord("d"), ord("D")):    # Delete enrollment
                    print("\n[verifier] Delete confirmation requested.")
                    from face_registration import FaceEnrollmentManager
                    enroll_mgr = FaceEnrollmentManager()
                    enroll_mgr.delete_enrollment()
                    self.reload_enrollment()

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run_secure_unlock_session(self, camera_index: int = 0) -> bool:
        """
        Executes full defense-in-depth secure unlock requiring BOTH
        enrolled identity verification AND active challenge-response liveness.
        Fails closed on any anomaly, timeout, or unknown face.
        """
        from liveness_detection import LivenessPipeline, LivenessStatus

        if not self.is_enrolled():
            print("[verifier] ERROR: No owner is enrolled. Please complete enrollment first.", file=sys.stderr)
            return False

        pipeline = LivenessPipeline(num_challenges=2)
        session, err = pipeline.start_session()
        if session is None:
            print(f"\n[verifier] SECURITY LOCKOUT: {err}", file=sys.stderr)
            return False

        print("\n" + "=" * 65)
        print("  FaceUnlockAI — Secure Unlock (Identity + Liveness)")
        print("=" * 65)
        print(f"Target Owner:       {self._owner_name}")
        print("Required Checks:    1. Enrolled Identity Match")
        print("                    2. Active Randomized Liveness Challenge")
        print("Anti-Spoof Defense: Timeouts, micro-motion jitter, anti-photo swap.")
        print("Controls:           Press 'Q' or ESC to cancel.")
        print("=" * 65 + "\n")

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[verifier] ERROR: Cannot open webcam index {camera_index}.", file=sys.stderr)
            return False

        window_name = "FaceUnlockAI — Secure Unlock with Liveness"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        unlocked = False
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue

                frame = cv2.flip(frame, 1)
                h_frame, w_frame = frame.shape[:2]
                detector = self._ensure_detector(w_frame, h_frame)

                retval, faces = detector.detect(frame)
                num_faces = len(faces) if (faces is not None and retval == 1) else 0

                id_result = self.verify_frame(frame)

                # Only advance liveness state if the person in view matches the enrolled owner
                if id_result.status == VerificationStatus.VERIFIED_OWNER:
                    live_decision = session.evaluate_frame(faces, (w_frame, h_frame))
                elif num_faces > 1:
                    live_decision = session.evaluate_frame(faces, (w_frame, h_frame))
                else:
                    # Non-owner or unknown in frame: evaluate with dummy/empty face to maintain timeout
                    live_decision = session.evaluate_frame(None, (w_frame, h_frame))

                display = frame.copy()

                # Choose theme color & status banner
                if live_decision.status == LivenessStatus.PASSED and id_result.status == VerificationStatus.VERIFIED_OWNER:
                    theme_color = (0, 220, 0)
                    banner_title = "ACCESS GRANTED — OWNER & LIVENESS VERIFIED"
                    sub_text = f"Welcome, {self._owner_name}! Unlock confirmed."
                elif id_result.status == VerificationStatus.REJECTED_UNKNOWN:
                    theme_color = (0, 0, 240)
                    banner_title = "ACCESS BLOCKED — UNKNOWN PERSON"
                    sub_text = "Identity mismatch. Liveness challenges suspended."
                elif live_decision.status == LivenessStatus.IN_PROGRESS:
                    theme_color = (0, 200, 255)
                    banner_title = f"VERIFYING: {live_decision.current_challenge.prompt if live_decision.current_challenge else 'Look at camera'}"
                    sub_text = f"Owner Match: {id_result.similarity:.2f} | Time: {live_decision.time_remaining:.1f}s"
                else:
                    theme_color = (0, 0, 240)
                    banner_title = f"VERIFICATION FAILED: {live_decision.status.value}"
                    sub_text = live_decision.message

                # Draw bounding box & landmarks
                if num_faces == 1:
                    x, y, w, h = [int(v) for v in faces[0][:4]]
                    cv2.rectangle(display, (x, y), (x + w, y + h), theme_color, 2)
                    landmarks = faces[0][4:14].reshape((5, 2)).astype(int)
                    for lx, ly in landmarks:
                        cv2.circle(display, (lx, ly), 3, (0, 255, 255), -1)

                # Top HUD Banner
                cv2.rectangle(display, (0, 0), (w_frame, 80), (15, 15, 15), -1)
                cv2.putText(display, banner_title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, theme_color, 2)
                cv2.putText(display, sub_text, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

                # Bottom status bar
                cv2.rectangle(display, (0, h_frame - 40), (w_frame, h_frame), (15, 15, 15), -1)
                cv2.putText(display, "[Q / ESC] Cancel Unlock", (20, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\n[verifier] Unlock session cancelled by user.")
                    session.consume()
                    break

                # Joint success check
                if live_decision.status == LivenessStatus.PASSED and id_result.status == VerificationStatus.VERIFIED_OWNER:
                    unlocked = True
                    pipeline.rate_limiter.record_success()
                    print("\n" + "=" * 65)
                    print("  [ACCESS GRANTED] Identity Verified + Liveness Confirmed!")
                    print(f"  Owner:       {self._owner_name}")
                    print(f"  Similarity:  {id_result.similarity:.3f}")
                    print("=" * 65 + "\n")
                    cv2.waitKey(2000)
                    break

                # Liveness failure terminal state
                if live_decision.status not in (LivenessStatus.IN_PROGRESS,):
                    is_blocked, remaining = pipeline.rate_limiter.record_failure()
                    print(f"\n[verifier] UNLOCK FAILED: {live_decision.status.value} ({live_decision.message})")
                    if is_blocked:
                        print(f"[verifier] SECURITY COOLDOWN: Blocked for {remaining:.0f} seconds.")
                    cv2.waitKey(2000)
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return unlocked


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceUnlockAI — Identity Verification & Secure Unlock")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--threshold", type=float, default=COSINE_THRESHOLD, help="Cosine threshold (default: 0.380)")
    parser.add_argument("--ambiguity-margin", type=float, default=AMBIGUITY_MARGIN, help="Ambiguity band (default: 0.035)")
    parser.add_argument("--liveness", action="store_true", help="Enforce full identity + liveness challenge unlock")

    args = parser.parse_args()
    verifier = FaceVerifier(threshold=args.threshold, ambiguity_margin=args.ambiguity_margin)

    if args.liveness:
        verifier.run_secure_unlock_session(camera_index=args.camera)
    else:
        verifier.run_live_verification(camera_index=args.camera)


if __name__ == "__main__":
    main()

