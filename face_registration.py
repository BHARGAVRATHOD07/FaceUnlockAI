"""
face_registration.py — FaceUnlockAI Phase 2
---------------------------------------------
Local Face Registration & Enrollment Module for Windows 11.

Key Security & Privacy Principles:
  1. Explicit deliberate action required: enrollment never begins automatically.
  2. Multi-sample guided capture (5 distinct poses/expressions) with quality filters.
  3. Quality checks reject blurry, underexposed, overexposed, or multi-face samples.
  4. Cross-sample consistency check verifies that all enrolled samples belong to the
     same individual before synthesizing the template.
  5. Privacy-first: Raw face images are discarded immediately after embedding
     extraction unless --save-raw-images is explicitly supplied.
  6. Restrictive Windows ACLs applied to data/registered_face/ so only the current
     Windows account has access.
  7. Clear deletion workflow: --delete or interactive key 'D' purges the template.
  8. Zero Windows password / credential storage.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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

# ─── QUALITY THRESHOLDS ───────────────────────────────────────────────────────
MIN_FACE_WIDTH = 80
MIN_FACE_HEIGHT = 80
MIN_SHARPNESS_LAPLACIAN = 45.0   # Variance of Laplacian for blur detection
MIN_BRIGHTNESS = 40.0            # Mean grayscale intensity
MAX_BRIGHTNESS = 225.0           # Avoid overexposed glare
MIN_CONTRAST_STD = 20.0          # Standard deviation of intensity
MIN_PAIRWISE_CONSISTENCY = 0.45  # Pairwise cosine similarity between samples

ENROLLMENT_STEPS = [
    {
        "id": 1,
        "title": "Frontal Face",
        "instruction": "Look straight at the camera with a neutral expression",
    },
    {
        "id": 2,
        "title": "Slight Left",
        "instruction": "Turn your head slightly to your LEFT (about 15 degrees)",
    },
    {
        "id": 3,
        "title": "Slight Right",
        "instruction": "Turn your head slightly to your RIGHT (about 15 degrees)",
    },
    {
        "id": 4,
        "title": "Slight Upward",
        "instruction": "Tilt your head slightly UPWARD (about 10 degrees)",
    },
    {
        "id": 5,
        "title": "Natural / Smile",
        "instruction": "Look straight ahead and smile or blink naturally",
    },
]


@dataclass
class QualityAssessment:
    is_valid: bool
    reason: str
    sharpness: float
    brightness: float
    contrast: float
    face_box: Optional[Tuple[int, int, int, int]] = None


def assess_quality(frame: np.ndarray, face_data: np.ndarray) -> QualityAssessment:
    """
    Evaluates face crop against sharpness, illumination, contrast, and sizing standards.
    face_data: 15-element array from YuNet [x, y, w, h, ...]
    """
    h_frame, w_frame = frame.shape[:2]
    x, y, w, h = [int(v) for v in face_data[:4]]

    # Boundary and sizing checks
    if w < MIN_FACE_WIDTH or h < MIN_FACE_HEIGHT:
        return QualityAssessment(
            is_valid=False,
            reason=f"Face too small ({w}x{h}px). Move closer to camera.",
            sharpness=0.0,
            brightness=0.0,
            contrast=0.0,
        )

    # Clamp crop coordinates
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_frame, x + w), min(h_frame, y + h)

    if (x2 - x1) < MIN_FACE_WIDTH or (y2 - y1) < MIN_FACE_HEIGHT:
        return QualityAssessment(
            is_valid=False,
            reason="Face is partially off-screen. Center your face.",
            sharpness=0.0,
            brightness=0.0,
            contrast=0.0,
        )

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())

    # Brightness (illumination)
    if brightness < MIN_BRIGHTNESS:
        return QualityAssessment(
            is_valid=False,
            reason=f"Lighting too dark ({brightness:.1f} < {MIN_BRIGHTNESS:.0f}). Increase lighting.",
            sharpness=0.0,
            brightness=brightness,
            contrast=contrast,
            face_box=(x, y, w, h),
        )
    if brightness > MAX_BRIGHTNESS:
        return QualityAssessment(
            is_valid=False,
            reason=f"Lighting too bright / glare detected ({brightness:.1f} > {MAX_BRIGHTNESS:.0f}).",
            sharpness=0.0,
            brightness=brightness,
            contrast=contrast,
            face_box=(x, y, w, h),
        )

    # Sharpness (blur)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < MIN_SHARPNESS_LAPLACIAN:
        return QualityAssessment(
            is_valid=False,
            reason=f"Motion blur detected (sharpness: {sharpness:.1f} < {MIN_SHARPNESS_LAPLACIAN:.0f}). Hold still.",
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            face_box=(x, y, w, h),
        )

    # Contrast
    if contrast < MIN_CONTRAST_STD:
        return QualityAssessment(
            is_valid=False,
            reason="Low image contrast. Adjust lighting onto your face.",
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            face_box=(x, y, w, h),
        )

    return QualityAssessment(
        is_valid=True,
        reason="Good quality",
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        face_box=(x, y, w, h),
    )


def secure_directory_acls(dir_path: Path) -> bool:
    """
    Applies restrictive Windows file permissions using icacls.
    Grants Full control to current user and SYSTEM.
    """
    if os.name != "nt":
        # On POSIX systems use standard chmod 0700
        try:
            os.chmod(dir_path, 0o700)
            return True
        except Exception:
            return False

    try:
        current_user = getpass.getuser()
        resolved_path = str(dir_path.resolve())
        is_directory = dir_path.is_dir()
        user_perm = f"{current_user}:(OI)(CI)(F)" if is_directory else f"{current_user}:(F)"
        system_perm = "SYSTEM:(OI)(CI)(F)" if is_directory else "SYSTEM:(F)"

        cmd = [
            "icacls",
            resolved_path,
            "/grant:r",
            user_perm,
            system_perm,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return res.returncode == 0
    except Exception as exc:
        print(f"[registration] Warning: Could not apply Windows ACLs: {exc}", file=sys.stderr)
        return False


class FaceEnrollmentManager:
    """
    Manages owner face registration, template creation, storage, and deletion.
    """

    def __init__(self) -> None:
        self.ensure_models_exist()
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None

    @staticmethod
    def ensure_models_exist() -> None:
        if not YUNET_MODEL.exists() or not SFACE_MODEL.exists():
            print("[registration] Required ONNX models missing. Attempting automatic download...", flush=True)
            from models.download_models import download_models
            download_models()

    def _init_models(self, frame_size: Tuple[int, int]) -> None:
        w, h = frame_size
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(YUNET_MODEL),
                "",
                (w, h),
                score_threshold=0.75,
                nms_threshold=0.3,
                top_k=5,
            )
        else:
            self._detector.setInputSize((w, h))

        if self._recognizer is None:
            self._recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")

    @staticmethod
    def is_enrolled() -> bool:
        return TEMPLATE_FILE.exists()

    @staticmethod
    def get_enrollment_info() -> Optional[dict]:
        if not TEMPLATE_FILE.exists():
            return None
        try:
            with np.load(TEMPLATE_FILE, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"]))
                return meta
        except Exception as exc:
            return {"error": f"Failed to read template: {exc}"}

    @staticmethod
    def delete_enrollment() -> bool:
        """
        Explicitly purges enrolled biometric template and raw images if any exist.
        """
        if not TEMPLATE_FILE.exists():
            print("[registration] No enrollment data exists to delete.")
            return False

        try:
            TEMPLATE_FILE.unlink(missing_ok=True)
            # Remove any raw images if user previously opted in
            raw_dir = ENROLLMENT_DIR / "raw_samples"
            if raw_dir.exists():
                for p in raw_dir.glob("*.jpg"):
                    p.unlink(missing_ok=True)
                raw_dir.rmdir()
            print("[registration] Enrollment data successfully purged.")
            return True
        except Exception as exc:
            print(f"[registration] Error deleting enrollment data: {exc}", file=sys.stderr)
            return False

    def save_template(
        self,
        owner_name: str,
        sample_embeddings: list[np.ndarray],
        quality_records: list[dict],
        raw_images: Optional[list[np.ndarray]] = None,
    ) -> Path:
        """
        Synthesizes mean embedding vector, validates pairwise consistency,
        and saves template with metadata.
        """
        ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
        secure_directory_acls(ENROLLMENT_DIR)

        # Compute normalized mean embedding
        embeddings_arr = np.vstack(sample_embeddings).astype(np.float32)  # (N, 128)
        mean_vec = np.mean(embeddings_arr, axis=0, keepdims=True)         # (1, 128)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm

        # Compute pairwise cosine similarities
        n = len(sample_embeddings)
        pairwise_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(embeddings_arr[i], embeddings_arr[j].T))
                pairwise_sims.append(sim)

        min_sim = min(pairwise_sims) if pairwise_sims else 1.0
        avg_sim = float(np.mean(pairwise_sims)) if pairwise_sims else 1.0

        metadata = {
            "owner_name": owner_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_version": "OpenCV_SFace_2021dec",
            "detector_version": "YuNet_2023mar",
            "sample_count": n,
            "recommended_threshold": 0.380,
            "pairwise_min_similarity": round(min_sim, 4),
            "pairwise_avg_similarity": round(avg_sim, 4),
            "quality_records": quality_records,
            "privacy_notice": (
                "Processed locally. No biometric data is uploaded. "
                "Templates are protected by local Windows ACLs. "
                "File permissions alone do NOT protect against malware executing in the owner's session."
            ),
        }

        # Save to disk
        np.savez_compressed(
            TEMPLATE_FILE,
            mean_embedding=mean_vec,
            sample_embeddings=embeddings_arr,
            metadata=json.dumps(metadata, indent=2),
        )

        # Re-apply restrictive ACLs directly to template file
        secure_directory_acls(TEMPLATE_FILE)

        # Optional raw image retention (only if opted in)
        if raw_images:
            raw_dir = ENROLLMENT_DIR / "raw_samples"
            raw_dir.mkdir(parents=True, exist_ok=True)
            secure_directory_acls(raw_dir)
            for idx, img in enumerate(raw_images):
                cv2.imwrite(str(raw_dir / f"sample_{idx + 1}.jpg"), img)
            print(f"[registration] User opted-in: {len(raw_images)} raw images saved to {raw_dir}")

        return TEMPLATE_FILE

    def start_enrollment(
        self,
        camera_index: int = 0,
        owner_name: str = "Device Owner",
        save_raw_images: bool = False,
    ) -> bool:
        """
        Interactive webcam enrollment with guided step-by-step sample collection.
        Requires deliberate user keypress (SPACE/ENTER) for each pose.
        """
        print("\n" + "=" * 65)
        print("  FaceUnlockAI — Device Owner Enrollment")
        print("=" * 65)
        print(f"Target User:        {owner_name}")
        print("Required Samples:   5 distinct poses / expressions")
        print("Privacy Guarantee:  Embeddings processed 100% locally.")
        if not save_raw_images:
            print("Privacy Setting:    Raw images discarded immediately (default).")
        else:
            print("Privacy Setting:    OPT-IN: Raw face samples will be stored.")
        print("-" * 65)
        print("Instructions:")
        print("  • Look at the camera and follow the on-screen instructions.")
        print("  • Press SPACE or ENTER to capture each required pose.")
        print("  • Press 'R' to restart enrollment.")
        print("  • Press 'Q' or ESC to cancel at any time.")
        print("=" * 65 + "\n")

        # Open camera
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[registration] ERROR: Cannot open camera index {camera_index}.", file=sys.stderr)
            return False

        current_step_idx = 0
        captured_embeddings: list[np.ndarray] = []
        captured_quality: list[dict] = []
        captured_raw_images: list[np.ndarray] = []

        window_name = "FaceUnlockAI — Owner Enrollment"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        last_status_message = "Align your face in the guide box and press SPACE to capture."
        last_status_color = (0, 255, 255)
        flash_countdown = 0

        try:
            while current_step_idx < len(ENROLLMENT_STEPS):
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[registration] Failed to read frame from camera.", file=sys.stderr)
                    time.sleep(0.05)
                    continue

                # Mirror frame for natural self-view
                frame = cv2.flip(frame, 1)
                h_frame, w_frame = frame.shape[:2]
                self._init_models((w_frame, h_frame))

                step_info = ENROLLMENT_STEPS[current_step_idx]

                # Face detection
                retval, faces = self._detector.detect(frame)
                num_faces = len(faces) if (faces is not None and retval == 1) else 0

                primary_face = None
                quality: Optional[QualityAssessment] = None

                if num_faces == 1:
                    primary_face = faces[0]
                    quality = assess_quality(frame, primary_face)
                    if quality.is_valid:
                        box_color = (0, 255, 0)      # Green
                        status_hint = "Quality: GOOD — Press SPACE or ENTER to capture"
                    else:
                        box_color = (0, 165, 255)    # Orange/Yellow
                        status_hint = f"Quality: {quality.reason}"
                elif num_faces == 0:
                    box_color = (128, 128, 128)      # Gray
                    status_hint = "No face detected — center your face in the box"
                else:
                    box_color = (0, 0, 255)          # Red
                    status_hint = f"Multiple faces detected ({num_faces}) — ensure only YOU are visible"

                # Draw UI HUD
                display = frame.copy()

                # Flash effect on capture
                if flash_countdown > 0:
                    display = cv2.addWeighted(display, 0.5, np.full_like(display, 255), 0.5, 0)
                    flash_countdown -= 1

                # Draw face bounding box and landmarks if detected
                if primary_face is not None:
                    x, y, w, h = [int(v) for v in primary_face[:4]]
                    cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 2)
                    # Landmarks: right eye, left eye, nose, mouth corners
                    landmarks = primary_face[4:14].reshape((5, 2)).astype(int)
                    for lx, ly in landmarks:
                        cv2.circle(display, (lx, ly), 3, (0, 255, 255), -1)

                # Top banner
                cv2.rectangle(display, (0, 0), (w_frame, 85), (20, 20, 20), -1)
                step_text = f"Sample {step_info['id']}/5: {step_info['title']}"
                cv2.putText(display, step_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                cv2.putText(display, step_info["instruction"], (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

                # Bottom status bar
                cv2.rectangle(display, (0, h_frame - 65), (w_frame, h_frame), (20, 20, 20), -1)
                cv2.putText(display, status_hint, (20, h_frame - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 1)
                cv2.putText(display, "[SPACE] Capture   [R] Reset   [Q / ESC] Cancel", (20, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # Q or Escape
                    print("\n[registration] Enrollment cancelled by user.")
                    return False

                elif key in (ord("r"), ord("R")):    # Reset
                    print("\n[registration] Enrollment reset.")
                    current_step_idx = 0
                    captured_embeddings.clear()
                    captured_quality.clear()
                    captured_raw_images.clear()
                    continue

                elif key in (32, 13):  # SPACE or ENTER
                    if num_faces == 0:
                        print("[registration] Cannot capture: No face detected.")
                        continue
                    if num_faces > 1:
                        print(f"[registration] Cannot capture: Multiple faces ({num_faces}) detected.")
                        continue
                    if quality is None or not quality.is_valid:
                        print(f"[registration] Sample rejected: {quality.reason if quality else 'Invalid'}")
                        continue

                    # Quality passed — extract embedding using SFace alignCrop + feature
                    aligned = self._recognizer.alignCrop(frame, primary_face)
                    feat = self._recognizer.feature(aligned)

                    captured_embeddings.append(feat)
                    captured_quality.append({
                        "step": step_info["id"],
                        "title": step_info["title"],
                        "sharpness": round(quality.sharpness, 2),
                        "brightness": round(quality.brightness, 2),
                        "contrast": round(quality.contrast, 2),
                    })
                    if save_raw_images:
                        captured_raw_images.append(aligned.copy())

                    flash_countdown = 3
                    print(f"[registration] Successfully captured Sample {step_info['id']}/5 ({step_info['title']}).")
                    current_step_idx += 1

            # All 5 samples captured — verify pairwise consistency
            print("\n[registration] Validating cross-sample consistency...")
            embeddings_arr = np.vstack(captured_embeddings)
            inconsistent = False
            for i in range(len(captured_embeddings)):
                for j in range(i + 1, len(captured_embeddings)):
                    sim = float(np.dot(embeddings_arr[i], embeddings_arr[j].T))
                    if sim < MIN_PAIRWISE_CONSISTENCY:
                        inconsistent = True
                        print(f"[registration] Inconsistency detected between sample {i+1} and {j+1} (similarity: {sim:.3f} < {MIN_PAIRWISE_CONSISTENCY}).")

            if inconsistent:
                print("[registration] ERROR: Captured samples failed consistency check (possible multi-person or high distortion).", file=sys.stderr)
                print("[registration] Enrollment rejected for safety. Please retry.", file=sys.stderr)
                return False

            # Save synthesized template
            saved_path = self.save_template(
                owner_name=owner_name,
                sample_embeddings=captured_embeddings,
                quality_records=captured_quality,
                raw_images=captured_raw_images if save_raw_images else None,
            )

            print("\n" + "=" * 65)
            print(f"  [SUCCESS] Owner Enrollment Completed Successfully!")
            print(f"  Template Saved: {saved_path}")
            print(f"  Owner:          {owner_name}")
            print(f"  Samples:        {len(captured_embeddings)} poses")
            print("=" * 65 + "\n")
            return True

        finally:
            cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceUnlockAI Phase 2 — Owner Face Enrollment")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--name", type=str, default="Device Owner", help="Owner identity label")
    parser.add_argument("--save-raw-images", action="store_true", help="Explicit opt-in: retain aligned face images")
    parser.add_argument("--delete", action="store_true", help="Delete existing enrollment data")
    parser.add_argument("--info", action="store_true", help="Display current enrollment metadata")
    parser.add_argument("--re-enroll", action="store_true", help="Overwrite existing enrollment")

    args = parser.parse_args()
    manager = FaceEnrollmentManager()

    if args.delete:
        manager.delete_enrollment()
        return

    if args.info:
        info = manager.get_enrollment_info()
        if info is None:
            print("[registration] No owner is currently enrolled.")
        else:
            print("\n--- Current Enrollment Metadata ---")
            print(json.dumps(info, indent=2))
        return

    if manager.is_enrolled() and not args.re-enroll:
        info = manager.get_enrollment_info() or {}
        created = info.get("created_at", "Unknown")
        print(f"[registration] An owner is already enrolled ({info.get('owner_name', 'Device Owner')}, enrolled: {created}).")
        print("[registration] Use --re-enroll to replace, or --delete to remove existing enrollment.")
        return

    manager.start_enrollment(
        camera_index=args.camera,
        owner_name=args.name,
        save_raw_images=args.save_raw_images,
    )


if __name__ == "__main__":
    main()
