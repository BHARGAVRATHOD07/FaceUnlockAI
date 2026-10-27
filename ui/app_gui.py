"""
app_gui.py — FaceUnlockAI Windows 11 Desktop Authentication Prototype
-----------------------------------------------------------------------
Maintainable, modern desktop application built with CustomTkinter.

Key Architecture:
  1. Complete separation of UI, Configuration, Biometric Engine, and Security Logs.
  2. Multi-view navigation: Dashboard, Enrollment, Secure Unlock, Audit Logs, Settings.
  3. Strict Camera Lifecycle: Explicit start/stop, persistent on-screen status badge,
     and guaranteed resource release on tab switch or window close.
  4. Joint Defense-in-Depth Verification: SFace identity + active liveness challenges.
  5. Privacy-preserving: zero raw image retention by default, zero Windows password/PIN
     storage, restrictive Windows ACLs.
  6. Permanent Prototype Warning: Clearly indicates this is a research prototype, not
     a certified replacement for Windows Hello.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

from config.config_manager import AppSettings, load_settings, save_settings
from face_registration import (
    FaceEnrollmentManager,
    ENROLLMENT_STEPS,
    TEMPLATE_FILE,
    assess_quality,
)
from face_recognition import (
    FaceVerifier,
    VerificationResult,
    VerificationStatus,
    COSINE_THRESHOLD,
    AMBIGUITY_MARGIN,
)
from liveness_detection import (
    LivenessChallenge,
    LivenessChallengeSession,
    LivenessDecision,
    LivenessPipeline,
    LivenessStatus,
    SECURITY_LOG_FILE,
    SecurityEventLogger,
)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class FaceUnlockApp(ctk.CTk):
    """
    Main application window implementing Windows 11 styled interface.
    """

    def __init__(self) -> None:
        super().__init__()

        # Load persisted settings
        self.settings = load_settings()
        ctk.set_appearance_mode(self.settings.ui_theme)
        ctk.set_default_color_theme("blue")

        self.title("FaceUnlockAI — Windows 11 Biometric Prototype")
        self.geometry("1100x720")
        self.minsize(980, 640)

        # Biometric & Hardware Engines
        self.enrollment_mgr = FaceEnrollmentManager()
        self.verifier = FaceVerifier(
            threshold=self.settings.cosine_threshold,
            ambiguity_margin=self.settings.ambiguity_margin,
        )
        self.liveness_pipeline = LivenessPipeline(
            num_challenges=self.settings.num_liveness_challenges,
            max_failures=self.settings.max_failed_attempts,
            cooldown_sec=self.settings.cooldown_duration_sec,
        )
        self.security_logger = SecurityEventLogger()

        # Camera state management
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_active = False
        self.active_camera_view: Optional[str] = None
        self._cam_lock = threading.Lock()

        # Build UI layout
        self._build_header()
        self._build_body()

        # Handle clean exit
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Show initial Dashboard
        self.show_view("dashboard")

    # ─── HEADER & NOTIFICATIONS ───────────────────────────────────────────────

    def _build_header(self) -> None:
        header_frame = ctk.CTkFrame(self, height=50, corner_radius=0, fg_color=("gray90", "gray14"))
        header_frame.pack(side="top", fill="x")

        # Prototype Warning Badge
        warning_badge = ctk.CTkLabel(
            header_frame,
            text="⚠ PROTOTYPE ONLY — NOT A REPLACEMENT FOR WINDOWS HELLO",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("#b25e00", "#ffb84d"),
            fg_color=("#ffe8cc", "#3a2500"),
            corner_radius=6,
            padx=10,
            pady=3,
        )
        warning_badge.pack(side="left", padx=15, pady=10)

        # Camera Active Indicator Pill
        self.camera_indicator_pill = ctk.CTkLabel(
            header_frame,
            text="○ Camera Offline",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray50", "gray60"),
            fg_color=("gray82", "gray22"),
            corner_radius=12,
            padx=12,
            pady=4,
        )
        self.camera_indicator_pill.pack(side="right", padx=15, pady=10)

    def _update_camera_indicator(self, active: bool, dev_idx: int = 0) -> None:
        if active:
            self.camera_indicator_pill.configure(
                text=f"● Camera Active (Dev {dev_idx})",
                text_color=("#008a00", "#47d147"),
                fg_color=("#d4fcd4", "#0a330a"),
            )
        else:
            self.camera_indicator_pill.configure(
                text="○ Camera Offline",
                text_color=("gray50", "gray60"),
                fg_color=("gray82", "gray22"),
            )

    # ─── BODY & NAVIGATION ────────────────────────────────────────────────────

    def _build_body(self) -> None:
        body_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        body_frame.pack(side="top", fill="both", expand=True)

        # Left Sidebar Navigation
        sidebar = ctk.CTkFrame(body_frame, width=220, corner_radius=0, fg_color=("gray95", "gray12"))
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Brand / App Title
        brand_label = ctk.CTkLabel(
            sidebar,
            text="FaceUnlock AI",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
        )
        brand_label.pack(anchor="w", padx=20, pady=(20, 5))

        sub_brand = ctk.CTkLabel(
            sidebar,
            text="Local Biometric Engine",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray60"),
        )
        sub_brand.pack(anchor="w", padx=20, pady=(0, 25))

        # Nav Buttons
        self.nav_buttons = {}
        nav_items = [
            ("dashboard", "🏠  Dashboard"),
            ("enrollment", "👤  Enrollment"),
            ("unlock", "🔓  Secure Unlock"),
            ("logs", "📋  Audit Logs"),
            ("settings", "⚙  Settings"),
        ]

        for view_key, label_text in nav_items:
            btn = ctk.CTkButton(
                sidebar,
                text=label_text,
                anchor="w",
                font=ctk.CTkFont(size=14, weight="normal"),
                height=40,
                corner_radius=8,
                fg_color="transparent",
                text_color=("gray20", "gray85"),
                hover_color=("gray85", "gray20"),
                command=lambda k=view_key: self.show_view(k),
            )
            btn.pack(fill="x", padx=12, pady=4)
            self.nav_buttons[view_key] = btn

        # Bottom System Info Badge
        enroll_status_text = "ENROLLED" if self.enrollment_mgr.is_enrolled() else "UNENROLLED"
        self.sidebar_status_badge = ctk.CTkLabel(
            sidebar,
            text=f"Owner: {enroll_status_text}\nOpenCV 4.11 | SFace\nYuNet 2023mar",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray55"),
            justify="left",
        )
        self.sidebar_status_badge.pack(side="bottom", anchor="w", padx=20, pady=20)

        # Right Main Container
        self.container = ctk.CTkFrame(body_frame, corner_radius=0, fg_color="transparent")
        self.container.pack(side="right", fill="both", expand=True, padx=20, pady=20)

    def show_view(self, view_name: str) -> None:
        """Switches active view and safely releases any camera hardware."""
        self.release_camera()

        for k, btn in self.nav_buttons.items():
            if k == view_name:
                btn.configure(fg_color=("gray80", "gray24"), text_color=("black", "white"))
            else:
                btn.configure(fg_color="transparent", text_color=("gray20", "gray85"))

        # Clear container
        for widget in self.container.winfo_children():
            widget.destroy()

        if view_name == "dashboard":
            self._render_dashboard()
        elif view_name == "enrollment":
            self._render_enrollment()
        elif view_name == "unlock":
            self._render_unlock()
        elif view_name == "logs":
            self._render_logs()
        elif view_name == "settings":
            self._render_settings()

    # ─── CAMERA HARDWARE MANAGEMENT ───────────────────────────────────────────

    def start_camera(self, target_view: str) -> bool:
        """Explicitly opens the webcam with mutex protection."""
        with self._cam_lock:
            if self.is_camera_active and self.cap is not None and self.cap.isOpened():
                return True

            cam_idx = self.settings.camera_index
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(cam_idx, backend)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_idx)

            if not cap.isOpened():
                print(f"[app_gui] Error: Unable to open camera {cam_idx}", file=sys.stderr)
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_height)

            self.cap = cap
            self.is_camera_active = True
            self.active_camera_view = target_view
            self._update_camera_indicator(True, cam_idx)
            return True

    def release_camera(self) -> None:
        """Guarantees complete release of camera hardware."""
        with self._cam_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.is_camera_active = False
            self.active_camera_view = None
            self._update_camera_indicator(False)

    def on_closing(self) -> None:
        """Window close hook releasing all resources."""
        self.release_camera()
        self.destroy()

    # ─── 1. DASHBOARD VIEW ────────────────────────────────────────────────────

    def _render_dashboard(self) -> None:
        is_enrolled = self.enrollment_mgr.is_enrolled()
        info = self.enrollment_mgr.get_enrollment_info() or {}

        title = ctk.CTkLabel(
            self.container,
            text="System Overview & Security Status",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.pack(anchor="w", pady=(0, 15))

        # 3 Card Summary Row
        card_row = ctk.CTkFrame(self.container, fg_color="transparent")
        card_row.pack(fill="x", pady=(0, 20))

        # Card 1: Owner Status
        c1 = ctk.CTkFrame(card_row, corner_radius=10)
        c1.pack(side="left", fill="both", expand=True, padx=(0, 10))
        ctk.CTkLabel(c1, text="OWNER STATUS", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray50").pack(anchor="w", padx=15, pady=(15, 5))
        owner_status_text = "Enrolled" if is_enrolled else "Not Enrolled"
        owner_status_color = ("#008a00", "#47d147") if is_enrolled else ("#c94a00", "#ff6600")
        ctk.CTkLabel(c1, text=owner_status_text, font=ctk.CTkFont(size=20, weight="bold"), text_color=owner_status_color).pack(anchor="w", padx=15, pady=(0, 5))
        owner_name = info.get("owner_name", "None") if is_enrolled else "No template registered"
        ctk.CTkLabel(c1, text=f"Identity: {owner_name}", font=ctk.CTkFont(size=12), text_color="gray60").pack(anchor="w", padx=15, pady=(0, 15))

        # Card 2: Liveness & Defense
        c2 = ctk.CTkFrame(card_row, corner_radius=10)
        c2.pack(side="left", fill="both", expand=True, padx=5)
        ctk.CTkLabel(c2, text="PRESENTATION ATTACK DEFENSE", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray50").pack(anchor="w", padx=15, pady=(15, 5))
        ctk.CTkLabel(c2, text="Active", font=ctk.CTkFont(size=20, weight="bold"), text_color=("#008a00", "#47d147")).pack(anchor="w", padx=15, pady=(0, 5))
        ctk.CTkLabel(c2, text=f"Challenges: {self.settings.num_liveness_challenges} Randomized", font=ctk.CTkFont(size=12), text_color="gray60").pack(anchor="w", padx=15, pady=(0, 15))

        # Card 3: Security & Privacy
        c3 = ctk.CTkFrame(card_row, corner_radius=10)
        c3.pack(side="left", fill="both", expand=True, padx=(10, 0))
        ctk.CTkLabel(c3, text="DATA SECURITY", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray50").pack(anchor="w", padx=15, pady=(15, 5))
        ctk.CTkLabel(c3, text="Local ACL Protected", font=ctk.CTkFont(size=20, weight="bold"), text_color=("#0066cc", "#3399ff")).pack(anchor="w", padx=15, pady=(0, 5))
        ctk.CTkLabel(c3, text="Zero Cloud Egress | No Passwords", font=ctk.CTkFont(size=12), text_color="gray60").pack(anchor="w", padx=15, pady=(0, 15))

        # Action Buttons Section
        actions_frame = ctk.CTkFrame(self.container, corner_radius=10)
        actions_frame.pack(fill="x", pady=10, padx=0)
        ctk.CTkLabel(actions_frame, text="Quick Operations", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=20, pady=(15, 10))

        btn_box = ctk.CTkFrame(actions_frame, fg_color="transparent")
        btn_box.pack(fill="x", padx=20, pady=(0, 15))

        ctk.CTkButton(
            btn_box,
            text="🔓 Launch Secure Unlock",
            height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda: self.show_view("unlock"),
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_box,
            text="👤 Owner Enrollment",
            height=42,
            fg_color=("gray75", "gray28"),
            hover_color=("gray65", "gray35"),
            text_color=("black", "white"),
            font=ctk.CTkFont(size=13),
            command=lambda: self.show_view("enrollment"),
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            btn_box,
            text="📋 View Audit Logs",
            height=42,
            fg_color=("gray75", "gray28"),
            hover_color=("gray65", "gray35"),
            text_color=("black", "white"),
            font=ctk.CTkFont(size=13),
            command=lambda: self.show_view("logs"),
        ).pack(side="left", padx=10)

        # Prototype Architecture Box
        info_box = ctk.CTkTextbox(self.container, height=220, font=ctk.CTkFont(family="Consolas", size=12))
        info_box.pack(fill="both", expand=True, pady=(15, 0))
        info_box.insert(
            "1.0",
            "=== FaceUnlockAI Architecture Overview ===\n\n"
            "• Face Detection:     OpenCV YuNet ONNX (face_detection_yunet_2023mar.onnx)\n"
            "• Landmark Warping:   5-Point Affine Landmark Alignment (Eye, Nose, Mouth)\n"
            "• Feature Extraction: OpenCV SFace MobileFaceNet (face_recognition_sface_2021dec.onnx)\n"
            "• Feature Dimension:  128-D L2-Normalized float32 vector\n"
            f"• Decision Threshold: Cosine Similarity >= {self.settings.cosine_threshold:.3f}\n"
            f"• Ambiguity Band:     [{self.settings.cosine_threshold - self.settings.ambiguity_margin:.3f}, {self.settings.cosine_threshold + self.settings.ambiguity_margin:.3f}]\n"
            "• Liveness Checking:  Dynamic head pose challenge-response (Yaw Left/Right, Pitch)\n"
            "• Anti-Spoofing:      Micro-motion variance gating + anti-swap position tracking\n"
            "• Rate Limiting:      3 failed attempts -> 30-second security cooldown\n"
            "• Storage Security:   Local Windows ACLs (icacls) | Zero password/PIN storage\n"
            "• Biometric Privacy:  Raw camera frames discarded immediately after embedding\n\n"
            "DISCLAIMER: This experimental prototype is designed for local evaluation.\n"
            "It does NOT replace hardware-level Windows Hello infrared/depth authentication."
        )
        info_box.configure(state="disabled")

    # ─── 2. ENROLLMENT VIEW ───────────────────────────────────────────────────

    def _render_enrollment(self) -> None:
        self.enroll_step_idx = 0
        self.enroll_captured_embeddings = []
        self.enroll_captured_quality = []

        top_bar = ctk.CTkFrame(self.container, fg_color="transparent")
        top_bar.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            top_bar,
            text="Device Owner Enrollment",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(side="left")

        # Video feed & Controls Layout
        content_box = ctk.CTkFrame(self.container, fg_color="transparent")
        content_box.pack(fill="both", expand=True)

        # Video Feed Canvas / Label (Left, 480x360)
        video_frame = ctk.CTkFrame(content_box, width=480, height=360, corner_radius=10)
        video_frame.pack(side="left", padx=(0, 15), pady=5)
        video_frame.pack_propagate(False)

        self.enroll_video_label = ctk.CTkLabel(video_frame, text="Camera Offline\nClick 'Start Camera' below")
        self.enroll_video_label.pack(fill="both", expand=True)

        # Right Control Panel
        panel = ctk.CTkFrame(content_box, corner_radius=10)
        panel.pack(side="right", fill="both", expand=True, pady=5)

        self.enroll_step_label = ctk.CTkLabel(
            panel,
            text="Step 1/5: Frontal Face",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#0066cc", "#3399ff"),
        )
        self.enroll_step_label.pack(anchor="w", padx=15, pady=(15, 2))

        self.enroll_instruction_label = ctk.CTkLabel(
            panel,
            text=ENROLLMENT_STEPS[0]["instruction"],
            font=ctk.CTkFont(size=12),
            text_color="gray60",
            wraplength=340,
            justify="left",
        )
        self.enroll_instruction_label.pack(anchor="w", padx=15, pady=(0, 10))

        # Progress Bar
        self.enroll_progress = ctk.CTkProgressBar(panel)
        self.enroll_progress.pack(fill="x", padx=15, pady=(0, 15))
        self.enroll_progress.set(0.0)

        # Quality Feedback Status Card
        self.enroll_quality_card = ctk.CTkFrame(panel, corner_radius=8, fg_color=("gray85", "gray18"))
        self.enroll_quality_card.pack(fill="x", padx=15, pady=(0, 15))

        self.enroll_quality_text = ctk.CTkLabel(
            self.enroll_quality_card,
            text="Quality Status: Waiting for Camera...",
            font=ctk.CTkFont(size=12),
            justify="left",
        )
        self.enroll_quality_text.pack(anchor="w", padx=12, pady=10)

        # Buttons
        self.enroll_cam_btn = ctk.CTkButton(
            panel,
            text="▶  Start Camera",
            height=38,
            command=self._toggle_enrollment_camera,
        )
        self.enroll_cam_btn.pack(fill="x", padx=15, pady=(0, 8))

        self.enroll_capture_btn = ctk.CTkButton(
            panel,
            text="📸  Capture Pose",
            height=40,
            font=ctk.CTkFont(weight="bold"),
            state="disabled",
            command=self._trigger_enrollment_capture,
        )
        self.enroll_capture_btn.pack(fill="x", padx=15, pady=(0, 8))

        self.enroll_reset_btn = ctk.CTkButton(
            panel,
            text="↺  Reset Enrollment",
            height=34,
            fg_color=("gray75", "gray28"),
            hover_color=("gray65", "gray35"),
            text_color=("black", "white"),
            command=self._reset_enrollment_state,
        )
        self.enroll_reset_btn.pack(fill="x", padx=15, pady=(0, 10))

    def _toggle_enrollment_camera(self) -> None:
        if not self.is_camera_active:
            if self.start_camera("enrollment"):
                self.enroll_cam_btn.configure(text="⏹  Stop Camera", fg_color=("#c94a00", "#ff6600"))
                self.enroll_capture_btn.configure(state="normal")
                self._run_enrollment_loop()
        else:
            self.release_camera()
            self.enroll_cam_btn.configure(text="▶  Start Camera", fg_color=("#1f6aa5", "#2980b9"))
            self.enroll_capture_btn.configure(state="disabled")
            self.enroll_video_label.configure(text="Camera Stopped", image="")

    def _reset_enrollment_state(self) -> None:
        self.enroll_step_idx = 0
        self.enroll_captured_embeddings.clear()
        self.enroll_captured_quality.clear()
        self.enroll_progress.set(0.0)
        self.enroll_step_label.configure(text="Step 1/5: Frontal Face")
        self.enroll_instruction_label.configure(text=ENROLLMENT_STEPS[0]["instruction"])
        self.enroll_quality_text.configure(text="Quality Status: Reset. Align face to begin.")

    def _run_enrollment_loop(self) -> None:
        if not self.is_camera_active or self.active_camera_view != "enrollment":
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            detector = self.verifier._ensure_detector(w, h)
            retval, faces = detector.detect(frame)
            num_faces = len(faces) if (faces is not None and retval == 1) else 0

            display = frame.copy()
            quality_status = "Waiting..."
            box_color = (128, 128, 128)

            if num_faces == 1:
                face = faces[0]
                q = assess_quality(frame, face)
                if q.is_valid:
                    box_color = (0, 255, 0)
                    quality_status = f"✓ Quality GOOD (Sharpness: {q.sharpness:.0f}, Lum: {q.brightness:.0f})"
                else:
                    box_color = (0, 165, 255)
                    quality_status = f"⚠ {q.reason}"

                x, y, fw, fh = [int(v) for v in face[:4]]
                cv2.rectangle(display, (x, y), (x + fw, y + fh), box_color, 2)
            elif num_faces > 1:
                quality_status = f"✕ Security Lockout: Multiple faces ({num_faces}) detected."
            else:
                quality_status = "Searching for face... Center in frame."

            self.enroll_quality_text.configure(text=f"Quality Status:\n{quality_status}")

            # Convert to CTk image
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((480, 360), Image.Resampling.LANCZOS)
            ctk_img = ImageTk.PhotoImage(img)
            self.enroll_video_label.configure(image=ctk_img, text="")
            self.enroll_video_label.image = ctk_img

        self.after(25, self._run_enrollment_loop)

    def _trigger_enrollment_capture(self) -> None:
        if not self.is_camera_active or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        detector = self.verifier._ensure_detector(w, h)
        retval, faces = detector.detect(frame)

        if faces is None or len(faces) != 1:
            self.enroll_quality_text.configure(text="Capture Failed: Ensure exactly 1 face is visible.")
            return

        face = faces[0]
        q = assess_quality(frame, face)
        if not q.is_valid:
            self.enroll_quality_text.configure(text=f"Capture Rejected: {q.reason}")
            return

        # Extract SFace embedding
        aligned = self.verifier._recognizer.alignCrop(frame, face)
        feat = self.verifier._recognizer.feature(aligned)

        step_info = ENROLLMENT_STEPS[self.enroll_step_idx]
        self.enroll_captured_embeddings.append(feat)
        self.enroll_captured_quality.append({
            "step": step_info["id"],
            "title": step_info["title"],
            "sharpness": round(q.sharpness, 1),
            "brightness": round(q.brightness, 1),
        })

        self.enroll_step_idx += 1
        progress_val = self.enroll_step_idx / len(ENROLLMENT_STEPS)
        self.enroll_progress.set(progress_val)

        if self.enroll_step_idx < len(ENROLLMENT_STEPS):
            next_step = ENROLLMENT_STEPS[self.enroll_step_idx]
            self.enroll_step_label.configure(text=f"Step {self.enroll_step_idx + 1}/5: {next_step['title']}")
            self.enroll_instruction_label.configure(text=next_step["instruction"])
            self.enroll_quality_text.configure(text=f"✓ Sample {self.enroll_step_idx}/5 captured successfully!")
        else:
            # All 5 captured — save synthesized template
            self.enrollment_mgr.save_template(
                owner_name="Device Owner",
                sample_embeddings=self.enroll_captured_embeddings,
                quality_records=self.enroll_captured_quality,
            )
            self.verifier.reload_enrollment()
            self.release_camera()
            self.enroll_step_label.configure(text="✓ Enrollment Complete!")
            self.enroll_instruction_label.configure(text="Biometric template synthesized and stored locally under Windows ACL protection.")
            self.enroll_quality_text.configure(text="[SUCCESS] Owner enrolled. You can now use Secure Unlock.")
            self.enroll_cam_btn.configure(state="disabled")
            self.enroll_capture_btn.configure(state="disabled")

    # ─── 3. SECURE UNLOCK VIEW ────────────────────────────────────────────────

    def _render_unlock(self) -> None:
        self.unlock_session: Optional[LivenessChallengeSession] = None
        self.unlock_in_progress = False

        top_bar = ctk.CTkFrame(self.container, fg_color="transparent")
        top_bar.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            top_bar,
            text="Identity Verification & Liveness Unlock",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(side="left")

        # Layout
        content_box = ctk.CTkFrame(self.container, fg_color="transparent")
        content_box.pack(fill="both", expand=True)

        video_frame = ctk.CTkFrame(content_box, width=500, height=380, corner_radius=10)
        video_frame.pack(side="left", padx=(0, 15), pady=5)
        video_frame.pack_propagate(False)

        self.unlock_video_label = ctk.CTkLabel(video_frame, text="Verification Idle\nClick 'Start Secure Unlock' below")
        self.unlock_video_label.pack(fill="both", expand=True)

        panel = ctk.CTkFrame(content_box, corner_radius=10)
        panel.pack(side="right", fill="both", expand=True, pady=5)

        # Status Banner
        self.unlock_banner = ctk.CTkLabel(
            panel,
            text="LOCKED",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#c94a00", "#ff6600"),
        )
        self.unlock_banner.pack(anchor="w", padx=15, pady=(15, 2))

        self.unlock_subtext = ctk.CTkLabel(
            panel,
            text="Device is locked. Click 'Start Secure Unlock' to begin.",
            font=ctk.CTkFont(size=12),
            text_color="gray60",
            wraplength=320,
            justify="left",
        )
        self.unlock_subtext.pack(anchor="w", padx=15, pady=(0, 15))

        # Identity Similarity Meter
        ctk.CTkLabel(panel, text="Cosine Similarity Match:", font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=15)
        self.unlock_sim_bar = ctk.CTkProgressBar(panel)
        self.unlock_sim_bar.pack(fill="x", padx=15, pady=(2, 5))
        self.unlock_sim_bar.set(0.0)

        self.unlock_sim_text = ctk.CTkLabel(panel, text=f"Score: 0.000 (Required: >= {self.settings.cosine_threshold:.3f})", font=ctk.CTkFont(size=11), text_color="gray60")
        self.unlock_sim_text.pack(anchor="w", padx=15, pady=(0, 15))

        # Liveness Challenge Prompt Card
        challenge_card = ctk.CTkFrame(panel, corner_radius=8, fg_color=("gray85", "gray18"))
        challenge_card.pack(fill="x", padx=15, pady=(0, 15))

        ctk.CTkLabel(challenge_card, text="LIVENESS CHALLENGE", font=ctk.CTkFont(size=10, weight="bold"), text_color="gray50").pack(anchor="w", padx=12, pady=(10, 2))
        self.unlock_challenge_prompt = ctk.CTkLabel(
            challenge_card,
            text="Awaiting activation...",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#0066cc", "#3399ff"),
            justify="left",
        )
        self.unlock_challenge_prompt.pack(anchor="w", padx=12, pady=(0, 5))

        self.unlock_timer_label = ctk.CTkLabel(challenge_card, text="Time Remaining: --", font=ctk.CTkFont(size=11), text_color="gray60")
        self.unlock_timer_label.pack(anchor="w", padx=12, pady=(0, 10))

        # Action Buttons
        self.unlock_start_btn = ctk.CTkButton(
            panel,
            text="🔓  Start Secure Unlock",
            height=42,
            font=ctk.CTkFont(weight="bold"),
            command=self._start_secure_unlock,
        )
        self.unlock_start_btn.pack(fill="x", padx=15, pady=(0, 10))

        self.unlock_stop_btn = ctk.CTkButton(
            panel,
            text="⏹  Cancel / Release Camera",
            height=34,
            fg_color=("gray75", "gray28"),
            hover_color=("gray65", "gray35"),
            text_color=("black", "white"),
            command=self._stop_secure_unlock,
        )
        self.unlock_stop_btn.pack(fill="x", padx=15, pady=(0, 10))

    def _start_secure_unlock(self) -> None:
        if not self.enrollment_mgr.is_enrolled():
            self.unlock_banner.configure(text="NOT ENROLLED", text_color=("#c94a00", "#ff6600"))
            self.unlock_subtext.configure(text="No owner enrolled. Please register an owner in Enrollment view.")
            return

        session, err = self.liveness_pipeline.start_session()
        if session is None:
            self.unlock_banner.configure(text="COOLDOWN ACTIVE", text_color=("#c94a00", "#ff6600"))
            self.unlock_subtext.configure(text=err or "Too many failed attempts. Temporary lockout active.")
            return

        if self.start_camera("unlock"):
            self.unlock_session = session
            self.unlock_in_progress = True
            self.unlock_start_btn.configure(state="disabled")
            self._run_unlock_loop()

    def _stop_secure_unlock(self) -> None:
        self.unlock_in_progress = False
        if self.unlock_session is not None:
            self.unlock_session.consume()
        self.release_camera()
        self.unlock_start_btn.configure(state="normal")
        self.unlock_video_label.configure(text="Camera Stopped", image="")
        self.unlock_banner.configure(text="LOCKED", text_color=("#c94a00", "#ff6600"))
        self.unlock_subtext.configure(text="Verification stopped.")

    def _run_unlock_loop(self) -> None:
        if not self.is_camera_active or self.active_camera_view != "unlock" or not self.unlock_in_progress:
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            detector = self.verifier._ensure_detector(w, h)
            retval, faces = detector.detect(frame)
            num_faces = len(faces) if (faces is not None and retval == 1) else 0

            # 1. Identity Check
            id_res = self.verifier.verify_frame(frame)
            self.unlock_sim_bar.set(max(0.0, min(1.0, id_res.similarity)))
            self.unlock_sim_text.configure(text=f"Score: {id_res.similarity:.3f} (Required: >= {self.settings.cosine_threshold:.3f})")

            # 2. Liveness Check
            if id_res.status == VerificationStatus.VERIFIED_OWNER:
                live_dec = self.unlock_session.evaluate_frame(faces, (w, h))
            elif num_faces > 1:
                live_dec = self.unlock_session.evaluate_frame(faces, (w, h))
            else:
                live_dec = self.unlock_session.evaluate_frame(None, (w, h))

            display = frame.copy()
            theme_color = (0, 0, 240)

            # Update challenge prompts
            if live_dec.current_challenge is not None:
                self.unlock_challenge_prompt.configure(text=live_dec.current_challenge.prompt)
                self.unlock_timer_label.configure(text=f"Time Remaining: {live_dec.time_remaining:.1f}s")
            else:
                self.unlock_challenge_prompt.configure(text=live_dec.message)

            if id_res.status == VerificationStatus.VERIFIED_OWNER and live_dec.status == LivenessStatus.PASSED:
                theme_color = (0, 220, 0)
                self.unlock_banner.configure(text="UNLOCKED — ACCESS GRANTED", text_color=("#008a00", "#47d147"))
                self.unlock_subtext.configure(text=f"Owner Verified: {id_res.owner_name} | Liveness Confirmed.")
                self.liveness_pipeline.rate_limiter.record_success()
                self._stop_secure_unlock()
                return

            elif id_res.status == VerificationStatus.REJECTED_UNKNOWN:
                self.unlock_banner.configure(text="ACCESS BLOCKED", text_color=("#c94a00", "#ff6600"))
                self.unlock_subtext.configure(text="Unknown person detected. Identity verification failed.")

            elif live_dec.status not in (LivenessStatus.IN_PROGRESS,):
                is_blocked, remaining = self.liveness_pipeline.rate_limiter.record_failure()
                self.unlock_banner.configure(text="VERIFICATION FAILED", text_color=("#c94a00", "#ff6600"))
                self.unlock_subtext.configure(text=f"{live_dec.status.value}: {live_dec.message}")
                self._stop_secure_unlock()
                return

            # Draw visual bounding box
            if num_faces == 1:
                x, y, fw, fh = [int(v) for v in faces[0][:4]]
                cv2.rectangle(display, (x, y), (x + fw, y + fh), theme_color, 2)

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((500, 380), Image.Resampling.LANCZOS)
            ctk_img = ImageTk.PhotoImage(img)
            self.unlock_video_label.configure(image=ctk_img, text="")
            self.unlock_video_label.image = ctk_img

        self.after(25, self._run_unlock_loop)

    # ─── 4. AUDIT LOGS VIEW ───────────────────────────────────────────────────

    def _render_logs(self) -> None:
        top_bar = ctk.CTkFrame(self.container, fg_color="transparent")
        top_bar.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(top_bar, text="Security Audit Logs", font=ctk.CTkFont(size=22, weight="bold")).pack(side="left")

        ctk.CTkButton(
            top_bar,
            text="↻ Refresh Logs",
            width=120,
            command=self._render_logs,
        ).pack(side="right")

        privacy_note = ctk.CTkLabel(
            self.container,
            text="Privacy Guarantee: Logs contain only metadata (event type, timestamps, session UUIDs). Never raw images or biometrics.",
            font=ctk.CTkFont(size=11),
            text_color="gray55",
        )
        privacy_note.pack(anchor="w", pady=(0, 10))

        log_box = ctk.CTkTextbox(self.container, font=ctk.CTkFont(family="Consolas", size=11))
        log_box.pack(fill="both", expand=True)

        if not SECURITY_LOG_FILE.exists():
            log_box.insert("1.0", "No security events recorded yet.")
        else:
            lines = SECURITY_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                log_box.insert("1.0", "Log file is currently empty.")
            else:
                for line in reversed(lines[-50:]):  # Show latest 50 events
                    try:
                        entry = json.loads(line)
                        ts = entry.get("timestamp", "")
                        ev = entry.get("event_type", "")
                        st = entry.get("status", "")
                        ch = entry.get("challenge", "None")
                        sid = entry.get("session_id", "")[:8]
                        log_box.insert("end", f"[{ts}] {ev:<28} Status: {st:<15} Challenge: {ch:<12} Session: {sid}...\n")
                    except Exception:
                        log_box.insert("end", f"{line}\n")

        log_box.configure(state="disabled")

    # ─── 5. SETTINGS VIEW ─────────────────────────────────────────────────────

    def _render_settings(self) -> None:
        title = ctk.CTkLabel(
            self.container,
            text="Application & Security Settings",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.pack(anchor="w", pady=(0, 15))

        settings_box = ctk.CTkScrollableFrame(self.container)
        settings_box.pack(fill="both", expand=True)

        # 1. Camera Index
        c_frame = ctk.CTkFrame(settings_box, fg_color="transparent")
        c_frame.pack(fill="x", pady=8)
        ctk.CTkLabel(c_frame, text="Camera Device Index:", font=ctk.CTkFont(weight="bold")).pack(side="left")
        self.cam_idx_entry = ctk.CTkEntry(c_frame, width=80)
        self.cam_idx_entry.insert(0, str(self.settings.camera_index))
        self.cam_idx_entry.pack(side="right")

        # 2. Cosine Threshold
        th_frame = ctk.CTkFrame(settings_box, fg_color="transparent")
        th_frame.pack(fill="x", pady=8)
        ctk.CTkLabel(th_frame, text="SFace Cosine Similarity Threshold (default: 0.380):", font=ctk.CTkFont(weight="bold")).pack(side="left")
        self.th_entry = ctk.CTkEntry(th_frame, width=80)
        self.th_entry.insert(0, f"{self.settings.cosine_threshold:.3f}")
        self.th_entry.pack(side="right")

        # 3. Liveness Challenges Count
        lc_frame = ctk.CTkFrame(settings_box, fg_color="transparent")
        lc_frame.pack(fill="x", pady=8)
        ctk.CTkLabel(lc_frame, text="Required Liveness Challenges (1-4):", font=ctk.CTkFont(weight="bold")).pack(side="left")
        self.lc_entry = ctk.CTkEntry(lc_frame, width=80)
        self.lc_entry.insert(0, str(self.settings.num_liveness_challenges))
        self.lc_entry.pack(side="right")

        # 4. Max Failures before Cooldown
        mf_frame = ctk.CTkFrame(settings_box, fg_color="transparent")
        mf_frame.pack(fill="x", pady=8)
        ctk.CTkLabel(mf_frame, text="Max Failed Attempts before Lockout:", font=ctk.CTkFont(weight="bold")).pack(side="left")
        self.mf_entry = ctk.CTkEntry(mf_frame, width=80)
        self.mf_entry.insert(0, str(self.settings.max_failed_attempts))
        self.mf_entry.pack(side="right")

        # 5. Cooldown Duration (seconds)
        cd_frame = ctk.CTkFrame(settings_box, fg_color="transparent")
        cd_frame.pack(fill="x", pady=8)
        ctk.CTkLabel(cd_frame, text="Lockout Cooldown Duration (seconds):", font=ctk.CTkFont(weight="bold")).pack(side="left")
        self.cd_entry = ctk.CTkEntry(cd_frame, width=80)
        self.cd_entry.insert(0, str(int(self.settings.cooldown_duration_sec)))
        self.cd_entry.pack(side="right")

        # 6. Save Button
        save_btn = ctk.CTkButton(
            settings_box,
            text="💾  Save Settings",
            height=38,
            font=ctk.CTkFont(weight="bold"),
            command=self._save_settings_from_ui,
        )
        save_btn.pack(fill="x", pady=(15, 25))

        # Divider
        ctk.CTkLabel(settings_box, text="— Danger Zone: Biometric Data Management —", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray50").pack(pady=(10, 10))

        # Delete Biometric Template Button
        del_btn = ctk.CTkButton(
            settings_box,
            text="🗑  Purge Enrolled Biometric Template",
            fg_color=("#c94a00", "#ff6600"),
            hover_color=("#a33c00", "#d45500"),
            height=38,
            font=ctk.CTkFont(weight="bold"),
            command=self._delete_biometric_template,
        )
        del_btn.pack(fill="x", pady=(5, 10))

    def _save_settings_from_ui(self) -> None:
        try:
            self.settings.camera_index = int(self.cam_idx_entry.get().strip())
            self.settings.cosine_threshold = float(self.th_entry.get().strip())
            self.settings.num_liveness_challenges = int(self.lc_entry.get().strip())
            self.settings.max_failed_attempts = int(self.mf_entry.get().strip())
            self.settings.cooldown_duration_sec = float(self.cd_entry.get().strip())
            self.settings.validate()
            save_settings(self.settings)

            # Update live engine thresholds
            self.verifier.threshold = self.settings.cosine_threshold
            self.liveness_pipeline.num_challenges = self.settings.num_liveness_challenges
            self.liveness_pipeline.rate_limiter.max_failures = self.settings.max_failed_attempts
            self.liveness_pipeline.rate_limiter.cooldown_sec = self.settings.cooldown_duration_sec

            ctk.CTkLabel(self.container, text="✓ Settings saved successfully!", text_color=("#008a00", "#47d147")).pack(pady=5)
        except Exception as exc:
            ctk.CTkLabel(self.container, text=f"Error saving settings: {exc}", text_color=("#c94a00", "#ff6600")).pack(pady=5)

    def _delete_biometric_template(self) -> None:
        self.enrollment_mgr.delete_enrollment()
        self.verifier.reload_enrollment()
        self.sidebar_status_badge.configure(text="Owner: UNENROLLED\nOpenCV 4.11 | SFace\nYuNet 2023mar")
        self.show_view("dashboard")


def launch_app() -> None:
    app = FaceUnlockApp()
    app.mainloop()


if __name__ == "__main__":
    launch_app()
