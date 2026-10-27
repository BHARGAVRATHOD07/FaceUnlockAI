"""
liveness_detection.py — FaceUnlockAI Phase 3
---------------------------------------------
Local Liveness & Presentation-Attack Detection (PAD) Module.

Architectural Guarantees:
  1. Independent Modularity: Separates face detection, identity verification, and
     liveness decisions into independently testable components.
  2. Challenge-Response State Machine: Issues randomized, non-predictable visual
     challenges (head yaw left/right, pitch up/down, center-hold).
  3. Timeouts & Tracking Continuity: Bounded per-challenge and session timeouts;
     fails immediately if tracking is interrupted, swapped, or lost.
  4. Anti-Replay / Freshness: Uses single-use UUID session tokens; past results
     cannot authorize new sessions.
  5. Multi-Face & Static Photo Rejection: Rejects multiple faces (shoulder-surfing)
     and static photos (zero micro-motion variance).
  6. Rate Limiting & Cooldown: Enforces configurable failure limits (e.g. 3 attempts)
     and timed lockouts (e.g. 30s) against brute-force presentation attacks.
  7. Privacy-Preserving Security Logging: Logs audit events in JSON Lines to
     logs/security_events.log with zero raw imagery or biometric vectors.
  8. Fail-Closed Policy: Any timeout, anomaly, or discrepancy results in denial.

Security Limitation Disclaimer:
  Software 2D RGB checks cannot guarantee spoof-proof security against advanced 3D
  silicone masks, high-fidelity deepfakes, or laser projections. Hardware IR depth
  sensing is required for full physical presentation-attack defense.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ─── PATHS ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
LOGS_DIR = PROJECT_ROOT / "logs"
SECURITY_LOG_FILE = LOGS_DIR / "security_events.log"
MODELS_DIR = PROJECT_ROOT / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

# ─── CONFIGURATION CONSTANTS ──────────────────────────────────────────────────
DEFAULT_PER_CHALLENGE_TIMEOUT_SEC = 5.0
DEFAULT_SESSION_TIMEOUT_SEC = 12.0
DEFAULT_MAX_FAILURES = 3
DEFAULT_COOLDOWN_SEC = 30.0

# Continuity thresholds
MAX_TRACKING_GAP_SEC = 0.5           # Max duration allowed with 0 faces
MAX_FACE_CENTER_JUMP_RATIO = 0.25    # Max frame width ratio jump between frames (photo swap)
MIN_STATIC_CHECK_FRAMES = 12         # Number of frames to evaluate static photo variance
MIN_MICRO_MOTION_VARIANCE = 0.00015  # Minimum variance of yaw ratio to reject static photos

# Pose geometric thresholds (ratios from 5 landmarks)
# Neutral yaw: [0.44, 0.56]
YAW_LEFT_THRESHOLD = 0.62            # Turn left in mirrored view
YAW_RIGHT_THRESHOLD = 0.38           # Turn right in mirrored view
PITCH_UP_THRESHOLD = 0.35            # Tilt head up
PITCH_DOWN_THRESHOLD = 0.65          # Tilt head down


class LivenessChallenge(Enum):
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    TILT_UP = "TILT_UP"
    TILT_DOWN = "TILT_DOWN"
    HOLD_STILL = "HOLD_STILL"

    @property
    def prompt(self) -> str:
        prompts = {
            LivenessChallenge.TURN_LEFT: "Turn your head to the LEFT",
            LivenessChallenge.TURN_RIGHT: "Turn your head to the RIGHT",
            LivenessChallenge.TILT_UP: "Tilt your head UPWARD",
            LivenessChallenge.TILT_DOWN: "Tilt your head DOWNWARD",
            LivenessChallenge.HOLD_STILL: "Look straight ahead and HOLD STILL",
        }
        return prompts[self]


class LivenessStatus(Enum):
    IN_PROGRESS = "IN_PROGRESS"
    PASSED = "PASSED"
    FAILED_TIMEOUT = "FAILED_TIMEOUT"
    FAILED_CONTINUITY_LOST = "FAILED_CONTINUITY_LOST"
    FAILED_MULTIPLE_FACES = "FAILED_MULTIPLE_FACES"
    FAILED_STATIC_PHOTO = "FAILED_STATIC_PHOTO"
    FAILED_WRONG_ACTION = "FAILED_WRONG_ACTION"
    BLOCKED_COOLDOWN = "BLOCKED_COOLDOWN"
    SESSION_EXPIRED = "SESSION_EXPIRED"


@dataclass
class PoseMetrics:
    yaw_ratio: float
    pitch_ratio: float
    roll_deg: float
    is_neutral: bool
    is_left: bool
    is_right: bool
    is_up: bool
    is_down: bool


@dataclass
class LivenessDecision:
    status: LivenessStatus
    session_id: str
    current_challenge: Optional[LivenessChallenge]
    challenges_completed: int
    total_challenges: int
    time_remaining: float
    message: str
    details: dict


def estimate_pose_from_landmarks(landmarks: np.ndarray) -> PoseMetrics:
    """
    Computes scale-invariant geometric pose ratios from 5 standard landmarks:
      0: right eye, 1: left eye, 2: nose tip, 3: right mouth corner, 4: left mouth corner
    """
    pts = landmarks.reshape((5, 2)).astype(np.float64)
    re, le, nt, rmc, lmc = pts

    eye_dx = le[0] - re[0]
    eye_dy = le[1] - re[1]
    eye_dist = np.hypot(eye_dx, eye_dy)

    if eye_dist < 1e-5:
        return PoseMetrics(0.5, 0.5, 0.0, True, False, False, False, False)

    # Yaw ratio: nose x relative to eye span
    yaw_ratio = float((nt[0] - re[0]) / eye_dx)

    # Pitch ratio: nose y relative to eyes-midpoint to mouth-midpoint vertical span
    eye_mid_y = (re[1] + le[1]) / 2.0
    mouth_mid_y = (rmc[1] + lmc[1]) / 2.0
    face_h = mouth_mid_y - eye_mid_y
    pitch_ratio = float((nt[1] - eye_mid_y) / face_h) if face_h > 1e-5 else 0.5

    # Roll angle in degrees
    roll_deg = float(np.degrees(np.arctan2(eye_dy, eye_dx)))

    is_left = yaw_ratio >= YAW_LEFT_THRESHOLD
    is_right = yaw_ratio <= YAW_RIGHT_THRESHOLD
    is_up = pitch_ratio <= PITCH_UP_THRESHOLD
    is_down = pitch_ratio >= PITCH_DOWN_THRESHOLD
    is_neutral = (0.42 <= yaw_ratio <= 0.58) and (0.38 <= pitch_ratio <= 0.62)

    return PoseMetrics(
        yaw_ratio=round(yaw_ratio, 3),
        pitch_ratio=round(pitch_ratio, 3),
        roll_deg=round(roll_deg, 2),
        is_neutral=is_neutral,
        is_left=is_left,
        is_right=is_right,
        is_up=is_up,
        is_down=is_down,
    )


class SecurityEventLogger:
    """
    Appends privacy-safe security and liveness audit events to JSON Lines log.
    Never logs raw frames, image buffers, or biometric vectors.
    """

    def __init__(self, log_path: Path = SECURITY_LOG_FILE) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: str,
        session_id: str,
        status: str,
        challenge: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "session_id": session_id,
            "status": status,
            "challenge": challenge,
            "details": details or {},
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            print(f"[security_logger] Warning: Could not write security log: {exc}", file=sys.stderr)


class RateLimiter:
    """
    Tracks consecutive failed verification/liveness attempts and enforces temporary cooldowns.
    """

    def __init__(
        self,
        max_failures: int = DEFAULT_MAX_FAILURES,
        cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
    ) -> None:
        self.max_failures = max_failures
        self.cooldown_sec = cooldown_sec
        self.failure_count = 0
        self.cooldown_until: float = 0.0

    def is_blocked(self, current_time: Optional[float] = None) -> Tuple[bool, float]:
        now = time.time() if current_time is None else current_time
        if now < self.cooldown_until:
            return True, round(self.cooldown_until - now, 1)
        return False, 0.0

    def record_failure(self, current_time: Optional[float] = None) -> Tuple[bool, float]:
        now = time.time() if current_time is None else current_time
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.cooldown_until = now + self.cooldown_sec
            self.failure_count = 0
            return True, self.cooldown_sec
        return False, 0.0

    def record_success(self) -> None:
        self.failure_count = 0
        self.cooldown_until = 0.0


class LivenessChallengeSession:
    """
    Manages an active challenge-response session with continuity tracking,
    anti-replay single-use tokens, and static-photo rejection.
    """

    def __init__(
        self,
        num_challenges: int = 2,
        challenge_timeout_sec: float = DEFAULT_PER_CHALLENGE_TIMEOUT_SEC,
        session_timeout_sec: float = DEFAULT_SESSION_TIMEOUT_SEC,
        logger: Optional[SecurityEventLogger] = None,
        challenges: Optional[List[LivenessChallenge]] = None,
    ) -> None:
        self.session_id = uuid.uuid4().hex
        self.logger = logger or SecurityEventLogger()
        self.challenge_timeout_sec = challenge_timeout_sec
        self.session_timeout_sec = session_timeout_sec

        # Challenge pool
        pool = [
            LivenessChallenge.TURN_LEFT,
            LivenessChallenge.TURN_RIGHT,
            LivenessChallenge.TILT_UP,
            LivenessChallenge.TILT_DOWN,
        ]
        if challenges is not None:
            self.challenges = list(challenges)
        else:
            # Generate randomized non-repeating sequence
            self.challenges = random.sample(pool, min(num_challenges, len(pool)))

        self.total_challenges = len(self.challenges)
        self.current_idx = 0
        self.is_consumed = False
        self.is_completed = False

        self.session_start_time: Optional[float] = None
        self.challenge_start_time: Optional[float] = None
        self.last_face_seen_time: Optional[float] = None
        self.last_face_center: Optional[Tuple[float, float]] = None

        # Micro-motion / static photo tracking
        self.yaw_history: List[float] = []
        self.gesture_hold_frames = 0
        self.waiting_for_neutral = True  # Requires neutral start before next challenge

        self.logger.log_event(
            event_type="LIVENESS_SESSION_CREATED",
            session_id=self.session_id,
            status="INIT",
            details={"challenges": [c.value for c in self.challenges]},
        )

    def current_challenge(self) -> Optional[LivenessChallenge]:
        if self.current_idx < len(self.challenges):
            return self.challenges[self.current_idx]
        return None

    def evaluate_frame(
        self,
        faces: Optional[np.ndarray],
        frame_dims: Tuple[int, int],
        current_time: Optional[float] = None,
    ) -> LivenessDecision:
        """
        Processes detected face landmarks for the active challenge step.
        Enforces continuity, detects static photo presentation attacks,
        and advances the challenge state machine.
        """
        now = time.time() if current_time is None else current_time

        if self.is_consumed:
            return LivenessDecision(
                status=LivenessStatus.SESSION_EXPIRED,
                session_id=self.session_id,
                current_challenge=None,
                challenges_completed=self.current_idx,
                total_challenges=self.total_challenges,
                time_remaining=0.0,
                message="Session already consumed. A new session must be initiated.",
                details={"consumed": True},
            )

        if self.session_start_time is None:
            self.session_start_time = now
            self.challenge_start_time = now

        session_elapsed = now - self.session_start_time
        session_remaining = max(0.0, self.session_timeout_sec - session_elapsed)
        challenge_elapsed = now - (self.challenge_start_time or now)
        challenge_remaining = max(0.0, self.challenge_timeout_sec - challenge_elapsed)

        # 1. Session timeout check
        if session_remaining <= 0.0 or challenge_remaining <= 0.0:
            self.is_consumed = True
            self.logger.log_event(
                event_type="LIVENESS_CHALLENGE_TIMEOUT",
                session_id=self.session_id,
                status="FAILED_TIMEOUT",
                challenge=self.current_challenge().value if self.current_challenge() else None,
            )
            return LivenessDecision(
                status=LivenessStatus.FAILED_TIMEOUT,
                session_id=self.session_id,
                current_challenge=self.current_challenge(),
                challenges_completed=self.current_idx,
                total_challenges=self.total_challenges,
                time_remaining=0.0,
                message="Liveness challenge timed out. Movement was not completed in time.",
                details={"challenge_elapsed": challenge_elapsed, "session_elapsed": session_elapsed},
            )

        num_faces = len(faces) if (faces is not None and len(faces) > 0) else 0

        # 2. Multi-face check (Security lockout)
        if num_faces > 1:
            self.is_consumed = True
            self.logger.log_event(
                event_type="LIVENESS_MULTIPLE_FACES",
                session_id=self.session_id,
                status="FAILED_MULTIPLE_FACES",
                details={"detected_faces": num_faces},
            )
            return LivenessDecision(
                status=LivenessStatus.FAILED_MULTIPLE_FACES,
                session_id=self.session_id,
                current_challenge=self.current_challenge(),
                challenges_completed=self.current_idx,
                total_challenges=self.total_challenges,
                time_remaining=challenge_remaining,
                message=f"Multiple faces ({num_faces}) detected. Security violation.",
                details={"faces_count": num_faces},
            )

        # 3. No face / continuity check
        if num_faces == 0:
            if self.last_face_seen_time is not None:
                gap = now - self.last_face_seen_time
                if gap > MAX_TRACKING_GAP_SEC:
                    self.is_consumed = True
                    self.logger.log_event(
                        event_type="LIVENESS_CONTINUITY_LOST",
                        session_id=self.session_id,
                        status="FAILED_CONTINUITY_LOST",
                        details={"tracking_gap": gap},
                    )
                    return LivenessDecision(
                        status=LivenessStatus.FAILED_CONTINUITY_LOST,
                        session_id=self.session_id,
                        current_challenge=self.current_challenge(),
                        challenges_completed=self.current_idx,
                        total_challenges=self.total_challenges,
                        time_remaining=challenge_remaining,
                        message="Tracking continuity broken. Face left the frame.",
                        details={"gap": gap},
                    )
            return LivenessDecision(
                status=LivenessStatus.IN_PROGRESS,
                session_id=self.session_id,
                current_challenge=self.current_challenge(),
                challenges_completed=self.current_idx,
                total_challenges=self.total_challenges,
                time_remaining=challenge_remaining,
                message="Searching for face...",
                details={},
            )

        # Face is present — check position jump (anti-photo swap)
        face_data = faces[0]
        x, y, w, h = [float(v) for v in face_data[:4]]
        center = (x + w / 2.0, y + h / 2.0)
        w_frame = frame_dims[0]

        if self.last_face_center is not None and w_frame > 0:
            dx = abs(center[0] - self.last_face_center[0])
            dy = abs(center[1] - self.last_face_center[1])
            dist_ratio = np.hypot(dx, dy) / w_frame
            if dist_ratio > MAX_FACE_CENTER_JUMP_RATIO:
                self.is_consumed = True
                self.logger.log_event(
                    event_type="LIVENESS_POSITION_JUMP",
                    session_id=self.session_id,
                    status="FAILED_CONTINUITY_LOST",
                    details={"jump_ratio": dist_ratio},
                )
                return LivenessDecision(
                    status=LivenessStatus.FAILED_CONTINUITY_LOST,
                    session_id=self.session_id,
                    current_challenge=self.current_challenge(),
                    challenges_completed=self.current_idx,
                    total_challenges=self.total_challenges,
                    time_remaining=challenge_remaining,
                    message="Sudden position jump detected. Presentation swap suspected.",
                    details={"jump_ratio": dist_ratio},
                )

        self.last_face_seen_time = now
        self.last_face_center = center

        # Landmark geometry & pose estimation
        landmarks = face_data[4:14]
        pose = estimate_pose_from_landmarks(landmarks)

        # Record micro-motion history
        self.yaw_history.append(pose.yaw_ratio)
        if len(self.yaw_history) > 30:
            self.yaw_history.pop(0)

        # 4. Static photo rejection: if user has responded over 15 frames but yaw has 0 variance
        if len(self.yaw_history) >= MIN_STATIC_CHECK_FRAMES:
            var = float(np.var(self.yaw_history))
            if var < MIN_MICRO_MOTION_VARIANCE and not pose.is_neutral:
                self.is_consumed = True
                self.logger.log_event(
                    event_type="LIVENESS_STATIC_PHOTO_DETECTED",
                    session_id=self.session_id,
                    status="FAILED_STATIC_PHOTO",
                    details={"yaw_variance": var},
                )
                return LivenessDecision(
                    status=LivenessStatus.FAILED_STATIC_PHOTO,
                    session_id=self.session_id,
                    current_challenge=self.current_challenge(),
                    challenges_completed=self.current_idx,
                    total_challenges=self.total_challenges,
                    time_remaining=challenge_remaining,
                    message="Static image detected (zero natural micro-motion). Attack rejected.",
                    details={"yaw_variance": var},
                )

        target = self.current_challenge()

        # Step A: If waiting for neutral face before challenge
        if self.waiting_for_neutral:
            if pose.is_neutral:
                self.waiting_for_neutral = False
                self.challenge_start_time = now  # Start timer once neutral is acquired
                return LivenessDecision(
                    status=LivenessStatus.IN_PROGRESS,
                    session_id=self.session_id,
                    current_challenge=target,
                    challenges_completed=self.current_idx,
                    total_challenges=self.total_challenges,
                    time_remaining=self.challenge_timeout_sec,
                    message=f"Challenge {self.current_idx + 1}/{self.total_challenges}: {target.prompt}",
                    details={"pose": asdict(pose)},
                )
            else:
                return LivenessDecision(
                    status=LivenessStatus.IN_PROGRESS,
                    session_id=self.session_id,
                    current_challenge=target,
                    challenges_completed=self.current_idx,
                    total_challenges=self.total_challenges,
                    time_remaining=challenge_remaining,
                    message="Please face the camera directly to begin challenge",
                    details={"pose": asdict(pose)},
                )

        # Step B: Evaluate target challenge action
        gesture_matches = False
        if target == LivenessChallenge.TURN_LEFT and pose.is_left:
            gesture_matches = True
        elif target == LivenessChallenge.TURN_RIGHT and pose.is_right:
            gesture_matches = True
        elif target == LivenessChallenge.TILT_UP and pose.is_up:
            gesture_matches = True
        elif target == LivenessChallenge.TILT_DOWN and pose.is_down:
            gesture_matches = True
        elif target == LivenessChallenge.HOLD_STILL and pose.is_neutral:
            gesture_matches = True

        challenge_satisfied = False
        if gesture_matches:
            self.gesture_hold_frames += 1
            # Check for static photo presentation attack after 3 frames of holding pose
            if self.gesture_hold_frames >= 3:
                var = float(np.var(self.yaw_history)) if len(self.yaw_history) >= 3 else 1.0
                if var < MIN_MICRO_MOTION_VARIANCE and target != LivenessChallenge.HOLD_STILL:
                    self.is_consumed = True
                    self.logger.log_event(
                        event_type="LIVENESS_STATIC_PHOTO_DETECTED",
                        session_id=self.session_id,
                        status="FAILED_STATIC_PHOTO",
                        details={"yaw_variance": var, "hold_frames": self.gesture_hold_frames},
                    )
                    return LivenessDecision(
                        status=LivenessStatus.FAILED_STATIC_PHOTO,
                        session_id=self.session_id,
                        current_challenge=self.current_challenge(),
                        challenges_completed=self.current_idx,
                        total_challenges=self.total_challenges,
                        time_remaining=challenge_remaining,
                        message="Static image detected (zero natural micro-motion). Attack rejected.",
                        details={"yaw_variance": var},
                    )
                challenge_satisfied = True
        else:
            self.gesture_hold_frames = 0

        if challenge_satisfied:
            self.logger.log_event(
                event_type="LIVENESS_CHALLENGE_SATISFIED",
                session_id=self.session_id,
                status="SATISFIED",
                challenge=target.value,
                details={"elapsed_sec": challenge_elapsed},
            )
            self.current_idx += 1
            self.waiting_for_neutral = True
            self.gesture_hold_frames = 0
            self.challenge_start_time = now
            self.yaw_history.clear()

            # If all challenges completed -> PASSED
            if self.current_idx >= self.total_challenges:
                self.is_completed = True
                self.is_consumed = True
                self.logger.log_event(
                    event_type="LIVENESS_SESSION_PASSED",
                    session_id=self.session_id,
                    status="PASSED",
                    details={"total_time_sec": now - (self.session_start_time or now)},
                )
                return LivenessDecision(
                    status=LivenessStatus.PASSED,
                    session_id=self.session_id,
                    current_challenge=None,
                    challenges_completed=self.current_idx,
                    total_challenges=self.total_challenges,
                    time_remaining=session_remaining,
                    message="Liveness challenge PASSED successfully.",
                    details={"total_time": now - (self.session_start_time or now)},
                )

        return LivenessDecision(
            status=LivenessStatus.IN_PROGRESS,
            session_id=self.session_id,
            current_challenge=self.current_challenge(),
            challenges_completed=self.current_idx,
            total_challenges=self.total_challenges,
            time_remaining=challenge_remaining,
            message=f"Challenge {self.current_idx + 1}/{self.total_challenges}: {self.current_challenge().prompt}",
            details={"pose": asdict(pose)},
        )

    def consume(self) -> None:
        """Explicitly marks session consumed to prevent any replay."""
        self.is_consumed = True


class LivenessPipeline:
    """
    Coordinator connecting YuNet face detection and challenge-response liveness.
    """

    def __init__(
        self,
        num_challenges: int = 2,
        max_failures: int = DEFAULT_MAX_FAILURES,
        cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
    ) -> None:
        self.num_challenges = num_challenges
        self.logger = SecurityEventLogger()
        self.rate_limiter = RateLimiter(max_failures, cooldown_sec)
        self._detector: Optional[cv2.FaceDetectorYN] = None

    def _ensure_detector(self, width: int, height: int) -> cv2.FaceDetectorYN:
        if not YUNET_MODEL.exists():
            raise FileNotFoundError(f"YuNet model missing at {YUNET_MODEL}")
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

    def start_session(self) -> Tuple[Optional[LivenessChallengeSession], Optional[str]]:
        is_blocked, remaining = self.rate_limiter.is_blocked()
        if is_blocked:
            self.logger.log_event(
                event_type="LIVENESS_ATTEMPT_BLOCKED",
                session_id="none",
                status="BLOCKED_COOLDOWN",
                details={"remaining_cooldown_sec": remaining},
            )
            return None, f"Too many failed attempts. Cooldown active for {remaining:.0f}s."

        session = LivenessChallengeSession(
            num_challenges=self.num_challenges,
            logger=self.logger,
        )
        return session, None

    def run_live_liveness_test(self, camera_index: int = 0) -> bool:
        """
        Interactive demonstration of liveness challenge flow.
        """
        print("\n" + "=" * 65)
        print("  FaceUnlockAI — Live Liveness Challenge Test")
        print("=" * 65)
        print("Required Actions: Complete randomized head movement challenges.")
        print("Security Checks:  Timeouts, tracking continuity, anti-photo static check.")
        print("Controls:         Press 'Q' or ESC to exit.")
        print("=" * 65 + "\n")

        session, err = self.start_session()
        if session is None:
            print(f"[liveness] {err}", file=sys.stderr)
            return False

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[liveness] ERROR: Cannot open webcam index {camera_index}.", file=sys.stderr)
            return False

        window_name = "FaceUnlockAI — Liveness Challenge"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        passed = False
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
                decision = session.evaluate_frame(faces, (w_frame, h_frame))

                display = frame.copy()

                # Choose theme color
                if decision.status == LivenessStatus.PASSED:
                    color = (0, 220, 0)      # Green
                elif decision.status == LivenessStatus.IN_PROGRESS:
                    color = (0, 200, 255)    # Amber
                else:
                    color = (0, 0, 240)      # Red (Failed)

                # Draw face bounding box & landmarks
                if faces is not None and len(faces) == 1:
                    x, y, w, h = [int(v) for v in faces[0][:4]]
                    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                    landmarks = faces[0][4:14].reshape((5, 2)).astype(int)
                    for lx, ly in landmarks:
                        cv2.circle(display, (lx, ly), 3, (0, 255, 255), -1)

                # Top HUD Banner
                cv2.rectangle(display, (0, 0), (w_frame, 80), (15, 15, 15), -1)
                title = f"Liveness: {decision.status.value}"
                cv2.putText(display, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, decision.message, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

                # Progress & Timer
                timer_text = f"Time: {decision.time_remaining:.1f}s"
                cv2.putText(display, timer_text, (w_frame - 150, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

                # Bottom status bar
                cv2.rectangle(display, (0, h_frame - 40), (w_frame, h_frame), (15, 15, 15), -1)
                cv2.putText(display, "[Q / ESC] Cancel Challenge", (20, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\n[liveness] Cancelled by user.")
                    session.consume()
                    break

                if decision.status == LivenessStatus.PASSED:
                    passed = True
                    self.rate_limiter.record_success()
                    print("\n[liveness] Liveness Challenge PASSED!")
                    cv2.waitKey(1500)
                    break

                if decision.status not in (LivenessStatus.IN_PROGRESS,):
                    is_blocked, remaining = self.rate_limiter.record_failure()
                    print(f"\n[liveness] Challenge FAILED: {decision.status.value} ({decision.message})")
                    if is_blocked:
                        print(f"[liveness] Cooldown active for {remaining:.0f} seconds.")
                    cv2.waitKey(2000)
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceUnlockAI Phase 3 — Liveness Challenge Layer")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--challenges", type=int, default=2, help="Number of challenges (default: 2)")

    args = parser.parse_args()
    pipeline = LivenessPipeline(num_challenges=args.challenges)
    pipeline.run_live_liveness_test(camera_index=args.camera)


if __name__ == "__main__":
    main()
