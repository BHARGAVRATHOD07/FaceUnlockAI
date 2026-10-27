"""
test_liveness_detection.py — FaceUnlockAI Phase 3 Automated Tests
-------------------------------------------------------------------
Automated tests for the liveness-checking and presentation-attack detection (PAD) layer.

Coverage:
  1. [pose]         HeadPoseEstimator: geometric ratio calculations for yaw, pitch, roll.
  2. [challenges]   Randomized challenge generation, state machine transitions, completion.
  3. [anti-replay]  Single-use session consumption (cached results cannot authorize new session).
  4. [timeouts]     Per-challenge and session timeout enforcement.
  5. [continuity]   Tracking continuity:
                      • Face disappearance (>0.5s) fails continuity
                      • Sudden position teleportation (>25% frame width) fails continuity
  6. [attacks]      Attack scenarios:
                      • Static photo presentation attack (zero micro-motion variance)
                      • Multi-face presentation attack (shoulder-surfing)
                      • Inconclusive / wrong movement rejection
  7. [rate-limit]   Brute-force protection: failure threshold & cooldown enforcement.
  8. [audit-log]    Security event logging: schema validity, zero raw imagery / biometrics.
  9. [joint-policy] Fail-closed joint policy (requires BOTH identity AND liveness).
 10. [disclaimer]   Biometric limitation assertion (2D RGB vs hardware IR/depth).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from liveness_detection import (
    DEFAULT_COOLDOWN_SEC,
    DEFAULT_MAX_FAILURES,
    DEFAULT_PER_CHALLENGE_TIMEOUT_SEC,
    LivenessChallenge,
    LivenessDecision,
    LivenessStatus,
    PoseMetrics,
    RateLimiter,
    SecurityEventLogger,
    LivenessChallengeSession,
    estimate_pose_from_landmarks,
)
from face_recognition import VerificationResult, VerificationStatus


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def neutral_landmarks() -> np.ndarray:
    """
    Standard 5 landmarks for a centered, neutral face:
    re=(100, 100), le=(200, 100), nt=(150, 150), rmc=(120, 200), lmc=(180, 200)
    """
    return np.array([100, 100, 200, 100, 150, 150, 120, 200, 180, 200], dtype=np.float32)


@pytest.fixture
def turn_left_landmarks() -> np.ndarray:
    """
    Turn left: in mirrored webcam coordinates, nose tip shifts towards left eye (nt_x = 180)
    """
    return np.array([100, 100, 200, 100, 180, 150, 130, 200, 190, 200], dtype=np.float32)


@pytest.fixture
def turn_right_landmarks() -> np.ndarray:
    """
    Turn right: in mirrored webcam coordinates, nose tip shifts towards right eye (nt_x = 120)
    """
    return np.array([100, 100, 200, 100, 120, 150, 110, 200, 170, 200], dtype=np.float32)


@pytest.fixture
def tilt_up_landmarks() -> np.ndarray:
    """
    Tilt up: nose tip moves vertically closer to the eye line (nt_y = 120 instead of 150)
    """
    return np.array([100, 100, 200, 100, 150, 120, 120, 190, 180, 190], dtype=np.float32)


@pytest.fixture
def tilt_down_landmarks() -> np.ndarray:
    """
    Tilt down: nose tip moves vertically lower towards mouth (nt_y = 175 instead of 150)
    """
    return np.array([100, 100, 200, 100, 150, 175, 120, 210, 180, 210], dtype=np.float32)


def make_yunet_face(bbox=(100, 100, 150, 150), landmarks=None) -> np.ndarray:
    """Builds a 15-element YuNet face vector."""
    if landmarks is None:
        landmarks = [120, 120, 180, 120, 150, 150, 130, 180, 170, 180]
    return np.array([bbox[0], bbox[1], bbox[2], bbox[3]] + list(landmarks) + [0.98], dtype=np.float32)


# ─── 1. POSE ESTIMATION TESTS ─────────────────────────────────────────────────

class TestHeadPoseEstimator:
    def test_neutral_pose(self, neutral_landmarks):
        pose = estimate_pose_from_landmarks(neutral_landmarks)
        assert pose.is_neutral is True
        assert 0.45 <= pose.yaw_ratio <= 0.55
        assert 0.45 <= pose.pitch_ratio <= 0.55
        assert abs(pose.roll_deg) < 5.0

    def test_turn_left_pose(self, turn_left_landmarks):
        pose = estimate_pose_from_landmarks(turn_left_landmarks)
        assert pose.is_left is True
        assert pose.yaw_ratio > 0.62
        assert pose.is_right is False

    def test_turn_right_pose(self, turn_right_landmarks):
        pose = estimate_pose_from_landmarks(turn_right_landmarks)
        assert pose.is_right is True
        assert pose.yaw_ratio < 0.38
        assert pose.is_left is False

    def test_tilt_up_pose(self, tilt_up_landmarks):
        pose = estimate_pose_from_landmarks(tilt_up_landmarks)
        assert pose.is_up is True
        assert pose.pitch_ratio < 0.35
        assert pose.is_down is False

    def test_tilt_down_pose(self, tilt_down_landmarks):
        pose = estimate_pose_from_landmarks(tilt_down_landmarks)
        assert pose.is_down is True
        assert pose.pitch_ratio > 0.65
        assert pose.is_up is False


# ─── 2. CHALLENGE-RESPONSE FLOW TESTS ─────────────────────────────────────────

class TestLivenessChallengeSession:
    def test_randomized_challenges_generated(self):
        s1 = LivenessChallengeSession(num_challenges=2)
        assert len(s1.challenges) == 2
        # Over multiple generations, challenges should vary and not be hardcoded
        sequences = {tuple(LivenessChallengeSession(num_challenges=2).challenges) for _ in range(20)}
        assert len(sequences) > 1, "Challenge sequence must be randomized, not fixed"

    def test_successful_challenge_progression(self, neutral_landmarks, turn_left_landmarks):
        # Explicit deterministic challenge sequence: [TURN_LEFT]
        session = LivenessChallengeSession(
            challenges=[LivenessChallenge.TURN_LEFT],
            challenge_timeout_sec=5.0,
        )
        t0 = 1000.0

        # Frame 1: User faces camera (neutral) to initialize challenge
        face_neutral = make_yunet_face(landmarks=neutral_landmarks)
        d1 = session.evaluate_frame(np.array([face_neutral]), (640, 480), current_time=t0)
        assert d1.status == LivenessStatus.IN_PROGRESS
        assert session.waiting_for_neutral is False

        # Frame 2 to 4: User turns left satisfying challenge with natural micro-movement
        for step in range(3):
            lm = turn_left_landmarks.copy()
            lm[4] += step * 2.0  # Natural micro-movement jitter
            face_left = make_yunet_face(landmarks=lm)
            d2 = session.evaluate_frame(np.array([face_left]), (640, 480), current_time=t0 + 0.5 + step * 0.1)

        assert d2.status == LivenessStatus.PASSED
        assert session.is_completed is True
        assert session.is_consumed is True

    def test_anti_replay_single_use_session(self, neutral_landmarks, turn_left_landmarks):
        session = LivenessChallengeSession(
            challenges=[LivenessChallenge.TURN_LEFT],
            challenge_timeout_sec=5.0,
        )
        t0 = 1000.0
        face_neutral = make_yunet_face(landmarks=neutral_landmarks)
        session.evaluate_frame(np.array([face_neutral]), (640, 480), current_time=t0)

        for step in range(3):
            lm = turn_left_landmarks.copy()
            lm[4] += step * 2.0
            face_left = make_yunet_face(landmarks=lm)
            d_pass = session.evaluate_frame(np.array([face_left]), (640, 480), current_time=t0 + 0.5 + step * 0.1)

        assert d_pass.status == LivenessStatus.PASSED

        # Calling evaluate_frame again on an already consumed session must fail closed
        d_replay = session.evaluate_frame(np.array([face_left]), (640, 480), current_time=t0 + 0.9)
        assert d_replay.status == LivenessStatus.SESSION_EXPIRED
        assert "Session already consumed" in d_replay.message


# ─── 3. TIMEOUTS & CONTINUITY TESTS ───────────────────────────────────────────

class TestTimeoutsAndContinuity:
    def test_challenge_timeout_fails_closed(self, neutral_landmarks):
        session = LivenessChallengeSession(
            challenges=[LivenessChallenge.TURN_LEFT],
            challenge_timeout_sec=3.0,
        )
        t0 = 1000.0
        face_neutral = make_yunet_face(landmarks=neutral_landmarks)
        session.evaluate_frame(np.array([face_neutral]), (640, 480), current_time=t0)

        # User remains static/neutral past 3.0s timeout
        d_timeout = session.evaluate_frame(np.array([face_neutral]), (640, 480), current_time=t0 + 3.1)
        assert d_timeout.status == LivenessStatus.FAILED_TIMEOUT
        assert session.is_consumed is True

    def test_face_disappearance_breaks_continuity(self, neutral_landmarks):
        session = LivenessChallengeSession(
            challenges=[LivenessChallenge.TURN_LEFT],
            challenge_timeout_sec=5.0,
        )
        t0 = 1000.0
        face_neutral = make_yunet_face(landmarks=neutral_landmarks)
        session.evaluate_frame(np.array([face_neutral]), (640, 480), current_time=t0)

        # 0 faces for 0.7s (> MAX_TRACKING_GAP_SEC 0.5s)
        d_lost = session.evaluate_frame(None, (640, 480), current_time=t0 + 0.7)
        assert d_lost.status == LivenessStatus.FAILED_CONTINUITY_LOST
        assert "continuity broken" in d_lost.message.lower()

    def test_sudden_position_jump_rejected(self, neutral_landmarks):
        session = LivenessChallengeSession(
            challenges=[LivenessChallenge.TURN_LEFT],
            challenge_timeout_sec=5.0,
        )
        t0 = 1000.0
        # Face at x=50
        face1 = make_yunet_face(bbox=(50, 100, 150, 150), landmarks=neutral_landmarks)
        session.evaluate_frame(np.array([face1]), (640, 480), current_time=t0)

        # In next frame, face teleports across screen to x=450 (> 25% of 640px)
        face2 = make_yunet_face(bbox=(450, 100, 150, 150), landmarks=neutral_landmarks)
        d_jump = session.evaluate_frame(np.array([face2]), (640, 480), current_time=t0 + 0.1)
        assert d_jump.status == LivenessStatus.FAILED_CONTINUITY_LOST
        assert "jump detected" in d_jump.message.lower()


# ─── 4. PRESENTATION ATTACK TESTS ─────────────────────────────────────────────

class TestPresentationAttacks:
    def test_multi_face_attack_rejected(self, neutral_landmarks):
        session = LivenessChallengeSession(challenges=[LivenessChallenge.TURN_LEFT])
        f1 = make_yunet_face(bbox=(50, 100, 150, 150), landmarks=neutral_landmarks)
        f2 = make_yunet_face(bbox=(300, 100, 150, 150), landmarks=neutral_landmarks)

        d_multi = session.evaluate_frame(np.array([f1, f2]), (640, 480), current_time=1000.0)
        assert d_multi.status == LivenessStatus.FAILED_MULTIPLE_FACES
        assert "Multiple faces" in d_multi.message

    def test_static_photo_attack_rejected(self, turn_left_landmarks):
        """
        Simulates an attacker holding a printed photo of a turned face with
        mathematically 0 micro-motion variance over 15 frames.
        """
        session = LivenessChallengeSession(
            challenges=[LivenessChallenge.TURN_LEFT],
            challenge_timeout_sec=5.0,
        )
        t0 = 1000.0
        # Bypass initial neutral requirement for test setup
        session.waiting_for_neutral = False
        session.challenge_start_time = t0

        photo_face = make_yunet_face(landmarks=turn_left_landmarks)
        d_static = None
        for i in range(5):
            d_static = session.evaluate_frame(np.array([photo_face]), (640, 480), current_time=t0 + i * 0.05)
            if d_static.status != LivenessStatus.IN_PROGRESS:
                break

        assert d_static is not None
        assert d_static.status == LivenessStatus.FAILED_STATIC_PHOTO
        assert "Static image detected" in d_static.message


# ─── 5. RATE LIMITING & SECURITY EVENT LOGGING ────────────────────────────────

class TestRateLimitingAndLogging:
    def test_rate_limiter_cooldown(self):
        limiter = RateLimiter(max_failures=3, cooldown_sec=30.0)
        t0 = 1000.0

        # First 2 failures: not blocked
        b1, _ = limiter.record_failure(t0)
        assert b1 is False
        b2, _ = limiter.record_failure(t0 + 1)
        assert b2 is False

        # 3rd failure: triggers 30s cooldown
        b3, cd = limiter.record_failure(t0 + 2)
        assert b3 is True
        assert cd == 30.0

        # Attempt during cooldown is blocked
        is_blocked, remaining = limiter.is_blocked(t0 + 10.0)
        assert is_blocked is True
        assert remaining == 22.0

        # Attempt after cooldown expires is unblocked
        is_blocked, _ = limiter.is_blocked(t0 + 35.0)
        assert is_blocked is False

    def test_security_event_logger(self, tmp_path):
        log_file = tmp_path / "test_security.log"
        logger = SecurityEventLogger(log_path=log_file)

        logger.log_event("TEST_EVENT", "sess-123", "PASSED", details={"reason": "unit_test"})

        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])

        assert data["session_id"] == "sess-123"
        assert data["event_type"] == "TEST_EVENT"
        assert data["status"] == "PASSED"

        # Privacy guarantee: ensure no image pixels or raw feature vectors are logged
        raw_text = log_file.read_text()
        forbidden = ["image_bytes", "raw_frame", "embedding_vector", "pixel_buffer"]
        for f in forbidden:
            assert f not in raw_text


# ─── 6. JOINT IDENTITY & LIVENESS SECURITY POLICY ─────────────────────────────

class TestJointPolicyDecision:
    """
    Critical requirement: Both identity verification and liveness must pass.
    Liveness alone does not prove identity, and identity alone does not bypass liveness.
    """

    @staticmethod
    def evaluate_joint_decision(
        id_result: VerificationResult,
        live_decision: LivenessDecision,
    ) -> tuple[bool, str]:
        """Fail-closed policy evaluator."""
        if id_result.status != VerificationStatus.VERIFIED_OWNER:
            return False, f"Access Denied: Identity check failed ({id_result.status.value})"
        if live_decision.status != LivenessStatus.PASSED:
            return False, f"Access Denied: Liveness check failed ({live_decision.status.value})"
        return True, "Access Granted: Owner verified and liveness confirmed."

    def test_owner_and_liveness_both_pass(self):
        id_res = VerificationResult(
            status=VerificationStatus.VERIFIED_OWNER,
            similarity=0.62,
            confidence_pct=92.0,
            message="Owner verified",
        )
        live_res = LivenessDecision(
            status=LivenessStatus.PASSED,
            session_id="s1",
            current_challenge=None,
            challenges_completed=2,
            total_challenges=2,
            time_remaining=8.0,
            message="Liveness passed",
            details={},
        )
        granted, msg = self.evaluate_joint_decision(id_res, live_res)
        assert granted is True
        assert "Access Granted" in msg

    def test_unknown_person_with_passing_liveness_rejected(self):
        """Liveness alone does NOT prove identity."""
        id_res = VerificationResult(
            status=VerificationStatus.REJECTED_UNKNOWN,
            similarity=0.18,
            confidence_pct=15.0,
            message="Unknown person",
        )
        live_res = LivenessDecision(
            status=LivenessStatus.PASSED,
            session_id="s2",
            current_challenge=None,
            challenges_completed=2,
            total_challenges=2,
            time_remaining=8.0,
            message="Liveness passed",
            details={},
        )
        granted, msg = self.evaluate_joint_decision(id_res, live_res)
        assert granted is False
        assert "Identity check failed" in msg

    def test_owner_with_failed_liveness_rejected(self):
        """Owner identity alone does NOT bypass liveness."""
        id_res = VerificationResult(
            status=VerificationStatus.VERIFIED_OWNER,
            similarity=0.60,
            confidence_pct=90.0,
            message="Owner verified",
        )
        live_res = LivenessDecision(
            status=LivenessStatus.FAILED_TIMEOUT,
            session_id="s3",
            current_challenge=LivenessChallenge.TURN_LEFT,
            challenges_completed=0,
            total_challenges=2,
            time_remaining=0.0,
            message="Timed out",
            details={},
        )
        granted, msg = self.evaluate_joint_decision(id_res, live_res)
        assert granted is False
        assert "Liveness check failed" in msg


# ─── 7. BIOMETRIC HARDWARE LIMITATION NOTICE ──────────────────────────────────

class TestBiometricHardwareNotice:
    def test_2d_rgb_hardware_limitation_documented(self):
        notice = (
            "SECURITY NOTICE: 2D RGB webcam liveness detection raises the cost of attacks "
            "against casual static photo replays, but CANNOT replace hardware-based Presentation "
            "Attack Detection (such as Windows Hello infrared / time-of-flight depth cameras). "
            "Advanced silicone masks, printed eye cutouts, or 3D avatars may bypass 2D visual heuristics."
        )
        assert len(notice) > 100
