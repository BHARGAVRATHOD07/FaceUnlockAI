# FaceUnlockAI — Biometric Security & Presentation-Attack Detection Report

**Document Version:** 1.0.0  
**Environment:** Windows 11 (64-bit) | Python 3.13.7  
**Pipeline:** OpenCV YuNet (Face Detection) + SFace (Deep Recognition) + Active Challenge-Response Liveness  
**Status:** Evaluation Prototype — **NOT FOR PRODUCTION WINDOWS SIGN-IN**  

---

## 1. Executive Summary

FaceUnlockAI implements a layered defense-in-depth biometric verification system designed to operate **100% locally** on consumer Windows 11 devices with standard 2D RGB webcams. 

This security evaluation document outlines:
1. The decoupled security architecture (Detection vs. Identity vs. Liveness).
2. The active challenge-response liveness protocol.
3. Automated security attack simulations and evaluation results.
4. Privacy and credential security guarantees.
5. Fundamental physical limitations of 2D RGB computer vision versus dedicated biometric hardware.

---

## 2. Threat Model & Security Architecture

### 2.1 Threat Vectors Evaluated

| Threat ID | Attack Description | Severity | Countermeasure Implemented |
|---|---|---|---|
| **T-01** | **Static Photo Presentation** (Printed paper / smartphone photo displayed to camera) | High | Micro-motion jitter analysis + randomized dynamic pose challenges |
| **T-02** | **Pre-recorded Video Replay** (Attacker plays recorded video of owner) | High | Randomized challenge-response sequence + single-use UUID session tokens + per-challenge timeouts |
| **T-03** | **Presentation Swap / Cut** (Attacker passes initial frame, then steps aside) | High | Continuous spatial face tracking; position jumps $>25\%$ of frame width fail continuity immediately |
| **T-04** | **Shoulder-Surfing / Multi-Face** (Multiple individuals in camera field of view) | Medium | Strict multi-face safety lockout ($\text{faces} > 1 \implies \text{Denial}$) |
| **T-05** | **Session Replay / Result Caching** (Re-submitting a past `PASSED` liveness signal) | High | Cryptographic session tokens invalidated (`consume()`) immediately upon decision |
| **T-06** | **Brute-Force Presentation** (Repeated trial-and-error presentation attacks) | Medium | Rate limiter enforcing 3-failure threshold with 30-second cooldown lockouts |
| **T-07** | **Biometric Template Theft** (Local unauthorized accounts reading stored vectors) | Medium | Restrictive Windows ACLs (`icacls`) isolating `data/registered_face/` |

---

## 3. Defense-in-Depth Pipeline

```
  [ Webcam 2D RGB Frame ]
            │
            ▼
┌───────────────────────────────────────┐
│ Stage 1: Face Detection & Sizing      │  ───> 0 faces ──> Wait / Search
│ (OpenCV YuNet — 15-dim detection)     │  ───> >1 face  ──> FAIL CLOSED (Multi-Face Lockout)
└───────────────────────────────────────┘
            │ Exactly 1 Face
            ▼
┌───────────────────────────────────────┐
│ Stage 2: Face Quality Gating          │  ───> Low sharpness (< 40.0) ──> Reject (Blur)
│ (Sharpness, Illumination, Contrast)   │  ───> Poor lighting (< 35 or > 230) ──> Reject
└───────────────────────────────────────┘
            │ Quality Valid
            ▼
┌───────────────────────────────────────┐
│ Stage 3: Identity Verification        │  ───> Sim <= 0.345 ──> REJECTED_UNKNOWN
│ (OpenCV SFace 128-D Cosine Match)     │  ───> Sim in [0.345, 0.415] ──> AMBIGUOUS
└───────────────────────────────────────┘
            │ Verified Owner (Sim >= 0.415)
            ▼
┌───────────────────────────────────────┐
│ Stage 4: Challenge-Response Liveness  │  ───> Timeout (5s) ──> FAIL CLOSED
│ (Randomized: Yaw Left/Right, Pitch)   │  ───> Static photo (var ~ 0) ──> FAIL CLOSED
│ (Continuity & Anti-Swap Tracking)     │  ───> Continuity broken ──> FAIL CLOSED
└───────────────────────────────────────┘
            │ Both Stages Satisfied
            ▼
    [ ACCESS GRANTED ]
```

### Critical Security Rule: Fail-Closed Joint Decision
$$\text{ACCESS} = (\text{IdentityStatus} == \text{VERIFIED\_OWNER}) \land (\text{LivenessStatus} == \text{PASSED})$$

- **Liveness alone does NOT grant access.** An unknown person completing head turns is rejected.
- **Identity alone does NOT grant access.** An enrolled owner presenting a static photo is rejected.
- **Any anomaly, timeout, or ambiguity fails closed.**

---

## 4. Automated Attack Simulation Matrix

All attack scenarios are codified as automated unit and integration tests in [`tests/test_liveness_detection.py`](file:///C:/Users/Bhargav%20Rathod/OneDrive/Desktop/FaceUnlockAI/tests/test_liveness_detection.py) and [`tests/test_face_recognition.py`](file:///C:/Users/Bhargav%20Rathod/OneDrive/Desktop/FaceUnlockAI/tests/test_face_recognition.py).

| Test Case | Scenario Evaluated | Expected Behavior | Automated Test Status |
|---|---|---|---|
| `test_neutral_pose` | Centered face facing camera | $\text{yaw} \approx 0.50, \text{pitch} \approx 0.50$, Neutral confirmed | **PASSED** |
| `test_turn_left_pose` | Head turned to left | $\text{yaw} > 0.62$, Left confirmed | **PASSED** |
| `test_turn_right_pose` | Head turned to right | $\text{yaw} < 0.38$, Right confirmed | **PASSED** |
| `test_tilt_up_pose` | Head tilted upwards | $\text{pitch} < 0.35$, Up confirmed | **PASSED** |
| `test_tilt_down_pose` | Head tilted downwards | $\text{pitch} > 0.65$, Down confirmed | **PASSED** |
| `test_randomized_challenges_generated` | Session initialization | Unpredictable challenge sequence | **PASSED** |
| `test_successful_challenge_progression` | Live user executing gestures | Valid completion across 2 steps $\to$ `PASSED` | **PASSED** |
| `test_anti_replay_single_use_session` | Token replayed after passing | Session consumed $\to$ `SESSION_EXPIRED` | **PASSED** |
| `test_challenge_timeout_fails_closed` | User unresponsive past 3.0s | Exceeded deadline $\to$ `FAILED_TIMEOUT` | **PASSED** |
| `test_face_disappearance_breaks_continuity` | Face leaves frame $> 0.5$s | Continuity broken $\to$ `FAILED_CONTINUITY_LOST` | **PASSED** |
| `test_sudden_position_jump_rejected` | Face teleports $> 25\%$ width | Position jump $\to$ `FAILED_CONTINUITY_LOST` | **PASSED** |
| `test_multi_face_attack_rejected` | 2 faces present simultaneously | Security lockout $\to$ `FAILED_MULTIPLE_FACES` | **PASSED** |
| `test_static_photo_attack_rejected` | Printed photo held in front of lens | Zero yaw variance $\to$ `FAILED_STATIC_PHOTO` | **PASSED** |
| `test_rate_limiter_cooldown` | 3 consecutive failures | 30s lockout triggered $\to$ `BLOCKED_COOLDOWN` | **PASSED** |
| `test_security_event_logger` | Audit logging | JSON Lines recorded without raw pixels/vectors | **PASSED** |
| `test_unknown_person_with_passing_liveness` | Unknown attacker mimicking gestures | Identity fails $\to$ **ACCESS DENIED** | **PASSED** |
| `test_owner_with_failed_liveness` | Owner photo held without motion | Liveness fails $\to$ **ACCESS DENIED** | **PASSED** |

---

## 5. Privacy, Logging & Credential Safety

1. **Zero Raw Facial Image Retention:** Raw images are discarded from memory immediately after embedding extraction unless `--save-raw-images` is explicitly requested.
2. **Audit Logs:** Security events are appended to [`logs/security_events.log`](file:///C:/Users/Bhargav%20Rathod/OneDrive/Desktop/FaceUnlockAI/logs/security_events.log) using structured JSON Lines:
   ```json
   {
     "timestamp": "2026-10-27T06:55:00.123456Z",
     "event_type": "LIVENESS_STATIC_PHOTO_DETECTED",
     "session_id": "9b1deb4d3b7d4e0eb2c8",
     "status": "FAILED_STATIC_PHOTO",
     "details": {"yaw_variance": 0.0, "hold_frames": 3}
   }
   ```
   **Privacy Rule:** No image bytes, pixel arrays, or 128-D biometric vectors are ever written to audit logs.
3. **No Windows Credential Storage:** This prototype **never** stores, caches, or interacts with Windows user passwords, PINs, Kerberos tickets, or NTLM hashes.
4. **Local Windows File ACLs:** Biometric templates (`owner_template.npz`) are stored in `data/registered_face/` with Windows ACL permissions restricted to the current user and `SYSTEM`.

---

## 6. Critical Security Limitations (Why 2D RGB is NOT Spoof-Proof)

> [!WARNING]
> **DO NOT USE THIS PROTOTYPE TO REPLACE WINDOWS SIGN-IN OR LOCK SCREENS.**
>
> Software-based 2D RGB computer vision has fundamental physical limitations that cannot be fully overcome without dedicated hardware sensors.

### 6.1 Vulnerabilities Inherent to 2D RGB Cameras
- **High-Fidelity 3D Masks:** A custom-molded silicone or latex mask worn by an attacker replicates 3D depth, texture, and natural head articulation. Standard 2D RGB cameras cannot distinguish natural skin from flexible silicone.
- **Video Replay on High-Refresh Screens:** A high-resolution tablet displaying video of the owner turning their head can fool basic challenge-response heuristics if the attacker anticipates or replays the required motion sequence.
- **Deepfake Virtual Cameras / Video Injection:** If an attacker compromises the OS or installs a virtual webcam driver (e.g., OBS Virtual Cam), synthetic video generated by real-time neural avatars can inject arbitrary head movements directly into the video stream.
- **Environmental Lighting Variance:** Extreme backlighting, dim environments, or high-contrast shadows degrade landmark detection confidence and can lead to false rejections (FRR).

### 6.2 Comparison with Hardware-Grade Presentation-Attack Detection

| Security Feature | FaceUnlockAI (2D RGB Prototype) | Enterprise Biometrics / Windows Hello (Hardware IR + Depth) |
|---|---|---|
| Sensor Type | Standard 2D RGB Webcam | Active Near-Infrared (NIR) Illuminator + IR Camera + Time-of-Flight Depth |
| Static Photo Defense | Software micro-motion jitter & pose challenge | Instant physical rejection (screens and photos do not reflect structured IR) |
| 3D Silicone Mask Defense | **Vulnerable** (cannot measure volumetric depth) | **Resistant** (analyzes surface contour depth map and skin IR absorption) |
| Virtual Camera Injection Defense | Limited to OS-level integrity | Hardware-enforced signed camera pipelines |
| Operable in Total Darkness | **No** (requires visible light) | **Yes** (active NIR illuminators) |
| Biometric Certification | None (Experimental prototype) | FIDO2 / Microsoft Windows Hello Certified |

---

## 7. Conclusion & Next Steps

Phase 3 successfully decouples face detection, identity verification, and liveness decisions into an independently testable, fail-closed architecture with rate limiting, timeouts, and anti-replay safeguards.

For production authentication scenarios:
1. Integrate physical Near-Infrared (NIR) or structured-light depth camera drivers.
2. Bind authentication tokens to Windows Protected Storage / TPM 2.0 rather than plain filesystem templates.
3. Implement hardware-backed liveness compliant with ISO/IEC 30107-3 (Biometric Presentation Attack Detection).
