# FaceUnlockAI — Windows 11 Biometric Authentication Prototype

> [!WARNING]
> ### EVALUATION & RESEARCH PROTOTYPE DISCLAIMER
> **FaceUnlockAI is an educational and security-research prototype. It is NOT a replacement for Windows Hello or any operating-system-level sign-in provider.**
> - **Zero Windows Credentials:** This application does **NOT** collect, store, simulate, inject, or bypass Windows passwords, PINs, Kerberos tickets, or DPAPI credentials.
> - **No Keystroke / Unlock Simulation:** This application does **NOT** simulate keystrokes, inject credentials into the Windows Logon Screen, or tamper with Windows Secure Desktop.
> - **Hardware Limitations:** Standard 2D RGB webcams lack structured light, time-of-flight (ToF) depth sensors, and active infrared (IR) illumination. Without dedicated biometric hardware, 2D RGB models remain susceptible to sophisticated presentation attacks (e.g. 3D latex masks, hyper-realistic video projections).

---

## 1. Overview & Architecture

**FaceUnlockAI** is a local biometric authentication and presentation-attack defense pipeline built specifically for Windows 11 using Python 3.13. It demonstrates defense-in-depth computer vision security by decoupling detection, identity verification, active liveness challenge-response, and security audit logging.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FaceUnlockAI Desktop GUI                        │
│             (Modern Windows 11 Interface via CustomTkinter)            │
└────────────────────────────────────────────────────────────────────────┘
     │                    │                     │                   │
     ▼                    ▼                     ▼                   ▼
┌──────────────┐   ┌──────────────┐      ┌─────────────┐     ┌─────────────┐
│ Registration │   │   Identity   │      │  Liveness   │     │  Security   │
│ & Alignment  │   │ Verification │      │  Challenge  │     │ Audit Logs  │
└──────────────┘   └──────────────┘      └─────────────┘     └─────────────┘
     │                    │                     │                   │
     ▼                    ▼                     ▼                   ▼
┌──────────────┐   ┌──────────────┐      ┌─────────────┐     ┌─────────────┐
│ OpenCV YuNet │   │ OpenCV SFace │      │ Head Pose   │     │ JSON Lines  │
│ Face Detect  │   │ 128-D Cosine │      │ Estimation  │     │ Zero Raw    │
│ (ONNX Local) │   │ (ONNX Local) │      │ State Mach. │     │ Biometrics  │
└──────────────┘   └──────────────┘      └─────────────┘     └─────────────┘
```

### Key Pillars
- **Polished Windows 11 GUI:** Built with `customtkinter`, featuring dark/light styling, fluid navigation, real-time video canvas, quality meters, challenge countdowns, and security log inspection.
- **Explicit Camera Safety:** Hardware webcam is never opened automatically without deliberate user interaction. A visible green/gray header badge displays `● Camera Active (Dev X)` or `○ Camera Offline`, and the camera is guaranteed to release on window close or tab change.
- **Fail-Closed Security Policy:** Unlock approval strictly requires **BOTH** identity confirmation (SFace cosine similarity $\ge 0.415$) **AND** completed dynamic challenge-response liveness within a cryptographically isolated session.
- **Biometric Privacy by Design:** Only 128-dimensional floating-point embeddings are stored; raw webcam frames are discarded from memory immediately after enrollment. Template files in `data/registered_face/` and configuration in `config/` are restricted using Windows Access Control Lists (`icacls`).

---

## 2. Desktop Interface Features

Launch the graphical application via:
```powershell
python main.py --gui
# Or: python main.py and choose option [1]
```

The GUI offers 5 dedicated views:

1. **Dashboard:**
   - Real-time status cards: Enrollment Status (Registered / Unregistered), Active Security Engine, Anti-Spoofing Status.
   - Quick operation buttons to jump straight to Secure Unlock or Enrollment.
   - Prototype architecture and hardware security info box.
2. **Guided Enrollment:**
   - Step-by-step guided enrollment across 5 distinct poses (Center Front, Slight Left, Slight Right, Slight Tilt Up, Slight Tilt Down).
   - Real-time quality meters for Sharpness, Illumination, Contrast, and Face Sizing.
   - Multi-sample progress bar with deliberate `[Capture Sample]` button.
3. **Secure Unlock:**
   - Live camera view with facial bounding box and landmark overlays.
   - Dual-status telemetry: Identity similarity meter vs verified threshold, and active liveness challenge prompt with countdown ring.
   - Anti-replay tracking: Prevents cached credentials and enforces rate-limiting lockouts.
4. **Security Audit Log:**
   - Live tabular viewer reading structured events from `logs/security_events.log`.
   - Complete visibility into challenge successes, timeouts, rate-limiting triggers, and presentation attacks.
   - Privacy-safe: Never contains raw imagery, facial vectors, or user names.
5. **Settings & Diagnostics:**
   - Configurable parameters: Camera device index, SFace matching threshold, required challenge count, challenge timeout, rate limit thresholds, and cooldown duration.
   - Biometric template management: Safely wipe enrolled templates with confirmation dialog.
   - Persistent settings storage in `config/settings.json` with ACL restrictions.

---

## 3. Quick Start (PowerShell on Windows 11)

### Prerequisites
- Windows 10/11 64-bit
- CPython 3.13 (or 3.12)
- Connected 2D RGB Webcam

### 1. Activate Environment
```powershell
cd "C:\Users\Bhargav Rathod\OneDrive\Desktop\FaceUnlockAI"
.\.venv313\Scripts\Activate.ps1
```

### 2. Launch the Application
```powershell
# Launch Desktop GUI directly:
python main.py --gui

# Or open interactive CLI launcher:
python main.py
```

### 3. Run Automated Tests
```powershell
# Run full offline test suite (76 unit & integration tests):
python -m pytest tests/ -v

# Run full test suite including live hardware webcam tests (80 tests):
python -m pytest tests/ -v --webcam
```

---

## 4. CLI Tools & Script Reference

In addition to the Desktop GUI, all components can be executed independently via command-line:

| Script | Purpose | Common Options |
|---|---|---|
| [`main.py`](file:///C:/Users/Bhargav%20Rathod/OneDrive/Desktop/FaceUnlockAI/main.py) | Interactive launcher menu | `--gui` (start desktop app), `--cli` (text menu) |
| [`camera_test.py`](file:///C:/Users/Bhargav%20Rathod/OneDrive/Desktop/FaceUnlockAI/camera_test.py) | Hardware webcam diagnostics | `--camera 0`, `--backend dshow`, `--list-all` |
| [`face_detection.py`](file:///C:/Users/Bhargav%20Rathod/OneDrive/Desktop/FaceUnlockAI/face_detection.py) | Standalone detector (YuNet / Haar) | `--backend yunet`, `--min-confidence 0.7` |
| [`face_registration.py`](file:///C:/Users/Bhargav%20Rathod/OneDrive/Desktop/FaceUnlockAI/face_registration.py) | 5-pose biometric enrollment | `--re-enroll`, `--delete`, `--info` |
| [`face_recognition.py`](file:///C:/Users/Bhargav%20Rathod/OneDrive/Desktop/FaceUnlockAI/face_recognition.py) | Identity verification & unlock | `--liveness` (require challenges), `--threshold 0.415` |
| [`liveness_detection.py`](file:///C:/Users/Bhargav%20Rathod/OneDrive/Desktop/FaceUnlockAI/liveness_detection.py) | Standalone challenge-response test | `--challenges 2`, `--timeout 6.0` |

---

## 5. Security & Threat Mitigation Summary

| Threat / Attack Vector | Mitigation in FaceUnlockAI |
|---|---|
| **Static Photo Attack** | Multi-frame micro-motion jitter analysis + randomized dynamic pose challenges |
| **Video Replay Attack** | Nonce-based challenge sequences + short per-challenge timeouts + single-use session tokens |
| **Cut / Subject Swap** | Continuous spatial tracking; frame-to-frame position displacement $>25\%$ fails continuity |
| **Multi-Face Spoofing** | Strict fail-closed policy: more than 1 detected face aborts verification immediately |
| **Session Hijacking / Replay** | Session UUID tokens invalidated (`consume()`) immediately upon outcome |
| **Brute-Force Presentation** | Cooldown lockout after 3 consecutive failures (30-second backoff) |
| **Local File Snooping** | Windows ACLs (`icacls`) restrict template and config access to the owner's Windows SID |

For full security analysis, see [SECURITY_REPORT.md](file:///C:/Users/Bhargav%20Rathod/OneDrive/Desktop/FaceUnlockAI/SECURITY_REPORT.md).

---

## 6. Directory Structure

```
FaceUnlockAI/
├── config/                  # Isolated settings management
│   ├── __init__.py
│   ├── config_manager.py    # JSON config loader, validator, and ACL setter
│   └── settings.json        # Runtime parameters (auto-generated)
├── data/
│   └── registered_face/     # Biometric template storage (owner_template.npz)
├── logs/                    # Audit logging (security_events.log - JSON Lines)
├── models/                  # Official ONNX model weights (YuNet, SFace, Haar)
├── tests/                   # Pytest test suite (80 automated tests)
│   ├── test_face_detection.py
│   ├── test_face_recognition.py
│   ├── test_liveness_detection.py
│   └── test_ui_and_config.py
├── ui/                      # Windows 11 CustomTkinter Desktop GUI
│   ├── __init__.py
│   └── app_gui.py           # 5-view desktop application with camera indicator
├── camera_test.py           # Independent hardware webcam tester
├── face_detection.py        # Face detector module (YuNet + Haar fallback)
├── face_registration.py     # 5-pose guided enrollment engine
├── face_recognition.py      # SFace identity verification & unlock engine
├── liveness_detection.py    # Active challenge-response presentation attack defense
├── main.py                  # Unified launcher (GUI / CLI)
├── requirements.txt         # Verified pinned Python dependencies
├── README.md                # Project documentation
├── SECURITY_REPORT.md       # Comprehensive security threat analysis
└── SETUP.md                 # Detailed environment setup & troubleshooting guide
```

---

## 7. License & Credits

- **Detection Model:** [libfacedetection / YuNet](https://github.com/ShiqiYu/libfacedetection) (Shiqi Yu, Apache 2.0 / OpenCV Zoo).
- **Recognition Model:** [SFace](https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface) (OpenCV Zoo).
- **UI Framework:** [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) (Tom Schimansky, MIT).
