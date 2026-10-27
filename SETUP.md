# FaceUnlockAI - Environment Setup Guide

## System Requirements

| Item | Required | Installed |
|---|---|---|
| OS | Windows 10/11 (64-bit) | Windows 11 |
| Python | 3.13.x CPython (MSC build) | 3.13.7 |
| Virtual Environment | `.venv313` | Created |

> **IMPORTANT - OneDrive Warning:**
> This project lives inside OneDrive. Virtual environments with large binary files (.pyd, .dll)
> can be corrupted by OneDrive Files-On-Demand. The .venv313 directory has a .gitignore
> that excludes all its files from sync, but OneDrive may still upload them.
> If cv2 or numpy fail to import after a reboot, reinstall the packages (see Troubleshooting).

---

## Quick Start (PowerShell)

### 1. Navigate to the project

```powershell
cd "C:\Users\Bhargav Rathod\OneDrive\Desktop\FaceUnlockAI"
```

### 2. Activate the virtual environment

```powershell
.\.venv313\Scripts\Activate.ps1
```

You will see `(.venv313)` prefix in your prompt.

### 3. Verify the environment (one-time check)

```powershell
python -c "import cv2, numpy, mediapipe; print('cv2:', cv2.__version__, '| numpy:', numpy.__version__, '| mediapipe:', mediapipe.__version__)"
```

Expected output:
```
cv2: 4.11.0 | numpy: 2.2.6 | mediapipe: 1.0.1
```

---

## Phase 1 - Run Scripts

### Camera Test (verify webcam access)

```powershell
.\.venv313\Scripts\Activate.ps1
python camera_test.py
```

Press **Q** or **Escape** to close the camera window.

### Face Detection (MediaPipe BlazeFace with Haar Fallback)

```powershell
.\.venv313\Scripts\Activate.ps1

# Run with primary detector (MediaPipe BlazeFace Tasks API)
python face_detection.py

# Or force OpenCV Haar Cascade backend:
python face_detection.py --backend haar

# Switch camera device index or resolution if needed:
python face_detection.py --camera 0 --width 1280 --height 720
```

Press **Q** or **Escape** to exit.

### Automated Unit & Integration Tests

```powershell
.\.venv313\Scripts\Activate.ps1

# Run test suite (models, error handling, synthetic frame checks):
python -m pytest tests/test_face_detection.py -v

# Run test suite including live hardware webcam tests:
python -m pytest tests/test_face_detection.py -v --webcam
```

---

## Phase 2 — Face Registration & Identity Verification

### Unified Application Launcher

Launch the interactive console menu to access all features:

```powershell
.\.venv313\Scripts\Activate.ps1
python main.py
```

### 1. Register / Enroll Device Owner

Requires deliberate user action. Guided 5-pose sample collection with real-time blur, brightness, and sizing checks:

```powershell
.\.venv313\Scripts\Activate.ps1
python face_registration.py
```

- **SPACE / ENTER:** Capture aligned sample
- **R:** Restart enrollment
- **Q / ESC:** Cancel enrollment

Optional flags:
```powershell
# Re-enroll owner (overwriting existing template):
python face_registration.py --re-enroll

# Delete enrolled biometric template:
python face_registration.py --delete

# View current template metadata:
python face_registration.py --info

# Explicit opt-in to store raw face sample images (disabled by default for privacy):
python face_registration.py --save-raw-images
```

### 2. Live Identity Verification (Simulated Unlock)

Runs real-time face alignment, SFace feature extraction, cosine similarity comparison, and safety lockouts:

```powershell
.\.venv313\Scripts\Activate.ps1
python face_recognition.py
```

- **Q / ESC:** Exit verification
- **E:** Switch to enrollment
- **D:** Delete current enrollment

---

## Phase 4 — Windows 11 Desktop GUI & Configuration

### 1. Launch the Desktop Application

Launch the modern CustomTkinter Windows 11 interface:

```powershell
.\.venv313\Scripts\Activate.ps1

# Launch GUI directly:
python main.py --gui

# Or run interactive launcher and choose option [1]:
python main.py
```

**Desktop Views:**
- **Dashboard:** Biometric system status, owner registration card, quick actions, and prototype disclaimer.
- **Guided Enrollment:** 5-pose sample capture with real-time blur, illumination, contrast, and sizing gauges.
- **Secure Unlock:** Live webcam canvas with facial bounding boxes, SFace cosine similarity score meter, and active randomized liveness challenge prompts with countdown timer.
- **Audit Logs:** Live viewer for structured security events recorded in `logs/security_events.log`.
- **Settings:** Configurable camera index, SFace threshold, challenge requirements, rate-limiting rules, and template deletion with confirmation.

### 2. Camera Indicator & Safety Guarantees

- **Visible Camera Pill:** The top-right header permanently displays `● Camera Active (Dev X)` in bright emerald green when the webcam hardware is active, and `○ Camera Offline` in muted gray when released.
- **Explicit User Action:** The webcam is never activated on startup or view switching; it only opens upon an explicit click on `[Start Verification]` or `[Start Capture]`.
- **Automatic Camera Teardown:** Navigating between views or closing the desktop window immediately disengages and releases the OpenCV VideoCapture device.

### 3. Comprehensive Automated Tests (80 Tests)

Runs 80 automated unit, integration, and UI tests across all 4 modules:

```powershell
.\.venv313\Scripts\Activate.ps1

# Run offline test suite (76 tests covering models, liveness, security, and UI/config):
python -m pytest tests/ -v

# Run full test suite including live hardware webcam tests (80 tests):
python -m pytest tests/ -v --webcam
```

---

## Phase 3 — Liveness Checking & Presentation-Attack Defense

### 1. Secure Unlock with Liveness Challenge (Full Security)

Runs joint verification requiring BOTH owner identity confirmation AND active completion of randomized head challenges:

```powershell
.\.venv313\Scripts\Activate.ps1

# Run secure unlock with challenge-response liveness:
python face_recognition.py --liveness
```

### 2. Standalone Liveness Challenge Test

Test the randomized head gesture challenge flow independently:

```powershell
.\.venv313\Scripts\Activate.ps1

# Run liveness challenge test:
python liveness_detection.py
```

### 3. Security Audit Logging

Security events, failed presentation attacks, timeouts, and rate-limiting cooldowns are automatically recorded to `logs/security_events.log` in structured JSON Lines format. Zero facial imagery or biometric vectors are ever logged.

See [SECURITY_REPORT.md](file:///C:/Users/Bhargav%20Rathod/OneDrive/Desktop/FaceUnlockAI/SECURITY_REPORT.md) for the complete threat model, attack simulation matrix, and 2D RGB hardware limitations.

---

## Security, Privacy, and Biometric Limitations

1. **Local Processing Only:** All feature extraction and template comparisons run locally via OpenCV and ONNX. No facial imagery or embeddings are uploaded.
2. **Raw Images Discarded:** By default, only mathematical embedding vectors (128-D float32) are stored; raw webcam frames are discarded immediately after enrollment unless the user explicitly opts in with `--save-raw-images`.
3. **Template Protection & Limitations:** Templates in `data/registered_face/` are restricted with Windows ACLs (`icacls`) to the owner's Windows user account. **Note:** File permissions protect against unauthorized local non-admin accounts, but they do NOT protect biometric templates from malware running within the user's own active session.
4. **No Password Storage:** This application never stores Windows passwords, PINs, authentication tokens, or reusable Windows credentials.
5. **Biometric Accuracy Disclaimer:** Automated tests using synthetic/mock inputs verify algorithmic flow, threshold boundaries, and error recovery; they do **NOT** establish biometric spoof resistance against presentation attacks (photos, videos, deepfakes, identical twins). Do not bind this module to Windows Sign-in without hardware IR liveness detection.

## Recreating .venv313 from Scratch

If the environment is ever corrupted, run these PowerShell commands in order:

```powershell
# 1. Delete the broken environment
Remove-Item -Recurse -Force ".venv313"

# 2. Create a fresh environment using the system Python 3.13
& "C:\Program Files\Python313\python.exe" -m venv .venv313

# 3. Activate
.\.venv313\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install numpy first (must be PyPI cp313 wheel, NOT MSYS2)
python -m pip install "numpy==2.2.6"

# 6. Install OpenCV core
python -m pip install "opencv-python==4.11.0.86"

# 7. Install mediapipe (pulls opencv-contrib-python 5.0 as dependency)
python -m pip install "mediapipe==1.0.1" "Pillow==10.4.0"

# 8. Verify
python -c "import cv2, numpy, mediapipe; print('cv2:', cv2.__version__, '| numpy:', numpy.__version__, '| mediapipe:', mediapipe.__version__)"
```

---

## Environment Audit Results

### Root Cause of Existing .venv Problems

The original `.venv` (Python 3.13, still present and untouched) had two issues:

1. **NumPy built with MINGW-W64**: `numpy 1.26.4` was compiled with the MSYS2 toolchain,
   not the official PyPI CPython wheel. This caused runtime crashes with messages:
   `Warning: Numpy built with MINGW-W64 on Windows 64 bits is experimental`
   and `RuntimeWarning: invalid value encountered in exp2`.

2. **Dual OpenCV distributions**: Both `opencv-python 4.11.0.86` and `opencv-contrib-python`
   were installed with incompatible versions in the same environment.

### .venv313 - Verified Clean State

| Package | Version | Wheel Tag | Source |
|---|---|---|---|
| Python | 3.13.7 | MSC v.1944 AMD64 | C:\Program Files\Python313 |
| numpy | 2.2.6 | cp313-cp313-win_amd64 | PyPI (official CPython wheel) |
| opencv-python | 4.11.0.86 | cp37-abi3-win_amd64 | PyPI |
| opencv-contrib-python | 5.0.0.93 | cp37-abi3-win_amd64 | PyPI (mediapipe dependency) |
| mediapipe | 1.0.1 | py3-none-win_amd64 | PyPI |
| Pillow | 10.4.0 | cp313-cp313-win_amd64 | PyPI |

### Why opencv-python AND opencv-contrib-python Are Both Installed

In OpenCV 5.0, the package was split: `opencv-python` provides the core `cv2.pyd` binary,
while `opencv-contrib-python` provides extension modules only. They must coexist.
`mediapipe 1.0.1` requires `opencv-contrib-python` as a dependency.
This is the correct and supported configuration for OpenCV 5.0+.

### MediaPipe API Migration Note

MediaPipe 1.0+ removed the legacy `mp.solutions.*` namespace. Use the Tasks API:

```python
# OLD (mediapipe < 1.0):
# mp.solutions.face_mesh.FaceMesh()

# NEW (mediapipe >= 1.0):
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# FaceDetector example:
options = mp_vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path='face_detector.task')
)
detector = mp_vision.FaceDetector.create_from_options(options)
```

---

## Troubleshooting

### import cv2 fails or AttributeError on __version__

The `cv2.pyd` may have been evicted by OneDrive. Reinstall:

```powershell
.\.venv313\Scripts\Activate.ps1
pip install --force-reinstall "opencv-python==4.11.0.86"
```

### NumPy MINGW warnings appear

You are using the wrong venv. Ensure `.venv313` is activated:

```powershell
.\.venv313\Scripts\Activate.ps1
python -c "import numpy; print(numpy.__file__)"
# Must show: ...\FaceUnlockAI\.venv313\Lib\site-packages\numpy\...
```

### ExecutionPolicy error when activating venv

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## Do NOT Use

- `C:\msys64\ucrt64\bin\python.exe` (Python 3.14, MSYS2 - incompatible numpy wheel)
- `C:\Program Files\Python313\python.exe` directly (always use the venv)
- Installing packages into the existing `.venv` (it has broken MINGW numpy)

---

## Phase 1 Start Command

```powershell
cd "C:\Users\Bhargav Rathod\OneDrive\Desktop\FaceUnlockAI"
.\.venv313\Scripts\Activate.ps1
python camera_test.py
```
