"""
camera_test.py — FaceUnlockAI Phase 1
--------------------------------------
Tests webcam availability and frame delivery independently of any face-detection
logic.  Run this first to confirm the camera is accessible before running
face_detection.py.

SECURITY NOTE:
  - Camera is opened only while this script is running.
  - No frames are saved to disk.
  - This tool does NOT perform identity verification or unlock Windows.

Usage:
    .venv313\\Scripts\\Activate.ps1
    python camera_test.py [--index INDEX]

Exit:
    Press Q or Escape to quit.
"""

import argparse
import sys
import time

import cv2

# ─── constants ────────────────────────────────────────────────────────────────
WINDOW_TITLE = "FaceUnlockAI – Camera Test"
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_OK = (0, 220, 80)        # green
COLOR_WARN = (0, 165, 255)     # amber
COLOR_ERR = (0, 0, 230)        # red (BGR)
COLOR_TEXT = (255, 255, 255)   # white

# Camera backend preference on Windows
_WIN_BACKENDS = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _overlay_text(frame, lines, start_y=30, scale=0.6, thickness=2):
    """Render a list of (text, color) tuples onto *frame* in-place."""
    y = start_y
    for text, color in lines:
        # thin black shadow for readability on any background
        cv2.putText(frame, text, (11, y + 1), FONT, scale, (0, 0, 0), thickness + 1,
                    cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), FONT, scale, color, thickness, cv2.LINE_AA)
        y += 28


def _try_open(index: int) -> cv2.VideoCapture | None:
    """
    Attempt to open camera *index* trying Windows-preferred backends in order.
    Returns an opened VideoCapture or None.
    """
    for backend in _WIN_BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


# ─── main ─────────────────────────────────────────────────────────────────────

def main(camera_index: int = 0) -> int:
    """
    Open the webcam, display its live feed with frame-rate and resolution info.

    Returns 0 on clean exit, 1 on error.
    """
    print(f"[camera_test] Attempting to open camera index {camera_index} …")

    cap = _try_open(camera_index)

    if cap is None:
        print(
            f"\n[ERROR] Cannot open camera index {camera_index}.\n"
            "Possible causes:\n"
            "  1. No webcam is connected or recognised by Windows.\n"
            "  2. Another application has exclusive access to the camera.\n"
            "  3. Windows camera privacy setting is OFF.\n"
            "     → Settings > Privacy & Security > Camera > Allow apps to access your camera.\n"
            "  4. Wrong camera index. Try --index 1 (or 2) if you have multiple cameras.\n"
        )
        return 1

    # Report actual capture properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_requested = cap.get(cv2.CAP_PROP_FPS)
    print(f"[camera_test] Camera opened: {width}x{height} @ {fps_requested:.0f} fps")
    print("[camera_test] Press Q or Escape to exit.")

    frame_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 30  # ~1 s at 30 fps before giving up

    t_start = time.perf_counter()
    fps_display = 0.0
    fps_update_interval = 1.0   # seconds between FPS recalculations
    last_fps_time = t_start

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(
                        f"\n[ERROR] Camera stopped delivering frames after "
                        f"{consecutive_failures} consecutive failures.\n"
                        "The camera may have been disconnected."
                    )
                    break
                # brief wait before retrying
                time.sleep(0.03)
                continue

            consecutive_failures = 0
            frame_count += 1

            # mirror for natural (selfie) orientation
            frame = cv2.flip(frame, 1)

            # update FPS every second
            now = time.perf_counter()
            elapsed = now - last_fps_time
            if elapsed >= fps_update_interval:
                fps_display = frame_count / (now - t_start)
                last_fps_time = now

            h, w = frame.shape[:2]

            # ── HUD overlay ──
            info_lines = [
                (f"Resolution : {w} x {h}", COLOR_TEXT),
                (f"FPS        : {fps_display:.1f}", COLOR_TEXT),
                (f"Frames     : {frame_count}", COLOR_TEXT),
                ("Camera     : OK", COLOR_OK),
                ("Press Q or Esc to exit", COLOR_WARN),
            ]
            _overlay_text(frame, info_lines, start_y=30)

            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 27 = Escape
                print("\n[camera_test] Exit requested by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed_total = time.perf_counter() - t_start
    print(
        f"[camera_test] Finished. {frame_count} frames captured "
        f"in {elapsed_total:.1f} s."
    )
    return 0


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FaceUnlockAI – Camera availability test"
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Camera device index (default: 0)"
    )
    args = parser.parse_args()
    sys.exit(main(camera_index=args.index))
