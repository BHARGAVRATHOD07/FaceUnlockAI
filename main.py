"""
main.py — FaceUnlockAI Unified Application Launcher
------------------------------------------------------
Interactive CLI menu and unified runner for FaceUnlockAI on Windows 11.

Options:
  1. Camera Hardware Diagnostic (camera_test.py)
  2. Live Face Detection Preview (face_detection.py)
  3. Device Owner Registration & Enrollment (face_registration.py)
  4. Live Identity Verification & Simulated Unlock (face_recognition.py)
  5. Re-enroll Owner
  6. Delete Biometric Enrollment Data
  7. Run Automated Pytest Test Suite
  0. Exit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
PYTHON_EXE = sys.executable


def run_subcommand(script_name: str, args: list[str] | None = None) -> int:
    cmd = [PYTHON_EXE, str(PROJECT_ROOT / script_name)]
    if args:
        cmd.extend(args)
    return subprocess.run(cmd).returncode


def print_banner() -> None:
    print("\n" + "=" * 62)
    print("      FaceUnlockAI — Biometric Face Verification System      ")
    print("=" * 62)
    print("  Environment: Windows 11 | Python 3.13 | OpenCV + SFace")
    print("  Privacy:     100% Local Processing | Zero Cloud Egress")
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceUnlockAI Unified Launcher")
    parser.add_argument("--gui", action="store_true", help="Launch polished Windows 11 Desktop GUI")
    parser.add_argument("--enroll", action="store_true", help="Launch face enrollment")
    parser.add_argument("--verify", action="store_true", help="Launch live identity verification")
    parser.add_argument("--detect", action="store_true", help="Launch face detection preview")
    parser.add_argument("--camera-test", action="store_true", help="Run camera diagnostic")
    parser.add_argument("--secure-unlock", action="store_true", help="Launch full secure unlock with liveness")
    parser.add_argument("--liveness-test", action="store_true", help="Run standalone liveness challenge test")
    parser.add_argument("--delete", action="store_true", help="Delete enrolled template")
    parser.add_argument("--re-enroll", action="store_true", help="Re-enroll owner")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")

    args = parser.parse_args()

    # Route direct CLI flags
    if args.gui:
        from ui.app_gui import launch_app
        launch_app()
        sys.exit(0)
    if args.camera_test:
        sys.exit(run_subcommand("camera_test.py", ["--camera", str(args.camera)]))
    if args.detect:
        sys.exit(run_subcommand("face_detection.py", ["--camera", str(args.camera)]))
    if args.enroll:
        sys.exit(run_subcommand("face_registration.py", ["--camera", str(args.camera)]))
    if args.re_enroll:
        sys.exit(run_subcommand("face_registration.py", ["--re-enroll", "--camera", str(args.camera)]))
    if args.delete:
        sys.exit(run_subcommand("face_registration.py", ["--delete"]))
    if args.verify:
        sys.exit(run_subcommand("face_recognition.py", ["--camera", str(args.camera)]))
    if args.secure_unlock:
        sys.exit(run_subcommand("face_recognition.py", ["--liveness", "--camera", str(args.camera)]))
    if args.liveness_test:
        sys.exit(run_subcommand("liveness_detection.py", ["--camera", str(args.camera)]))
    if args.test:
        cmd = [PYTHON_EXE, "-m", "pytest", "tests/", "-v"]
        sys.exit(subprocess.run(cmd).returncode)

    # Interactive Menu
    while True:
        print_banner()
        from face_registration import FaceEnrollmentManager
        is_enrolled = FaceEnrollmentManager.is_enrolled()
        status_text = "ENROLLED" if is_enrolled else "NOT ENROLLED"
        print(f"  Owner Status: [{status_text}]\n")

        print("  [1] Launch Desktop GUI Application (ui/app_gui.py)")
        print("  [2] Camera Hardware Diagnostic (camera_test.py)")
        print("  [3] Face Detection Preview (face_detection.py)")
        print("  [4] Register / Enroll Device Owner (face_registration.py)")
        print("  [5] Quick Identity Verification Preview (face_recognition.py)")
        print("  [6] Secure Face Unlock with Liveness Challenge (Full Security)")
        print("  [7] Standalone Liveness Challenge Test (liveness_detection.py)")
        print("  [8] Re-enroll Device Owner")
        print("  [9] Delete Enrollment Data (Purge Biometrics)")
        print("  [10] Run Automated Pytest Test Suite")
        print("  [0] Exit")
        print("-" * 62)

        try:
            choice = input("Select an option [0-10]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice == "1":
            from ui.app_gui import launch_app
            launch_app()
        elif choice == "2":
            run_subcommand("camera_test.py", ["--camera", str(args.camera)])
        elif choice == "3":
            run_subcommand("face_detection.py", ["--camera", str(args.camera)])
        elif choice == "4":
            run_subcommand("face_registration.py", ["--camera", str(args.camera)])
        elif choice == "5":
            run_subcommand("face_recognition.py", ["--camera", str(args.camera)])
        elif choice == "6":
            run_subcommand("face_recognition.py", ["--liveness", "--camera", str(args.camera)])
        elif choice == "7":
            run_subcommand("liveness_detection.py", ["--camera", str(args.camera)])
        elif choice == "8":
            run_subcommand("face_registration.py", ["--re-enroll", "--camera", str(args.camera)])
        elif choice == "9":
            run_subcommand("face_registration.py", ["--delete"])
        elif choice == "10":
            subprocess.run([PYTHON_EXE, "-m", "pytest", "tests/", "-v"])
        elif choice in ("0", "q", "exit"):
            print("\nGoodbye!")
            break
        else:
            print(f"Invalid option: {choice}. Please enter a number between 0 and 10.")


if __name__ == "__main__":
    main()
