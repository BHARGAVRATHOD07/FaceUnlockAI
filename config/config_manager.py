"""
config_manager.py — Configuration Management for FaceUnlockAI
---------------------------------------------------------------
Loads, validates, and persists user settings in config/settings.json,
isolated from application code and biometric templates.
Applies restrictive Windows ACLs to config directory.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_DIR = PROJECT_ROOT / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.json"


@dataclass
class AppSettings:
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    cosine_threshold: float = 0.380
    ambiguity_margin: float = 0.035
    liveness_enabled: bool = True
    num_liveness_challenges: int = 2
    challenge_timeout_sec: float = 5.0
    max_failed_attempts: int = 3
    cooldown_duration_sec: float = 30.0
    save_raw_images: bool = False
    ui_theme: str = "dark"

    def validate(self) -> None:
        if self.camera_index < 0:
            self.camera_index = 0
        self.cosine_threshold = max(0.20, min(0.60, float(self.cosine_threshold)))
        self.ambiguity_margin = max(0.01, min(0.10, float(self.ambiguity_margin)))
        self.num_liveness_challenges = max(1, min(4, int(self.num_liveness_challenges)))
        self.challenge_timeout_sec = max(2.0, min(15.0, float(self.challenge_timeout_sec)))
        self.max_failed_attempts = max(1, min(10, int(self.max_failed_attempts)))
        self.cooldown_duration_sec = max(5.0, min(120.0, float(self.cooldown_duration_sec)))
        if self.ui_theme not in ("dark", "light", "system"):
            self.ui_theme = "dark"


def load_settings() -> AppSettings:
    """Loads settings from config/settings.json or creates default if missing."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_FILE.exists():
        settings = AppSettings()
        save_settings(settings)
        return settings

    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        settings = AppSettings(**{k: v for k, v in data.items() if k in AppSettings.__dataclass_fields__})
        settings.validate()
        return settings
    except Exception as exc:
        print(f"[config] Warning: Failed to parse settings file ({exc}), falling back to defaults.", file=sys.stderr)
        return AppSettings()


def save_settings(settings: AppSettings) -> bool:
    """Saves settings to config/settings.json and secures permissions."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    settings.validate()
    try:
        SETTINGS_FILE.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
        from face_registration import secure_directory_acls
        secure_directory_acls(CONFIG_DIR)
        return True
    except Exception as exc:
        print(f"[config] Error saving settings: {exc}", file=sys.stderr)
        return False
