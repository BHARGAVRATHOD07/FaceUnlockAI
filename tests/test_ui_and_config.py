"""
test_ui_and_config.py — Automated Tests for FaceUnlockAI Desktop UI & Config
-----------------------------------------------------------------------------
Tests coverage:
  1. [config]      AppSettings validation, defaults, load/save JSON isolation.
  2. [security]    Separation of config, templates, and logs; absence of credentials.
  3. [ui]          Desktop interface instantiation, view switching, and camera lifecycle.
  4. [indicator]   Camera indicator states (Active vs Offline) and cleanup on exit.
  5. [disclaimer]  Presence of the prominent prototype disclaimer banner.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.config_manager import AppSettings, load_settings, save_settings


# ─── 1. CONFIGURATION ISOLATION TESTS ─────────────────────────────────────────

class TestConfigManagement:
    def test_default_settings_values(self):
        s = AppSettings()
        assert s.camera_index == 0
        assert s.cosine_threshold == 0.380
        assert s.ambiguity_margin == 0.035
        assert s.liveness_enabled is True
        assert s.num_liveness_challenges == 2
        assert s.max_failed_attempts == 3
        assert s.cooldown_duration_sec == 30.0
        assert s.save_raw_images is False
        assert s.ui_theme == "dark"

    def test_validation_clamps_invalid_ranges(self):
        s = AppSettings(
            camera_index=-5,
            cosine_threshold=0.99,   # Out of normal range [0.20, 0.60]
            ambiguity_margin=-0.5,   # Out of normal range [0.01, 0.10]
            num_liveness_challenges=10, # Out of range [1, 4]
            max_failed_attempts=0,   # Out of range [1, 10]
            ui_theme="invalid_theme",
        )
        s.validate()

        assert s.camera_index == 0
        assert s.cosine_threshold <= 0.60
        assert s.ambiguity_margin >= 0.01
        assert s.num_liveness_challenges <= 4
        assert s.max_failed_attempts >= 1
        assert s.ui_theme in ("dark", "light", "system")

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        test_file = tmp_path / "test_settings.json"
        monkeypatch.setattr("config.config_manager.SETTINGS_FILE", test_file)
        monkeypatch.setattr("config.config_manager.CONFIG_DIR", tmp_path)

        settings = AppSettings(
            camera_index=1,
            cosine_threshold=0.410,
            num_liveness_challenges=3,
        )
        success = save_settings(settings)
        assert success is True
        assert test_file.exists()

        loaded = load_settings()
        assert loaded.camera_index == 1
        assert loaded.cosine_threshold == 0.410
        assert loaded.num_liveness_challenges == 3

    def test_settings_isolated_from_code_and_templates(self, tmp_path, monkeypatch):
        test_file = tmp_path / "settings.json"
        monkeypatch.setattr("config.config_manager.SETTINGS_FILE", test_file)
        monkeypatch.setattr("config.config_manager.CONFIG_DIR", tmp_path)

        save_settings(AppSettings())
        data = json.loads(test_file.read_text(encoding="utf-8"))

        # Verify no credentials or sensitive biometric vectors stored in config
        forbidden = ["password", "pin", "credential", "embedding", "template", "secret"]
        for k in data.keys():
            for f in forbidden:
                assert f not in k.lower(), f"Forbidden key in config: {k}"


# ─── 2. DESKTOP GUI COMPONENT TESTS ───────────────────────────────────────────

@pytest.fixture(scope="module")
def app_instance():
    """Creates a single withdrawn (headless) CTk application instance for the module."""
    import customtkinter as ctk
    from ui.app_gui import FaceUnlockApp

    app = FaceUnlockApp()
    app.withdraw()
    yield app
    app.release_camera()
    try:
        app.destroy()
    except Exception:
        pass


class TestDesktopUI:

    def test_window_initialization(self, app_instance):
        assert "FaceUnlockAI" in app_instance.title()
        assert app_instance.is_camera_active is False
        assert app_instance.cap is None
        assert "Offline" in app_instance.camera_indicator_pill.cget("text")

    def test_view_switching_releases_camera(self, app_instance):
        # Mock camera active
        app_instance.is_camera_active = True
        app_instance.cap = MagicMock()

        # Switch to logs view
        app_instance.show_view("logs")
        assert app_instance.is_camera_active is False
        assert app_instance.cap is None
        assert "Offline" in app_instance.camera_indicator_pill.cget("text")

    def test_prototype_warning_badge_present(self, app_instance):
        # Verify warning banner is displayed
        found_warning = False
        for widget in app_instance.winfo_children():
            # Search children inside header frame
            for child in widget.winfo_children():
                if hasattr(child, "cget"):
                    try:
                        text = child.cget("text")
                        if "PROTOTYPE" in text and "WINDOWS HELLO" in text:
                            found_warning = True
                            break
                    except Exception:
                        pass
        assert found_warning is True, "Prototype disclaimer banner must be displayed"

    def test_release_camera_guaranteed_cleanup(self, app_instance):
        mock_cap = MagicMock()
        app_instance.cap = mock_cap
        app_instance.is_camera_active = True

        app_instance.release_camera()

        mock_cap.release.assert_called_once()
        assert app_instance.cap is None
        assert app_instance.is_camera_active is False
        assert "Offline" in app_instance.camera_indicator_pill.cget("text")
