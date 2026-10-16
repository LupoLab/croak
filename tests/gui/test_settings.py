"""Remembered window geometry and preferences."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QWidget

from croak.gui import settings


def test_available_size_without_any_screen_falls_back(qtbot, monkeypatch):
    window = QWidget()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "screen", lambda: None)
    monkeypatch.setattr(settings.QGuiApplication, "primaryScreen", lambda: None)
    assert settings._available_size(window) == settings.DEFAULT_WINDOW_SIZE
