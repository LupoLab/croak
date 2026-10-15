"""Application identity: icon, names and the Windows AppUserModelID."""

from __future__ import annotations

import ctypes
import sys
import types

import pytest

pytest.importorskip("PyQt6")

from croak.gui import branding


def test_app_icon_carries_every_size(qapp):
    icon = branding.app_icon()
    assert not icon.isNull()
    sizes = {s.width() for s in icon.availableSizes()}
    assert set(branding._ICON_SIZES) <= sizes


def test_app_icon_missing_svg_gives_a_null_icon(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(branding, "ICON_PATH", tmp_path / "missing.svg")
    assert branding.app_icon().isNull()


def test_windows_app_id_is_a_no_op_elsewhere(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    # Would raise AttributeError if it reached ctypes.windll.
    monkeypatch.delattr(ctypes, "windll", raising=False)
    branding._set_windows_app_id()


def _fake_windll(setter):
    return types.SimpleNamespace(
        shell32=types.SimpleNamespace(SetCurrentProcessExplicitAppUserModelID=setter)
    )


def test_windows_app_id_is_set_on_windows(monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", _fake_windll(seen.append), raising=False)
    branding._set_windows_app_id()
    assert seen == [branding._APP_USER_MODEL_ID]


def test_windows_app_id_failure_is_cosmetic(monkeypatch):
    def fail(_app_id):
        raise OSError("no shell32")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", _fake_windll(fail), raising=False)
    branding._set_windows_app_id()  # swallowed: a failure here is cosmetic
