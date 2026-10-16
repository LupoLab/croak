"""The ``croak.gui`` package entry points: run_wizard, main, lazy exports, -m."""

from __future__ import annotations

import runpy
import sys

import pytest

pytest.importorskip("PyQt6")

import croak.gui as gui
from croak.gui import wizard as wizard_mod


def test_run_wizard_shows_maximised_and_runs_the_event_loop(qapp, monkeypatch):
    shown = []
    monkeypatch.setattr(
        wizard_mod.Wizard, "showMaximized", lambda self: shown.append(self)
    )
    monkeypatch.setattr(type(qapp), "exec", lambda self: 42)
    assert gui.run_wizard(["croak-gui"]) == 42
    assert len(shown) == 1
    # apply_identity ran: the application carries croak's name
    assert qapp.applicationName() == "croak"
    shown[0].deleteLater()


def test_main_delegates_to_run_wizard(monkeypatch):
    monkeypatch.setattr(gui, "run_wizard", lambda argv=None: 7)
    assert gui.main() == 7


def test_run_wizard_without_qt_says_how_to_install(monkeypatch):
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", None)
    with pytest.raises(ImportError, match=r"croak\[gui\]"):
        gui.run_wizard([])


def test_lazy_exports_resolve_and_unknown_names_raise():
    from croak.gui.state import WizardState
    from croak.gui.wizard import Wizard

    assert gui.Wizard is Wizard
    assert gui.WizardState is WizardState
    with pytest.raises(AttributeError, match="no_such_name"):
        _ = gui.no_such_name


def test_python_dash_m_runs_the_wizard(monkeypatch):
    monkeypatch.setattr(gui, "run_wizard", lambda: 3)
    # runpy warns when re-running a module that is already imported.
    monkeypatch.delitem(sys.modules, "croak.gui.__main__", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("croak.gui", run_name="__main__")
    assert exit_info.value.code == 3


def test_importing_the_main_module_does_not_launch(monkeypatch):
    import importlib

    launched = []
    monkeypatch.setattr(gui, "run_wizard", lambda: launched.append(1))
    monkeypatch.delitem(sys.modules, "croak.gui.__main__", raising=False)
    importlib.import_module("croak.gui.__main__")
    assert launched == []
