"""The Stage base page: help dialog, entry-page hooks and the shared width."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from croak.gui import settings
from croak.gui.base import Stage
from croak.gui.state import WizardState


class _HelpedStage(Stage):
    HELP_TITLE = "Demo"
    HELP_INTRO = "Intro"
    HELP = [("Section", [("Label", "Text")])]


def test_stage_without_help_has_no_dialog(qtbot):
    stage = Stage(WizardState())
    qtbot.addWidget(stage)
    assert stage.show_help() is None


def test_stage_help_dialog_is_reused(qtbot):
    stage = _HelpedStage(WizardState())
    qtbot.addWidget(stage)
    first = stage.show_help()
    assert first is not None and first.isVisible()
    assert stage.show_help() is first
    first.close()


def test_stage_default_hooks(qtbot):
    stage = Stage(WizardState())
    qtbot.addWidget(stage)
    assert stage.next_label() == "Next →"
    assert stage.can_advance() is True
    assert stage.advance() is None
    assert stage.on_enter() is None


def test_stage_shows_with_the_shared_controls_width(qtbot):
    settings.set_controls_width(300)
    stage = Stage(WizardState())
    qtbot.addWidget(stage)
    stage.resize(1200, 600)
    stage.show()
    qtbot.waitExposed(stage)
    assert stage.splitter.sizes()[0] == 300
    # Showing again with the width already applied changes nothing.
    stage.hide()
    stage.show()
    assert stage.splitter.sizes()[0] == 300
    # A rested divider drag is remembered for every stage.
    stage.splitter.setSizes([280, 900])
    stage._save_controls_width()
    assert settings.controls_width() == 280
