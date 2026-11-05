"""The Stage base page: help dialog, entry-page hooks and the sheet layout."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QLineEdit

from croak.gui.base import Stage
from croak.gui.canvas import group
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


def test_stage_builds_on_the_sheet_layout(qtbot):
    stage = _HelpedStage(WizardState())
    qtbot.addWidget(stage)
    # Sections go in the form's column; the ready line is the status.
    assert stage.controls is stage.form.content
    assert stage.status_label is stage.form.ready
    stage.set_status("Loaded.")
    assert stage.form.ready.text() == "Loaded." and stage.form.ready.state == "ok"
    stage.set_status("Failed: no file", "error")
    assert stage.form.ready.property("error") is True
    plot, strip = QLabel("plot"), QLabel("strip")
    stage.set_plot_area(plot, strip)
    assert stage.isAncestorOf(plot) and stage.isAncestorOf(strip)


def test_stage_help_reaches_the_section_rows(qtbot):
    stage = _HelpedStage(WizardState())
    qtbot.addWidget(stage)
    section, form = group("Section")
    field = QLineEdit()
    form.addRow("Label", field)
    stage.controls.addWidget(section)
    assert stage.apply_help() == {("Section", "Label")}
    assert field.toolTip() == "Text"
