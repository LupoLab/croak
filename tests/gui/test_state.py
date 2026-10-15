"""WizardState properties and stage navigation."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from croak.gui.state import WizardState


def test_geometry_and_truth_come_from_the_load():
    state = WizardState()
    assert state.geometry is None and state.truth is None
    state.load_data = {"geometry": "g", "truth": "t"}
    assert state.geometry == "g" and state.truth == "t"


def test_stage_setter_clamps_and_emits_only_on_change(qtbot):
    state = WizardState()
    seen = []
    state.stage_changed.connect(seen.append)
    state.stage = 1  # unchanged: no signal
    state.stage = 99  # clamped to the last stage
    state.stage = 6  # unchanged again
    assert seen == [6]
    assert state.stage == 6
