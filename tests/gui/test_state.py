"""WizardState properties."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from croak.gui.state import WizardState


def test_geometry_and_truth_come_from_the_load():
    state = WizardState()
    assert state.geometry is None and state.truth is None
    state.load_data = {"geometry": "g", "truth": "t"}
    assert state.geometry == "g" and state.truth == "t"
