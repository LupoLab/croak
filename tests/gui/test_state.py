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


def test_invalidate_tracedata_drops_the_trace_and_what_was_built_on_it(qtbot):
    state = WizardState()
    emitted = []
    state.tracedata_changed.connect(lambda: emitted.append(True))
    state.invalidate_tracedata()  # nothing prepared: nothing to drop or announce
    assert emitted == []
    state.load_data = {"trace": 1}
    state.tracedata, state.result, state.processed = "td", "result", "processed"
    state.invalidate_tracedata()
    assert (state.tracedata, state.result, state.processed) == (None, None, None)
    assert state.load_data == {"trace": 1}
    assert emitted == [True]
