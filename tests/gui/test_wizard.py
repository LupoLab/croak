"""Tests for the wizard shell (:mod:`croak.gui.wizard`).

Navigation, the leave-session confirmation, saved-session restore (every
early-exit path) and the saved-result reuse check. The end-to-end flows live
in ``test_wizard_flow.py``; these target the branches those do not reach.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from croak import save
from croak.gui import stage_preprocess, wizard
from croak.gui.state import WizardState
from croak.gui.wizard import Wizard, _result_mismatch


@pytest.fixture
def w(qtbot):
    wiz = Wizard()
    qtbot.addWidget(wiz)
    return wiz


def _point_at_experiment(state, experiment_h5, experiment):
    p = state.load
    p.frog_path = experiment_h5
    p.ifrog_name, p.lam_name, p.scan_name = "trace", "wavelength", "delay"
    p.lam_unit, p.scan_unit, p.scan_type = "nm", "fs", "delay"
    p.interaction = experiment.interaction


# -- _result_mismatch -----------------------------------------------------------
def _td(trace, omega):
    return SimpleNamespace(trace=trace, grid=SimpleNamespace(omega=omega))


@pytest.mark.parametrize(
    ("saved", "reason"),
    [
        ({}, "the saved result does not store the trace it was fitted to"),
        (
            {"trace_meas": np.ones((2, 2))},
            "the saved result is on a 2x2 grid but this trace regrids to 2x3",
        ),
        (
            {"trace_meas": np.full((2, 3), 2.0)},
            "preprocessing no longer reproduces the trace the result was fitted to",
        ),
        (
            {"trace_meas": np.ones((2, 3)), "omega": np.arange(3.0)},
            "the saved result is on a different frequency grid",
        ),
        (
            {"trace_meas": np.ones((2, 3)), "omega": np.array([1.0, 2.5])},
            "the saved result is on a different frequency grid",
        ),
        (
            {
                "trace_meas": np.ones((2, 3)),
                "omega": np.array([1.0, 2.0]),
                "algorithm": "lm",
            },
            "the saved result came from lm but the loaded settings select copra",
        ),
        ({"trace_meas": np.ones((2, 3)), "omega": np.array([1.0, 2.0])}, ""),
        ({"trace_meas": np.ones((2, 3)), "algorithm": "copra"}, ""),
    ],
)
def test_wizard_result_mismatch_explains_why_a_result_is_refused(saved, reason):
    td = _td(np.ones((2, 3)), np.array([1.0, 2.0]))
    assert _result_mismatch(saved, td, SimpleNamespace(solver="copra")) == reason


def test_wizard_result_mismatch_handles_an_all_zero_trace():
    # a zero peak/span falls back to an absolute tolerance of 1e-9
    td = _td(np.zeros((1, 2)), np.zeros(2))
    saved = {"trace_meas": np.zeros((1, 2)), "omega": np.zeros(2)}
    assert _result_mismatch(saved, td, SimpleNamespace(solver="copra")) == ""


# -- navigation -------------------------------------------------------------------
def test_wizard_go_next_on_an_entry_page_waits_until_it_can_advance(w):
    w.load_simulated()  # nothing chosen yet: cannot advance
    w.go_next()
    assert w.stack.currentWidget() is w.simulated


def test_wizard_go_next_advances_a_ready_stage(w, experiment_h5, experiment):
    w.start_load()
    _point_at_experiment(w.state, experiment_h5, experiment)
    w.stages[0]._load_preview()
    w.go_next()
    assert w.state.stage == 2
    assert w.stack.currentWidget() is w.stages[1]


def test_wizard_go_next_stays_on_a_stage_that_is_not_ready(w):
    w.start_load()  # Stage 1 with no trace loaded
    w.go_next()
    assert w.state.stage == 1


def _answer_discard(monkeypatch, *, leave):
    """Make the leave-session box answer without showing it."""

    def exec_(box):
        label = "Back to menu" if leave else "Cancel"
        button = next(b for b in box.buttons() if b.text().replace("&", "") == label)
        button.click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)


def test_wizard_leave_to_welcome_asks_before_discarding_data(
    w, monkeypatch, experiment_h5, experiment
):
    w.goto_stage(1)
    _point_at_experiment(w.state, experiment_h5, experiment)
    w.stages[0]._load_preview()
    assert w.state.has_data
    _answer_discard(monkeypatch, leave=False)
    w.leave_to_welcome()
    assert w.stack.currentWidget() is w.stages[0]  # cancelled: still on Load
    _answer_discard(monkeypatch, leave=True)
    w.leave_to_welcome()
    assert w.stack.currentIndex() == 0  # the welcome menu
    assert w.state.load_data is not None  # kept until a menu entry is chosen


# -- loading a saved session --------------------------------------------------------
def test_wizard_load_previous_cancelled_keeps_a_fresh_state(w, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    w.state.retrieve.maxiters = 7
    w.load_previous()
    assert w.state.retrieve.maxiters == WizardState().retrieve.maxiters


def test_wizard_load_options_restores_a_synthetic_session(w, monkeypatch, tmp_path):
    src = WizardState()
    src.entry = "synthetic"
    src.retrieve.maxiters = 77
    opts = str(tmp_path / "options.toml")
    save.save_options(src.to_options(), opts)
    (tmp_path / "result.h5").write_bytes(b"")  # a sibling result is looked for
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (opts, ""))
    w.load_previous()
    assert w.state.retrieve.maxiters == 77
    assert w.stack.currentWidget() is w.synthetic
    assert "synthetic generator's settings are not saved" in (
        w.synthetic.status_label.text()
    )


def test_wizard_restore_experimental_without_a_trace_file(w):
    w.state.entry = "experimental"
    w._restore_session()
    assert w.state.stage == 1
    assert w.stages[0].status_label.text() == (
        "Options loaded. Choose a trace file and run the stages."
    )


def test_wizard_restore_experimental_reports_an_unreadable_trace(w, tmp_path):
    w.state.entry = "experimental"
    w.state.load.frog_path = str(tmp_path / "moved.h5")
    w._restore_session()
    assert w.state.stage == 1
    assert (
        w.stages[0]
        .status_label.text()
        .startswith("Options loaded, but reloading the trace failed:")
    )
    assert w.state.load_data is None


def test_wizard_restore_stops_when_preprocess_fails(
    w, monkeypatch, experiment_h5, experiment
):
    def broken(params, data, **kwargs):
        raise RuntimeError("bad window")

    monkeypatch.setattr(stage_preprocess.pipeline, "build_tracedata", broken)
    w.state.entry = "experimental"
    _point_at_experiment(w.state, experiment_h5, experiment)
    w._restore_session()
    assert w.state.load_data is not None
    assert w.state.tracedata is None
    assert w.state.stage == 3
    assert w.stages[2].status_label.text() == (
        "Options loaded and data restored; preprocess did not complete."
    )


def test_wizard_restore_simulated_without_a_scan_file(w):
    w.state.entry = "simulated"
    w._restore_session()
    assert w.stack.currentWidget() is w.simulated
    assert w.simulated.status_label.text() == (
        "Options loaded. Choose a simulated scan file and run the stages."
    )


def test_wizard_restore_simulated_reports_an_unreadable_scan(w, tmp_path):
    w.state.entry = "simulated"
    w.state.simulated.frog_path = str(tmp_path / "moved.h5")
    w._restore_session()
    assert w.stack.currentWidget() is w.simulated
    assert w.simulated.status_label.text().startswith(
        "Options loaded, but reloading the simulated scan failed:"
    )


def test_wizard_restore_simulated_reports_a_failed_raw_trace(
    w, monkeypatch, simulated_scan_h5
):
    def broken(params):
        raise ValueError("no native grid")

    monkeypatch.setattr(wizard, "assemble_simulated_tracedata", broken)
    w.state.entry = "simulated"
    w.state.simulated.frog_path = simulated_scan_h5
    w.state.simulated.raw_direct = True
    w._restore_session()
    assert w.stack.currentWidget() is w.simulated
    assert w.simulated.status_label.text() == (
        "Options loaded, but the raw simulated trace failed: no native grid"
    )
    assert w.state.tracedata is None


def test_wizard_reload_saved_result_refuses_without_trace_or_file(w, tmp_path):
    assert w._reload_saved_result(str(tmp_path / "result.h5")) is False
    w.state.tracedata = SimpleNamespace(grid=None)
    assert w._reload_saved_result(str(tmp_path / "missing.h5")) is False
