"""Tests for the Preprocess stage (:mod:`croak.gui.stage_preprocess`).

Driven directly on a :class:`WizardState`; the off-thread worker path is
exercised with real :class:`~croak.gui.worker.PreprocessWorker` threads.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import sip

from croak.gui import stage_preprocess
from croak.gui.stage_load import seed_preproc_defaults
from croak.gui.stage_preprocess import StagePreprocess
from croak.gui.state import WizardState
from croak.session.pipeline import assemble_load_data, build_tracedata


@pytest.fixture
def stage(qtbot):
    s = StagePreprocess(WizardState())
    qtbot.addWidget(s)
    return s


@pytest.fixture
def loaded(stage, experiment_h5, experiment):
    """The stage with the synthetic experimental trace loaded and seeded."""
    p = stage.state.load
    p.frog_path = experiment_h5
    p.ifrog_name, p.lam_name, p.scan_name = "trace", "wavelength", "delay"
    p.lam_unit, p.scan_unit, p.scan_type = "nm", "fs", "delay"
    p.interaction = experiment.interaction
    stage.state.set_load_data(assemble_load_data(p))
    seed_preproc_defaults(stage.state)
    return stage


def test_stage_preprocess_without_data_asks_for_a_trace(stage):
    for action in (
        stage.on_enter,
        stage.apply,
        stage._reset_defaults,
        stage._auto_tau_exclude,
    ):
        stage.set_status("")
        action()
        assert stage.status_label.text() == "Load a trace first."
    assert stage.state.tracedata is None


def test_stage_preprocess_range_handlers_store_values_and_schedule(stage):
    handlers = {
        stage._on_lam_range: ("lam_min_nm", "lam_max_nm"),
        stage._on_tau_range: ("tau_min_fs", "tau_max_fs"),
        stage._on_lamm_range: ("lamm_min_nm", "lamm_max_nm"),
        stage._on_taum_range: ("taum_min_fs", "taum_max_fs"),
    }
    p = stage.state.preproc
    for handler, (lo_attr, hi_attr) in handlers.items():
        stage._timer.stop()
        handler(1.0, 2.0)
        assert (getattr(p, lo_attr), getattr(p, hi_attr)) == (1.0, 2.0)
        assert stage._timer.isActive()


def test_stage_preprocess_schedule_is_silent_while_seeding(stage):
    stage._timer.stop()
    stage._seeding = True
    stage._schedule()
    assert not stage._timer.isActive()


def test_stage_preprocess_fringe_removers_are_mutually_exclusive(stage):
    stage.fringes_check.setChecked(True)
    stage.defringe_check.setChecked(True)
    assert not stage.fringes_check.isChecked()
    assert stage.state.preproc.defringe and not stage.state.preproc.filter_fringes
    stage.fringes_check.setChecked(True)
    assert not stage.defringe_check.isChecked()
    assert stage.state.preproc.filter_fringes and not stage.state.preproc.defringe
    # unticking with the other already off leaves it off
    stage.fringes_check.setChecked(False)
    assert not stage.defringe_check.isChecked()


def test_stage_preprocess_auto_tau_exclude_reports_failure(loaded, monkeypatch):
    def refuse(trace, tau):
        raise ValueError("flat marginal")

    monkeypatch.setattr(stage_preprocess, "suggested_tau_exclude", refuse)
    before = loaded.baseline_tauex_spin.value()
    loaded._auto_tau_exclude()
    assert loaded.status_label.text() == (
        "Could not set the hold-out automatically: flat marginal"
    )
    assert loaded.baseline_tauex_spin.value() == before


def test_stage_preprocess_auto_tau_exclude_converts_position_scans(loaded, monkeypatch):
    seen = {}

    def record(trace, tau):
        seen["tau"] = tau
        return 30e-15

    monkeypatch.setattr(stage_preprocess, "suggested_tau_exclude", record)
    data = dict(loaded.state.load_data)
    data["input_unit"] = "position"
    data["scanaxis"] = data["scanaxis"] * 0 + 1.5e-6  # 1.5 µm of stage travel
    loaded.state.load_data = data
    loaded._auto_tau_exclude()
    # τ = 2z/c, so 1.5 µm of travel is 10 fs of delay
    assert seen["tau"][0] == pytest.approx(10.007e-15, rel=1e-3)
    assert loaded.baseline_tauex_spin.value() == pytest.approx(30.0)


def test_stage_preprocess_blocking_apply_reports_failure(loaded, monkeypatch):
    def broken(params, data, **kw):
        raise RuntimeError("bad window")

    monkeypatch.setattr(stage_preprocess.pipeline, "build_tracedata", broken)
    loaded.apply(block=True)
    assert loaded.status_label.text() == "Preprocess failed: bad window"
    assert loaded.state.tracedata is None


def test_stage_preprocess_on_enter_computes_once(loaded):
    loaded.on_enter()
    td = loaded.state.tracedata
    assert td is not None
    assert loaded.status_label.text().startswith("Regridded to")
    loaded.on_enter()  # a trace exists: re-seeds the widgets, no recompute
    assert loaded.state.tracedata is td
    # a recompute with the view already built updates the artists in place
    assert loaded._view.attached
    figure_axes = list(loaded.canvas.figure.axes)
    loaded.apply(block=True)
    assert loaded.state.tracedata is not td
    assert loaded.canvas.figure.axes == figure_axes


def test_stage_preprocess_reset_restores_defaults(loaded):
    loaded.state.preproc.tau_min_fs += 5.0
    loaded._reset_defaults()
    assert loaded.state.preproc.tau_min_fs == loaded.state.preproc_defaults.tau_min_fs
    assert loaded.state.tracedata is not None


def test_stage_preprocess_async_preview_then_exact_refine(loaded, qtbot):
    loaded.apply()
    assert loaded.status_label.text() == "Preprocessing…"
    # the fast preview is drawn, then the exact refinement replaces the trace
    qtbot.waitUntil(lambda: not loaded._workers, timeout=20000)
    td = loaded.state.tracedata
    assert td is not None
    exact = build_tracedata(loaded._pending[0], loaded._pending[1])
    assert td.trace.shape == exact.trace.shape


def test_stage_preprocess_ignores_superseded_and_reports_failed_results(loaded):
    loaded._req_id = 5
    loaded._on_preprocess_done(object(), 4, True)  # stale: ignored
    loaded._on_preprocess_failed("old error", 4)
    assert loaded.state.tracedata is None
    assert loaded.status_label.text() == ""
    loaded._on_preprocess_failed("new error", 5)
    assert loaded.status_label.text() == "Preprocess failed: new error"


def test_stage_preprocess_fast_result_without_pending_does_not_refine(loaded):
    td = build_tracedata(loaded.state.preproc, loaded.state.load_data)
    loaded._pending = None
    loaded._on_preprocess_done(td, loaded._req_id, True)
    assert loaded.state.tracedata is td
    assert loaded._workers == []


def test_stage_preprocess_replot_without_a_trace_only_attaches(loaded):
    loaded.canvas.render(loaded._replot_view)
    assert loaded._view.attached
    assert loaded.state.tracedata is None


def test_stage_preprocess_discard_worker_tolerates_destroyed_workers(loaded):
    worker = stage_preprocess.PreprocessWorker(
        loaded.state.preproc, loaded.state.load_data, 1, parent=loaded
    )
    loaded._workers.append(worker)
    sip.delete(worker)  # destroyed with its stage before the queued slot ran
    loaded._discard_worker(worker)
    assert loaded._workers == []
    # a worker no longer tracked (and still alive) is just scheduled for deletion
    other = stage_preprocess.PreprocessWorker(
        loaded.state.preproc, loaded.state.load_data, 2, parent=loaded
    )
    loaded._discard_worker(other)
    assert not sip.isdeleted(other)


def test_stage_preprocess_baseline_toggle_gates_its_controls(stage):
    dependents = (
        stage.baseline_smooth_spin,
        stage.baseline_ratio_spin,
        stage.baseline_tauex_spin,
        stage.baseline_tauex_auto,
        stage.baseline_clip_check,
    )
    initial = stage.baseline_check.isChecked()
    for checked in (not initial, initial):  # each a real toggle
        stage.baseline_check.setChecked(checked)
        assert stage.state.preproc.baseline is checked
        assert all(w.isEnabled() is checked for w in dependents)
