"""Tests for the Retrieve stage (:mod:`croak.gui.stage_retrieve`).

The coupled controls, the capability gating, the geometry adopted from a
simulated file, and the completion / read-out / save paths. One short COPRA
retrieval of the synthetic experimental trace (computed once per module) stands
in for a real run; the threaded worker itself is exercised in
``test_wizard_flow.py``.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QFileDialog

from croak.gui import stage_retrieve
from croak.gui.stage_load import seed_preproc_defaults
from croak.gui.stage_retrieve import StageRetrieve
from croak.gui.state import WizardState
from croak.processing import TruthPulse
from croak.session.params import RetrieveParams
from croak.session.pipeline import assemble_load_data, build_tracedata, run_retrieval

DASH = stage_retrieve.DASH


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """``(load_data, preproc, tracedata, result)`` for a short COPRA run."""
    from conftest import make_synthetic_experiment, write_experiment_h5

    exp = make_synthetic_experiment()
    path = write_experiment_h5(tmp_path_factory.mktemp("retrieve") / "f.h5", exp)
    state = WizardState()
    p = state.load
    p.frog_path = path
    p.ifrog_name, p.lam_name, p.scan_name = "trace", "wavelength", "delay"
    p.lam_unit, p.scan_unit, p.scan_type = "nm", "fs", "delay"
    p.interaction = exp.interaction
    state.set_load_data(assemble_load_data(p))
    seed_preproc_defaults(state)
    td = build_tracedata(state.preproc, state.load_data)
    params = RetrieveParams(solver="copra", maxiters=5)
    result = run_retrieval(params, td, rng=np.random.default_rng(0))
    return state.load_data, state.preproc, td, result


@pytest.fixture
def stage(qtbot):
    s = StageRetrieve(WizardState())
    qtbot.addWidget(s)
    return s


@pytest.fixture
def with_trace(stage, run):
    stage.state.tracedata = run[2]
    return stage


# -- coupled controls and gating -----------------------------------------------
def test_stage_retrieve_bspline_basis_locks_phase_only(stage):
    stage.solver_combo.setCurrentText("lbfgs-ad")
    stage.basis_combo.setCurrentText("B-spline")
    assert stage.state.retrieve.phase_basis == "bspline"
    assert stage.mode_combo.currentText() == "Phase only"
    assert not stage.mode_combo.isEnabled()
    assert stage.nodes_spin.isEnabled()
    stage.basis_combo.setCurrentText("Pointwise")
    assert stage.state.retrieve.phase_basis == "pointwise"
    assert stage.mode_combo.isEnabled()
    assert stage.mode_combo.toolTip() == stage._mode_tip


def test_stage_retrieve_dispersive_toggle_gates_thickness_fitting(stage):
    stage.solver_combo.setCurrentText("lbfgs-ad")
    stage._on_dispersive_changed(False)
    assert not stage.fit_thickness_check.isEnabled()
    stage._on_dispersive_changed(True)
    assert stage.state.retrieve.dispersive is True
    assert stage.fit_thickness_check.isEnabled()


def test_stage_retrieve_unknown_solver_leaves_the_gating_alone(stage):
    enabled = stage.reg_box.isEnabled()
    stage.state.retrieve.solver = "no-such-solver"
    stage._update_solver_caps()
    assert stage.reg_box.isEnabled() is enabled


def test_stage_retrieve_zero_values_do_not_recouple_npoints(stage):
    p = stage.state.retrieve
    npoints, dz = p.npoints, p.delta_z_um
    stage._on_thickness_changed(0.0)
    stage._on_dz_changed(0.0)
    stage._on_npoints_changed(0)
    assert p.thickness_um == 0.0 and p.delta_z_um == 0.0 and p.npoints == 0
    assert stage.npoints_spin.value() == npoints  # derived widgets untouched
    assert stage.dz_spin.value() == pytest.approx(dz)


def test_stage_retrieve_smearing_and_focal_are_mutually_exclusive(stage):
    stage.focal_check.setChecked(True)
    assert stage.state.retrieve.focal
    stage.smearing_check.setChecked(True)
    assert not stage.focal_check.isChecked() and not stage.state.retrieve.focal
    stage.focal_check.setChecked(True)
    assert not stage.smearing_check.isChecked()
    assert not stage.state.retrieve.smearing
    stage.focal_check.setChecked(False)
    assert not stage.state.retrieve.focal


def test_stage_retrieve_collection_manual_enables_the_diameter(stage):
    stage._on_collection_changed("manual")
    assert stage.state.retrieve.collection == "manual"
    assert stage.collection_diam_spin.isEnabled()
    stage._on_collection_changed("file")
    assert not stage.collection_diam_spin.isEnabled()


@pytest.mark.parametrize(
    ("focal", "source", "expected"),
    [
        (False, "beamlet", 0.5),
        (True, "beamlet", 1.0),
        (True, "beamlet_reimaged", 0.0),
        (False, "beamlet_reimaged", -0.5),
    ],
)
def test_stage_retrieve_frame_auto_follows_model_and_source(
    stage, focal, source, expected
):
    stage.state.retrieve.focal = focal
    stage.state.simulated.spectrum_source = source
    stage.frame_auto_check.setChecked(True)
    stage._apply_frame_auto()
    assert not stage.frame_p_spin.isEnabled()
    assert stage.state.retrieve.spectrum_frame_p == pytest.approx(expected)


def test_stage_retrieve_manual_frame_is_left_alone(stage):
    stage.frame_auto_check.setChecked(False)
    stage.state.retrieve.spectrum_frame_p = 0.3
    stage._apply_frame_auto()
    assert stage.frame_p_spin.isEnabled()
    assert stage.state.retrieve.spectrum_frame_p == 0.3


def test_stage_retrieve_unreachable_smear_width_keeps_the_spacing(stage, monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("below the geometric floor")

    monkeypatch.setattr(stage_retrieve.smearing, "square_boxcars_spacing", refuse)
    spacing = stage.state.retrieve.smear_hole_spacing_mm
    stage._on_smear_sigma_changed(0.01)
    assert stage.state.retrieve.smear_sigma_delay_fs == 0.01
    assert stage.state.retrieve.smear_hole_spacing_mm == spacing


# -- geometry from a simulated file --------------------------------------------
def _geom(**kw):
    base = dict(
        thickness_um=None,
        has_mask=True,
        mask_diameter_mm=None,
        mask_spacing_mm=None,
        layout=None,
        f_foc_mm=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_stage_retrieve_geometry_none_changes_nothing(stage):
    assert stage.apply_simulated_geometry(None) == ""


def test_stage_retrieve_geometry_reports_each_change(stage):
    p = stage.state.retrieve
    note = stage.apply_simulated_geometry(
        _geom(
            thickness_um=p.thickness_um,  # unchanged: not reported
            mask_diameter_mm=p.smear_hole_diameter_mm,
            mask_spacing_mm=p.smear_hole_spacing_mm,
            layout="no-such-layout" if p.smear_layout != "x" else "y",
            f_foc_mm=p.focal_f_mm + 100.0,
        )
    )
    assert note.startswith("From the scan: kernel layout ")
    assert f"focal length {p.focal_f_mm:.4g} mm" in note
    # the same geometry again reports nothing new
    assert (
        stage.apply_simulated_geometry(
            _geom(
                mask_diameter_mm=p.smear_hole_diameter_mm,
                mask_spacing_mm=p.smear_hole_spacing_mm,
                layout=p.smear_layout,
                f_foc_mm=p.focal_f_mm,
            )
        )
        == ""
    )


def test_stage_retrieve_geometry_without_mask_says_so(stage):
    p = stage.state.retrieve
    note = stage.apply_simulated_geometry(_geom(has_mask=False))
    assert note == (
        "Note: the file does not record the mask geometry — check the hole "
        "diameter and spacing by hand."
    )
    note = stage.apply_simulated_geometry(
        _geom(has_mask=False, thickness_um=p.thickness_um + 5.0)
    )
    assert note.startswith("From the scan: thickness ")
    assert note.endswith("check the hole diameter and spacing by hand.")


# -- running -------------------------------------------------------------------
def test_stage_retrieve_run_without_a_trace(stage):
    stage._run()
    assert stage.status_label.text() == "No trace to retrieve."


def test_stage_retrieve_truth_init_needs_a_complex_truth(with_trace):
    with_trace.state.retrieve.truth_init = True
    with_trace._run()
    assert with_trace.status_label.text().startswith(
        "Truth initialisation needs a known complex field"
    )
    assert with_trace._worker is None


class _Signal:
    def connect(self, _slot):
        pass


class _Worker:
    """A stand-in for the threaded worker that records its arguments."""

    def __init__(self, td, p, **kwargs):
        self.kwargs = kwargs
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def report_preview_cost(self, seconds):
        self.cost = seconds

    def __getattr__(self, name):
        return _Signal() if name != "start" else (lambda: None)


def test_stage_retrieve_reuse_without_a_compatible_result(with_trace, monkeypatch, run):
    monkeypatch.setattr(stage_retrieve, "RetrievalWorker", _Worker)
    with_trace.state.retrieve.reuse_result = True
    with_trace._run()
    assert with_trace.status_label.text() == (
        "No compatible previous result to reuse; using the default guess."
    )
    assert with_trace._worker.kwargs["guess_override"] is None
    # a result on the same grid but with nothing extra to seed says nothing
    with_trace.state.result = run[3]
    with_trace.set_status("")
    with_trace._run()
    assert with_trace._worker.kwargs["extras_seed"] is run[3]
    assert with_trace.status_label.text() == ""
    with_trace._stop()
    assert with_trace._worker.stop_requested


def test_stage_retrieve_stop_without_a_worker_is_a_no_op(stage):
    stage._stop()
    assert stage._worker is None


def test_stage_retrieve_seeded_extras_names_thickness_and_smearing(stage, run):
    p = stage.state.retrieve
    p.dispersive = p.fit_thickness = True
    p.smearing = p.fit_smearing = True
    seed = replace(run[3], thickness=33e-6, smear_scale=1.25)
    text = stage._seeded_extras(seed)
    assert "thickness 33.00 µm" in text
    assert "smearing ×1.250" in text


# -- completion, preview and read-out ------------------------------------------
def test_stage_retrieve_preview_without_trace_is_ignored(stage, run):
    stage._on_preview(run[3])
    assert not stage._live_full


def test_stage_retrieve_preview_without_worker_still_shows(with_trace, run):
    with_trace._worker = None
    with_trace._on_preview(run[3])
    assert with_trace._live_full
    assert with_trace.status_label.text().endswith("(live preview)")


def test_stage_retrieve_preview_reports_its_cost_to_the_worker(with_trace, run):
    with_trace._worker = _Worker(None, None)
    with_trace._on_preview(run[3])
    assert with_trace._worker.cost >= 0.0


def test_stage_retrieve_convergence_waits_for_errors(stage, monkeypatch):
    shown = []
    monkeypatch.setattr(stage.pages, "show_progress", shown.append)
    stage._live_full = False
    stage._errors = []
    stage._draw_convergence()  # nothing to plot yet
    assert shown == []
    stage._errors = [0.1, 0.05]
    stage._draw_convergence()
    assert shown == [[0.1, 0.05]]


def test_stage_retrieve_stopped_run_is_kept_and_labelled(with_trace, run):
    with_trace._on_stopped(run[3], 1.25)
    assert with_trace.state.result is not None
    assert with_trace.save_btn.isEnabled()
    status = with_trace.status_label.text()
    assert status.startswith("Stopped. R = ")
    assert status.endswith("1.25 s")


def test_stage_retrieve_status_lists_fitted_extras(with_trace, run):
    p = with_trace.state.retrieve
    p.smearing = True
    fitted = replace(run[3], thickness=12e-6, tau0=2e-15, smear_scale=1.1)
    with_trace._on_finished(fitted, 0.5)
    status = with_trace.status_label.text()
    assert "thickness 12.00 µm" in status
    assert "τ₀ 2.00 fs" in status
    assert "smearing σ" in status and "(×1.100)" in status


def test_stage_retrieve_update_view_needs_result_and_trace(stage, run):
    stage._update_view()  # no result yet
    stage._base_result = run[3]
    stage._update_view()  # no trace either
    assert stage.state.result is None
    with pytest.raises(RuntimeError, match="no trace data"):
        stage._plot_data(run[3], None)


def test_stage_retrieve_smear_width_is_none_for_shg(stage):
    stage.state.retrieve.smearing = True
    td = SimpleNamespace(interaction="shg", omega0_pulse=2.35e15)
    assert stage._smear_width_fs(1.0, td) is None


def test_stage_retrieve_elapsed_clock_ticks(stage, monkeypatch):
    seen = {}
    monkeypatch.setattr(stage.readout, "set_scalars", seen.update)
    monkeypatch.setattr(stage_retrieve.time, "perf_counter", lambda: 12.34)
    stage._t0 = 10.0
    stage._tick_elapsed()
    assert seen == {"elapsed": "2.3 s"}


@pytest.mark.parametrize("value", [None, float("nan")])
def test_stage_retrieve_formatters_show_a_dash_when_undefined(value):
    assert StageRetrieve._nm(value) == DASH
    assert StageRetrieve._fs(value) == DASH
    assert StageRetrieve._power(value, 1.0) == DASH


def test_stage_retrieve_curve_readout_without_truth_spectrum(
    with_trace, run, monkeypatch
):
    with_trace._on_finished(run[3], 0.5)
    pr = with_trace.state.processed
    t = np.linspace(-50e-15, 50e-15, 101)
    # intensity only: no spectrum on the truth
    truth = TruthPulse(t=t, It=np.exp(-((t / 5e-15) ** 2)), fwhm=8.3e-15)
    cells = {}
    monkeypatch.setattr(
        with_trace.readout, "set_cells", lambda row, values: cells.update({row: values})
    )
    with_trace._update_curve_readout(pr, truth)
    # without a known spectrum the truth row carries temporal cells only
    assert set(cells["truth"]) == {"fwhm", "peak_power"}


def test_stage_retrieve_extras_readout_marks_fitted_values(with_trace, run):
    p = with_trace.state.retrieve
    p.smearing = True
    seen = {}
    with_trace.readout.set_scalars = lambda values: seen.update(values)
    fitted = replace(run[3], thickness=12e-6, tau0=2e-15, smear_scale=1.1)
    with_trace._update_extras_readout(fitted)
    assert seen["thickness"] == "12.00 µm (fit)"
    assert seen["tau0"] == "2.00 fs (fit)"
    assert seen["smearing"].endswith("(×1.100, fit)")
    # a kernel the trace cannot carry (SHG) falls back to the bare factor
    with_trace.state.tracedata = replace(run[2], interaction="shg")
    with_trace._update_extras_readout(fitted)
    assert seen["smearing"] == "×1.100 (fit)"
    # nothing fitted, no slab: dashes and the nominal fixed values
    p.dispersive = False
    p.smearing = False
    with_trace._update_extras_readout(None)
    assert seen["thickness"] == DASH
    assert seen["tau0"] == "0.00 fs (fixed)"
    assert seen["smearing"] == DASH


# -- retarget and save ---------------------------------------------------------
def test_stage_retrieve_retarget_emits_the_chosen_folder(stage, monkeypatch, qtbot):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    stage._retarget()
    assert stage.status_label.text() == ""
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: "/data/next"
    )
    with qtbot.waitSignal(stage.state.retarget_requested) as blocker:
        stage._retarget()
    assert blocker.args == ["/data/next"]
    assert stage.status_label.text() == "Retargeting to /data/next …"


def test_stage_retrieve_save_needs_a_result_and_a_folder(
    with_trace, monkeypatch, run, tmp_path
):
    calls = []
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *a, **k: calls.append(1) or "",
    )
    with_trace._save()  # no result: the dialog never opens
    assert calls == []
    with_trace._on_finished(run[3], 0.5)
    with_trace._save()  # cancelled
    assert calls == [1]
    assert list(tmp_path.iterdir()) == []


def test_stage_retrieve_smear_summary_after_the_kernel_is_switched_off(stage):
    stage.state.retrieve.smearing = False
    td = SimpleNamespace(interaction="pg", omega0_pulse=2.35e15)
    assert stage._smear_summary(1.1, td) == "smearing ×1.100"


@pytest.mark.parametrize(
    ("message", "status"), [("stopped", "Stopped."), ("diverged", "Failed: diverged")]
)
def test_stage_retrieve_failure_reenables_run(stage, message, status):
    stage.run_btn.setEnabled(False)
    stage.stop_btn.setEnabled(True)
    stage._on_failed(message)
    assert stage.run_btn.isEnabled() and not stage.stop_btn.isEnabled()
    assert stage.status_label.text() == status
