"""Tests for the marginal-check stage (:mod:`croak.gui.stage_marginal_check`).

The stage only needs a :class:`WizardState` carrying ``load_data``; most tests
build that dictionary directly — a smooth trace and, optionally, a Gaussian
independent spectrum — so each branch of the QC overlay can be reached without
a file.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from croak.gui import stage_marginal_check
from croak.gui.display import set_display
from croak.gui.stage_marginal_check import StageMarginalCheck, _peak_norm
from croak.gui.state import WizardState
from croak.plotting.display import Display
from croak.session.pipeline import assemble_load_data

LAM = np.linspace(700e-9, 900e-9, 64)


def _data(interaction="pg", *, spectrum=True, trace=None, input_unit="delay"):
    """A loaded-trace dictionary: Gaussian rows × Gaussian delay profile."""
    delays = np.linspace(-40e-15, 40e-15, 21)
    if trace is None:
        rows = np.exp(-(((LAM - 800e-9) / 40e-9) ** 2))
        trace = np.outer(rows, np.exp(-((delays / 15e-15) ** 2)))
    spec = np.exp(-(((LAM - 800e-9) / 30e-9) ** 2))
    return {
        "lam": LAM,
        "scanaxis": delays,
        "input_unit": input_unit,
        "trace": trace,
        "lam_spec": LAM if spectrum else None,
        "Ilam_spec": spec if spectrum else None,
        "interaction": interaction,
    }


@pytest.fixture
def stage(qtbot):
    s = StageMarginalCheck(WizardState())
    qtbot.addWidget(s)
    return s


def _enter(stage, data, entry="experimental"):
    stage.state.entry = entry
    stage.state.set_load_data(data)
    stage.on_enter()


def _legend(view):
    """The names of the curves the marginal view shows."""
    return [c.name for c in view.plot.curves]


@pytest.fixture
def from_file(stage, experiment_h5, experiment):
    """The stage on the experimental file, with an independent spectrum added."""
    p = stage.state.load
    p.frog_path = experiment_h5
    p.ifrog_name, p.lam_name, p.scan_name = "trace", "wavelength", "delay"
    p.lam_unit, p.scan_unit, p.scan_type = "nm", "fs", "delay"
    p.interaction = experiment.interaction
    data = assemble_load_data(p)
    lam = data["lam"]
    centre = float(np.average(lam, weights=data["trace"].sum(axis=1)))
    data["lam_spec"] = lam
    data["Ilam_spec"] = np.exp(-(((lam - centre) / 20e-9) ** 2))
    _enter(stage, data)
    return stage


# -- lifecycle -----------------------------------------------------------------
def test_stage_marginal_check_without_data_asks_for_a_trace(stage):
    stage.on_enter()
    assert stage.status_label.text() == "Load a trace first."
    stage._draw()  # nothing to draw
    assert stage.trace_view.plot.data is None


def test_stage_marginal_check_measured_trace_defaults_and_controls(stage):
    stage.state.load.third_order = True
    stage.state.load.third_order_exp = 3.5
    _enter(stage, _data())
    assert stage._trust_nm == (190.0, 1000.0)
    assert stage._loader_params() is stage.state.load
    assert stage.third_order_check.isEnabled() and stage.third_order_check.isChecked()
    assert stage.exp_spin.value() == pytest.approx(3.5)
    assert stage.auto_btn.isEnabled()
    assert stage.status_label.text().startswith("PG marginal check")
    assert "trace marginal" in _legend(stage.marg_view)


def test_stage_marginal_check_trust_is_kept_for_the_same_trace(stage):
    _enter(stage, _data(), entry="synthetic")
    # noise-free: the full loaded range, floored/ceiled to whole nm (700e-9/1e-9
    # is 699.99..., so the floor is 699)
    assert stage._trust_nm == (699.0, 900.0)
    stage._on_trust(720.0, 880.0)
    stage.on_enter()  # same trace: the manual trust region is kept
    assert stage._trust_nm == (720.0, 880.0)


def test_stage_marginal_check_synthetic_trace_disables_the_correction(stage):
    _enter(stage, _data(spectrum=False), entry="synthetic")
    assert stage._loader_params() is None
    assert not stage.third_order_check.isEnabled()
    assert not stage.auto_btn.isEnabled()
    data = stage.state.load_data
    # with no loader there is nothing to re-apply: every handler is a no-op
    stage._on_third_order_toggled(True)
    stage._on_exp_changed(4.0)
    stage._auto_exponent()
    stage._reassemble()
    assert stage.state.load_data is data
    assert stage.status_label.text() == (
        "No independent spectrum: showing the trace marginal only."
    )


def test_stage_marginal_check_simulated_entry_uses_simulated_params(stage):
    stage.state.entry = "simulated"
    assert stage._loader_params() is stage.state.simulated


def test_stage_marginal_check_shg_correction_needs_a_spectrum(stage):
    stage.state.preproc.marginal_correct = True
    _enter(stage, _data("shg", spectrum=False))
    assert not stage.marginal_correct_check.isEnabled()
    assert stage.state.preproc.marginal_correct is False  # cleared, else preprocess
    _enter(stage, _data("shg"))
    assert stage.marginal_correct_check.isEnabled()
    stage.marginal_correct_check.setChecked(True)
    assert stage.state.preproc.marginal_correct is True
    assert "trace marginal (corrected)" in _legend(stage.marg_view)
    assert "(exact)" in stage.status_label.text()  # SHG predicts the width exactly


# -- display ------------------------------------------------------------------
def test_stage_marginal_check_wavelength_axis_and_log_scale(stage):
    _enter(stage, _data())
    set_display(Display(axis="wavelength"))
    stage.scale_combo.setCurrentText("log")
    plot = stage.marg_view.plot
    assert plot.item.getAxis("bottom").labelText == "Wavelength (nm)"
    assert plot.log_y
    assert plot.box.viewRange()[0] == pytest.approx(stage._trust_nm)
    # the log window is 1e-3 … 1.5, in decades
    assert plot.box.viewRange()[1] == pytest.approx([-3.0, np.log10(1.5)])


def test_stage_marginal_check_position_scan_and_blank_trace(stage):
    # A blank, position-scanned trace: the image has no in-band signal to
    # zoom to and the marginal has too few positive samples for its moments.
    _enter(stage, _data(trace=np.zeros((LAM.size, 21)), input_unit="position"))
    assert stage.trace_view.title.text() == "Measured trace"
    assert stage.status_label.text() == (
        "No independent spectrum: showing the trace marginal only."
    )


def test_stage_marginal_check_trace_outside_trust_scales_to_all_of_it(stage):
    _enter(stage, _data(), entry="synthetic")
    stage._on_trust(100.0, 200.0)  # no sample in band
    assert stage.trace_view.title.text() == "Measured trace"
    assert stage.trace_view.plot.data is not None


# -- prediction edge cases ------------------------------------------------------
def test_stage_marginal_check_predict_returns_none_without_usable_spectrum(
    stage, monkeypatch
):
    spec = np.exp(-(((LAM - 800e-9) / 30e-9) ** 2))
    # all of the spectrum outside its trust region
    stage._spec_trust_nm = (1200.0, 1300.0)
    assert stage._predict(LAM, spec, "pg") is None
    # a single in-band sample spans no band to predict on
    stage._spec_trust_nm = (700.0, 900.0)
    assert stage._predict(np.array([800e-9]), np.array([1.0]), "pg") is None

    def refuse(*args, **kwargs):
        raise ValueError("degenerate spectrum")

    monkeypatch.setattr(stage_marginal_check, "predicted_marginal", refuse)
    assert stage._predict(LAM, spec, "pg") is None


@pytest.mark.parametrize(
    ("sigma_meas", "flag"),
    [
        (0.5, "⚠ below σ_S (clipping?)"),
        (5.0, "⚠ above TL (scatter/background?)"),
        (2.0, "✓ in bracket"),
    ],
)
def test_stage_marginal_check_width_bracket_flags(stage, sigma_meas, flag):
    pred = {
        "centroid_nm": 800.0,
        "centroid_omega": 2.35e15,
        "centroid_is_hard": True,
        "sigma_spectrum": 1.0,
        "width_bracket": (1.0, 3.0),
        "width_is_hard": False,
        "rms_width": 2.0,
    }
    stage._annotate_checks(801.0, sigma_meas, pred, "pg")
    text = stage.status_label.text()
    assert text.startswith("PG marginal check")
    assert "(Δ +1.0 nm, hard)" in text
    assert text.endswith(flag)


@pytest.mark.parametrize(("sigma_meas", "mark"), [(2.0, "✓"), (0.5, "⚠ below σ_S")])
def test_stage_marginal_check_sd_reports_only_the_lower_bound(stage, sigma_meas, mark):
    pred = {
        "centroid_nm": 800.0,
        "centroid_omega": 2.35e15,
        "centroid_is_hard": False,
        "sigma_spectrum": 1.0,
        "width_bracket": None,
        "width_is_hard": False,
        "rms_width": 2.0,
    }
    stage._annotate_checks(800.0, sigma_meas, pred, "sd")
    text = stage.status_label.text()
    assert "TL/soft" in text
    assert text.endswith(f"(σ_S) {mark}")


# -- third-order correction ----------------------------------------------------
def test_stage_marginal_check_toggle_and_exponent_reassemble(from_file):
    stage = from_file
    data = stage.state.load_data
    stage.third_order_check.setChecked(True)
    assert stage.state.load.third_order is True
    tilted = stage.state.load_data
    assert tilted is not data
    stage._on_exp_changed(4.0)
    assert stage.state.load.third_order_exp == 4.0
    assert stage.state.load_data is not tilted
    # with the correction off an exponent change does not reload
    stage.third_order_check.setChecked(False)
    untilted = stage.state.load_data
    stage._on_exp_changed(5.0)
    assert stage.state.load_data is untilted


def test_stage_marginal_check_reassemble_reports_failure(from_file, tmp_path):
    stage = from_file
    data = stage.state.load_data
    stage.state.load.frog_path = str(tmp_path / "moved.h5")
    stage._reassemble()
    assert stage.status_label.text().startswith("Reload failed:")
    assert stage.state.load_data is data


def test_stage_marginal_check_reassemble_simulated(stage, monkeypatch):
    rebuilt = _data()
    monkeypatch.setattr(
        stage_marginal_check, "assemble_simulated_load_data", lambda p: rebuilt
    )
    stage.state.entry = "simulated"
    stage._reassemble()
    assert stage.state.load_data is rebuilt


def test_stage_marginal_check_auto_needs_a_spectrum(stage):
    _enter(stage, _data(spectrum=False))
    stage._auto_exponent()
    assert stage.status_label.text() == (
        "Load an independent spectrum to auto-fit the exponent."
    )


def test_stage_marginal_check_auto_reports_a_failed_fit(stage, monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("no root in range")

    monkeypatch.setattr(stage_marginal_check, "auto_third_order_exponent", refuse)
    _enter(stage, _data())
    stage._auto_exponent()
    assert stage.status_label.text() == "Auto-fit failed: no root in range"
    assert stage.state.load.third_order is False


def test_stage_marginal_check_auto_removes_the_old_tilt_first(from_file, monkeypatch):
    stage = from_file
    stage.state.load.third_order = True
    stage.state.load.third_order_exp = 2.0
    data = stage.state.load_data
    seen = {}

    def fit(lam, marg, *args, **kwargs):
        seen["marg"] = marg
        return 3.14159

    monkeypatch.setattr(stage_marginal_check, "auto_third_order_exponent", fit)
    stage._auto_exponent()
    # the marginal handed to the fit is the measured one with (λ/µm)^2 undone
    expected = data["trace"].sum(axis=1) / (data["lam"] / 1e-6) ** 2.0
    np.testing.assert_allclose(seen["marg"], expected, rtol=1e-12)
    assert stage.state.load.third_order_exp == 3.14  # rounded to the spin's 0.01
    assert stage.exp_spin.value() == pytest.approx(3.14)
    assert stage.third_order_check.isChecked()


# -- helpers --------------------------------------------------------------------
def test_peak_norm_falls_back_outside_the_mask():
    y = np.array([0.0, 2.0, 4.0])
    np.testing.assert_allclose(_peak_norm(y, np.array([True, True, False])), y / 2)
    # an empty or non-positive mask selection normalises to the overall peak
    np.testing.assert_allclose(_peak_norm(y, np.zeros(3, bool)), y / 4)
    np.testing.assert_allclose(_peak_norm(y, np.array([True, False, False])), y / 4)
    # all-zero data is returned unchanged
    np.testing.assert_allclose(_peak_norm(np.zeros(3), np.ones(3, bool)), 0.0)
    assert _peak_norm(np.zeros(0), np.zeros(0, bool)).size == 0


# -- separate trust regions for the trace and the spectrum ---------------------
def _shg_data_an_octave_apart():
    """An SHG trace near 400 nm with its fundamental spectrum near 800 nm."""
    data = _data("shg")
    data["lam"] = LAM / 2.0
    return data


def test_stage_marginal_check_spectrum_trust_defaults_to_the_spectrum(stage):
    _enter(stage, _shg_data_an_octave_apart())
    assert stage._trust_nm == (190.0, 1000.0)
    assert stage._spec_trust_nm == (700.0, 900.0)
    assert stage.spec_trust.isEnabled()
    # with the trust regions an octave apart the prediction still forms (one
    # shared region cannot hold a 400 nm trace and an 800 nm spectrum)
    stage._on_trust(340.0, 460.0)
    spec = np.exp(-(((LAM - 800e-9) / 30e-9) ** 2))
    assert stage._predict(LAM, spec, "shg") is not None


def test_stage_marginal_check_spectrum_trust_is_kept_for_the_same_spectrum(stage):
    _enter(stage, _data())
    stage._on_spec_trust(720.0, 880.0)
    assert stage._spec_trust_nm == (720.0, 880.0)
    stage.on_enter()  # same spectrum: the manual region is kept
    assert stage._spec_trust_nm == (720.0, 880.0)
    data = _data()
    data["lam_spec"] = np.linspace(600e-9, 1000e-9, 64)
    _enter(stage, data)  # a different spectrum re-defaults
    assert stage._spec_trust_nm == (600.0, 1000.0)


def test_stage_marginal_check_spectrum_trust_is_off_without_a_spectrum(stage):
    _enter(stage, _data(spectrum=False))
    assert not stage.spec_trust.isEnabled()


def test_stage_marginal_check_auto_passes_both_trust_regions(stage, monkeypatch):
    seen = {}

    def fit(*args, **kwargs):
        seen.update(kwargs)
        return 1.0

    monkeypatch.setattr(stage_marginal_check, "auto_third_order_exponent", fit)
    _enter(stage, _shg_data_an_octave_apart())
    stage._on_trust(340.0, 460.0)
    stage._on_spec_trust(720.0, 880.0)
    stage._auto_exponent()
    assert seen["trace_window"] == pytest.approx((340e-9, 460e-9))
    assert seen["spectrum_window"] == pytest.approx((720e-9, 880e-9))


def test_stage_marginal_check_spectrum_trust_outside_the_spectrum_shows_no_prediction(
    stage,
):
    _enter(stage, _data())
    stage._on_spec_trust(1200.0, 1300.0)  # no spectrum sample in the region
    legend = _legend(stage.marg_view)
    assert "spectrum" in legend and "predicted" not in legend
    assert stage.status_label.text() == (
        "No independent spectrum: showing the trace marginal only."
    )


@pytest.mark.parametrize(
    ("entry", "extent", "expected"),
    [
        # measured: clipped to the 190-1000 nm detector band
        ("experimental", (600e-9, 1200e-9), (600.0, 1000.0)),
        # measured, wholly beyond 1 um (a 1.6 um fundamental): its full range
        ("experimental", (1400e-9, 1800e-9), (1400.0, 1800.0)),
        # noise-free: the full range
        ("synthetic", (600e-9, 1200e-9), (600.0, 1200.0)),
    ],
)
def test_stage_marginal_check_spectrum_trust_default(stage, entry, extent, expected):
    data = _data()
    data["lam_spec"] = np.linspace(*extent, 64)
    _enter(stage, data, entry=entry)
    assert stage._spec_trust_nm == expected
