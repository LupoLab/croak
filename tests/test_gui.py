"""Offscreen smoke tests for the PyQt6 wizard.

Headless (``QT_QPA_PLATFORM=offscreen``); drive a synthetic dataset through
Load → Marginal check → Preprocess → Retrieve → Dispersion. Smoke-level, not
pixel tests. The six linear stages are ``w.stages[0..5]`` = Load, Marginal
check, Preprocess, Retrieve, Dispersion, Uncertainty (stage numbers 1..6).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

import croak

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings, QSize, Qt

from croak.gui import settings
from croak.gui.base import CONTROLS_MIN_WIDTH
from croak.gui.canvas import MplCanvas, collapsible
from croak.gui.stage_load import assemble_load_data, seed_preproc_defaults
from croak.gui.stage_marginal_check import StageMarginalCheck
from croak.gui.state import WizardState
from croak.gui.wizard import Wizard
from croak.uncertainty import UncertaintyResult


def _uresult(method, sd, rng):
    """A lightweight UncertaintyResult with a temporal band for stage-6 GUI tests."""
    samples = 1.5e-15 + sd * rng.standard_normal(400)
    t = np.linspace(-15e-15, 15e-15, 30)
    prof = np.clip(
        np.exp(-((t / 1.5e-15) ** 2)) + 0.02 * rng.standard_normal((400, t.size)),
        0.0,
        None,
    )
    return UncertaintyResult(
        statistic="fwhm",
        point_estimate=1.5e-15,
        samples=samples,
        interval_68=(1.45e-15, 1.55e-15),
        interval_95=(1.4e-15, 1.6e-15),
        interval_method="percentile",
        method=method,
        n_resamples=400,
        n_converged=400,
        frog_errors=np.empty(0),
        base_error=0.01,
        profiles=prof,
        t_profile=t,
    )


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path):
    """Point QSettings at a per-test directory so no test touches real preferences.

    ``croak.gui.settings`` only ever uses the explicit ``QSettings(IniFormat,
    UserScope, org, app)`` constructor, which honours this redirect.
    """
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path / "settings"),
    )


def test_wizard_opens_on_welcome(qtbot):
    w = Wizard()
    qtbot.addWidget(w)
    assert len(w.stages) == 6
    # opens on the welcome page (index 0); the stage header is not yet set
    assert w.stack.currentIndex() == 0
    # entering the load workflow shows Stage 1
    w.start_load()
    assert w.stack.currentWidget() is w.stages[0]
    assert "Stage 1" in w.header.text()
    assert not w.next_btn.isEnabled()


def test_synthetic_generation_routes_to_marginal_check(qtbot):
    """The synthetic generator builds a trace and lands on the marginal-check stage."""
    w = Wizard()
    qtbot.addWidget(w)
    w.show_synthetic()
    assert w.stack.currentWidget() is w.synthetic
    # defaults: 10 fs Gaussian at 800 nm, SHG, λ window 350–450 nm
    w.synthetic.advance()
    assert w.state.load_data is not None
    assert w.state.truth is not None
    assert w.state.stage == 2
    assert w.stack.currentWidget() is w.stages[1]  # StageMarginalCheck
    # the trace is regridded only on entering the (next) preprocess stage
    assert w.state.tracedata is None
    w.state.stage = 3  # advance to preprocess
    assert w.state.tracedata is not None
    assert w.stack.currentWidget() is w.stages[2]  # StagePreprocess


def test_synthetic_generates_independent_spectrum_and_steps(qtbot):
    """Synthetic SHG traces carry a mirrored ×2 fundamental spectrum; steps link."""
    w = Wizard()
    qtbot.addWidget(w)
    w.show_synthetic()
    s = w.synthetic
    # step <-> count interlock at fixed range
    s._on_delay_step(2.0)  # 120 fs span / 2 fs -> 61 delays
    assert s.ndelay.value() == 61
    # mirror: SHG fundamental window is 2x the (300-550 nm) signal window
    assert s._spec_wl == pytest.approx((600.0, 1100.0))
    assert s.nspec.value() == s.nwl.value()
    s.advance()
    data = w.state.load_data
    assert data["lam_spec"] is not None and data["Ilam_spec"] is not None
    assert data["lam_spec"].min() / 1e-9 == pytest.approx(600.0, abs=1.0)
    assert data["lam_spec"].max() / 1e-9 == pytest.approx(1100.0, abs=1.0)
    # the known-truth intensity FWHM matches the requested 10 fs duration
    assert w.state.truth.fwhm / 1e-15 == pytest.approx(10.0, rel=2e-2)


def test_marginal_correction_lives_on_marginal_stage(qtbot):
    """The SHG marginal correction control is on the marginal stage, not preprocess."""
    w = Wizard()
    qtbot.addWidget(w)
    w.show_synthetic()
    w.synthetic.advance()  # SHG synthetic trace with an independent spectrum
    marginal = w.stages[1]
    preprocess = w.stages[2]
    assert hasattr(marginal, "marginal_correct_check")
    assert not hasattr(preprocess, "marginal_check")
    # SHG + independent spectrum -> the control is enabled
    assert marginal.marginal_correct_check.isEnabled()
    marginal.marginal_correct_check.setChecked(True)
    assert w.state.preproc.marginal_correct is True


def test_synthetic_back_returns_to_generator(qtbot):
    """In a synthetic session, Back from the marginal stage returns to the generator."""
    w = Wizard()
    qtbot.addWidget(w)
    w.show_synthetic()
    w.synthetic.advance()
    assert w.state.stage == 2  # marginal check

    w.state.stage = 3  # advance to preprocess
    w.go_back()  # -> marginal check
    assert w.stack.currentWidget() is w.stages[1]
    w.go_back()  # -> the synthetic generator, not the experimental loader
    assert w.stack.currentWidget() is w.synthetic
    # the in-progress data survives the round trip (no reset on Back)
    assert w.state.truth is not None


def test_simulated_load_routes_to_marginal_check(
    qtbot, simulated_scan_h5, simulated_truth
):
    """Loading a simulated scan lands on the marginal-check stage."""
    w = Wizard()
    qtbot.addWidget(w)
    w.load_simulated()
    assert w.stack.currentWidget() is w.simulated
    sim = w.simulated
    sim.path_edit.setText(simulated_scan_h5)
    sim.band.set_values(
        simulated_truth.lam_min / 1e-9 - 10, simulated_truth.lam_max / 1e-9 + 10
    )
    sim._band = (
        simulated_truth.lam_min / 1e-9 - 10,
        simulated_truth.lam_max / 1e-9 + 10,
    )
    sim._preview()
    assert sim.can_advance()
    sim.advance()
    assert w.state.load_data is not None
    # the chosen options are recorded on the state for provenance
    assert w.state.simulated.interaction == "pg"
    assert w.state.simulated.frog_path == simulated_scan_h5
    assert w.state.stage == 2
    assert w.stack.currentWidget() is w.stages[1]  # StageMarginalCheck
    # preprocess regrids on advancing to it
    w.state.stage = 3
    assert w.state.tracedata is not None
    assert w.stack.currentWidget() is w.stages[2]  # StagePreprocess


def test_simulated_thickness_selector(
    qtbot, simulated_scan_multi_h5, simulated_scan_h5
):
    """The thickness selector lists ``zsave`` slices and disables for legacy files."""
    from conftest import SIMULATED_ZSAVE

    w = Wizard()
    qtbot.addWidget(w)
    w.load_simulated()
    sim = w.simulated

    # multi-thickness file: one item per slice, default = the full (exit) thickness
    sim.path_edit.setText(simulated_scan_multi_h5)
    sim._populate_file_options(simulated_scan_multi_h5)
    assert sim.thickness.isEnabled()
    assert sim.thickness.count() == SIMULATED_ZSAVE.size
    assert "full" in sim.thickness.currentText()
    assert sim._params().z_thickness_um == pytest.approx(SIMULATED_ZSAVE.max() * 1e6)
    # choosing the entrance slice (listed last) changes the loaded thickness
    sim.thickness.setCurrentIndex(sim.thickness.count() - 1)
    assert "entrance" in sim.thickness.currentText()
    assert sim._params().z_thickness_um == pytest.approx(0.0)

    # legacy file: nothing to choose -> disabled, falls back to the exit slice
    sim._populate_file_options(simulated_scan_h5)
    assert not sim.thickness.isEnabled()
    assert sim._params().z_thickness_um is None


def test_simulated_truth_source_selector(
    qtbot, tmp_path, simulated_scan_h5, simulated_truth
):
    """The truth picker offers the beamlet by default and disables when only one."""
    import h5py
    from conftest import write_simulated_h5

    w = Wizard()
    qtbot.addWidget(w)
    w.load_simulated()
    sim = w.simulated

    # legacy file stores only the source temporal truth -> disabled, falls to source
    sim.path_edit.setText(simulated_scan_h5)
    sim._populate_file_options(simulated_scan_h5)
    assert not sim.truth_source.isEnabled()
    assert sim._params().truth_source == "source"

    # a file that also stores It_beamlet -> beamlet offered and chosen by default
    path = write_simulated_h5(tmp_path / "beamlet.h5", simulated_truth)
    with h5py.File(path, "a") as f:
        f["grid"]["It_beamlet"] = simulated_truth.It
    sim.path_edit.setText(path)
    sim._populate_file_options(path)
    assert sim.truth_source.isEnabled()
    assert sim._params().truth_source == "beamlet"
    sim.truth_source.setCurrentText("source")
    assert sim._params().truth_source == "source"


def test_dispersive_dz_npoints_coupling(qtbot):
    """Δz, thickness and Npoints stay tied (npoints = thickness / Δz)."""
    w = Wizard()
    qtbot.addWidget(w)
    stage = w.stages[3]  # StageRetrieve
    p = stage.state.retrieve

    # defaults are self-consistent: 10 µm / 1 µm = 10 points
    assert stage.dz_spin.value() == pytest.approx(1.0)
    assert stage.npoints_spin.value() == 10

    # editing the thickness holds Δz fixed and grows npoints
    stage.thickness_spin.setValue(40.0)
    assert stage.dz_spin.value() == pytest.approx(1.0)
    assert stage.npoints_spin.value() == 40
    assert p.npoints == 40 and p.thickness_um == pytest.approx(40.0)

    # editing Δz recomputes npoints (thickness held)
    stage.dz_spin.setValue(4.0)
    assert stage.npoints_spin.value() == 10
    assert p.delta_z_um == pytest.approx(4.0)

    # editing npoints back-computes Δz so the displayed step is the real spacing
    stage.npoints_spin.setValue(20)
    assert stage.dz_spin.value() == pytest.approx(2.0)
    assert p.delta_z_um == pytest.approx(2.0)


def test_smearing_geometry_sigma_coupling(qtbot):
    """Hole diameter, spacing and sigma stay tied through the square-BOXCARS map."""
    w = Wizard()
    qtbot.addWidget(w)
    stage = w.stages[3]  # StageRetrieve
    p = stage.state.retrieve

    # Defaults are the reference DUV instrument: 1 mm holes, 0.5 mm edge-to-edge.
    assert stage.hole_diameter_spin.value() == pytest.approx(1.0)
    assert stage.hole_spacing_spin.value() == pytest.approx(0.5)

    # Widening the mask increases the delay width linearly in d/D. The stored
    # width is exact; the spin box shows it to three decimals.
    stage.hole_spacing_spin.setValue(1.5)
    wide = p.smear_sigma_delay_fs
    assert stage.smear_sigma_spin.value() == pytest.approx(wide, abs=5e-5)
    stage.hole_spacing_spin.setValue(0.5)
    narrow = p.smear_sigma_delay_fs
    d_narrow, d_wide = 0.5 * (0.5 + 1.0), 0.5 * (1.5 + 1.0)
    assert wide / narrow == pytest.approx(d_wide / d_narrow, rel=1e-9)

    # A bigger hole means a smaller focal spot and less smearing.
    stage.hole_diameter_spin.setValue(2.0)
    assert p.smear_sigma_delay_fs < narrow

    # Editing sigma back-computes the spacing, holding the diameter.
    stage.hole_diameter_spin.setValue(1.0)
    stage.hole_spacing_spin.setValue(3.0)
    stage.smear_sigma_spin.setValue(narrow)
    # Exact to the displayed precision of the sigma box (four decimals).
    assert stage.hole_spacing_spin.value() == pytest.approx(0.5, abs=1e-3)
    assert p.smear_hole_spacing_mm == pytest.approx(0.5, abs=1e-3)


def test_smearing_gates_the_fit_checkbox_and_solver_support(qtbot):
    """The group follows solver capability; the fit box follows the enable."""
    w = Wizard()
    qtbot.addWidget(w)
    stage = w.stages[3]

    stage.solver_combo.setCurrentText("lbfgs-ad")
    assert stage.smear_box.isEnabled()
    # Fitting the width needs a kernel to scale.
    assert not stage.fit_smearing_check.isEnabled()
    stage.smearing_check.setChecked(True)
    assert stage.state.retrieve.smearing is True
    assert stage.fit_smearing_check.isEnabled()

    # COPRA's magnitude projection cannot represent the incoherent node sum.
    stage.solver_combo.setCurrentText("copra")
    assert not stage.smear_box.isEnabled()
    assert not stage.fit_smearing_check.isEnabled()


def test_simulated_trace_window_lists_iomega_full(
    qtbot, simulated_scan_multi_h5, simulated_scan_h5
):
    """The trace-window dropdown offers ``Iω_full`` only when the file stores it."""
    w = Wizard()
    qtbot.addWidget(w)
    w.load_simulated()
    sim = w.simulated

    sim.path_edit.setText(simulated_scan_multi_h5)
    sim._populate_file_options(simulated_scan_multi_h5)
    items = [sim.window_key.itemText(i) for i in range(sim.window_key.count())]
    assert "Iω_full" in items
    sim.window_key.setCurrentText("Iω_full")
    assert sim._params().window_key == "Iω_full"

    # legacy file lacks Iω_full: it drops out and the selection falls back
    sim._populate_file_options(simulated_scan_h5)
    items = [sim.window_key.itemText(i) for i in range(sim.window_key.count())]
    assert "Iω_full" not in items
    assert sim._params().window_key in ("Iω_win", "Iω_win_reimaged")


def test_simulated_band_defaults_to_full_range(qtbot, simulated_scan_multi_h5):
    """Picking a file seeds the load band to the file's full λ range."""
    from croak import io

    w = Wizard()
    qtbot.addWidget(w)
    w.load_simulated()
    sim = w.simulated
    sim._populate_file_options(simulated_scan_multi_h5)
    lam_lo, lam_hi = io.read_simulated_wavelength_range(simulated_scan_multi_h5)
    blo, bhi = sim.band.values()
    assert blo == pytest.approx(float(np.floor(lam_lo)))
    assert bhi == pytest.approx(float(np.ceil(lam_hi)))


def test_simulated_raw_direct_routes_to_retrieve(qtbot, simulated_scan_h5):
    """The raw checkbox loads straight into Retrieve with tracedata + truth set."""
    from croak import io

    w = Wizard()
    qtbot.addWidget(w)
    w.load_simulated()
    sim = w.simulated
    sim.path_edit.setText(simulated_scan_h5)
    sim._populate_file_options(simulated_scan_h5)
    sim.raw_direct.setChecked(True)
    sim.advance()
    # bypassed marginal-check (2) and preprocess (3): landed on Retrieve (4)
    assert w.state.stage == 4
    assert w.stack.currentWidget() is w.stages[3]  # StageRetrieve
    assert w.state.tracedata is not None
    assert w.state.truth is not None  # truth still carried for the overlay
    # native grid: as-simulated (no regrid), so the grid matches the scan's ω axis
    scan = io.read_simulated_scan(simulated_scan_h5)
    assert w.state.tracedata.grid.n == scan.omega.size


def test_simulated_back_returns_to_loader(qtbot, simulated_scan_h5, simulated_truth):
    """In a simulated session, Back from the marginal stage returns to the loader."""
    w = Wizard()
    qtbot.addWidget(w)
    w.load_simulated()
    w.simulated.path_edit.setText(simulated_scan_h5)
    w.simulated.advance()
    assert w.state.stage == 2

    w.state.stage = 3  # advance to preprocess
    w.go_back()  # -> marginal check
    assert w.stack.currentWidget() is w.stages[1]
    w.go_back()  # -> the simulated loader, not the experimental Stage 1
    assert w.stack.currentWidget() is w.simulated


def test_marginal_check_predicts_and_auto_fits(
    qtbot, simulated_scan_h5, simulated_truth
):
    """The marginal-check stage predicts a marginal and Auto sets a finite exponent."""
    w = Wizard()
    qtbot.addWidget(w)
    w.load_simulated()
    sim = w.simulated
    sim.path_edit.setText(simulated_scan_h5)
    sim.band.set_values(
        simulated_truth.lam_min / 1e-9 - 10, simulated_truth.lam_max / 1e-9 + 10
    )
    sim._band = (
        simulated_truth.lam_min / 1e-9 - 10,
        simulated_truth.lam_max / 1e-9 + 10,
    )
    sim.advance()
    assert w.state.stage == 2
    mc = w.stages[1]
    assert isinstance(mc, StageMarginalCheck)
    data = w.state.load_data
    # the simulated loader supplies the known input spectrum -> prediction works
    assert data["Ilam_spec"] is not None
    pred = mc._predict(data["lam_spec"], data["Ilam_spec"], data["interaction"])
    assert pred is not None
    assert np.isfinite(pred["centroid_nm"])
    assert mc.auto_btn.isEnabled()
    # Auto enables the third-order correction and sets a finite, in-range exponent
    mc._auto_exponent()
    assert w.state.simulated.third_order
    assert 0.0 <= w.state.simulated.third_order_exp <= 12.0


def test_new_session_clears_previous_data(qtbot, experiment_h5, experiment):
    """Starting a new process from the menu clears prior data and settings."""
    w = Wizard()
    qtbot.addWidget(w)

    # run a synthetic generation, leaving load_data/truth populated
    w.show_synthetic()
    w.synthetic.advance()
    assert w.state.truth is not None
    assert w.state.load_data is not None

    # switch to the experimental workflow from the menu -> everything cleared
    w.start_load()
    assert w.state.truth is None
    assert w.state.load_data is None
    assert w.state.tracedata is None
    assert w.state.result is None
    assert w.state.load.frog_path == ""
    assert w.stack.currentWidget() is w.stages[0]  # StageLoad, fresh


def test_solver_combo_lists_all_algorithms(qtbot):
    from PyQt6.QtWidgets import QComboBox

    w = Wizard()
    qtbot.addWidget(w)
    stage = w.stages[3]  # StageRetrieve
    items = {
        combo.itemText(i)
        for combo in stage.findChildren(QComboBox)
        for i in range(combo.count())
    }
    # the four retrieval algorithms are offered; the removed "dcopra" is not
    assert {"copra", "lbfgs", "lbfgs-ad", "lm"} <= items
    assert "dcopra" not in items


def test_every_stage_help_entry_maps_to_a_control(qtbot):
    """Each HELP (section, label) must match a real control, and set its tooltip.

    Guards against the help text and the controls drifting apart: a renamed or
    removed option leaves an orphan HELP entry that this test catches.
    """
    from croak.gui.help import flatten

    w = Wizard()
    qtbot.addWidget(w)
    # the simulated loader is an entry page (not in w.stages) but declares HELP too
    for stage in [*w.stages, w.simulated]:
        expected = set(flatten(stage.HELP))
        assert expected, f"{type(stage).__name__} has no HELP content"
        matched = stage.apply_help()
        orphans = expected - matched
        assert not orphans, f"{type(stage).__name__} help has no control: {orphans}"


def test_help_dialog_builds_and_shows(qtbot):
    """The header '?' opens a non-empty summary dialog for the current stage."""
    from PyQt6.QtWidgets import QTextBrowser

    from croak.gui.help import HelpDialog

    w = Wizard()
    qtbot.addWidget(w)
    w.goto_stage(4)  # Retrieve
    w._show_help()
    dlg = w.stages[3]._help_dialog
    assert isinstance(dlg, HelpDialog)
    assert dlg.isVisible()
    # the rendered summary mentions section headings from the Retrieve stage
    text = dlg.findChild(QTextBrowser).toPlainText()
    assert "Solver" in text
    assert "Regularisation" in text


def test_load_options_refreshes_controls(qtbot, tmp_path):
    """'Load saved…' must update the on-screen controls, not just the params."""
    from PyQt6.QtWidgets import QComboBox, QSpinBox

    from croak import save

    # write an options file with distinctive retrieval settings
    src = WizardState()
    src.retrieve.solver = "lm"
    src.retrieve.maxiters = 222
    src.retrieve.reltol = 1e-6
    src.retrieve.abstol = 1e-10
    src.retrieve.reg_spectrum = 0.25
    opts_path = str(tmp_path / "options.toml")
    save.save_options(src.to_options(), opts_path)

    w = Wizard()
    qtbot.addWidget(w)
    # emulate _load_options without the file dialog
    w.state.from_options(save.load_options(opts_path))
    w._rebuild_stages()

    stage = w.stages[3]  # StageRetrieve
    assert stage.findChildren(QComboBox)[0].currentText() == "lm"
    spin_values = {s.value() for s in stage.findChildren(QSpinBox)}
    assert 222 in spin_values  # maxiters
    assert -6 in spin_values  # reltol exponent
    assert -10 in spin_values  # abstol exponent
    # the params themselves round-tripped
    assert w.state.retrieve.solver == "lm"
    assert w.state.retrieve.reltol == pytest.approx(1e-6)
    assert w.state.retrieve.reg_spectrum == pytest.approx(0.25)
    # the λ-spectrum control is present in the Regularisation group
    from PyQt6.QtWidgets import QDoubleSpinBox

    dspin_values = {round(s.value(), 3) for s in stage.findChildren(QDoubleSpinBox)}
    assert 0.25 in dspin_values


def test_load_options_restores_full_session(qtbot, experiment_h5, experiment, tmp_path):
    """'Load saved…' replays load → preprocess → retrieve to restore the session."""
    from croak import save

    # Build and save a complete session in one wizard.
    w1 = Wizard()
    qtbot.addWidget(w1)
    s1 = w1.state
    s1.load.frog_path = experiment_h5
    w1.stages[0]._populate_frog()
    s1.load.interaction = experiment.interaction
    w1.stages[0]._load_preview()
    s1.preproc.lam_min_nm = experiment.lam_min / 1e-9
    s1.preproc.lam_max_nm = experiment.lam_max / 1e-9
    s1.stage = 3  # preprocess
    assert s1.tracedata is not None
    s1.retrieve.solver = "copra"
    s1.retrieve.maxiters = 40
    opts_path = str(tmp_path / "options.toml")
    save.save_options(s1.to_options(), opts_path)

    # Restore into a fresh wizard (emulating _load_options without the dialog).
    w2 = Wizard()
    qtbot.addWidget(w2)
    w2.state.from_options(save.load_options(opts_path))
    w2._rebuild_stages()
    with qtbot.waitSignal(w2.state.result_changed, timeout=60000):
        w2._restore_session()

    # Same data, options and (re-computed) result are all restored.
    assert w2.state.load_data is not None
    assert w2.state.tracedata is not None
    assert w2.state.result is not None
    assert w2.state.stage == 4
    assert w2.state.preproc.lam_min_nm == pytest.approx(experiment.lam_min / 1e-9)
    assert w2.state.load.interaction == experiment.interaction


def test_load_options_restores_defringe_baseline_widgets(
    qtbot, experiment_h5, experiment, tmp_path
):
    """A saved options.toml restores the new de-fringe / baseline controls."""
    from croak import save

    w1 = Wizard()
    qtbot.addWidget(w1)
    s1 = w1.state
    s1.load.frog_path = experiment_h5
    w1.stages[0]._populate_frog()
    s1.load.interaction = experiment.interaction
    w1.stages[0]._load_preview()
    # set the new preprocessing options to non-default values
    s1.preproc.defringe = True
    s1.preproc.defringe_fraction = 0.4
    s1.preproc.defringe_on_alias = "clamp"
    s1.preproc.baseline = True
    s1.preproc.baseline_smoothness = 2.5e2
    s1.preproc.baseline_ratio = 4e-3
    s1.preproc.baseline_clip = True
    s1.preproc.baseline_tau_exclude_fs = 5.0
    opts_path = str(tmp_path / "options.toml")
    save.save_options(s1.to_options(), opts_path)

    # Restore into a fresh wizard and rebuild the stage widgets from the file.
    w2 = Wizard()
    qtbot.addWidget(w2)
    w2.state.from_options(save.load_options(opts_path))
    w2._rebuild_stages()

    p2 = w2.state.preproc
    assert p2.defringe is True
    assert p2.defringe_fraction == pytest.approx(0.4)
    assert p2.defringe_on_alias == "clamp"
    assert p2.baseline is True
    assert p2.baseline_smoothness == pytest.approx(2.5e2)
    assert p2.baseline_ratio == pytest.approx(4e-3)
    assert p2.baseline_clip is True
    assert p2.baseline_tau_exclude_fs == pytest.approx(5.0)

    # the rebuilt preprocess-stage widgets reflect the loaded options, with the
    # dependent controls enabled because their toggles loaded as on
    pre2 = w2.stages[2]
    assert pre2.defringe_check.isChecked()
    assert pre2.defringe_frac_spin.value() == pytest.approx(0.4)
    assert pre2.defringe_alias_combo.currentText() == "clamp"
    assert pre2.baseline_check.isChecked()
    assert pre2.baseline_smooth_spin.value() == pytest.approx(2.5e2)
    assert pre2.baseline_clip_check.isChecked()
    assert pre2.defringe_frac_spin.isEnabled()
    assert pre2.baseline_smooth_spin.isEnabled()


def test_load_previous_reuses_saved_result_without_rerun(
    qtbot, experiment_h5, experiment, tmp_path
):
    """A sibling result.h5 is rehydrated on load, skipping the solver re-run."""
    from croak import save

    w1 = Wizard()
    qtbot.addWidget(w1)
    s1 = w1.state
    s1.load.frog_path = experiment_h5
    w1.stages[0]._populate_frog()
    s1.load.interaction = experiment.interaction
    w1.stages[0]._load_preview()
    s1.preproc.lam_min_nm = experiment.lam_min / 1e-9
    s1.preproc.lam_max_nm = experiment.lam_max / 1e-9
    s1.stage = 3  # preprocess
    s1.retrieve.solver = "copra"
    s1.retrieve.maxiters = 40
    s1.stage = 4  # retrieve
    with qtbot.waitSignal(s1.result_changed, timeout=60000):
        w1.stages[3]._run()
    save.save_result(
        s1.result,
        str(tmp_path / "result.h5"),
        processed=s1.processed,
        force=True,
    )
    save.save_options(s1.to_options(), str(tmp_path / "options.toml"))

    # Restore into a fresh wizard; spy on the solver re-run.
    w2 = Wizard()
    qtbot.addWidget(w2)
    w2.state.from_options(save.load_options(str(tmp_path / "options.toml")))
    w2._rebuild_stages()
    reran = {"called": False}
    w2.stages[3]._run = lambda *a, **k: reran.__setitem__("called", True)
    with qtbot.waitSignal(w2.state.result_changed, timeout=60000):
        w2._restore_session(str(tmp_path / "result.h5"))

    assert w2.state.result is not None  # result loaded from disk
    assert reran["called"] is False  # rehydrated, not re-run
    assert w2.state.stage == 4
    # the rehydrated result matches the saved spectrum
    np.testing.assert_allclose(w2.state.result.spectrum, s1.result.spectrum)


def test_load_previous_reruns_when_saved_result_grid_moved(
    qtbot, experiment_h5, experiment, tmp_path
):
    """A saved result on a stale delay grid must re-run, not crash.

    The sub-sample tau=0 centring is data-dependent, so a preprocessing change
    can move the crop boundary by one sample and a session saved by an older
    croak reloads onto a trace one delay longer than the one it was fitted to.
    Nothing in the options file records this. Adopting the result anyway used to
    raise inside compute_mu, because the Retrieve stage pairs the saved
    retrieval with the freshly preprocessed trace.
    """
    import h5py

    from croak import save

    w1 = Wizard()
    qtbot.addWidget(w1)
    s1 = w1.state
    s1.load.frog_path = experiment_h5
    w1.stages[0]._populate_frog()
    s1.load.interaction = experiment.interaction
    w1.stages[0]._load_preview()
    s1.preproc.lam_min_nm = experiment.lam_min / 1e-9
    s1.preproc.lam_max_nm = experiment.lam_max / 1e-9
    s1.stage = 3
    s1.retrieve.solver = "copra"
    s1.retrieve.maxiters = 20
    s1.stage = 4
    with qtbot.waitSignal(s1.result_changed, timeout=60000):
        w1.stages[3]._run()
    rp = str(tmp_path / "result.h5")
    save.save_result(s1.result, rp, processed=s1.processed, force=True)
    save.save_options(s1.to_options(), str(tmp_path / "options.toml"))

    # Drop one delay from the saved trace: the axis drift, reproduced exactly.
    with h5py.File(rp, "r+") as f:
        for key in ("trace_retr", "trace_meas"):
            if key in f:
                trimmed = f[key][:, :-1]
                del f[key]
                f[key] = trimmed
        if "tau" in f:
            trimmed = f["tau"][:-1]
            del f["tau"]
            f["tau"] = trimmed

    w2 = Wizard()
    qtbot.addWidget(w2)
    w2.state.from_options(save.load_options(str(tmp_path / "options.toml")))
    w2._rebuild_stages()
    reran = {"called": False}
    w2.stages[3]._run = lambda *a, **k: reran.__setitem__("called", True)
    w2._restore_session(rp)  # must not raise

    assert reran["called"] is True  # fell back to a re-run
    assert "regrids to" in w2.stages[3].status_label.text()  # and said why


def test_rehydrated_result_matches_a_rerun_of_the_same_session(
    qtbot, experiment_h5, experiment, tmp_path
):
    """The fast path must not be a different answer from the slow path.

    Reloading a saved result skips the solver, so the two routes into the
    Retrieve stage have to agree: same measured trace, same retrieved spectrum,
    same trace error. If they can diverge, the reload is a silent second
    implementation of the retrieval and cannot be trusted.
    """
    from croak import save

    w1 = Wizard()
    qtbot.addWidget(w1)
    s1 = w1.state
    s1.load.frog_path = experiment_h5
    w1.stages[0]._populate_frog()
    s1.load.interaction = experiment.interaction
    w1.stages[0]._load_preview()
    s1.preproc.lam_min_nm = experiment.lam_min / 1e-9
    s1.preproc.lam_max_nm = experiment.lam_max / 1e-9
    s1.stage = 3
    s1.retrieve.solver = "copra"
    s1.retrieve.maxiters = 20
    s1.stage = 4
    with qtbot.waitSignal(s1.result_changed, timeout=60000):
        w1.stages[3]._run()
    rp = str(tmp_path / "result.h5")
    save.save_result(s1.result, rp, processed=s1.processed, force=True)
    save.save_options(s1.to_options(), str(tmp_path / "options.toml"))

    w2 = Wizard()
    qtbot.addWidget(w2)
    w2.state.from_options(save.load_options(str(tmp_path / "options.toml")))
    w2._rebuild_stages()
    with qtbot.waitSignal(w2.state.result_changed, timeout=60000):
        w2._restore_session(rp)

    # the trace the reloaded session preprocessed is the one that was fitted
    np.testing.assert_allclose(w2.state.tracedata.trace, s1.tracedata.trace)
    np.testing.assert_allclose(w2.state.result.spectrum, s1.result.spectrum)
    assert w2.state.result.error == pytest.approx(s1.result.error)
    # and the derived quantities the stage displays agree too
    assert w2.state.processed.fwhm_retr == pytest.approx(s1.processed.fwhm_retr)


def _load_simulated_session(w, simulated_scan_h5, simulated_truth):
    """Drive the simulated loader page to the marginal-check stage."""
    w.load_simulated()
    sim = w.simulated
    sim.path_edit.setText(simulated_scan_h5)
    band = (
        simulated_truth.lam_min / 1e-9 - 10,
        simulated_truth.lam_max / 1e-9 + 10,
    )
    sim.band.set_values(*band)
    sim._band = band
    sim._preview()
    sim.advance()
    return band


def test_simulated_session_saves_its_trace_provenance(
    qtbot, simulated_scan_h5, simulated_truth, tmp_path
):
    """A simulated session records which scan (and how) it was loaded from."""
    from croak import save

    w = Wizard()
    qtbot.addWidget(w)
    band = _load_simulated_session(w, simulated_scan_h5, simulated_truth)

    opts_path = str(tmp_path / "options.toml")
    save.save_options(w.state.to_options(), opts_path)
    loaded = save.load_options(opts_path)

    assert loaded["entry"] == "simulated"
    assert loaded["simulated"]["frog_path"] == simulated_scan_h5
    assert loaded["simulated"]["interaction"] == "pg"
    assert loaded["simulated"]["lam_min_nm"] == pytest.approx(band[0])
    assert loaded["simulated"]["lam_max_nm"] == pytest.approx(band[1])


def test_load_previous_restores_a_simulated_session(
    qtbot, simulated_scan_h5, simulated_truth, tmp_path
):
    """A saved simulated session reloads its own scan — not the empty Load stage."""
    from croak import save

    w1 = Wizard()
    qtbot.addWidget(w1)
    _load_simulated_session(w1, simulated_scan_h5, simulated_truth)
    s1 = w1.state
    s1.preproc.lam_min_nm = simulated_truth.lam_min / 1e-9
    s1.preproc.lam_max_nm = simulated_truth.lam_max / 1e-9
    s1.stage = 3  # preprocess
    assert s1.tracedata is not None
    s1.retrieve.solver = "copra"
    s1.retrieve.maxiters = 40
    s1.stage = 4  # retrieve
    with qtbot.waitSignal(s1.result_changed, timeout=60000):
        w1.stages[3]._run()
    save.save_result(
        s1.result, str(tmp_path / "result.h5"), processed=s1.processed, force=True
    )
    save.save_options(s1.to_options(), str(tmp_path / "options.toml"))

    # Restore into a fresh wizard (emulating _load_options without the dialog).
    w2 = Wizard()
    qtbot.addWidget(w2)
    w2.state.from_options(save.load_options(str(tmp_path / "options.toml")))
    w2._rebuild_stages()
    with qtbot.waitSignal(w2.state.result_changed, timeout=60000):
        w2._restore_session(str(tmp_path / "result.h5"))

    assert w2.state.entry == "simulated"
    assert w2.state.simulated.frog_path == simulated_scan_h5
    assert w2.state.load_data is not None
    assert w2.state.tracedata is not None
    assert w2.state.result is not None
    assert w2.state.stage == 4
    # the loader page's controls reflect the restored options, so going Back and
    # reloading does not silently revert to the defaults
    assert w2.simulated.path_edit.text() == simulated_scan_h5
    assert w2.simulated.interaction.currentText() == "pg"
    # the rehydrated result is the saved one (no solver re-run needed)
    np.testing.assert_allclose(w2.state.result.spectrum, w1.state.result.spectrum)


def test_load_previous_restores_a_raw_direct_simulated_session(
    qtbot, simulated_scan_h5, tmp_path
):
    """The raw simulated path restores its native-grid trace, skipping preprocess."""
    from croak import save

    w1 = Wizard()
    qtbot.addWidget(w1)
    w1.load_simulated()
    w1.simulated.path_edit.setText(simulated_scan_h5)
    w1.simulated.raw_direct.setChecked(True)
    w1.simulated._preview()
    w1.simulated.advance()
    assert w1.state.stage == 4  # straight to Retrieve
    opts_path = str(tmp_path / "options.toml")
    save.save_options(w1.state.to_options(), opts_path)

    w2 = Wizard()
    qtbot.addWidget(w2)
    w2.state.from_options(save.load_options(opts_path))
    w2._rebuild_stages()
    w2.stages[3]._run = lambda *a, **k: None  # don't re-run the solver here
    w2._restore_session()

    assert w2.state.simulated.raw_direct is True
    assert w2.state.tracedata is not None
    np.testing.assert_allclose(w2.state.tracedata.omega, w1.state.tracedata.omega)


def test_retarget_reports_that_simulated_sessions_cannot_move(
    qtbot, simulated_scan_h5, simulated_truth
):
    """Retarget is experimental-only; a simulated session is told, not replayed."""
    w = Wizard()
    qtbot.addWidget(w)
    _load_simulated_session(w, simulated_scan_h5, simulated_truth)
    replayed = {"called": False}
    w._restore_session = lambda *a, **k: replayed.__setitem__("called", True)

    w._retarget_session("/data/run99")

    assert replayed["called"] is False
    assert (
        "cannot retarget a simulated session"
        in w.stages[w.state.stage - 1].status_label.text()
    )


def test_dispersion_round_trips_through_wizard_state(tmp_path):
    """The [dispersion] settings survive WizardState → options.toml → WizardState."""
    import croak
    from croak import save
    from croak.session.params import BeamPathMirror

    src = WizardState()
    src.dispersion.gdd_fs2 = 123.0
    src.dispersion.tod_fs3 = -45.0
    src.dispersion.material_thickness_mm = {"SiO2": 0.5}
    src.dispersion.gas_name = croak.gases.GASES[0]
    src.dispersion.gas_path_cm = 10.0
    src.dispersion.mirrors = [
        BeamPathMirror(
            name=next(iter(croak.mirrors.BUILTIN_MIRRORS)),
            bounces=2,
            direction="remove",
        )
    ]
    path = str(tmp_path / "options.toml")
    save.save_options(src.to_options(), path)

    dst = WizardState()
    dst.from_options(save.load_options(path))
    assert dst.dispersion == src.dispersion


def test_dispersion_stage_syncs_and_restores_controls(qtbot):
    """The dispersion stage mirrors its controls to/from DispersionParams."""
    import croak
    from croak.session.params import BeamPathMirror

    w = Wizard()
    qtbot.addWidget(w)
    stage = w.stages[4]  # StageDispersion
    w.state.dispersion.gdd_fs2 = 80.0
    w.state.dispersion.material_thickness_mm = {"SiO2": 0.3}
    w.state.dispersion.mirrors = [
        BeamPathMirror(
            name=next(iter(croak.mirrors.BUILTIN_MIRRORS)),
            bounces=2,
            direction="remove",
        )
    ]
    # restore params -> controls
    stage._restore_from_params()
    assert stage.gdd.value() == pytest.approx(80.0)
    assert stage._mat_ctrls["SiO2"].value() == pytest.approx(0.3)
    assert len(stage.mirror_stack.to_specs()) == 1
    # sync controls -> params (round-trips the mirror back out as a spec)
    w.state.dispersion.gdd_fs2 = 0.0  # clobber, then re-sync from controls
    stage._sync_to_params()
    assert w.state.dispersion.gdd_fs2 == pytest.approx(80.0)
    assert w.state.dispersion.material_thickness_mm == {"SiO2": 0.3}
    assert w.state.dispersion.mirrors[0].bounces == 2
    assert w.state.dispersion.mirrors[0].direction == "remove"


def test_retrieve_stage_retarget_reruns_on_new_dataset(qtbot, tmp_path, experiment):
    """The Retrieve stage's retarget replays the session on a copied dataset folder."""
    import shutil

    from conftest import write_experiment_h5

    # Lay the dataset out under run01, drive a session up to a cleaned trace.
    run01 = tmp_path / "run01"
    run01.mkdir()
    h5 = write_experiment_h5(run01 / "acq.h5", experiment)

    w = Wizard()
    qtbot.addWidget(w)
    s = w.state
    s.load.frog_path = h5
    w.stages[0]._populate_frog()
    s.load.interaction = experiment.interaction
    w.stages[0]._load_preview()
    s.preproc.lam_min_nm = experiment.lam_min / 1e-9
    s.preproc.lam_max_nm = experiment.lam_max / 1e-9
    s.retrieve.solver = "copra"
    s.retrieve.maxiters = 10
    s.stage = 3  # preprocess → tracedata
    assert s.tracedata is not None

    # Copy the dataset folder and retarget the session onto it (the handler the
    # Retrieve-stage "Retarget…" button reaches via the retarget_requested signal).
    run02 = tmp_path / "run02"
    shutil.copytree(run01, run02)
    with qtbot.waitSignal(s.result_changed, timeout=60000):
        s.retarget_requested.emit(str(run02))

    assert s.load.frog_path == str(run02 / "acq.h5")
    assert s.load_data is not None
    assert s.result is not None  # reran load → preprocess → retrieve on the new data
    assert s.stage == 4


def test_load_autodetects_datasets_and_units(qtbot, experiment_h5, experiment):
    w = Wizard()
    qtbot.addWidget(w)
    stage = w.stages[0]
    w.state.load.frog_path = experiment_h5
    stage._populate_frog()
    p = w.state.load
    # shortest-name matching picks the right datasets over the decoy
    assert p.ifrog_name == "trace"
    assert p.lam_name == "wavelength"
    assert p.scan_name == "delay"
    # units auto-guessed from values/name
    assert p.lam_unit == "nm"
    assert p.scan_unit == "fs"
    assert p.scan_type == "delay"


def test_assemble_and_seed_defaults(experiment_h5, experiment):
    state = WizardState()
    p = state.load
    p.frog_path = experiment_h5
    p.ifrog_name, p.lam_name, p.scan_name = "trace", "wavelength", "delay"
    p.lam_unit, p.scan_unit, p.scan_type = "nm", "fs", "delay"
    p.interaction = experiment.interaction
    data = assemble_load_data(p)
    assert data["trace"].shape == experiment.trace.shape
    assert data["input_unit"] == "delay"
    assert data["lam"].max() < 2e-6

    state.set_load_data(data)
    seed_preproc_defaults(state)
    pp = state.preproc
    # τm window seeded from the scan range; trange is twice that span
    assert pp.taum_min_fs < 0 < pp.taum_max_fs
    assert pp.trange_fs == pytest.approx(2 * (pp.taum_max_fs - pp.taum_min_fs))
    # grid λ seeded from the trace λ range (× ω0 scale = 1 for PG)
    lam_nm = data["lam"] / 1e-9
    assert pp.lam_min_nm == pytest.approx(np.floor(lam_nm.min()), abs=1)
    assert pp.lamm_min_nm == pytest.approx(np.floor(lam_nm.min()), abs=1)


def test_preprocess_widgets_show_seeded_wavelengths(qtbot, experiment_h5, experiment):
    w = Wizard()
    qtbot.addWidget(w)
    state = w.state
    state.load.frog_path = experiment_h5
    w.stages[0]._populate_frog()
    state.load.interaction = experiment.interaction
    w.stages[0]._load_preview()
    lam_nm = state.load_data["lam"] / 1e-9
    state.stage = 3  # preprocess: triggers _seed_ranges -> widgets
    lo, hi = w.stages[2].lam_range.values()
    # the grid-λ widget must reflect the seeded trace extents, not the 700-900 default
    assert lo == pytest.approx(np.floor(lam_nm.min()), abs=2)
    assert hi == pytest.approx(np.ceil(lam_nm.max()), abs=2)
    assert (lo, hi) != (700.0, 900.0)


def test_preprocess_interactive_change_recomputes_off_thread(
    qtbot, experiment_h5, experiment
):
    """An interactive (debounced) change recomputes via the worker thread."""
    w = Wizard()
    qtbot.addWidget(w)
    state = w.state
    state.load.frog_path = experiment_h5
    w.stages[0]._populate_frog()
    state.load.interaction = experiment.interaction
    w.stages[0]._load_preview()
    state.stage = 3  # on_enter runs the initial (synchronous) preprocess
    stage = w.stages[2]
    assert state.tracedata is not None
    td_before = state.tracedata
    assert stage._view.attached  # the reusable view was built on the first draw

    # Narrow the grid-λ range: schedules the 200 ms debounce, which fires the
    # off-thread recompute. Spinning the event loop runs both timer and worker.
    lo, hi = stage.lam_range.values()
    with qtbot.waitSignal(state.tracedata_changed, timeout=10000):
        stage._on_lam_range(lo + 5.0, hi - 5.0)
    assert state.tracedata is not td_before  # a fresh trace (fast preview) produced
    # The fast preview is followed by an exact-baseline refine worker; wait for
    # both to finish, then the worker list is cleaned up.
    qtbot.waitUntil(lambda: not stage._workers, timeout=10000)
    assert not stage._workers


def test_retrieve_solver_combo_greys_regularisation(qtbot):
    """COPRA (no reg) disables the Regularisation group; LBFGS enables it."""
    w = Wizard()
    qtbot.addWidget(w)
    rs = w.stages[3]
    rs.solver_combo.setCurrentText("copra")
    assert not rs.reg_box.isEnabled()
    rs.solver_combo.setCurrentText("copra-jax")
    assert not rs.reg_box.isEnabled()
    rs.solver_combo.setCurrentText("lbfgs")
    assert rs.reg_box.isEnabled()


def test_retrieve_view_restored_on_reentry(qtbot, experiment_h5, experiment):
    """Navigating back to preprocess and forward keeps the result and re-shows it."""
    w = Wizard()
    qtbot.addWidget(w)
    st = w.state
    st.load.frog_path = experiment_h5
    w.stages[0]._populate_frog()
    st.load.interaction = experiment.interaction
    w.stages[0]._load_preview()
    st.stage = 3  # preprocess (synchronous)
    rs = w.stages[3]
    rs.state.retrieve.solver = "copra"
    rs.state.retrieve.maxiters = 15
    st.stage = 4  # retrieve
    with qtbot.waitSignal(st.result_changed, timeout=30000):
        rs._run()
    assert st.result is not None
    assert w.next_btn.isEnabled()  # can proceed to dispersion

    # simulate a fresh stage instance that hasn't drawn the result yet
    rs._base_result = None
    rs.save_btn.setEnabled(False)
    rs.on_enter()
    assert rs._base_result is st.result  # view restored
    assert rs.save_btn.isEnabled()
    assert w.next_btn.isEnabled()


def test_retrieve_reuse_previous_result_guess(qtbot, experiment_h5, experiment):
    """'Reuse previous result' seeds the next run from the prior spectrum."""
    w = Wizard()
    qtbot.addWidget(w)
    st = w.state
    st.load.frog_path = experiment_h5
    w.stages[0]._populate_frog()
    st.load.interaction = experiment.interaction
    w.stages[0]._load_preview()
    st.stage = 3
    rs = w.stages[3]
    rs.state.retrieve.solver = "copra"
    rs.state.retrieve.maxiters = 15
    st.stage = 4
    with qtbot.waitSignal(st.result_changed, timeout=30000):
        rs._run()
    first = st.result.spectrum.copy()

    # now reuse it as the guess for another run
    rs.state.retrieve.reuse_result = True
    rs.state.retrieve.maxiters = 5
    with qtbot.waitSignal(st.result_changed, timeout=30000):
        rs._run()
    # a result was produced on the same grid (the reuse path ran without error)
    assert st.result.spectrum.shape == first.shape


def test_retrieve_initial_guess_selector_is_mutually_exclusive(qtbot):
    """Selecting one GUI seed clears every previously selected seed flag."""
    w = Wizard()
    qtbot.addWidget(w)
    rs = w.stages[3]
    p = rs.state.retrieve

    rs.guess_combo.setCurrentText("Random initial phase")
    assert p.random_init
    assert not p.perfect_init and not p.truth_init and not p.reuse_result

    rs.guess_combo.setCurrentText("Truth (known complex field)")
    assert p.truth_init
    assert not p.random_init and not p.perfect_init and not p.reuse_result

    rs.guess_combo.setCurrentText("Reuse previous result")
    assert p.reuse_result
    assert not p.random_init and not p.perfect_init and not p.truth_init


def test_retrieve_reuse_passes_the_previous_fitted_extras(qtbot, monkeypatch):
    """'Reuse previous result' hands the last result on as the warm-start seed."""
    from croak.gui import stage_retrieve as sr

    w = Wizard()
    qtbot.addWidget(w)
    st = w.state
    rs = w.stages[3]

    class _Grid:
        n = 8

    class _TD:
        grid = _Grid()

    st.tracedata = _TD()
    previous = _seed_result_for_reuse(n=_Grid.n)
    st.result = previous
    st.retrieve.reuse_result = True
    st.retrieve.fit_tau0 = True

    captured = {}

    class _Signal:
        def connect(self, _slot):
            pass

    class _Worker:
        """Stands in for the threaded worker: records its kwargs, runs nothing."""

        def __init__(self, td, p, **kwargs):
            captured.update(kwargs)

        def __getattr__(self, _name):  # every signal connects; start() is a no-op
            return _Signal() if _name != "start" else (lambda: None)

    monkeypatch.setattr(sr, "RetrievalWorker", _Worker)
    rs._run()

    assert captured["extras_seed"] is previous
    np.testing.assert_allclose(captured["guess_override"], previous.spectrum)
    # the status names what was seeded, so a warm start is never silent
    assert "τ₀" in rs.status_label.text()


def _seed_result_for_reuse(*, n: int):
    """A previous result on an ``n``-point grid carrying a fitted τ₀."""
    import croak
    from croak.result import RetrievalResult

    return RetrievalResult(
        spectrum=np.ones(n, dtype=complex),
        grid=croak.Grid(n, dt=1e-15),
        delays=np.zeros(3),
        interaction="pg",
        algorithm="lbfgs-ad",
        error=1e-3,
        tau0=1.5e-15,
    )


def test_retrieve_status_reports_the_fitted_smearing_width(qtbot):
    """A fitted smearing strength is reported as a delay width and its factor."""
    from croak.session.pipeline import smearing_kernel

    w = Wizard()
    qtbot.addWidget(w)
    rs = w.stages[3]
    p = rs.state.retrieve
    p.smearing = True
    p.smear_hole_diameter_mm = 1.0
    p.smear_hole_spacing_mm = 2.0

    class _TD:
        interaction = "pg"
        omega0_pulse = 2.0 * np.pi * 3e8 / 800e-9

    td = _TD()
    text = rs._smear_summary(1.457, td)

    sigma_fs = smearing_kernel(p, td).sigma_delta * 1.457 / 1e-15
    assert f"{sigma_fs:.2f} fs" in text
    assert "×1.457" in text


def test_position_scan_seeds_delay(experiment_h5, experiment):

    # rewrite the file with a position axis named "z" (mm)
    state = WizardState()
    p = state.load
    p.frog_path = experiment_h5
    p.ifrog_name, p.lam_name, p.scan_name = "trace", "wavelength", "delay"
    p.scan_type = "position"  # treat the (fs-valued) delay column as a position
    p.lam_unit, p.scan_unit = "nm", "fs"
    p.interaction = experiment.interaction
    data = assemble_load_data(p)
    assert data["input_unit"] == "position"
    state.set_load_data(data)
    seed_preproc_defaults(state)
    # τ = 2z/c → seeded delay window differs from the raw scan range
    assert state.preproc.taum_min_fs < 0 < state.preproc.taum_max_fs


def test_full_flow(qtbot, experiment_h5, experiment):
    w = Wizard()
    qtbot.addWidget(w)
    state = w.state

    # Stage 1: load via the real flow
    state.load.frog_path = experiment_h5
    w.stages[0]._populate_frog()
    state.load.interaction = experiment.interaction
    w.stages[0]._load_preview()
    assert state.load_data is not None
    assert w.next_btn.isEnabled()

    # narrow the retrieval grid to the useful band (keeps the test fast)
    state.preproc.lam_min_nm = experiment.lam_min / 1e-9
    state.preproc.lam_max_nm = experiment.lam_max / 1e-9

    # Stage 3: preprocess
    state.stage = 3
    assert state.tracedata is not None

    # Stage 4: retrieve (threaded)
    state.stage = 4
    state.retrieve.solver = "copra"
    state.retrieve.maxiters = 40
    with qtbot.waitSignal(state.result_changed, timeout=60000):
        w.stages[3]._run()
    assert state.result is not None
    assert state.result.error < 0.2

    # Stage 5: dispersion tuning
    state.stage = 5
    disp = w.stages[4]
    disp.gdd.set_value(10.0)
    disp._changed()
    assert "fs" in disp.fwhm_label.text()

    # the mirror stack: a chirped-mirror row registers and updates the pulse
    from croak import dispersion

    row = disp.mirror_stack.add_row()
    row.type_combo.setCurrentText("PC70")  # covers 800 nm
    assert row.key in dispersion.MIRRORS
    assert row.key in dispersion.MIRROR_AMPLITUDES  # reflectivity is modelled too
    assert "nm" in row.range_label.text()
    row.dir_combo.setCurrentText("add")
    row.bounces.setValue(4)
    assert disp.mirror_stack.bounces_dict() == {row.key: 4}
    assert "fs" in disp.fwhm_label.text()
    # 'Auto' picks the peak-power bounce count and leaves the pulse finite
    disp._auto_row(row)
    assert 0 <= row.bounces.value() <= 200
    assert "fs" in disp.fwhm_label.text()


def test_dispersion_gas_updates_pulse(qtbot):
    """Selecting a gas registers it, and pressure/path-length update the pulse."""
    import croak
    from croak.maths import wlfreq

    w = Wizard()
    qtbot.addWidget(w)
    g = croak.Grid(48, dt=0.6e-15)
    ew = croak.gaussian_pulse(g, 7e-15)
    delays = np.linspace(-30e-15, 30e-15, 30)
    trace = croak.maketrace(g.omega, delays, ew, "pg")
    res = croak.retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        guess=ew,
        omega0=float(wlfreq(800e-9)),
        maxiters=30,
    )
    w.state.set_result(res, croak.process_result(res, measured=trace))
    disp = w.stages[4]
    disp.on_enter()

    # selecting a gas registers a pressure-baked material under the active-gas key
    disp.gas_combo.setCurrentText("Helium")
    assert disp._gas_key is not None
    assert disp._gas_name == "Helium"

    # a higher pressure re-registers; a non-zero path length updates the pulse
    disp.gas_pressure.set_value(5.0)
    disp._on_gas_pressure(5.0)
    disp.gas_path.set_value(50.0)
    disp._changed()
    assert "fs" in disp.fwhm_label.text()
    assert disp._gas_key in disp._extra_dispersion()[0]

    # Auto stays within the path control's bounds and leaves the pulse finite
    disp._auto_gas()
    lo, hi = disp.gas_path.bounds()
    assert lo <= disp.gas_path.value() <= hi
    assert "fs" in disp.fwhm_label.text()

    # switching back to none unregisters the gas
    disp.gas_combo.setCurrentText("(none)")
    assert disp._gas_key is None


def test_dispersion_mirror_stack_removes_beam_path_mirrors(qtbot):
    """The mirror stack back-propagates several coating mirrors at once."""
    import croak
    from croak import dispersion
    from croak.maths import wlfreq

    w = Wizard()
    qtbot.addWidget(w)
    g = croak.Grid(48, dt=0.6e-15)
    ew = croak.gaussian_pulse(g, 7e-15)
    delays = np.linspace(-30e-15, 30e-15, 30)
    trace = croak.maketrace(g.omega, delays, ew, "pg")
    res = croak.retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        guess=ew,
        omega0=float(wlfreq(800e-9)),
        maxiters=30,
    )
    w.state.set_result(res, croak.process_result(res, measured=trace))
    disp = w.stages[4]
    disp.on_enter()
    stack = disp.mirror_stack

    # a typical stack: 1× Si @ 0°, 6× MgF2/Al @ 0°, 2× MgF2/Al @ 45° (s), all removed
    specs = [("Si @ 0°", "s", 1), ("MgF2/Al @ 0°", "s", 6), ("MgF2/Al @ 45°", "s", 2)]
    rows = []
    for name, pol, n in specs:
        row = stack.add_row()
        row.type_combo.setCurrentText(name)
        row.pol_combo.setCurrentText(pol)
        row.bounces.setValue(n)
        rows.append(row)

    bounces = stack.bounces_dict()
    assert len(bounces) == 3
    # 'remove' is the default direction → negative (back-propagating) counts
    assert sorted(bounces.values()) == [-6, -2, -1]
    # coatings register phase *and* amplitude (reflectivity is divided out)
    for row in rows:
        assert row.key in dispersion.MIRRORS
        assert row.key in dispersion.MIRROR_AMPLITUDES
    # the stack actually changes the pulse and keeps the labels finite
    assert not np.allclose(disp._modified().spectrum, disp._base)
    assert "fs" in disp.fwhm_label.text()

    # flipping one row to 'add' flips the sign of its contribution
    rows[0].dir_combo.setCurrentText("add")
    assert stack.bounces_dict()[rows[0].key] == 1

    # removing a row unregisters both its phase and amplitude
    key = rows[2].key
    rows[2].remove_requested.emit(rows[2])
    assert key not in dispersion.MIRRORS
    assert key not in dispersion.MIRROR_AMPLITUDES
    assert len(stack.bounces_dict()) == 2

    # reset clears the whole stack and unregisters its rows' keys
    keys = [row.key for row in rows]
    disp._reset()
    assert stack.bounces_dict() == {}
    assert all(k not in dispersion.MIRRORS for k in keys)
    assert all(k not in dispersion.MIRROR_AMPLITUDES for k in keys)


def test_sci_spinbox_decade_aware_stepping(qtbot):
    from croak.gui.widgets import SciSpinBox

    sb = SciSpinBox(0.0, 1.0, 5e-3)
    qtbot.addWidget(sb)
    assert sb.textFromValue(5e-3) == "5.0e-03"
    sb.stepBy(1)
    assert sb.value() == pytest.approx(6e-3)  # within the decade, not 1e-2
    sb.setValue(9e-3)
    sb.stepBy(1)
    assert sb.value() == pytest.approx(1e-2)  # rolls up a decade
    sb.setValue(1e-2)
    sb.stepBy(-1)
    assert sb.value() == pytest.approx(9e-3)  # rolls down a decade
    # arbitrary mantissa survives a round-trip through the text field
    assert sb.valueFromText("7e-4") == pytest.approx(7e-4)


def test_reload_same_trace_keeps_preproc(qtbot, experiment_h5, experiment):
    """Reloading the *same* trace must not discard the user's preprocess windows."""
    w = Wizard()
    qtbot.addWidget(w)
    state = w.state
    state.load.frog_path = experiment_h5
    w.stages[0]._populate_frog()
    state.load.interaction = experiment.interaction
    w.stages[0]._load_preview()

    # user narrows the grid and turns on a filter
    state.preproc.lam_min_nm = 333.0
    state.preproc.filter_dc = True

    # reloading the same trace (e.g. after toggling a correction) keeps them
    w.stages[0]._load_preview()
    assert state.preproc.lam_min_nm == pytest.approx(333.0)
    assert state.preproc.filter_dc is True

    # changing the interaction makes it a *different* trace → defaults reseeded
    state.load.interaction = "shg" if experiment.interaction != "shg" else "pg"
    w.stages[0]._load_preview()
    assert state.preproc.lam_min_nm != pytest.approx(333.0)
    assert state.preproc.filter_dc is False


def test_preprocess_defringe_baseline_controls(qtbot, experiment_h5, experiment):
    """The new de-fringe / baseline controls drive params, gate, and round-trip."""
    w = Wizard()
    qtbot.addWidget(w)
    state = w.state
    state.load.frog_path = experiment_h5
    w.stages[0]._populate_frog()
    state.load.interaction = experiment.interaction
    w.stages[0]._load_preview()
    pre = w.stages[2]

    # legacy filters are relabelled
    assert pre.fringes_check.text().endswith("(legacy)")
    assert pre.dc_check.text().endswith("(legacy)")

    # dependent controls start disabled (their toggles are off)
    assert not pre.defringe_frac_spin.isEnabled()
    assert not pre.baseline_smooth_spin.isEnabled()

    # de-fringe: toggle drives the param and enables its controls
    pre.defringe_check.setChecked(True)
    assert state.preproc.defringe is True
    assert pre.defringe_frac_spin.isEnabled()
    assert pre.defringe_alias_combo.isEnabled()
    pre.defringe_frac_spin.setValue(0.35)
    assert state.preproc.defringe_fraction == pytest.approx(0.35)

    # de-fringe and the legacy fringe notch are mutually exclusive in the UI
    pre.fringes_check.setChecked(True)
    assert state.preproc.filter_fringes is True
    assert state.preproc.defringe is False
    assert pre.defringe_check.isChecked() is False

    # baseline: toggle drives the param and gates its four parameter controls
    pre.baseline_check.setChecked(True)
    assert state.preproc.baseline is True
    for ctl in (
        pre.baseline_smooth_spin,
        pre.baseline_ratio_spin,
        pre.baseline_tauex_spin,
        pre.baseline_clip_check,
    ):
        assert ctl.isEnabled()
    pre.baseline_smooth_spin.setValue(5e3)
    pre.baseline_tauex_spin.setValue(8.0)
    pre.baseline_clip_check.setChecked(True)
    assert state.preproc.baseline_smoothness == pytest.approx(5e3)
    assert state.preproc.baseline_tau_exclude_fs == pytest.approx(8.0)
    assert state.preproc.baseline_clip is True

    # reset restores defaults into both params and widgets (incl. gating)
    pre._reset_defaults()
    assert state.preproc.defringe is False
    assert state.preproc.baseline is False
    assert not pre.defringe_check.isChecked()
    assert not pre.baseline_check.isChecked()
    assert not pre.defringe_frac_spin.isEnabled()
    assert not pre.baseline_smooth_spin.isEnabled()


def test_preprocess_reset_button_restores_defaults(qtbot, experiment_h5, experiment):
    w = Wizard()
    qtbot.addWidget(w)
    state = w.state
    state.load.frog_path = experiment_h5
    w.stages[0]._populate_frog()
    state.load.interaction = experiment.interaction
    w.stages[0]._load_preview()
    default_lam = state.preproc_defaults.lam_min_nm

    state.stage = 3
    pre = w.stages[2]
    state.preproc.lam_min_nm = 321.0
    state.preproc.filter_dc = True
    pre._reset_defaults()
    assert state.preproc.lam_min_nm == pytest.approx(default_lam)
    assert state.preproc.filter_dc is False
    # the widgets reflect the reset, not just the params
    assert not pre.dc_check.isChecked()


def test_retrieve_reset_button_restores_defaults(qtbot, experiment_h5, experiment):
    w = Wizard()
    qtbot.addWidget(w)
    ret = w.stages[3]
    state = w.state
    state.retrieve.solver = "lm"
    state.retrieve.maxiters = 999
    state.retrieve.R_omega = False  # a non-default (the default is now True)
    ret._seed_widgets()  # reflect the changes in the widgets
    ret._reset_defaults()
    assert state.retrieve.solver == "warm-lbfgs"
    assert state.retrieve.maxiters == 300
    assert state.retrieve.R_omega is True
    assert ret.solver_combo.currentText() == "warm-lbfgs"
    assert ret.maxiters_spin.value() == 300
    assert ret.romega_check.isChecked()


# ---------------------------------------------------------------------------
# Stage 6 — combine and save uncertainty estimates
# ---------------------------------------------------------------------------
def test_stage6_combine_estimates(qtbot):
    """Ticking two retained estimates and combining produces a 'combined' result."""
    w = Wizard()
    qtbot.addWidget(w)
    stage = w.stages[5]
    rng = np.random.default_rng(0)
    w.state.uncertainty_results = {
        "parametric": _uresult("parametric", 0.05e-15, rng),
        "thickness": _uresult("thickness", 0.12e-15, rng),
    }
    stage._refresh_estimates()
    assert stage.est_list.count() == 2
    for i in range(stage.est_list.count()):
        stage.est_list.item(i).setCheckState(Qt.CheckState.Checked)
    stage._combine()
    combined = w.state.uncertainty_results.get("combined")
    assert combined is not None
    assert combined.method == "combined"
    assert set(combined.components) == {"parametric", "thickness"}
    assert stage.est_list.count() == 3  # combined now listed too


def test_stage6_save_appends_uncertainty(qtbot, tmp_path, monkeypatch):
    """The stage-6 Save writes the intervals (and bands) into result.h5."""
    from PyQt6.QtWidgets import QFileDialog

    import croak
    from croak.maths import wlfreq

    w = Wizard()
    qtbot.addWidget(w)
    stage = w.stages[5]
    g = croak.Grid(48, dt=0.6e-15)
    ew = croak.gaussian_pulse(g, 7e-15)
    delays = np.linspace(-30e-15, 30e-15, 30)
    trace = croak.maketrace(g.omega, delays, ew, "pg")
    res = croak.retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        guess=ew,
        omega0=float(wlfreq(800e-9)),
        maxiters=30,
    )
    w.state.set_result(res, croak.process_result(res, measured=trace))
    rng = np.random.default_rng(1)
    w.state.set_uncertainty_result(_uresult("parametric", 0.05e-15, rng))
    w.state.set_uncertainty_result(_uresult("thickness", 0.12e-15, rng))
    stage._refresh_estimates()

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )
    stage._save()

    data = croak.save.load_result(str(tmp_path / "result.h5"))
    assert set(data["uncertainty"]) == {"parametric", "thickness"}
    assert "band_median" in data["uncertainty"]["thickness"]
    assert (tmp_path / "options.toml").exists()


def test_preprocess_reseeds_every_control_on_entry(qtbot):
    """Entering Preprocess pushes the parameters back into *every* widget.

    Regression: the controls are bound one-way (widget -> parameter) through
    edge-triggered signals, so a parameter reset elsewhere (``seed_preproc_defaults``
    on a new trace) used to leave the checkboxes showing the user's old ticks while
    a different set of options was actually applied — the user had to un-tick and
    re-tick each one to force it through.
    """
    w = Wizard()
    qtbot.addWidget(w)
    w.show_synthetic()
    w.synthetic.advance()
    w.state.stage = 3  # preprocess
    stage = w.stages[2]

    # mutate the parameters behind the widgets' backs, as a re-load would
    p = w.state.preproc
    p.defringe = not stage.defringe_check.isChecked()
    p.baseline = not stage.baseline_check.isChecked()
    p.use_threshold = not stage.threshold_check.isChecked()
    p.subsample_tau = 3
    p.defringe_fraction = 0.42

    stage.on_enter()

    assert stage.defringe_check.isChecked() is p.defringe
    assert stage.baseline_check.isChecked() is p.baseline
    assert stage.threshold_check.isChecked() is p.use_threshold
    assert stage.factor_spin.value() == 3
    assert stage.defringe_frac_spin.value() == pytest.approx(0.42)
    # dependent controls follow their parent toggle
    assert stage.defringe_frac_spin.isEnabled() is p.defringe
    assert stage.baseline_smooth_spin.isEnabled() is p.baseline


def test_simulated_reload_keeps_preproc_settings(qtbot, simulated_scan_h5):
    """Re-loading the same simulated scan preserves the user's preprocess choices.

    Same rule as the experimental loader: only a genuinely different trace (here a
    narrower loaded λ band, which moves the data-seeded defaults) resets them.
    """
    w = Wizard()
    qtbot.addWidget(w)
    w.load_simulated()
    sim = w.simulated
    sim.path_edit.setText(simulated_scan_h5)
    sim._populate_file_options(simulated_scan_h5)
    sim.advance()

    # the user turns on a filter on the preprocess stage
    w.state.preproc.defringe = True
    w.state.preproc.use_threshold = True

    # Back to the loader and forward again with the same options -> settings kept
    w.go_back()  # stage 2 -> the simulated loader (this session's "stage 1")
    assert w.stack.currentWidget() is w.simulated
    sim.advance()
    assert w.state.preproc.defringe is True
    assert w.state.preproc.use_threshold is True

    # a different loaded band is a different trace -> data-seeded defaults return
    w.go_back()
    lo, hi = sim.band.values()
    sim.band.set_values(lo + 5.0, hi - 5.0)
    sim._band = (lo + 5.0, hi - 5.0)
    sim.advance()
    assert w.state.preproc.defringe is False
    assert w.state.preproc.use_threshold is False


def test_entry_pages_use_the_wizard_footer(qtbot, simulated_scan_h5):
    """Both entry pages navigate through the shared bottom-right Back/Next."""
    w = Wizard()
    qtbot.addWidget(w)

    # -- synthetic generator --
    w.show_synthetic()
    assert w.footer.isVisible() or not w.isVisible()  # chrome shown, not hidden
    assert w._header_bar.isVisibleTo(w)
    assert w.next_btn.text() == "Generate →"
    assert w.next_btn.isEnabled()
    w.go_back()
    assert w.stack.currentWidget() is w.welcome
    w.show_synthetic()
    w.go_next()  # Next == Generate
    assert w.state.stage == 2
    assert w.stack.currentWidget() is w.stages[1]

    # -- simulated loader --
    w.load_simulated()
    sim = w.simulated
    assert not w.next_btn.isEnabled()  # nothing previewed yet
    # "&&" is Qt's escape for a literal "&" in a button label
    assert w.next_btn.text() == "Load && continue →"
    sim.path_edit.setText(simulated_scan_h5)
    sim._populate_file_options(simulated_scan_h5)
    sim._preview()
    assert w.next_btn.isEnabled()
    # the raw path changes where Next lands, so it relabels
    sim.raw_direct.setChecked(True)
    assert w.next_btn.text() == "Load raw → Retrieve →"
    sim.raw_direct.setChecked(False)
    # "&&" is Qt's escape for a literal "&" in a button label
    assert w.next_btn.text() == "Load && continue →"
    w.go_back()
    assert w.stack.currentWidget() is w.welcome


def test_slider_controls_fill_the_control_column(qtbot):
    """Range sliders span their group box rather than collapsing to the spins.

    Regression: ``QFormLayout``'s default field-growth policy is style-dependent
    (``FieldsStayAtSizeHint`` on macOS), which ignored the controls' ``Expanding``
    size policy and left the sliders as narrow as the two spin boxes.
    """
    w = Wizard()
    qtbot.addWidget(w)
    w.show_synthetic()
    w.synthetic.advance()
    w.state.stage = 3
    stage = w.stages[2]
    w.resize(1500, 950)
    w.show()
    qtbot.waitExposed(w)
    assert stage.splitter.sizes()[0] == settings.DEFAULT_CONTROLS_WIDTH

    box = stage.lam_range.parentWidget()
    assert box is not None
    # the slider reaches (nearly) the group box's inner width
    assert stage.lam_range.slider.width() > 0.85 * box.width()
    assert stage.lam_range.slider.width() > 2 * stage.lam_range.lo_spin.width()


def test_widget_labels_escape_literal_ampersands(qtbot):
    """A literal "&" in a widget label is written "&&", and help still matches.

    Regression: Qt reads a lone ``&`` in a check box, push button, radio button,
    group-box title or form-row label as a keyboard-mnemonic marker and does not
    draw it, so "Skip filtering & regrid" rendered as "Skip filtering  regrid".
    Escaping it as ``&&`` fixes the rendering, and :func:`croak.gui.help._plain`
    undoes the escape so the (unescaped) HELP keys still resolve to their controls.
    """
    from PyQt6.QtWidgets import (
        QCheckBox,
        QFormLayout,
        QGroupBox,
        QLabel,
        QPushButton,
        QRadioButton,
    )

    from croak.gui.help import _plain

    assert _plain("a && b") == "a & b"

    w = Wizard()
    qtbot.addWidget(w)
    pages = [w.welcome, w.synthetic, w.simulated, *w.stages]

    # no mnemonic-parsing label anywhere carries an unescaped ampersand
    unescaped = []
    for page in pages:
        for cls in (QCheckBox, QPushButton, QRadioButton):
            unescaped += [
                (type(page).__name__, x.text())
                for x in page.findChildren(cls)
                if "&" in x.text().replace("&&", "")
            ]
        unescaped += [
            (type(page).__name__, x.title())
            for x in page.findChildren(QGroupBox)
            if "&" in x.title().replace("&&", "")
        ]
        for form in page.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                label = item.widget() if item is not None else None
                if isinstance(label, QLabel) and "&" in label.text().replace("&&", ""):
                    unescaped.append((type(page).__name__, label.text()))
    assert not unescaped, f"unescaped '&' in widget labels: {unescaped}"

    # the two labels that actually contain an ampersand render it exactly once...
    raw_direct = w.simulated.raw_direct
    assert raw_direct.text() == "Skip filtering && regrid (raw → retrieval)"
    assert _plain(raw_direct.text()) == "Skip filtering & regrid (raw → retrieval)"
    units = next(
        b for b in w.stages[0].findChildren(QGroupBox) if "interaction" in b.title()
    )
    assert _plain(units.title()) == "Units & interaction"

    # ...and their (unescaped) help entries still reach the controls
    assert "skipping the marginal-check" in raw_direct.toolTip()
    assert w.stages[0].lam_unit_combo.toolTip()  # a row inside "Units & interaction"


def _readout_row(readout, name):
    """The four formatted cells of one curve row, in column order."""
    from croak.gui.readout import _COLUMNS

    return [readout._cells[name, key].text() for key, *_ in _COLUMNS]


def _run_synthetic_retrieval(w, qtbot, *, energy=None, maxiters=15):
    """Generate a synthetic trace and retrieve it; returns the Retrieve stage."""
    w.show_synthetic()
    w.synthetic.advance()
    w.state.stage = 3  # preprocess (synchronous)
    rs = w.stages[3]
    rs.state.retrieve.solver = "copra"
    rs.state.retrieve.maxiters = maxiters
    if energy is not None:
        rs.state.load.energy_j = energy
    w.state.stage = 4
    with qtbot.waitSignal(w.state.result_changed, timeout=30000):
        rs._run()
    return rs


def test_retrieval_readout_fills_from_a_synthetic_run(qtbot):
    """Each curve row carries exactly the quantities that are defined for it."""
    from croak.gui.readout import DASH

    w = Wizard()
    qtbot.addWidget(w)
    rs = _run_synthetic_retrieval(w, qtbot)
    lam_peak, lam_fwhm, fwhm, power = range(4)

    r = _readout_row(rs.readout, "R")
    assert all(cell != DASH for cell in r[:3])  # spectral + temporal, no energy
    assert r[power] == DASH

    # the measured spectrum has no phase -> no temporal envelope
    m = _readout_row(rs.readout, "M")
    assert m[lam_peak] != DASH and m[lam_fwhm] != DASH
    assert m[fwhm] == DASH and m[power] == DASH

    # the transform limit shares R's spectrum, so it is reported once, on R
    tl = _readout_row(rs.readout, "TL")
    assert tl[lam_peak] == DASH and tl[lam_fwhm] == DASH
    assert tl[fwhm] != DASH

    # the synthetic generator supplies a known truth, spectrum included
    truth = _readout_row(rs.readout, "truth")
    assert truth[lam_peak] != DASH and truth[fwhm] != DASH
    # ...and it agrees with the truth pulse it was built from
    assert float(truth[fwhm]) == pytest.approx(w.state.truth.fwhm / 1e-15, rel=1e-2)

    scalars = {k: v.text() for k, v in rs.readout._scalars.items()}
    assert scalars["iteration"] == str(len(w.state.result.errors))
    assert scalars["error"].endswith("%")
    assert scalars["elapsed"].endswith(" s")


def test_retrieval_readout_shows_the_truth_errors(qtbot):
    """A known pulse fills the eps read-outs beside the FROG error.

    The synthetic generator supplies a complex truth, so all three are defined.
    They answer a different question from ``R``: it scores the fit to the trace,
    they score the distance to the true pulse.
    """
    from croak.gui.readout import DASH

    w = Wizard()
    qtbot.addWidget(w)
    rs = _run_synthetic_retrieval(w, qtbot)

    scalars = {k: v.text() for k, v in rs.readout._scalars.items()}
    for key in ("eps_It", "eps_Iw", "eps_Ew"):
        assert scalars[key] != DASH
        assert 0.0 <= float(scalars[key]) <= 1.5
    # they agree with the metric functions the strip is reporting
    from croak.truth_metrics import truth_errors

    expected = truth_errors(w.state.result, w.state.truth)
    assert float(scalars["eps_It"]) == pytest.approx(expected.eps_It, rel=1e-2)


def test_retrieval_readout_dashes_the_truth_errors_without_a_truth(qtbot):
    """An experimental load has no known pulse, so nothing is invented."""
    from croak.gui.readout import DASH

    w = Wizard()
    qtbot.addWidget(w)
    rs = w.stages[3]
    rs._update_truth_error_readout(None, None)
    scalars = {k: v.text() for k, v in rs.readout._scalars.items()}
    assert all(scalars[k] == DASH for k in ("eps_It", "eps_Iw", "eps_Ew"))


def test_retrieval_readout_peak_power_column(qtbot):
    """The peak-power column needs a pulse energy, and shares one SI prefix."""
    from croak.gui.readout import DASH

    w = Wizard()
    qtbot.addWidget(w)
    rs = _run_synthetic_retrieval(w, qtbot, energy=1.2e-6)

    header = rs.readout._header["peak_power"].text()
    assert header.startswith("Peak power (") and header.endswith("W)")
    for row in ("R", "TL", "truth"):
        assert _readout_row(rs.readout, row)[3] != DASH
    assert _readout_row(rs.readout, "M")[3] == DASH  # no temporal envelope

    # dropping the energy takes the whole column away again
    rs.state.load.energy_j = 0.0
    rs._update_view()
    assert rs.readout._header["peak_power"].text() == "Peak power"
    for row in ("R", "M", "TL", "truth"):
        assert _readout_row(rs.readout, row)[3] == DASH


def test_retrieval_readout_truth_row_blank_without_a_truth(
    qtbot, experiment_h5, experiment
):
    """An experimental trace has no known truth, so that row stays blank."""
    from croak.gui.readout import DASH

    w = Wizard()
    qtbot.addWidget(w)
    st = w.state
    st.load.frog_path = experiment_h5
    w.stages[0]._populate_frog()
    st.load.interaction = experiment.interaction
    w.stages[0]._load_preview()
    st.stage = 3
    rs = w.stages[3]
    rs.state.retrieve.solver = "copra"
    rs.state.retrieve.maxiters = 15
    st.stage = 4
    with qtbot.waitSignal(st.result_changed, timeout=30000):
        rs._run()

    assert st.truth is None
    assert _readout_row(rs.readout, "truth") == [DASH] * 4
    assert _readout_row(rs.readout, "R")[0] != DASH  # the rest still fills


def test_retrieval_readout_progress_without_a_result(qtbot):
    """Per-iteration updates move iteration/error with no processed result."""
    w = Wizard()
    qtbot.addWidget(w)
    rs = w.stages[3]
    rs._errors = []
    rs._on_progress(7, 0.0123, 0.0100)
    assert rs.readout._scalars["iteration"].text() == "7"
    assert rs.readout._scalars["error"].text() == "1.2300%"


def test_retrieval_readout_cleared_when_a_run_starts(qtbot, monkeypatch):
    """Starting a run blanks the previous numbers rather than leaving them stale."""
    from croak.gui import stage_retrieve as sr
    from croak.gui.readout import DASH

    w = Wizard()
    qtbot.addWidget(w)
    rs = _run_synthetic_retrieval(w, qtbot)
    assert _readout_row(rs.readout, "R")[0] != DASH

    # a worker that never runs, so we can inspect the just-cleared state
    class _Idle:
        def __init__(self, *a, **k):
            self.progress = self.preview = self.finished_ok = _Sig()
            self.stopped = self.failed = _Sig()

        def start(self):
            pass

    class _Sig:
        def connect(self, _slot):
            pass

    monkeypatch.setattr(sr, "RetrievalWorker", _Idle)
    rs._run()
    assert _readout_row(rs.readout, "R") == [DASH] * 4
    assert rs.readout._scalars["iteration"].text() == "0"
    assert rs._elapsed_timer.isActive()
    rs._elapsed_timer.stop()


def test_retrieval_readout_marks_extras_fitted_or_fixed(qtbot):
    """τ₀ / thickness / smearing say whether the solver was free to move them."""
    from croak.gui.readout import DASH

    w = Wizard()
    qtbot.addWidget(w)
    rs = w.stages[3]
    p = rs.state.retrieve
    scalars = rs.readout._scalars

    # no dispersive slab modelled at all -> nothing to report
    p.dispersive = False
    rs._update_extras_readout(None)
    assert scalars["thickness"].text() == DASH
    assert scalars["tau0"].text() == "0.00 fs (fixed)"
    assert scalars["smearing"].text() == DASH

    # a slab, held fixed at the value in the controls
    p.dispersive = True
    p.thickness_um = 42.0
    rs._update_extras_readout(None)
    assert scalars["thickness"].text() == "42.00 µm (fixed)"

    # ...and one the solver fitted
    class _Fitted:
        thickness = 55e-6
        tau0 = 1.25e-15
        smear_scale = None

    rs._update_extras_readout(_Fitted())
    assert scalars["thickness"].text() == "55.00 µm (fit)"
    assert scalars["tau0"].text() == "1.25 fs (fit)"


def test_gas_controls_reach_millibar_and_long_paths(qtbot):
    """Pressure resolves 1 mbar; the gas path reaches ±999 cm."""
    w = Wizard()
    qtbot.addWidget(w)
    stage = w.stages[4]  # StageDispersion

    assert stage.gas_pressure.bounds() == (0.0, 10.0)
    for mbar in (1.0, 3.0, 5.0, 1234.0):
        stage.gas_pressure.set_value(mbar / 1000.0)
        assert stage.gas_pressure.value() == pytest.approx(mbar / 1000.0)
        # the slider tracks it too, one step per millibar
        assert stage.gas_pressure.slider.value() == int(round(mbar))

    assert stage.gas_path.bounds() == (-999.0, 999.0)
    for cm in (999.0, -999.0, 250.5):
        stage.gas_path.set_value(cm)
        assert stage.gas_path.value() == pytest.approx(cm)


def test_retrieval_readout_smearing_survives_an_shg_trace(qtbot):
    """Reporting smearing on an SHG trace must not raise.

    Regression: the read-out asks for the smearing width whenever the option is
    ticked, but ``square_boxcars_kernel`` refuses SHG (no three-arm form), so an
    SHG trace with smearing enabled used to raise straight out of the update.
    """
    from croak.gui.readout import DASH

    w = Wizard()
    qtbot.addWidget(w)
    w.show_synthetic()
    w.synthetic.advance()  # the default synthetic trace is SHG
    w.state.stage = 3
    rs = w.stages[3]
    rs.state.retrieve.smearing = True

    rs._update_extras_readout(None)  # must not raise
    assert rs.readout._scalars["smearing"].text() == DASH
    # and the status-line helper, which shares _smear_width_fs, degrades too
    assert rs._smear_summary(1.5, w.state.tracedata) == "smearing ×1.500"


def test_preprocess_tau_exclude_auto_button(qtbot, experiment_h5, experiment):
    """'Auto' fills the baseline hold-out from the loaded trace's delay marginal."""
    w = Wizard()
    qtbot.addWidget(w)
    st = w.state
    st.load.frog_path = experiment_h5
    w.stages[0]._populate_frog()
    st.load.interaction = experiment.interaction
    w.stages[0]._load_preview()
    st.stage = 3  # preprocess
    stage = w.stages[2]

    # the whole baseline group, Auto included, is gated on the checkbox
    stage.baseline_check.setChecked(False)
    assert not stage.baseline_tauex_auto.isEnabled()
    stage.baseline_check.setChecked(True)
    assert stage.baseline_tauex_auto.isEnabled()

    assert st.preproc.baseline_tau_exclude_fs == 0.0  # off by default
    stage._auto_tau_exclude()
    value = stage.baseline_tauex_spin.value()
    assert value > 0.0
    assert st.preproc.baseline_tau_exclude_fs == pytest.approx(value)

    # ...and it is twice the delay-marginal FWHM of the loaded trace
    from croak import io
    from croak.preprocess import suggested_tau_exclude

    data = st.load_data
    scan = data["scanaxis"]
    tau = io.position_to_delay(scan) if data["input_unit"] == "position" else scan
    assert value == pytest.approx(suggested_tau_exclude(data["trace"], tau) / 1e-15)


def test_preprocess_tau_exclude_auto_without_a_trace(qtbot):
    """Auto with nothing loaded reports it rather than raising."""
    w = Wizard()
    qtbot.addWidget(w)
    stage = w.stages[2]
    assert w.state.load_data is None
    stage._auto_tau_exclude()
    assert "Load a trace" in stage.status_label.text()
    assert stage.baseline_tauex_spin.value() == 0.0


def test_app_icon_loads_at_every_size(qtbot):
    """The packaged icon renders — guards the asset and Qt's SVG plugin."""
    from croak.gui.branding import ICON_PATH, app_icon

    assert ICON_PATH.is_file(), ICON_PATH
    icon = app_icon()
    assert not icon.isNull()
    # a vector-backed QIcon reports no availableSizes, so the icon is built from
    # explicit pixmaps; that is what platforms pick their raster from
    assert icon.availableSizes()
    for size in (16, 256):
        pixmap = icon.pixmap(size, size)
        assert not pixmap.isNull()
        # Retina renders at a device-pixel ratio, so compare in logical units
        assert pixmap.width() / pixmap.devicePixelRatio() == pytest.approx(size)


def test_apply_identity_names_the_application(qtbot):
    """Name, display name and window icon reach the running QApplication."""
    from PyQt6.QtWidgets import QApplication

    from croak.gui.branding import APP_NAME, ORGANISATION_NAME, apply_identity

    app = QApplication.instance()
    assert app is not None
    before = (
        QApplication.applicationName(),
        QApplication.applicationDisplayName(),
        QApplication.windowIcon(),
        QApplication.organizationName(),
    )
    try:
        apply_identity()
        assert QApplication.organizationName() == ORGANISATION_NAME
        assert QApplication.applicationName() == APP_NAME
        assert QApplication.applicationDisplayName() == APP_NAME
        assert not QApplication.windowIcon().isNull()
        assert QApplication.applicationVersion() == croak.__version__
    finally:  # application-wide state: do not leak into the other tests
        QApplication.setApplicationName(before[0])
        QApplication.setApplicationDisplayName(before[1])
        QApplication.setWindowIcon(before[2])
        QApplication.setOrganizationName(before[3])


def test_wizard_carries_the_icon_under_a_foreign_qapplication(qtbot):
    """A Wizard built under someone else's QApplication still has the icon.

    run_wizard sets it application-wide, which covers the windows it opens; the
    window-level call is what covers an embedder (and this test suite, whose
    QApplication comes from qtbot).
    """
    w = Wizard()
    qtbot.addWidget(w)
    assert not w.windowIcon().isNull()


def test_importing_croak_gui_does_not_import_qt():
    """`import croak.gui` must stay Qt-free so the library works headless.

    The lazy ``__getattr__`` in ``croak/gui/__init__.py`` exists for this; a new
    module imported at the top of it would silently break the guarantee.
    """
    import subprocess

    code = (
        "import croak.gui, sys; print(any(m.startswith('PyQt6') for m in sys.modules))"
    )
    out = subprocess.run(  # noqa: S603 — sys.executable and a literal, nothing untrusted
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False", out.stdout


def test_simulated_load_adopts_recorded_geometry(qtbot, simulated_truth, tmp_path):
    """Loading a simulated scan sets thickness and mask from the file itself.

    These are the two settings that cannot be checked by looking at the trace:
    a kernel built from the wrong hole spacing cost ~2% of retrieved duration
    in one sweep while *improving* the trace error.
    """
    from conftest import SIMULATED_ZSAVE, write_simulated_h5

    path = write_simulated_h5(
        tmp_path / "geom.h5",
        simulated_truth,
        zsave=SIMULATED_ZSAVE,
        geometry=(1e-3, 1e-3, "tg"),
    )
    w = Wizard()
    qtbot.addWidget(w)
    w.load_simulated()
    sim = w.simulated
    stage = w.stages[3]  # StageRetrieve
    # start from settings that do NOT match the file, so adoption is visible
    stage.thickness_spin.setValue(40.0)
    stage.hole_spacing_spin.setValue(0.5)
    sim.path_edit.setText(path)
    sim._populate_file_options(path)
    lo = simulated_truth.lam_min / 1e-9 - 10
    hi = simulated_truth.lam_max / 1e-9 + 10
    sim.band.set_values(lo, hi)
    sim._band = (lo, hi)
    # pick the 10 µm slice (SIMULATED_ZSAVE index 2), not the 40 µm exit
    sim.thickness.setCurrentIndex(sim.thickness.findData(10.0))
    sim._preview()
    sim.advance()

    p = w.state.retrieve
    assert p.thickness_um == pytest.approx(10.0)
    assert p.npoints == 10  # Δz held at 1 µm
    assert p.smear_hole_diameter_mm == pytest.approx(1.0)
    assert p.smear_hole_spacing_mm == pytest.approx(1.0)
    assert stage.thickness_spin.value() == pytest.approx(10.0)
    assert stage.hole_spacing_spin.value() == pytest.approx(1.0)


def test_simulated_load_says_when_the_mask_is_not_recorded(
    qtbot, simulated_scan_multi_h5, simulated_truth
):
    """A file without mask metadata sets the thickness and says the mask is unknown."""
    from conftest import SIMULATED_ZSAVE

    w = Wizard()
    qtbot.addWidget(w)
    w.load_simulated()
    sim = w.simulated
    stage = w.stages[3]
    stage.hole_spacing_spin.setValue(0.5)
    sim.path_edit.setText(simulated_scan_multi_h5)
    sim._populate_file_options(simulated_scan_multi_h5)
    lo = simulated_truth.lam_min / 1e-9 - 10
    hi = simulated_truth.lam_max / 1e-9 + 10
    sim.band.set_values(lo, hi)
    sim._band = (lo, hi)
    sim._preview()
    sim.advance()

    # the thickness still follows the slice; the mask is left alone and flagged
    assert w.state.retrieve.thickness_um == pytest.approx(SIMULATED_ZSAVE.max() * 1e6)
    assert w.state.retrieve.smear_hole_spacing_mm == pytest.approx(0.5)
    assert "does not record the mask" in sim.status_label.text()


def test_solver_switch_rescales_regularisation(qtbot):
    """LM and the gradient solvers get their own weights, remembered per family.

    The two minimise different objectives (R + λP against R² + λP), so a weight
    carried across is wrong by about 1/2R — three orders of magnitude at a good
    trace error, which shows up as an LM run that is silently over-regularised.
    """
    from croak.session.params import REG_DEFAULTS

    w = Wizard()
    qtbot.addWidget(w)
    stage = w.stages[3]
    p = stage.state.retrieve
    grad_spec, grad_amp = REG_DEFAULTS["gradient"]
    lm_spec, lm_amp = REG_DEFAULTS["lm"]
    assert (p.reg_spectrum, p.reg_amp) == (grad_spec, grad_amp)

    stage.solver_combo.setCurrentText("lm")
    assert (p.reg_spectrum, p.reg_amp) == (lm_spec, lm_amp)
    assert stage.reg_spectrum_spin.value() == pytest.approx(lm_spec)

    # hand-tuning inside a family is remembered across a round trip
    stage.reg_spectrum_spin.setValue(3e-5)
    stage.solver_combo.setCurrentText("warm-lbfgs")
    assert (p.reg_spectrum, p.reg_amp) == (grad_spec, grad_amp)
    stage.solver_combo.setCurrentText("lm-optx")  # same family as lm
    assert p.reg_spectrum == pytest.approx(3e-5)

    # moves within a family leave the weights alone
    stage.solver_combo.setCurrentText("warm-lbfgs")
    stage.reg_spectrum_spin.setValue(0.05)
    stage.solver_combo.setCurrentText("copra")
    assert p.reg_spectrum == pytest.approx(0.05)


# -- responsive canvas ---------------------------------------------------------
@pytest.mark.parametrize(
    ("size", "small"), [((1366, 768), True), ((1920, 1080), False)]
)
def test_canvas_density_follows_the_window_size(qtbot, size, small):
    """Plot text shrinks in a small window and stays at full size in a large one."""
    w = Wizard()
    qtbot.addWidget(w)
    w.show_synthetic()
    w.synthetic.advance()  # marginal check: two stacked (5, 4)-inch canvases
    canvas = w.stages[1].marg_canvas
    w.resize(*size)
    w.show()
    qtbot.waitExposed(w)
    if small:
        qtbot.waitUntil(lambda: canvas.scale.font < 0.8)
    else:
        qtbot.waitUntil(lambda: canvas.scale.font == 1.0)
    # once the debounced re-plot has run, the title carries the scaled size
    qtbot.waitUntil(
        lambda: (
            bool(canvas.figure.axes)
            and canvas.figure.axes[0].title.get_fontsize()
            == pytest.approx(12.0 * canvas.scale.font)
        )
    )


def test_canvas_replots_when_a_resize_changes_the_density(qtbot):
    canvas = MplCanvas(figsize=(12, 7))
    qtbot.addWidget(canvas)
    calls = []
    canvas.render(lambda fig: calls.append(fig.add_subplot(111).set_title("t")))
    canvas.resize(1200, 700)  # exactly the design size: no re-plot on show
    canvas.show()
    qtbot.waitExposed(canvas)
    assert canvas.scale.font == 1.0
    assert len(calls) == 1
    canvas.resize(600, 350)  # half size -> density clamps to the 0.7 minimum
    qtbot.waitUntil(lambda: len(calls) == 2)
    assert canvas.scale.font == pytest.approx(0.7)
    assert canvas.figure.axes[0].title.get_fontsize() == pytest.approx(8.4)


def test_canvas_resize_without_a_density_change_does_not_replot(qtbot):
    canvas = MplCanvas(figsize=(12, 7))
    qtbot.addWidget(canvas)
    calls = []
    canvas.render(lambda fig: calls.append(fig.add_subplot(111)))
    canvas.resize(1200, 700)
    canvas.show()
    qtbot.waitExposed(canvas)
    canvas.resize(1180, 690)  # a few pixels: same quantised density
    qtbot.wait(3 * MplCanvas.REPLOT_DELAY_MS)
    assert len(calls) == 1


def test_canvas_save_figure_exports_at_the_design_size(qtbot, tmp_path):
    """The export is drawn afresh at the design size, not the shrunken screen size."""
    canvas = MplCanvas(figsize=(12, 7))
    qtbot.addWidget(canvas)
    seen = []

    def plot(fig):
        seen.append(tuple(float(v) for v in fig.get_size_inches()))
        fig.add_subplot(111).set_title("t")

    canvas.render(plot)
    canvas.resize(600, 350)
    canvas.show()
    qtbot.waitExposed(canvas)
    qtbot.waitUntil(lambda: canvas.scale.font == pytest.approx(0.7))
    path = tmp_path / "fig.pdf"
    canvas.save_figure(path, dpi=72)
    assert path.stat().st_size > 0
    assert seen[-1] == pytest.approx((12.0, 7.0))
    blank = MplCanvas()
    qtbot.addWidget(blank)
    with pytest.raises(RuntimeError, match="nothing has been rendered"):
        blank.save_figure(tmp_path / "none.pdf", dpi=72)


# -- remembered layout -----------------------------------------------------------
@pytest.mark.parametrize(
    ("size", "available", "expected"),
    [
        ((1500, 950), (800, 800), (720, 720)),  # first run on a small screen
        ((700, 600), (800, 800), (700, 600)),  # already fits: untouched
        ((1500, 950), (2560, 1400), (1500, 950)),  # never enlarged
    ],
)
def test_fit_to_screen_never_enlarges_and_caps_at_the_screen_fill(
    size, available, expected
):
    assert settings.fit_to_screen(QSize(*size), QSize(*available)) == QSize(*expected)


def test_settings_file_lives_under_the_redirected_path(tmp_path):
    """Guards the isolation fixture: nothing here can reach real preferences."""
    assert settings.user_settings().fileName().startswith(str(tmp_path))


def test_new_window_is_clamped_to_the_screen(qtbot):
    """With nothing saved, a fresh window fills at most 90 % of the screen."""
    from PyQt6.QtWidgets import QApplication

    screen = QApplication.primaryScreen()
    assert screen is not None
    available = screen.availableGeometry().size()  # 800x800 offscreen
    w = Wizard()
    qtbot.addWidget(w)
    assert w.width() <= settings.SCREEN_FILL * available.width()
    assert w.height() <= settings.SCREEN_FILL * available.height()
    assert w.width() < settings.DEFAULT_WINDOW_SIZE.width()


def test_window_geometry_round_trips_and_is_clamped_on_restore(qtbot):
    """Closing saves the geometry; the next window restores it, kept on screen."""
    from PyQt6.QtWidgets import QMainWindow

    w = Wizard()
    qtbot.addWidget(w)
    w.resize(700, 600)  # the width is clamped up to the window's minimum
    w.show()
    qtbot.waitExposed(w)
    shown = (w.width(), w.height())
    assert shown != (settings.DEFAULT_WINDOW_SIZE.width(), 600)
    w.close()
    again = Wizard()
    qtbot.addWidget(again)
    assert (again.width(), again.height()) == shown
    # a geometry saved on a larger display never restores wider than this screen
    big = QMainWindow()
    qtbot.addWidget(big)
    big.resize(1500, 950)
    big.show()
    qtbot.waitExposed(big)
    settings.save_window(big)
    clamped = Wizard()
    qtbot.addWidget(clamped)
    assert clamped.width() <= 800
    assert clamped.height() <= 800


# -- adaptive layout: splitter and strips ----------------------------------------
def _marginal_stage_shown(qtbot, width, height):
    """A wizard on the marginal-check stage, shown at ``width`` x ``height``."""
    w = Wizard()
    qtbot.addWidget(w)
    w.show_synthetic()
    w.synthetic.advance()
    w.resize(width, height)
    w.show()
    qtbot.waitExposed(w)
    return w, w.stages[1]


def test_stage_control_column_is_a_collapsible_splitter(qtbot):
    """The controls sit on a splitter: 315 px by default, min 240, closable."""
    _w, stage = _marginal_stage_shown(qtbot, 1366, 768)
    sp = stage.splitter
    assert sp.isCollapsible(0)
    assert not sp.isCollapsible(1)
    assert sp.sizes()[0] == settings.DEFAULT_CONTROLS_WIDTH
    assert sp.widget(0).minimumWidth() == CONTROLS_MIN_WIDTH
    before = stage.marg_canvas.width()
    sp.setSizes([0, 1])
    qtbot.waitUntil(lambda: sp.sizes()[0] == 0)
    qtbot.waitUntil(lambda: stage.marg_canvas.width() >= before + 300)


def test_control_column_width_is_shared_and_remembered(qtbot):
    """A dragged divider is remembered, reaches the other stages and later runs."""
    w, stage = _marginal_stage_shown(qtbot, 1366, 768)
    sp = stage.splitter
    sp.setSizes([260, 1])
    sp.splitterMoved.emit(260, 1)  # setSizes() is silent; a real drag emits this
    qtbot.waitUntil(lambda: settings.controls_width() == 260)
    w.state.stage = 3  # preprocess: shown for the first time, adopts the width
    qtbot.waitUntil(lambda: w.stages[2].splitter.sizes()[0] == 260)
    again, again_stage = _marginal_stage_shown(qtbot, 1366, 768)
    assert again_stage.splitter.sizes()[0] == 260


def test_no_stage_needs_more_than_a_small_display(qtbot):
    """No page (or widget on it) demands a window wider or taller than 1366x768."""
    from PyQt6.QtWidgets import QWidget

    w = Wizard()
    qtbot.addWidget(w)
    w.show_synthetic()
    w.synthetic.advance()
    for page in [*w.stages, w.synthetic, w.simulated, w.welcome]:
        w.stack.setCurrentWidget(page)
        hint = w.minimumSizeHint()
        assert (hint.width(), hint.height()) <= (1366, 768), type(page).__name__
        for child in page.findChildren(QWidget):
            assert child.minimumSizeHint().width() <= 1366, type(child).__name__


def test_marginal_check_canvases_share_a_vertical_splitter(qtbot):
    _w, stage = _marginal_stage_shown(qtbot, 1366, 768)
    sp = stage.plot_splitter
    assert sp.orientation() == Qt.Orientation.Vertical
    assert sp.count() == 2
    assert all(size > 200 for size in sp.sizes())


def test_collapsible_strip_folds_and_remembers_its_state(qtbot):
    from PyQt6.QtWidgets import QLabel

    strip = collapsible("Read-out", QLabel("x"), settings_key="test/strip")
    qtbot.addWidget(strip)
    assert strip.isExpanded()
    strip.collapse(animate=False)
    assert not strip.isExpanded()
    assert strip.content().maximumHeight() == 0
    again = collapsible("Read-out", QLabel("y"), settings_key="test/strip")
    qtbot.addWidget(again)
    assert not again.isExpanded()  # remembered
    plain = collapsible("Read-out", QLabel("z"))
    qtbot.addWidget(plain)
    assert plain.isExpanded()  # no key: the default


# -- Retrieve pages ----------------------------------------------------------------
def _titles(figure):
    return [ax.get_title() for ax in figure.axes if ax.get_title()]


def _page_titles(key):
    page = croak.plotting.RETRIEVAL_PAGES[key]
    return [croak.plotting.RETRIEVAL_PANELS[k].title for k in page.keys]


class _IdleWorker:
    """A retrieval worker that never runs, so a just-started state can be inspected."""

    class _Sig:
        def connect(self, _slot):
            pass

    def __init__(self, *a, **k):
        self.progress = self.preview = self.finished_ok = self._Sig()
        self.stopped = self.failed = self._Sig()

    def start(self):
        pass


def test_retrieve_pages_show_three_tabs_and_draw_the_visible_page(qtbot):
    w = Wizard()
    qtbot.addWidget(w)
    rs = _run_synthetic_retrieval(w, qtbot)
    pages = rs.pages
    assert pages.page_keys() == ("traces", "pulse", "diagnostics")
    assert pages.current_page() == "traces"
    assert _titles(pages.canvas("traces").figure) == _page_titles("traces")
    assert pages.is_stale("pulse") and pages.is_stale("diagnostics")
    assert not pages.canvas("pulse").figure.axes  # not drawn until selected


def test_retrieve_pages_redraw_a_stale_page_when_selected(qtbot):
    w = Wizard()
    qtbot.addWidget(w)
    rs = _run_synthetic_retrieval(w, qtbot)
    rs.pages.set_current_page("pulse")
    assert not rs.pages.is_stale("pulse")
    assert _titles(rs.pages.canvas("pulse").figure) == _page_titles("pulse")
    rs.pages.set_current_page("diagnostics")
    assert _titles(rs.pages.canvas("diagnostics").figure) == _page_titles("diagnostics")


def test_retrieve_pages_pop_out_opens_a_window_that_follows_the_result(qtbot):
    w = Wizard()
    qtbot.addWidget(w)
    rs = _run_synthetic_retrieval(w, qtbot)
    dlg = rs.pages.pop_out("pulse")
    assert dlg.isWindow()
    assert dlg.isVisible()
    assert _titles(dlg.canvas.figure) == _page_titles("pulse")
    before = dlg.canvas.figure.axes[0]
    rs._update_view()  # the pop-out follows, updated in place
    assert dlg.canvas.figure.axes[0] is before
    assert rs.pages.pop_out("pulse") is dlg  # one window per page
    dlg.close()
    qtbot.waitUntil(lambda: "pulse" not in rs.pages._popouts)


def test_retrieve_progress_curve_shows_on_the_visible_page(qtbot):
    w = Wizard()
    qtbot.addWidget(w)
    w.show_synthetic()
    w.synthetic.advance()
    w.state.stage = 3
    w.state.stage = 4
    rs = w.stages[3]
    rs.pages.set_current_page("pulse")
    rs._errors = []
    rs._on_progress(1, 0.1, 0.1)
    rs._on_progress(2, 0.05, 0.05)
    fig = rs.pages.canvas("pulse").figure
    qtbot.waitUntil(lambda: _titles(fig) == ["Convergence"])  # after the debounce
    line = fig.axes[0].lines[0]
    assert line.get_xydata().shape == (2, 2)
    rs._on_progress(3, 0.02, 0.02)
    qtbot.waitUntil(lambda: line.get_xydata().shape == (3, 2))  # grown in place
    assert fig.axes[0].lines[0] is line
    # switching tabs while running shows the curve there too
    rs.pages.set_current_page("traces")
    assert _titles(rs.pages.canvas("traces").figure) == ["Convergence"]


def test_retrieve_run_clears_the_pages(qtbot, monkeypatch):
    from croak.gui import stage_retrieve as sr

    w = Wizard()
    qtbot.addWidget(w)
    rs = _run_synthetic_retrieval(w, qtbot)
    assert rs.pages.canvas("traces").figure.axes
    monkeypatch.setattr(sr, "RetrievalWorker", _IdleWorker)
    rs._run()
    assert all(not rs.pages.canvas(k).figure.axes for k in rs.pages.page_keys())
    assert rs.pages.data is None


def test_retrieve_readout_collapse_frees_canvas_height(qtbot):
    w = Wizard()
    qtbot.addWidget(w)
    rs = _run_synthetic_retrieval(w, qtbot)
    w.resize(1366, 768)
    w.show()
    qtbot.waitExposed(w)
    assert rs.readout_strip.isExpanded()
    before = rs.pages.height()
    rs.readout_strip.collapse(animate=False)
    qtbot.waitUntil(lambda: rs.pages.height() > before + 100)
    assert settings.value("retrieve/readout_expanded", True) is False


def test_retrieve_save_writes_the_full_overview_pdf(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    from croak.gui import stage_retrieve as sr

    w = Wizard()
    qtbot.addWidget(w)
    rs = _run_synthetic_retrieval(w, qtbot)
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )
    seen = []
    real = sr.plotting.plot_retrieval_page

    def spy(data, page, **kwargs):
        fig = real(data, page, **kwargs)
        seen.append((page, fig))
        return fig

    monkeypatch.setattr(sr.plotting, "plot_retrieval_page", spy)
    rs._save()
    assert (tmp_path / "retrieval.pdf").stat().st_size > 0
    page, fig = seen[-1]
    assert page is croak.plotting.RETRIEVAL_OVERVIEW
    assert len(_titles(fig)) == 12


# -- live preview in place -----------------------------------------------------------
def test_retrieve_live_preview_updates_the_page_in_place(qtbot):
    """A later snapshot of the same run reuses the drawn axes and artists."""
    from dataclasses import replace

    w = Wizard()
    qtbot.addWidget(w)
    rs = _run_synthetic_retrieval(w, qtbot)
    r1 = w.state.result
    assert r1 is not None
    r2 = replace(r1, errors=list(r1.errors[:5]), error=float(r1.errors[4]))
    rs._on_preview(r1)
    figure = rs.pages.canvas("traces").figure
    axes = list(figure.axes)
    retrieved_lin = next(
        ax for ax in figure.axes if ax.get_title() == "Retrieved (lin)"
    )
    label = retrieved_lin.texts[0]  # the bold error/shape annotation
    rs._on_preview(r2)
    assert list(figure.axes) == axes
    assert rs._live_full
    assert f"{r2.error * 100:.2f}%" in label.get_text()
    assert "iter 5" in rs.status_label.text()
    # a stale page drawn later shows the latest snapshot
    rs.pages.set_current_page("pulse")
    assert rs.pages.data is not None
    assert rs.pages.data.result is r2


def test_retrieve_post_filter_updates_in_place(qtbot):
    w = Wizard()
    qtbot.addWidget(w)
    rs = _run_synthetic_retrieval(w, qtbot)
    figure = rs.pages.canvas("traces").figure
    axes = list(figure.axes)
    status = rs.status_label.text()
    rs.pf_lam.setChecked(True)  # re-plots the stored result through _update_view
    assert list(figure.axes) == axes
    assert rs.state.retrieve.post_filter_lam is True
    assert rs.status_label.text() != status or "Done" in rs.status_label.text()


def test_retrieve_convergence_redraw_is_throttled(qtbot, monkeypatch):
    """A thousand iterations in a burst draw the curve once, after the debounce."""
    w = Wizard()
    qtbot.addWidget(w)
    w.show_synthetic()
    w.synthetic.advance()
    w.state.stage = 3
    w.state.stage = 4
    rs = w.stages[3]
    calls = []
    monkeypatch.setattr(
        rs.pages, "show_progress", lambda errors: calls.append(len(errors))
    )
    rs._errors = []
    for i in range(1000):
        rs._on_progress(i + 1, 0.5 / (i + 1), 0.5 / (i + 1))
    assert calls == []
    assert rs._conv_timer.isActive()
    qtbot.waitUntil(lambda: calls == [1000])
    qtbot.wait(3 * rs._conv_timer.interval())
    assert calls == [1000]  # and nothing more without new iterations
