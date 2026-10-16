"""Tests for the simulated-scan entry page (:mod:`croak.gui.stage_simulated`).

The page lives inside a :class:`~croak.gui.wizard.Wizard` (it calls back into
it to relabel and advance), so each test drives ``w.simulated``.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QFileDialog

from croak.gui import stage_simulated
from croak.gui.wizard import Wizard
from croak.io import SIMULATED_WINDOW_KEYS
from croak.session.params import SimulatedLoadParams


@pytest.fixture
def wizard(qtbot):
    w = Wizard()
    qtbot.addWidget(w)
    w.load_simulated()
    return w


@pytest.fixture
def junk_h5(tmp_path):
    """An HDF5 file that is not a simulated scan at all."""
    path = tmp_path / "junk.h5"
    with h5py.File(path, "w") as f:
        f["x"] = np.arange(3.0)
    return str(path)


def test_stage_simulated_on_enter_without_a_file_prompts(wizard):
    sim = wizard.simulated
    sim.on_enter()
    assert sim.status_label.text() == "Choose a simulated scan file (.h5) to begin."
    sim._preview()  # nothing chosen: no-op
    assert not sim.can_advance()


def test_stage_simulated_on_enter_previews_the_chosen_file(wizard, simulated_scan_h5):
    sim = wizard.simulated
    sim.path_edit.setText(simulated_scan_h5)
    sim.on_enter()
    assert sim.can_advance()
    # a legacy file has no thickness slices: the generic prompt
    assert sim.status_label.text() == (
        "Review the trace, then Load & continue to filter it."
    )
    titles = [ax.get_title() for ax in sim.canvas.figure.axes]
    assert "Simulated FROG trace (lin)" in titles


def test_stage_simulated_unreadable_file_falls_back_and_reports(wizard, junk_h5):
    sim = wizard.simulated
    before = sim._band
    sim.path_edit.setText(junk_h5)
    sim.on_enter()
    # every file-driven selector falls back to its placeholder
    items = [sim.window_key.itemText(i) for i in range(sim.window_key.count())]
    assert items == list(SIMULATED_WINDOW_KEYS)
    assert not sim.thickness.isEnabled()
    assert not sim.truth_source.isEnabled()
    assert sim._band == before
    # and the preview surfaces the load error instead of enabling Load
    assert sim.status_label.text().startswith("Load failed:")
    assert not sim.can_advance()


def test_stage_simulated_advance_reports_load_failure(wizard, junk_h5):
    sim = wizard.simulated
    sim.path_edit.setText(junk_h5)
    stage_before = wizard.state.stage
    sim.advance()
    assert sim.status_label.text().startswith("Load failed:")
    assert wizard.state.load_data is None
    assert wizard.state.stage == stage_before


def test_stage_simulated_raw_load_failure_stays_on_the_page(
    wizard, simulated_scan_h5, monkeypatch
):
    def broken(params):
        raise ValueError("no native grid")

    monkeypatch.setattr(stage_simulated, "assemble_simulated_tracedata", broken)
    # no geometry note either: keep the status free for the raw-load error
    monkeypatch.setattr(
        wizard.stages[3], "apply_simulated_geometry", lambda geometry: ""
    )
    sim = wizard.simulated
    sim.path_edit.setText(simulated_scan_h5)
    sim._populate_file_options(simulated_scan_h5)
    sim.raw_direct.setChecked(True)
    sim.advance()
    assert sim.status_label.text() == "Raw load failed: no native grid"
    assert wizard.state.tracedata is None
    assert wizard.stack.currentWidget() is sim


def test_stage_simulated_advance_without_geometry_note(
    wizard, simulated_scan_h5, monkeypatch
):
    monkeypatch.setattr(
        wizard.stages[3], "apply_simulated_geometry", lambda geometry: ""
    )
    sim = wizard.simulated
    sim.path_edit.setText(simulated_scan_h5)
    sim._populate_file_options(simulated_scan_h5)
    sim.set_status("before")
    sim.advance()
    assert wizard.stack.currentWidget() is wizard.stages[1]
    assert sim.status_label.text() == "before"


def test_stage_simulated_pick_file_previews_or_cancels(
    wizard, simulated_scan_h5, monkeypatch
):
    sim = wizard.simulated
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    sim._pick_file()
    assert sim.path_edit.text() == ""
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: (simulated_scan_h5, "")
    )
    sim._pick_file()
    assert sim.path_edit.text() == simulated_scan_h5
    assert sim.can_advance()


def test_stage_simulated_band_change_schedules_a_preview(wizard):
    sim = wizard.simulated
    sim._timer.stop()
    sim._on_band(600.0, 900.0)
    assert sim._band == (600.0, 900.0)
    assert sim._timer.isActive()
    assert sim._params().lam_min_nm == 600.0


def test_stage_simulated_seed_widgets_without_a_file(wizard):
    sim = wizard.simulated
    p = SimulatedLoadParams(frog_path="", lam_min_nm=650.0, lam_max_nm=950.0)
    p.reverse_trace = None  # unset: off
    sim.seed_widgets(p)
    assert sim.path_edit.text() == ""
    assert not sim.reverse.isChecked()
    assert sim._band == (650.0, 950.0)


def test_stage_simulated_seed_widgets_selects_saved_thickness(
    wizard, simulated_scan_multi_h5
):
    from conftest import SIMULATED_ZSAVE

    sim = wizard.simulated
    p = SimulatedLoadParams(frog_path=simulated_scan_multi_h5)
    # a thickness the file does not list leaves the default (full) slice
    p.z_thickness_um = 12345.0
    sim.seed_widgets(p)
    assert "full" in sim.thickness.currentText()
    entrance = float(SIMULATED_ZSAVE.min() * 1e6)
    p.z_thickness_um = entrance
    sim.seed_widgets(p)
    assert sim._params().z_thickness_um == pytest.approx(entrance)


def test_stage_simulated_missing_file_falls_back_to_placeholders(wizard, tmp_path):
    sim = wizard.simulated
    missing = str(tmp_path / "gone.h5")
    sim.path_edit.setText(missing)
    sim._populate_file_options(missing)
    items = [sim.window_key.itemText(i) for i in range(sim.window_key.count())]
    assert items == list(SIMULATED_WINDOW_KEYS)
    assert not sim.thickness.isEnabled()


def test_stage_simulated_multi_slice_preview_names_the_slice(
    wizard, simulated_scan_multi_h5
):
    sim = wizard.simulated
    sim.path_edit.setText(simulated_scan_multi_h5)
    sim.on_enter()
    assert sim.status_label.text().startswith("Loaded the ")
    assert "(full) slice" in sim.status_label.text()


def test_stage_simulated_advance_reports_adopted_geometry(
    wizard, simulated_scan_multi_h5, monkeypatch
):
    monkeypatch.setattr(
        wizard.stages[3],
        "apply_simulated_geometry",
        lambda geometry: "Adopted from the file: thickness 20 µm.",
    )
    sim = wizard.simulated
    sim.path_edit.setText(simulated_scan_multi_h5)
    sim._populate_file_options(simulated_scan_multi_h5)
    sim.advance()
    assert sim.status_label.text() == "Adopted from the file: thickness 20 µm."


def _sim_data(lam, trace, lam_spec=None, Ilam_spec=None):
    return {
        "lam": lam,
        "scanaxis": np.linspace(-20e-15, 20e-15, trace.shape[1]),
        "trace": trace,
        "lam_spec": lam_spec,
        "Ilam_spec": Ilam_spec,
    }


def test_stage_simulated_draw_handles_a_blank_trace_and_spectrum(wizard):
    # all-zero trace and input spectrum: no normalisation, full-range limits
    lam = np.linspace(700e-9, 900e-9, 6)
    sim = wizard.simulated
    sim._draw(_sim_data(lam, np.zeros((6, 4)), lam, np.zeros(6)))
    axd = {ax.get_title(): ax for ax in sim.canvas.figure.axes}
    assert "Simulated FROG trace (lin)" in axd
    legend = sim.canvas.figure.axes[3].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["trace marginal", "input"]


def test_stage_simulated_draw_skips_a_spectrum_without_intensity(wizard):
    lam = np.linspace(700e-9, 900e-9, 6)
    trace = np.outer(np.hanning(6), np.hanning(4))
    sim = wizard.simulated
    sim._draw(_sim_data(lam, trace, lam_spec=lam, Ilam_spec=None))
    assert sim.canvas.figure.axes[3].get_legend() is None
