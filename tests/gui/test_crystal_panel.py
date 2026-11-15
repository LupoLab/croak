"""Tests for the Data stage's crystal panel (:mod:`croak.gui.crystal_panel`).

The panel edits the session's retrieval settings directly, enables only the
controls the settings use, and describes the crystal and the beams at the best
carrier the session has so far.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from croak.grid import Grid
from croak.gui.crystal_panel import CRYSTAL_HELP, CrystalPanel
from croak.gui.help import flatten
from croak.gui.state import WizardState
from croak.session.params import RetrieveParams


@pytest.fixture
def panel(qtbot):
    widget = CrystalPanel(WizardState())
    qtbot.addWidget(widget)
    widget.edits = []
    widget.edited.connect(lambda: widget.edits.append(True))
    return widget


def _prepared(panel):
    """Give the session a stand-in prepared SHG trace at 800 nm."""
    panel.state.tracedata = SimpleNamespace(
        interaction="shg", grid=Grid(64, dt=3e-15), omega0_pulse=2.3546e15, Iomega=None
    )
    panel.refresh()


def test_crystal_panel_controls_store_their_settings(panel):
    p = panel.state.retrieve
    panel.crystal_check.setChecked(True)
    panel.crystal_combo.setCurrentText("BBO")
    panel.thickness_spin.setValue(55.0)
    panel.cut_angle_spin.setValue(33.4)
    panel.angle_mode_combo.setCurrentText("External tilt")
    panel.tilt_spin.setValue(-1.5)
    panel.polarization_combo.setCurrentText("Perpendicular to the crossing plane")
    panel.delayed_beam_combo.setCurrentText("Away from the optic axis")
    panel.separation_spin.setValue(8.0)
    panel.focal_f_spin.setValue(250.0)
    panel.beam_diameter_spin.setValue(1.5)
    assert p.dispersive is True
    assert (p.crystal, p.thickness_um, p.cut_angle_deg) == ("BBO", 55.0, 33.4)
    assert (p.angle_mode, p.tilt_deg) == ("tilt", -1.5)
    assert (p.polarization, p.delayed_beam) == ("perpendicular", "away_from_axis")
    assert (p.beam_separation_mm, p.focal_f_mm, p.beam_diameter_mm) == (
        8.0,
        250.0,
        1.5,
    )
    assert panel.edits  # every edit is announced
    assert panel.tilt_spin.isEnabled() and not panel.phase_match_spin.isEnabled()
    panel.angle_mode_combo.setCurrentText("Phase-matched wavelength")
    panel.phase_match_spin.setValue(780.0)
    assert (p.angle_mode, p.phase_match_nm) == ("phase-match", 780.0)
    assert not panel.tilt_spin.isEnabled()


def test_crystal_panel_follows_replaced_settings(panel):
    # the session may swap its settings object (a reset, a reopened session)
    panel.state.retrieve = RetrieveParams()
    panel.thickness_spin.setValue(33.0)
    assert panel.state.retrieve.thickness_um == 33.0


def test_crystal_panel_delayed_beam_matters_only_in_the_principal_plane(panel):
    panel.separation_spin.setValue(5.0)
    panel.polarization_combo.setCurrentText("In the crossing plane")
    assert not panel.delayed_beam_combo.isEnabled()
    panel.polarization_combo.setCurrentText("Perpendicular to the crossing plane")
    assert panel.delayed_beam_combo.isEnabled()
    panel.separation_spin.setValue(0.0)  # collinear: no crossing plane at all
    assert not panel.polarization_combo.isEnabled()
    assert not panel.delayed_beam_combo.isEnabled()
    assert panel.geometry_readout.text() == "Collinear beams: no delay smearing"


def test_crystal_panel_readouts_before_any_data(panel):
    panel.separation_spin.setValue(5.0)
    panel.focal_f_spin.setValue(100.0)
    # d = 5 mm, D = 3 mm, f = 100 mm at 800 nm: sigma = lambda d / (sqrt2 pi D c)
    assert panel.geometry_readout.text() == (
        "Crossing half-angle α = 1.432°\nDelay smearing σ = 1.00 fs rms (2.36 fs FWHM)"
    )
    text = panel.crystal_readout.text()
    assert "θ inside" in text and "path =" in text
    assert "depth nodes: after Prepare" in text
    assert text.endswith("at 800 nm (the phase-matched wavelength)")


def test_crystal_panel_readouts_at_the_prepare_band_once_loaded(panel):
    panel.state.load_data = {"interaction": "shg"}
    panel.state.preproc.lam_min_nm, panel.state.preproc.lam_max_nm = 700.0, 933.3
    panel.refresh()
    # the band's centre in frequency: 1/λ = (1/700 + 1/933.3) / 2
    assert panel.crystal_readout.text().endswith(
        "at 800 nm (the centre of the Prepare band)"
    )


def test_crystal_panel_readouts_of_a_prepared_trace(panel):
    _prepared(panel)
    panel.thickness_spin.setValue(100.0)
    text = panel.crystal_readout.text()
    assert "depth nodes = " in text
    assert text.endswith("(the prepared trace's carrier)")
    # a zero thickness needs no depth nodes
    panel.thickness_spin.setValue(0.0)
    assert "depth nodes" not in panel.crystal_readout.text()


def test_crystal_panel_explains_settings_that_give_no_crystal(panel):
    # a cut the wavelength cannot phase-match at any tilt is explained, not raised
    panel.cut_angle_spin.setValue(80.0)
    assert panel.crystal_readout.text().startswith("No valid crystal:")


def test_crystal_panel_refresh_shows_the_stored_settings(panel):
    panel.state.retrieve = RetrieveParams(
        dispersive=True,
        thickness_um=42.0,
        cut_angle_deg=31.0,
        angle_mode="tilt",
        tilt_deg=2.0,
        polarization="perpendicular",
        delayed_beam="away_from_axis",
        beam_separation_mm=6.0,
        focal_f_mm=300.0,
    )
    panel.edits.clear()
    panel.refresh()
    assert not panel.edits  # showing is not editing
    assert panel.crystal_check.isChecked()
    assert panel.thickness_spin.value() == 42.0
    assert panel.cut_angle_spin.value() == 31.0
    assert panel.angle_mode_combo.currentText() == "External tilt"
    assert panel.tilt_spin.value() == 2.0
    assert panel.polarization_combo.currentText() == (
        "Perpendicular to the crossing plane"
    )
    assert panel.delayed_beam_combo.currentText() == "Away from the optic axis"
    assert panel.separation_spin.value() == 6.0
    assert panel.focal_f_spin.value() == 300.0


def test_crystal_panel_unknown_stored_setting_is_explained(panel):
    panel.state.retrieve = RetrieveParams(angle_mode="bogus")
    panel.refresh()
    assert panel.crystal_readout.text().startswith("No valid crystal: angle_mode")


def test_crystal_panel_is_shown_for_shg_only(panel):
    panel.set_shg(False)
    assert panel.isHidden()
    panel.state.retrieve.thickness_um = 21.0
    panel.set_shg(True)
    assert not panel.isHidden()
    assert panel.thickness_spin.value() == 21.0  # refreshed as it is shown


def test_crystal_panel_controls_carry_their_help(panel):
    help_map = flatten(CRYSTAL_HELP)
    assert (
        panel.cut_angle_spin.toolTip()
        == help_map[("Nonlinear crystal", "Cut angle θc (°)")]
    )
    assert (
        panel.beam_diameter_spin.toolTip()
        == help_map[("Beam geometry", "Beam diameter (mm)")]
    )
