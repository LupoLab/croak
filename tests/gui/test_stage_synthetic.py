"""The synthetic generator: sampling interlocks, physics options, noise, failures."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")


from croak.gui.stage_synthetic import StageSynthetic
from croak.gui.state import WizardState


@pytest.fixture
def stage(qtbot):
    widget = StageSynthetic(WizardState())
    qtbot.addWidget(widget)
    # Count where the generator says the session goes on success.
    widget.went = []
    widget.advanced.connect(widget.went.append)
    # Small, fast settings: a coarse trace is plenty to exercise the logic.
    widget.ndelay.setValue(32)
    widget.nwl.setValue(64)
    widget._timer.stop()
    return widget


def test_step_to_count_clamps_and_rejects_non_positive_steps():
    assert StageSynthetic._step_to_count(100.0, 10.0, 8, 1024) == 11
    assert StageSynthetic._step_to_count(100.0, 1e-6, 8, 1024) == 1024
    assert StageSynthetic._step_to_count(100.0, 1000.0, 8, 1024) == 8
    assert StageSynthetic._step_to_count(100.0, 0.0, 8, 1024) == 8


def test_delay_count_and_step_are_interlocked(stage):
    stage.delay_range.changed.emit(-50.0, 50.0)
    assert stage._delay == (-50.0, 50.0)
    # the step box shows three decimals
    assert stage.delay_step.value() == pytest.approx(100.0 / 31, abs=1e-3)
    stage.ndelay.setValue(101)
    assert stage.delay_step.value() == pytest.approx(1.0)
    stage.delay_step.setValue(2.0)
    assert stage.ndelay.value() == 51
    assert stage._timer.isActive()  # every edit schedules a debounced preview


def test_wavelength_sampling_follows_and_mirrors_onto_the_spectrum(stage):
    stage.interaction.setCurrentText("shg")
    stage.wl_range.changed.emit(350.0, 450.0)
    # SHG: the fundamental sits at twice the signal wavelength.
    assert stage._spec_wl == (700.0, 900.0)
    stage.nwl.setValue(101)
    assert stage.wl_step.value() == pytest.approx(1.0)
    assert stage.nspec.value() == 101
    stage.wl_step.setValue(2.0)
    assert stage.nwl.value() == 51 and stage.nspec.value() == 51


def test_unmirrored_spectrum_has_its_own_sampling(stage):
    stage.spec_mirror.setChecked(False)
    assert stage.spec_range.isEnabled()
    stage.spec_range.changed.emit(600.0, 1000.0)
    stage.nspec.setValue(41)
    assert stage.spec_step.value() == pytest.approx(10.0)
    stage.spec_step.setValue(20.0)
    assert stage.nspec.value() == 21
    # Changing the trace window no longer moves the spectrum window.
    stage.wl_range.changed.emit(300.0, 450.0)
    assert stage._spec_wl == (600.0, 1000.0)
    # Re-mirroring snaps it back to the (SHG-doubled) trace window.
    stage.spec_mirror.setChecked(True)
    assert stage._spec_wl == (600.0, 900.0)


def test_material_and_sech_shape_change_the_pulse(stage):
    _, ew_plain, _ = stage._pulse()
    stage.shape.setCurrentText("Sech²")
    stage.use_material.setChecked(True)
    stage.thickness.set_value(1.0)
    *_, mats = stage._coeffs()
    assert mats == {stage.material.currentText(): pytest.approx(1e-3)}
    grid, ew, _ = stage._pulse()
    # Material dispersion only adds phase: the spectral intensity is unchanged in
    # shape, but the field itself differs.
    assert not np.allclose(ew, ew_plain)
    # A zero thickness means no material, even with the box ticked.
    stage.thickness.set_value(0.0)
    assert stage._coeffs()[-1] is None


def test_spectrum_outside_the_window_is_none(stage):
    grid, ew, omega0 = stage._pulse()
    stage.spec_mirror.setChecked(False)
    # Below the shortest wavelength the auto-sized grid represents (~190 nm for
    # the default 10 fs pulse), so the interpolated spectrum is identically zero.
    stage._spec_wl = (100.0, 150.0)
    lam_spec, ilam_spec = stage._spectrum(grid, ew, omega0)
    assert lam_spec is None and ilam_spec is None


def test_noise_is_off_by_default_and_normalised_when_on(stage):
    trace = np.linspace(0.0, 1.0, 50).reshape(5, 10)
    assert stage._add_noise(trace) is trace
    stage.noise_add.set_value(0.05)
    stage.noise_pois.set_value(1000.0)
    noisy = stage._add_noise(trace)
    assert noisy.shape == trace.shape
    assert noisy.min() >= 0.0 and noisy.max() == pytest.approx(1.0)
    assert not np.allclose(noisy, trace)


def test_noise_on_an_all_zero_trace_stays_zero(stage):
    stage.noise_pois.set_value(100.0)
    zeros = np.zeros((4, 4))
    assert np.array_equal(stage._add_noise(zeros), zeros)


def test_preview_failure_is_reported(stage, monkeypatch):
    def fail():
        raise ValueError("bad grid")

    monkeypatch.setattr(stage, "_pulse", fail)
    stage.on_enter()
    assert stage.status_label.text() == "Generation failed: bad grid"
    assert stage.views["trace"].plot.data is None  # nothing drawn


def test_generate_failure_stays_on_the_page(stage, monkeypatch):
    def fail():
        raise ValueError("bad grid")

    monkeypatch.setattr(stage, "_pulse", fail)
    stage.advance()
    assert stage.status_label.text() == "Generation failed: bad grid"
    assert stage.went == []


def test_generate_with_an_empty_window_asks_to_widen_it(stage):
    stage.interaction.setCurrentText("pg")
    stage._wl = (100.0, 150.0)  # below every wavelength the grid represents
    stage.advance()
    assert "widen the range" in stage.status_label.text()
    assert stage.went == []
    assert stage.state.load_data is None


def test_generate_hands_the_trace_and_truth_on(stage):
    stage.advance()
    assert stage.went == ["check"]
    data = stage.state.load_data
    assert data["trace"].shape == (64, 32)
    assert data["trace"].max() == pytest.approx(1.0)
    assert data["truth"] is not None
    assert stage.state.load.interaction == stage.interaction.currentText()


def test_additive_noise_alone(stage):
    trace = np.linspace(0.0, 1.0, 50).reshape(5, 10)
    stage.noise_add.set_value(0.05)
    noisy = stage._add_noise(trace)
    assert noisy.min() >= 0.0 and noisy.max() == pytest.approx(1.0)


def test_preview_draws_the_trace_continue_hands_on(stage):
    stage.on_enter()
    shown = stage.views["trace"].plot
    stage.advance()
    lam_nm = stage.state.load_data["lam"] / 1e-9
    # The λ window's own samples, not the whole simulation grid, in which the
    # signal band was one or two rows of a featureless stripe.
    assert shown.data.shape == stage.state.load_data["trace"].shape
    assert shown.y_axis[0] == pytest.approx(lam_nm[0])
    assert shown.y_axis[-1] == pytest.approx(lam_nm[-1])


def test_preview_with_an_empty_window_asks_to_widen_it(stage):
    stage.interaction.setCurrentText("pg")
    stage._wl = (100.0, 150.0)  # below every wavelength the grid represents
    stage.on_enter()
    assert "widen the range" in stage.status_label.text()
    assert stage.status_label.state == "warn"


def test_preview_draws_the_four_panels(stage):
    stage.on_enter()
    titles = {view.title.text() for view in stage.views.values()}
    assert titles == {"Temporal", "Spectral", "Spectrogram", "Synthetic FROG trace"}
    assert stage.views["trace"].plot.data is not None
    assert stage.status_label.text().startswith("Adjust the pulse")
    assert stage._last is not None


def test_shg_traces_can_be_made_through_the_crystal_panels_crystal(stage):
    from croak.crystals import depth_points
    from croak.forward import maketrace
    from croak.session.pipeline import crystal_setup

    stage.interaction.setCurrentText("shg")
    panel = stage.crystal_panel
    assert not panel.isHidden()
    grid, ew, omega0 = stage._pulse()
    _, delays, ideal = stage._signal_trace(grid, ew, omega0)
    stage._timer.stop()
    panel.crystal_check.setChecked(True)
    panel.thickness_spin.setValue(100.0)
    # the panel's edits are the session's crystal, and they redraw the preview
    assert stage.state.retrieve.dispersive is True
    assert stage.state.retrieve.thickness_um == 100.0
    assert stage._timer.isActive()
    _, _, made = stage._signal_trace(grid, ew, omega0)
    # the crystal is the one the session's settings describe
    setup = crystal_setup(stage.state.retrieve)
    expected = maketrace(
        grid.omega,
        delays,
        ew,
        "shg",
        material=setup,
        thickness=100e-6,
        npoints=depth_points(setup, grid.omega, omega0, 100e-6),
        omega0=omega0,
    )
    pos = grid.omega + 2 * omega0 > 0
    order = np.argsort(1.0 / (grid.omega + 2 * omega0)[pos])
    # the same computation in the same order: equal to rounding, relative to the peak
    want = expected[pos][order]
    np.testing.assert_allclose(made, want, rtol=0.0, atol=1e-12 * want.max())
    # 100 um of phase matching changes the trace
    assert not np.allclose(made, ideal)
    # a zero thickness is the thin medium again, as is the crystal unmodelled
    panel.thickness_spin.setValue(0.0)
    np.testing.assert_array_equal(stage._signal_trace(grid, ew, omega0)[2], ideal)
    panel.thickness_spin.setValue(100.0)
    panel.crystal_check.setChecked(False)
    np.testing.assert_array_equal(stage._signal_trace(grid, ew, omega0)[2], ideal)


def test_the_crystal_is_for_shg_only(stage):
    stage.interaction.setCurrentText("pg")
    assert stage.crystal_panel.isHidden()
    stage.state.retrieve.dispersive = True
    stage.state.retrieve.thickness_um = 100.0
    assert stage._crystal_kwargs(*stage._pulse()[::2]) == {}
    stage.interaction.setCurrentText("shg")
    assert not stage.crystal_panel.isHidden()


def test_entering_shows_crystal_settings_changed_elsewhere(stage):
    stage.state.retrieve.thickness_um = 42.0
    stage.on_enter()
    assert stage.crystal_panel.thickness_spin.value() == 42.0
