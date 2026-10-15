"""The dispersion stage: tuning controls, auto-tuners, gases, RI materials, save."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from croak import materials, refractive_db
from croak.forward import maketrace
from croak.grid import Grid
from croak.gui import stage_dispersion as sd
from croak.gui.stage_dispersion import StageDispersion
from croak.gui.state import WizardState
from croak.maths import wlfreq
from croak.processing import process_result
from croak.pulses import gaussian_pulse
from croak.retrieve import retrieve
from croak.session.params import BeamPathMirror

#: The test pulse's chirp (s²): positive GDD, so compression needs negative GDD
#: or a negative thickness of a normally dispersive glass.
GDD = 40e-30


@pytest.fixture(scope="module")
def chirped_result():
    g = Grid(128, dt=0.5e-15)
    ew = gaussian_pulse(g, 8e-15, phases=(GDD,))
    delays = np.linspace(-60e-15, 60e-15, 48)
    trace = maketrace(g.omega, delays, ew, "pg")
    return retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="copra",
        maxiters=10,
        guess=ew,
        omega0=wlfreq(800e-9),
        progress=False,
    )


def _make_stage(qtbot, result=None):
    state = WizardState()
    if result is not None:
        state.set_result(result)
    stage = StageDispersion(state)
    qtbot.addWidget(stage)
    # Nelder-Mead from the centre is fast and plenty for a pure-GDD pulse.
    stage.method.setCurrentText("nelder-mead")
    return stage


@pytest.fixture
def stage(qtbot, chirped_result):
    widget = _make_stage(qtbot, chirped_result)
    widget.on_enter()
    yield widget
    _cleanup(widget)


def _cleanup(stage):
    """Leave the module-global mirror and material registries as they were.

    The mirror stack is cleared first, with its signals blocked: its ``changed``
    would otherwise recompute against materials already unregistered.
    """
    stage.mirror_stack.blockSignals(True)
    stage.mirror_stack.clear()
    for key in (stage._ri_key, stage._gas_key):
        if key:
            materials.unregister_material(key)


def _fwhm(stage) -> float:
    return process_result(stage._modified()).fwhm_retr


def test_without_a_result_everything_is_inert(qtbot):
    stage = _make_stage(qtbot)
    stage.on_enter()
    assert stage.status_label.text() == "Retrieve a pulse first."
    for action in (
        stage._auto_taylor,
        stage._auto_ri,
        stage._auto_gas,
        stage._save,
        lambda: stage._auto_material("SiO2"),
        lambda: stage._auto_row(stage.mirror_stack.add_row()),
    ):
        action()
    assert not stage.canvas.rendered
    with pytest.raises(RuntimeError, match="no retrieval result"):
        stage._modified()
    with pytest.raises(RuntimeError, match="no retrieval result"):
        stage._applied_phase()
    with pytest.raises(RuntimeError, match="no base spectrum"):
        stage._mirror_base(None)
    stage.mirror_stack.clear()


def test_base_without_a_result_is_inert_for_the_tuners(qtbot, chirped_result):
    # A base spectrum but the result gone (a new session): the tuners do nothing.
    stage = _make_stage(qtbot, chirped_result)
    stage.on_enter()
    stage.state.result = None
    before = stage.gdd.value()
    stage._auto_taylor()
    stage._auto_material("SiO2")
    stage._auto_ri()
    assert stage.gdd.value() == before
    _cleanup(stage)


def test_entering_shows_the_pulse_and_power_labels(stage):
    assert stage.canvas.rendered
    assert stage.fwhm_label.text().endswith(" fs")
    assert stage.peak_label.text() == "–"
    stage.energy.setValue(1e-6)
    assert stage.state.load.energy_j == pytest.approx(1e-6)
    assert stage.peak_label.text().endswith("W")
    assert stage.peak_tl_label.text().endswith("W")


def test_auto_taylor_compresses_the_chirped_pulse(stage):
    before = _fwhm(stage)
    stage._auto_taylor()
    # The pulse carries +40 fs² of GDD; the optimum cancels most of it.
    assert stage.gdd.value() == pytest.approx(-GDD / 1e-30, abs=10.0)
    assert _fwhm(stage) < 0.8 * before
    assert stage.status_label.text().startswith("Auto: GDD")


def test_auto_material_uses_a_negative_glass_thickness(stage):
    before = _fwhm(stage)
    stage._auto_material("SiO2")
    # Normal dispersion: compensating positive GDD needs negative thickness.
    assert stage._mat_ctrls["SiO2"].value() < 0
    assert _fwhm(stage) < before
    assert stage.state.dispersion.material_thickness_mm["SiO2"] < 0


def test_material_with_no_gdd_falls_back_to_the_unit_span(stage, monkeypatch):
    monkeypatch.setattr(sd.dispersion, "material_gdd", lambda *a: 0.0)
    # Without a GDD estimate the search box is the ±1 mm floor. Nelder–Mead is a
    # local polish that may step past its box, so only check it seeded there and
    # did not wander far (the true optimum is ~ -1 mm for this pulse).
    stage._auto_material("SiO2")
    assert -3.0 < stage._mat_ctrls["SiO2"].value() < 0.0


def test_gas_select_pressure_auto_and_clear(stage):
    stage.gas_combo.setCurrentText("Air")
    assert stage._gas_key == "_active_gas"
    n_1bar = materials.refractive_index(stage._gas_key)(800e-9)
    stage.gas_pressure.set_value(2.0)
    stage.gas_pressure.changed.emit(2.0)
    n_2bar = materials.refractive_index(stage._gas_key)(800e-9)
    # n - 1 scales with pressure (ideal gas).
    # n - 1 scales (to first order) with pressure: the ideal-gas density doubles,
    # and the Lorentz–Lorenz correction at this tiny n - 1 is ~1e-4 relative.
    np.testing.assert_allclose(n_2bar - 1, 2 * (n_1bar - 1), rtol=1e-3)
    stage._auto_gas()
    assert stage.gas_path.value() != 0.0
    assert stage._gas_key in stage._extra_dispersion()[0]
    stage.gas_combo.setCurrentText(sd._GAS_NONE)
    assert stage._gas_key is None and stage._gas_name == ""
    # Pressure edits with no gas only redraw.
    stage.gas_pressure.changed.emit(1.5)
    assert stage._gas_key is None


def test_auto_gas_without_a_gas_does_nothing(stage):
    stage._auto_gas()
    assert stage.gas_path.value() == 0.0


@pytest.mark.parametrize("direction", ["remove", "add"])
def test_auto_row_finds_the_best_bounce_count(stage, direction):
    row = stage.mirror_stack.add_row()
    row.dir_combo.setCurrentText(direction)
    row.bounces.setMaximum(3)  # keep the scan short
    stage._auto_row(row)
    assert 0 <= row.bounces.value() <= 3
    # The chosen count is the best of the scan.
    best = float(np.max(stage._modified().intensity_t))
    for n in range(4):
        row.set_bounces(n, silent=True)
        assert float(np.max(stage._modified().intensity_t)) <= best * (1 + 1e-12)


def test_mirror_base_applies_only_the_mirrors(stage):
    r = stage.state.result
    assert stage._mirror_base(r) is stage._base
    row = stage.mirror_stack.add_row()
    row.set_bounces(2)
    assert not np.allclose(stage._mirror_base(r), stage._base)


def test_reset_clears_every_control(stage):
    stage.gdd.set_value(100.0)
    stage._mat_ctrls["SiO2"].set_value(1.0)
    stage.gas_combo.setCurrentText("Helium")
    stage.mirror_stack.add_row()
    stage._reset()
    assert stage.gdd.value() == 0.0
    assert stage._mat_ctrls["SiO2"].value() == 0.0
    assert stage.gas_combo.currentText() == sd._GAS_NONE
    assert stage._gas_key is None
    assert stage.mirror_stack.to_specs() == []
    assert stage.gas_pressure.value() == 1.0


def test_saved_settings_are_restored_on_entry(qtbot, chirped_result):
    stage = _make_stage(qtbot, chirped_result)
    d = stage.state.dispersion
    d.gdd_fs2 = -30.0
    d.material_thickness_mm = {"BK7": 0.5}
    d.gas_name = "Helium"
    d.gas_pressure_bar = 3.0
    d.gas_path_cm = 10.0
    d.mirrors = [BeamPathMirror(name="", polarization="s", bounces=1)]
    stage.on_enter()
    assert stage.gdd.value() == pytest.approx(-30.0)
    assert stage._mat_ctrls["BK7"].value() == pytest.approx(0.5)
    assert stage.gas_combo.currentText() == "Helium"
    assert stage._gas_key == "_active_gas"
    assert len(stage.mirror_stack.to_specs()) == 1
    _cleanup(stage)


# -- refractiveindex.info (a fake catalogue: the extra is not installed) -------


def _n_glass(lam_m):
    """A normally dispersive Cauchy glass, n = 1.45 + 3600 nm² / λ²."""
    lam_nm = np.asarray(lam_m, dtype=float) / 1e-9
    return 1.45 + 3600.0 / lam_nm**2


@pytest.fixture
def fake_ridb(monkeypatch):
    calls = []

    def make_material(sid, bid, pid):
        calls.append((sid, bid, pid))
        if pid == "broken":
            raise ValueError("no dispersion formula")
        return _n_glass, (400e-9, 1200e-9)

    monkeypatch.setattr(refractive_db, "available", lambda: True)
    monkeypatch.setattr(refractive_db, "shelves", lambda: [("main", "Main")])
    monkeypatch.setattr(refractive_db, "books", lambda sid: [("Glass", "A glass")])
    monkeypatch.setattr(
        refractive_db,
        "pages",
        lambda sid, bid: [
            ("Cauchy", "Cauchy fit"),
            ("broken", "Broken"),
            ("Cauchy2", "Second fit"),
        ],
    )
    monkeypatch.setattr(refractive_db, "make_material", make_material)
    return calls


def test_ri_unavailable_explains_how_to_install(qtbot, monkeypatch):
    monkeypatch.setattr(refractive_db, "available", lambda: False)
    stage = _make_stage(qtbot)
    assert not stage.ri_load_btn.isEnabled()
    assert stage.ri_load_btn.text() == "Install 'ridb' extra"
    stage._load_ri_db()
    assert "croak[ridb]" in stage.status_label.text()


def test_ri_catalogue_failure_is_reported(qtbot, fake_ridb, monkeypatch):
    def fail():
        raise OSError("download failed")

    monkeypatch.setattr(refractive_db, "shelves", fail)
    stage = _make_stage(qtbot)
    assert stage.ri_load_btn.isEnabled()
    stage._load_ri_db()
    assert stage.status_label.text() == (
        "refractiveindex.info unavailable: download failed"
    )


def test_ri_cascade_registers_the_material_and_auto_tunes(stage, fake_ridb):
    stage._load_ri_db()
    assert stage.status_label.text() == "1 shelves loaded."
    assert stage._ri_key == "RI:Glass/Cauchy"
    assert stage.ri_range_label.text() == "400–1200 nm"
    assert stage.state.dispersion.ri_page == "Cauchy"
    stage._auto_ri()
    assert stage.ri_thickness.value() < 0  # normal dispersion compensates +GDD
    assert stage._ri_key in stage._extra_dispersion()[0]


def test_ri_auto_without_a_usable_gdd_searches_a_fixed_span(
    stage, fake_ridb, monkeypatch
):
    stage._load_ri_db()
    monkeypatch.setattr(sd.dispersion, "material_gdd", lambda *a: float("nan"))
    stage._auto_ri()  # the ±20 mm fallback box; the optimum is a thin negative slab
    assert -20.0 < stage.ri_thickness.value() < 0.0


def test_ri_page_that_fails_to_load_keeps_the_previous_material(stage, fake_ridb):
    stage._load_ri_db()
    stage.ri_page.setCurrentIndex(1)  # the broken page
    assert stage.status_label.text() == "Could not load material: no dispersion formula"
    assert stage._ri_key == "RI:Glass/Cauchy"


def test_ri_selecting_another_page_replaces_the_registration(stage, fake_ridb):
    stage._load_ri_db()
    stage.ri_page.setCurrentIndex(2)
    assert stage._ri_key == "RI:Glass/Cauchy2"
    # The first page's material is no longer registered.
    with pytest.raises(ValueError, match="unknown material"):
        materials.refractive_index("RI:Glass/Cauchy")


def test_ri_auto_without_a_result_does_nothing(stage, fake_ridb):
    stage._load_ri_db()
    stage.state.result = None
    stage._auto_ri()
    assert stage.ri_thickness.value() == 0.0


def test_ri_shelf_with_no_books_stops_the_cascade(stage, fake_ridb, monkeypatch):
    monkeypatch.setattr(refractive_db, "books", lambda sid: [])
    stage._load_ri_db()
    assert stage.ri_book.count() == 0
    assert stage._ri_key is None


def test_ri_handlers_without_a_catalogue_do_nothing(stage):
    stage._on_ri_shelf()
    stage._on_ri_book()
    stage._on_ri_page()
    stage._auto_ri()
    assert stage._ri_key is None


def test_ri_material_is_restored_from_saved_settings(stage, fake_ridb):
    d = stage.state.dispersion
    d.ri_shelf, d.ri_book, d.ri_page = "main", "Glass", "Cauchy"
    stage._restore_ri(d)
    assert stage._ri_key == "RI:Glass/Cauchy"
    # Restoring again replaces the registration rather than leaking a second one.
    stage._restore_ri(d)
    assert fake_ridb[-1] == ("main", "Glass", "Cauchy")
    # A page that no longer loads is skipped silently (best-effort restore).
    d.ri_page = "broken"
    stage._restore_ri(d)
    assert stage._ri_key == "RI:Glass/Cauchy"


# -- save ------------------------------------------------------------------------


def test_save_cancelled_writes_nothing(stage, monkeypatch, tmp_path):
    monkeypatch.setattr(
        sd.QFileDialog, "getExistingDirectory", staticmethod(lambda *a: "")
    )
    stage._save()
    assert list(tmp_path.iterdir()) == []


def test_save_writes_result_options_and_figure(stage, monkeypatch, tmp_path):
    stage.gdd.set_value(-20.0)
    monkeypatch.setattr(
        sd.QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *a: str(tmp_path)),
    )
    stage._save()
    for name in ("result_dispersion.h5", "options.toml", "dispersion.pdf"):
        assert (tmp_path / name).exists()
    assert "gdd_fs2 = -20.0" in (tmp_path / "options.toml").read_text()
    assert stage.status_label.text() == f"Saved to {tmp_path}"
