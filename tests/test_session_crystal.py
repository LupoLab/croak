"""The SHG crystal in a session: settings to CrystalSetup, and into the retrieval."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from croak import crystals
from croak.crystals import CrystalSetup
from croak.grid import Grid
from croak.maths import wlfreq
from croak.session import pipeline as session_pipeline
from croak.session.options import SessionOptions
from croak.session.params import RetrieveParams
from croak.session.pipeline import (
    bootstrap_solver_kwargs,
    covariance_kwargs,
    crystal_setup,
    fits_thickness,
    retrieval_medium,
    retrieval_npoints,
)

W0 = float(wlfreq(800e-9))


def _td(interaction="shg"):
    """A stand-in for TraceData: the fields the session reads here."""
    return SimpleNamespace(
        interaction=interaction,
        grid=Grid(64, dt=3e-15),
        omega0_pulse=W0,
        Iomega=None,
    )


def test_crystal_setup_phase_matched_and_crossing_in_the_principal_plane():
    p = RetrieveParams(
        polarization="perpendicular",
        delayed_beam="away_from_axis",
        beam_separation_mm=10.0,
        focal_f_mm=100.0,
    )
    setup = crystal_setup(p)
    assert setup.crystal is crystals.builtin_crystal("BBO")
    assert setup.cut_angle == pytest.approx(math.radians(29.2))
    assert setup.phase_match_wavelength == pytest.approx(800e-9)
    assert setup.tilt is None
    assert setup.crossing_half_angle == pytest.approx(math.atan(10.0 / 200.0))
    assert setup.in_principal_plane
    assert setup.delayed_beam == "away_from_axis"


def test_crystal_setup_with_a_tilt_and_the_polarisation_in_the_crossing_plane():
    p = RetrieveParams(angle_mode="tilt", tilt_deg=2.5, polarization="parallel")
    setup = crystal_setup(p)
    assert setup.tilt == pytest.approx(math.radians(2.5))
    assert setup.phase_match_wavelength is None
    assert not setup.in_principal_plane
    assert setup.delayed_beam is None  # the side matters only in the principal plane


def test_crystal_setup_without_beam_separation_is_collinear():
    setup = crystal_setup(RetrieveParams(beam_separation_mm=0.0))
    assert setup.geometry is None
    assert setup.polarization is None and setup.delayed_beam is None


def test_crystal_setup_refuses_an_unknown_angle_mode():
    with pytest.raises(ValueError, match="angle_mode must be"):
        crystal_setup(RetrieveParams(angle_mode="auto"))


@pytest.mark.parametrize(
    ("dispersive", "interaction", "expected"),
    [(False, "shg", type(None)), (True, "pg", str), (True, "shg", CrystalSetup)],
)
def test_retrieval_medium_is_a_slab_or_a_crystal(dispersive, interaction, expected):
    p = RetrieveParams(dispersive=dispersive, material="BK7")
    medium = retrieval_medium(p, _td(interaction))
    assert isinstance(medium, expected)
    if expected is str:
        assert medium == "BK7"


def test_retrieval_npoints_gives_a_crystal_the_nodes_it_needs():
    td = _td()
    p = RetrieveParams(dispersive=True, thickness_um=100.0, npoints=2)
    medium = retrieval_medium(p, td)
    needed = crystals.depth_points(medium, td.grid.omega, W0, 100e-6)
    assert needed > 2
    # the crystal's own count, whatever the slab's (GUI-tied) npoints says
    assert retrieval_npoints(p, td, medium, 100e-6) == needed
    many = RetrieveParams(dispersive=True, thickness_um=100.0, npoints=needed + 50)
    assert retrieval_npoints(many, td, medium, 100e-6) == needed
    # a slab keeps the settings' count
    assert retrieval_npoints(p, _td("pg"), "BK7", 100e-6) == 2


def test_a_crystals_thickness_is_not_fitted_yet():
    p = RetrieveParams(solver="lbfgs-ad", dispersive=True, fit_thickness=True)
    assert fits_thickness(p, "BK7") is True
    assert fits_thickness(p, crystal_setup(p)) is False
    assert fits_thickness(p, None) is False
    td = _td()
    assert bootstrap_solver_kwargs(p, td)["fit_thickness"] is False
    assert covariance_kwargs(p, td)["fit_thickness"] is False


def test_run_retrieval_does_not_fit_a_crystals_thickness(monkeypatch):
    captured: dict = {}

    def _fake(td, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(spectrum=np.ones(64))

    monkeypatch.setattr(session_pipeline, "retrieve_from_tracedata", _fake)
    p = RetrieveParams(solver="lbfgs-ad", dispersive=True, fit_thickness=True)
    session_pipeline.run_retrieval(p, _td())
    assert captured["fit_thickness"] is False


def test_a_retrieval_crossing_in_the_principal_plane_checks_the_tilt(monkeypatch):
    # A 5 fs-wide spectrum in 300 um crossing at 2.86 degrees: the first-order
    # tilt model leaves about 2 % (tests/test_forward_crystal.py), so it warns.
    from croak.crystals import CrystalApproximationWarning
    from croak.pulses import gaussian_pulse

    td = _td()
    td.delays = np.linspace(-30e-15, 30e-15, 21)
    td.trace = np.ones((td.grid.n, td.delays.size))
    spectrum = gaussian_pulse(td.grid, 5e-15)
    monkeypatch.setattr(
        session_pipeline,
        "retrieve_from_tracedata",
        lambda td, **kw: SimpleNamespace(spectrum=spectrum),
    )
    p = RetrieveParams(
        solver="lbfgs-ad",
        dispersive=True,
        thickness_um=300.0,
        polarization="perpendicular",
        beam_separation_mm=10.0,
    )
    with pytest.warns(CrystalApproximationWarning, match="first order only"):
        session_pipeline.run_retrieval(p, td)


def test_the_thickness_uncertainty_needs_dispersive_propagation():
    rp = RetrieveParams(solver="lbfgs-ad", dispersive=False)
    up = session_pipeline.UncertaintyParams(method="thickness")
    td = _td()
    td.trace = np.ones((4, 3))
    with pytest.raises(ValueError, match="needs dispersive propagation"):
        session_pipeline.run_uncertainty(up, rp, SimpleNamespace(omega0=W0), td)


def test_run_retrieval_passes_the_crystal_and_its_nodes(monkeypatch):
    captured: dict = {}

    def _fake(td, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(spectrum=np.ones(64))

    monkeypatch.setattr(session_pipeline, "retrieve_from_tracedata", _fake)
    p = RetrieveParams(solver="lbfgs-ad", dispersive=True, thickness_um=50.0)
    td = _td()
    session_pipeline.run_retrieval(p, td)
    assert isinstance(captured["material"], CrystalSetup)
    assert captured["thickness"] == pytest.approx(50e-6)
    assert captured["npoints"] == crystals.depth_points(
        captured["material"], td.grid.omega, W0, 50e-6
    )


def test_uncertainty_kwargs_carry_the_crystal():
    p = RetrieveParams(solver="lbfgs-ad", dispersive=True, thickness_um=50.0)
    td = _td()
    for kwargs in (bootstrap_solver_kwargs(p, td), covariance_kwargs(p, td)):
        assert isinstance(kwargs["material"], CrystalSetup)
        assert kwargs["npoints"] >= crystals.depth_points(
            kwargs["material"], td.grid.omega, W0, 50e-6
        )


def test_thickness_uncertainty_draws_through_the_crystal(monkeypatch):
    captured: dict = {}

    def _fake(result, measured, **kwargs):
        captured.update(kwargs)
        return "uncertainty"

    monkeypatch.setattr(session_pipeline, "estimate_fwhm_uncertainty", _fake)
    rp = RetrieveParams(solver="lbfgs-ad", dispersive=True, thickness_um=50.0)
    up = session_pipeline.UncertaintyParams(method="thickness")
    td = _td()
    td.trace = np.ones((4, 3))
    result = SimpleNamespace(omega0=W0)
    session_pipeline.run_uncertainty(up, rp, result, td)
    assert isinstance(captured["material"], CrystalSetup)


def test_crystal_settings_round_trip_through_options_toml(tmp_path):
    retrieve = RetrieveParams(
        dispersive=True,
        crystal="BBO",
        cut_angle_deg=33.4,
        angle_mode="tilt",
        tilt_deg=-1.5,
        phase_match_nm=780.0,
        polarization="perpendicular",
        delayed_beam="away_from_axis",
        beam_separation_mm=7.5,
    )
    options = SessionOptions().with_(retrieve=retrieve)
    path = str(tmp_path / "options.toml")
    options.to_toml(path)
    assert SessionOptions.from_toml(path).retrieve == retrieve


def _tilt_case(n=256, dt=1.5e-15):
    """A 6 fs pulse on a wide grid, crossing at 2.86 degrees in the principal plane."""
    from croak.pulses import gaussian_pulse

    grid = Grid(n, dt=dt)
    td = SimpleNamespace(
        interaction="shg",
        grid=grid,
        omega0_pulse=W0,
        delays=np.linspace(-30e-15, 30e-15, 21),
        trace=np.ones((n, 21)),
        Iomega=None,
    )
    p = RetrieveParams(
        dispersive=True,
        thickness_um=300.0,
        polarization="perpendicular",
        beam_separation_mm=10.0,
    )
    return p, td, SimpleNamespace(spectrum=gaussian_pulse(grid, 6e-15))


def test_crystal_tilt_error_on_the_cropped_band_matches_the_full_grid():
    # the crop keeps every pair whose amplitude exceeds 1e-3 of the peak and all
    # their sum frequencies, so it changes the estimate only at that level
    p, td, result = _tilt_case()
    setup = crystal_setup(p)
    delays = np.linspace(td.delays.min(), td.delays.max(), 9)
    full = crystals.tilt_field_error(
        setup, td.grid.omega, W0, 300e-6, np.abs(result.spectrum), delays
    )
    cropped = session_pipeline.crystal_tilt_error(p, td, result)
    assert full > 0.01
    assert cropped == pytest.approx(full, rel=1e-3)


@pytest.mark.parametrize("broken", ["parallel", "zero trace", "nan spectrum"])
def test_crystal_tilt_error_is_none_when_there_is_nothing_to_check(broken):
    p, td, result = _tilt_case(n=64, dt=3e-15)
    if broken == "parallel":
        p = RetrieveParams(dispersive=True, polarization="parallel")
    elif broken == "zero trace":
        td.trace = np.zeros_like(td.trace)
    else:
        result = SimpleNamespace(spectrum=np.full(td.grid.n, np.nan))
    assert session_pipeline.crystal_tilt_error(p, td, result) is None


def test_run_retrieval_can_leave_the_tilt_check_to_the_caller(monkeypatch):
    import warnings

    p, td, result = _tilt_case(n=64, dt=3e-15)
    monkeypatch.setattr(
        session_pipeline, "retrieve_from_tracedata", lambda td, **kw: result
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session_pipeline.run_retrieval(p, td, check_tilt=False)
