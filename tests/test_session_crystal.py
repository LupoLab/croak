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


def test_retrieval_npoints_raises_a_crystals_nodes_to_what_it_needs():
    td = _td()
    p = RetrieveParams(dispersive=True, thickness_um=100.0, npoints=2)
    medium = retrieval_medium(p, td)
    needed = crystals.depth_points(medium, td.grid.omega, W0, 100e-6)
    assert needed > 2
    assert retrieval_npoints(p, td, medium, 100e-6) == needed
    # never lowered, and a slab keeps the settings' count
    many = RetrieveParams(dispersive=True, thickness_um=100.0, npoints=needed + 5)
    assert retrieval_npoints(many, td, medium, 100e-6) == needed + 5
    assert retrieval_npoints(p, _td("pg"), "BK7", 100e-6) == 2


def test_run_retrieval_passes_the_crystal_and_its_nodes(monkeypatch):
    captured: dict = {}

    def _fake(td, **kwargs):
        captured.update(kwargs)
        return "result"

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
