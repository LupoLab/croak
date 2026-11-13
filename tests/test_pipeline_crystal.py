"""Retrieval of an SHG trace made in a Type I crystal, through every solver.

The trace is simulated with the crystal model of :mod:`croak.crystals` (a synthetic
BBO-like Sellmeier pair, 30 µm, crossing in the principal plane) and retrieved with
the same crystal passed as ``material``, which the solvers forward to the forward
models unchanged.
"""

from __future__ import annotations

import functools
import math
from dataclasses import replace

import numpy as np
import pytest
from test_crystals import BBO, W0

from croak import crystals, preprocess
from croak.crystals import CrystalSetup
from croak.forward import ForwardModel
from croak.geometry import TwoBeamGeometry
from croak.grid import Grid
from croak.maths import wlfreq
from croak.covariance import parameter_covariance
from croak.pipeline import retrieve_from_tracedata
from croak.pulses import gaussian_pulse
from croak.retrieve import ALGORITHMS
from croak.uncertainty import thickness_bootstrap

THICKNESS = 30e-6
CRYSTAL = CrystalSetup(
    BBO,
    math.radians(29.2),
    phase_match_wavelength=800e-9,
    geometry=TwoBeamGeometry(beam_separation=5e-3, focal_length=0.1),
    polarization="perpendicular",
    delayed_beam="toward_axis",
)


def _tracedata(thickness):
    """A 10 fs pulse with 30 fs^2 of GDD, its SHG trace made in the crystal."""
    g = Grid(128, dt=1.5e-15)
    ew = gaussian_pulse(g, 10e-15, phases=(30e-30,))
    delays = 1.5e-15 * np.arange(-24, 25)
    npoints = crystals.depth_points(CRYSTAL, g.omega, W0, thickness)
    model = ForwardModel(
        g.omega,
        delays,
        "shg",
        material=CRYSTAL,
        thickness=thickness,
        npoints=npoints,
        omega0=W0,
    )
    trace = model.trace(ew, normalize=False)
    omega_abs = g.omega + 2.0 * W0  # the SHG trace sits at twice the carrier
    lam = wlfreq(omega_abs)
    order = np.argsort(lam)
    return preprocess.load_and_clean(
        trace[order] / trace.max(),
        lam[order],
        delays,
        "shg",
        # the retrieval grid's range, in fundamental wavelengths
        lam_min=600e-9,
        lam_max=1200e-9,
        input_unit="delay",
        threshold=1e-6,
    )


@pytest.fixture(scope="module")
def crystal_tracedata():
    return _tracedata(THICKNESS)


def _depth_points(td, thickness=THICKNESS):
    return crystals.depth_points(CRYSTAL, td.grid.omega, td.omega0_pulse, thickness)


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_every_solver_retrieves_with_a_crystal(crystal_tracedata, algorithm, monkeypatch):
    # Spy on the solver's constructor: the crystal must reach it unchanged (a
    # solver that dropped `material` would otherwise still return finite numbers).
    original = ALGORITHMS[algorithm]
    received: dict = {}

    class Spy(original):  # type: ignore[valid-type, misc]
        @functools.wraps(original.__init__)
        def __init__(self, **kwargs):
            received.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setitem(ALGORITHMS, algorithm, Spy)
    td = crystal_tracedata
    res = retrieve_from_tracedata(
        td,
        algorithm=algorithm,
        maxiters=30,
        material=CRYSTAL,
        thickness=THICKNESS,
        npoints=_depth_points(td),
        rng=np.random.default_rng(0),
    )
    assert received["material"] is CRYSTAL
    assert received["thickness"] == THICKNESS
    assert np.all(np.isfinite(res.spectrum))
    assert np.isfinite(res.error)


def test_parameter_covariance_runs_through_the_crystal(crystal_tracedata):
    td = crystal_tracedata
    kw = dict(material=CRYSTAL, thickness=THICKNESS, npoints=_depth_points(td))
    res = retrieve_from_tracedata(
        td, algorithm="lbfgs-ad", maxiters=200, rng=np.random.default_rng(0), **kw
    )
    cov = parameter_covariance(res, td.trace, noise=1e-3, omega0=td.omega0_pulse, **kw)
    assert np.all(np.isfinite(cov.cov)) and cov.rank > 0


def test_the_crystal_model_fits_its_own_trace_better_than_ideal_shg():
    # In 100 um the crystal's phase matching and group-velocity mismatch shape the
    # trace; retrieving with the crystal model must fit it better than ideal SHG.
    td = _tracedata(100e-6)
    common = dict(algorithm="lbfgs-ad", maxiters=300, rng=np.random.default_rng(0))
    with_crystal = retrieve_from_tracedata(
        td,
        material=CRYSTAL,
        thickness=100e-6,
        npoints=_depth_points(td, 100e-6),
        **common,
    )
    ideal = retrieve_from_tracedata(td, **common)
    assert with_crystal.error < 0.5 * ideal.error


def test_thickness_bootstrap_resolves_each_drawn_crystal(crystal_tracedata):
    # A draw thicker than the central crystal has a larger phase mismatch to
    # resolve; the bootstrap must raise the depth nodes for it rather than let the
    # forward model refuse the under-resolved draw.
    td = crystal_tracedata
    npoints = _depth_points(td)
    base = retrieve_from_tracedata(
        td,
        algorithm="lbfgs-ad",
        maxiters=20,
        material=CRYSTAL,
        thickness=THICKNESS,
        npoints=npoints,
        rng=np.random.default_rng(0),
    )
    out = thickness_bootstrap(
        base,
        td.trace,
        central_thickness=THICKNESS,
        thickness_sigma=40e-6,
        material=CRYSTAL,
        n_resamples=3,
        rng=np.random.default_rng(3),
        algorithm="lbfgs-ad",
        maxiters=5,
        npoints=npoints,
    )
    drawn = np.asarray(out.thicknesses)
    assert drawn.max() > THICKNESS
    needed = crystals.depth_points(CRYSTAL, td.grid.omega, td.omega0_pulse, drawn.max())
    assert needed > npoints  # the thickest draw would be refused at npoints
    assert np.all(np.isfinite(out.samples))


def test_a_crystal_refuses_a_marginal_corrected_trace(crystal_tracedata):
    # The correction already divides out an estimate of the phase-matching
    # response, which the crystal model would apply again.
    td = replace(crystal_tracedata, marginal_corrected=True)
    with pytest.raises(ValueError, match="marginal-corrected"):
        retrieve_from_tracedata(
            td,
            algorithm="lbfgs-ad",
            material=CRYSTAL,
            thickness=THICKNESS,
            npoints=_depth_points(td),
        )
