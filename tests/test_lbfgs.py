"""Convergence tests for the L-BFGS solver."""

import numpy as np
import pytest

from croak.forward import maketrace
from croak.grid import Grid
from croak.lbfgs import LBFGS
from croak.maths import wlfreq
from croak.pulses import gaussian_pulse


@pytest.fixture
def setup():
    g = Grid(128, dt=0.3e-15)
    ew = gaussian_pulse(g, 2.5e-15)
    delays = np.linspace(-15e-15, 15e-15, 87)
    return g, ew, delays


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_lbfgs_near_initial(setup, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    near = gaussian_pulse(g, 1.5e-15)
    res = LBFGS(maxiters=300).run(trace, g.omega, delays, interaction, guess=near)
    assert res.error < 5e-3


@pytest.mark.parametrize("interaction", ["pg", "shg"])
def test_lbfgs_random_initial(setup, interaction):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, interaction)
    rng = np.random.default_rng(1)
    res = LBFGS(maxiters=500).run(
        trace, g.omega, delays, interaction, guess=None, rng=rng
    )
    assert res.error < 5e-2


def test_lbfgs_chirped_phase_only(setup):
    g, _, delays = setup
    ew = gaussian_pulse(g, 2.5e-15, phases=[3e-30])
    trace = maketrace(g.omega, delays, ew, "pg")
    # correct amplitude, perturbed phase
    guess = gaussian_pulse(g, 2.5e-15, phases=[1e-30])
    res = LBFGS(maxiters=300, phase_only=True).run(
        trace, g.omega, delays, "pg", guess=guess
    )
    assert res.error < 1e-2


def _spectrum_l2(res, I_true):
    """Peak-normalised L2 distance between retrieved |E(ω)| and √I_true."""
    a = np.abs(res.spectrum)
    a /= a.max()
    s = np.sqrt(I_true)
    s /= s.max()
    return float(np.linalg.norm(a - s))


def test_lbfgs_spectral_reg_improves_spectrum(setup):
    # Structured (double-peak) spectrum, deliberately wrong-amplitude guess on a
    # NOISY trace: a clean PG trace already determines |E(ω)|, so noise is what
    # makes the amplitude under-constrained and gives the spectral-match penalty
    # something to pull on — with it, |E(ω)| lands close to the true spectrum.
    g, _, delays = setup
    ew = gaussian_pulse(g, 2.5e-15) + 0.6 * gaussian_pulse(g, 1.2e-15)
    ew = ew / np.max(np.abs(ew))
    I_true = np.abs(ew) ** 2
    clean = maketrace(g.omega, delays, ew, "pg")
    rng = np.random.default_rng(0)
    trace = np.clip(clean + 0.03 * rng.standard_normal(clean.shape), 0.0, None)
    near = gaussian_pulse(g, 4.0e-15)  # wrong amplitude
    r0 = LBFGS(maxiters=400).run(trace, g.omega, delays, "pg", guess=near)
    r1 = LBFGS(maxiters=400, reg_spectrum=0.5, spectrum_target=I_true).run(
        trace, g.omega, delays, "pg", guess=near
    )
    assert r1.error < 5e-2  # near the noise floor
    assert _spectrum_l2(r1, I_true) < _spectrum_l2(r0, I_true)


def test_lbfgs_spectral_reg_none_target_is_noop(setup):
    g, ew, delays = setup
    trace = maketrace(g.omega, delays, ew, "pg")
    near = gaussian_pulse(g, 1.5e-15)
    r0 = LBFGS(maxiters=200).run(trace, g.omega, delays, "pg", guess=near)
    r1 = LBFGS(maxiters=200, reg_spectrum=0.3, spectrum_target=None).run(
        trace, g.omega, delays, "pg", guess=near
    )
    assert np.allclose(r0.spectrum, r1.spectrum)


def test_lbfgs_dispersive():
    g = Grid(128, dt=0.5e-15)
    omega0 = wlfreq(800e-9)
    ew = gaussian_pulse(g, 3.0e-15)
    delays = np.linspace(-20e-15, 20e-15, 80)
    trace = maketrace(
        g.omega,
        delays,
        ew,
        "pg",
        material="SiO2",
        thickness=10e-6,
        npoints=20,
        omega0=omega0,
    )
    near = gaussian_pulse(g, 2.0e-15)
    res = LBFGS(
        maxiters=400, material="SiO2", thickness=10e-6, npoints=20, omega0=omega0
    ).run(trace, g.omega, delays, "pg", guess=near)
    assert res.error < 5e-2
