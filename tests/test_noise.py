"""Noise-robustness tests, following Geib et al., Optica 6, 495 (2019).

For additive Gaussian trace noise the best achievable FROG error is the error of
the *true* pulse evaluated against the noisy trace (the noise floor ``R0``). A
correct retrieval should reach approximately this floor and not substantially
exceed it.
"""

import numpy as np
import pytest

from croak.copra import COPRA
from croak.forward import maketrace
from croak.grid import Grid
from croak.lbfgs import LBFGS
from croak.metrics import frog_error
from croak.pulses import gaussian_pulse


@pytest.fixture(scope="module")
def noisy_problem():
    g = Grid(128, dt=0.5e-15)
    ew = gaussian_pulse(g, 3.0e-15, phases=[5e-30, 1e-44])  # GDD + TOD
    delays = np.linspace(-20e-15, 20e-15, 64)
    clean = maketrace(g.omega, delays, ew, "pg")
    return g, ew, delays, clean


def _add_noise(clean, sigma, seed):
    rng = np.random.default_rng(seed)
    noisy = clean + sigma * clean.max() * rng.standard_normal(clean.shape)
    return np.clip(noisy, 0.0, None)


def test_noise_floor_increases_with_sigma(noisy_problem):
    g, ew, delays, clean = noisy_problem
    floors = []
    for sigma in (0.0, 0.005, 0.01, 0.03):
        noisy = _add_noise(clean, sigma, seed=0)
        floors.append(frog_error(noisy, clean))
    assert np.all(np.diff(floors) > 0)


def test_copra_noiseless_converges(noisy_problem):
    g, ew, delays, clean = noisy_problem
    near = gaussian_pulse(g, 2.0e-15)
    res = COPRA(maxiters=200).run(
        clean, g.omega, delays, "pg", guess=near, rng=np.random.default_rng(10)
    )
    assert res.error < 5e-3


@pytest.mark.parametrize("sigma", [0.01, 0.03])
def test_copra_reaches_noise_floor(noisy_problem, sigma):
    g, ew, delays, clean = noisy_problem
    noisy = _add_noise(clean, sigma, seed=1)
    r0 = frog_error(noisy, clean)
    near = gaussian_pulse(g, 2.0e-15)
    res = COPRA(maxiters=250).run(
        noisy, g.omega, delays, "pg", guess=near, rng=np.random.default_rng(11)
    )
    assert res.error < r0 + 5e-3
    assert res.error < 3 * r0


@pytest.mark.parametrize("sigma", [0.01, 0.03])
def test_lbfgs_reaches_noise_floor(noisy_problem, sigma):
    g, ew, delays, clean = noisy_problem
    noisy = _add_noise(clean, sigma, seed=2)
    r0 = frog_error(noisy, clean)
    near = gaussian_pulse(g, 2.0e-15)
    res = LBFGS(maxiters=300).run(
        noisy, g.omega, delays, "pg", guess=near, rng=np.random.default_rng(12)
    )
    assert res.error < r0 + 5e-3
    assert res.error < 3 * r0


def test_copra_and_lbfgs_comparable(noisy_problem):
    g, ew, delays, clean = noisy_problem
    noisy = _add_noise(clean, 0.01, seed=3)
    near = gaussian_pulse(g, 2.0e-15)
    r_copra = (
        COPRA(maxiters=250)
        .run(noisy, g.omega, delays, "pg", guess=near, rng=np.random.default_rng(13))
        .error
    )
    r_lbfgs = (
        LBFGS(maxiters=300)
        .run(noisy, g.omega, delays, "pg", guess=near, rng=np.random.default_rng(14))
        .error
    )
    assert 0.1 < r_copra / r_lbfgs < 10.0
