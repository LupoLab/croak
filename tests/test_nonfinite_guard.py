"""The non-finite objective guard for the NLopt line search."""

from __future__ import annotations

import numpy as np
import pytest

from croak import Grid, retrieve
from croak.forward_jax import maketrace_jax
from croak.pulses import gaussian_pulse
from croak.solver import NONFINITE_PENALTY, guard_nonfinite


def test_guard_passes_finite_values_through():
    grad = np.ones(4)
    state: dict = {}
    assert guard_nonfinite(1.5, grad, grad, state) is None
    assert np.all(grad == 1.0) and state == {}


def test_guard_traps_nonfinite_total_and_gradient():
    state: dict = {}
    grad = np.ones(4)
    with pytest.warns(RuntimeWarning, match="non-finite"):
        penalty = guard_nonfinite(float("nan"), None, grad, state)
    assert penalty == NONFINITE_PENALTY and np.isfinite(penalty)
    assert np.all(grad == 0.0)
    # second trip in the same solve: silent
    bad = np.array([1.0, np.inf])
    penalty = guard_nonfinite(0.1, bad, bad, state)
    assert penalty == NONFINITE_PENALTY and np.all(bad == 0.0)


def test_retrieval_survives_nlopt_line_search_failure():
    """Regression: the seed-0 dispersive retrieval that aborted mid line search.

    On the platform where this was observed (Grid(512, 0.5 fs), 50 um SiO2,
    random seed 0), NLopt's internal line search failed at eval ~13 and the
    retrieval died with a bare ``runtime_error``. It must now finish by
    salvaging the best iterate (the run ends early with a warning, which is
    platform-dependent and not asserted) instead of raising.
    """
    C = 299_792_458.0
    omega0 = 2 * np.pi * C / 260e-9
    g = Grid(512, dt=0.5e-15)
    ew = gaussian_pulse(g, 1.0e-15)
    delays = np.arange(-40e-15, 40.0001e-15, 0.5e-15)
    kw = dict(material="SiO2", thickness=50e-6, npoints=50)
    trace = maketrace_jax(g.omega, delays, ew, "pg", omega0=omega0, **kw)
    result = retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="lbfgs-ad",
        guess=None,
        rng=np.random.default_rng(0),
        omega0=omega0,
        maxiters=80,
        R_omega=False,
        progress=False,
        **kw,
    )
    assert np.isfinite(result.error) and result.error < 0.1
