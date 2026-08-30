"""Tests for the :func:`croak.retrieve` wrapper and uniform result type."""

import numpy as np
import pytest

import croak
from croak.result import RetrievalResult


@pytest.fixture
def problem():
    g = croak.Grid(128, dt=0.3e-15)
    ew = croak.gaussian_pulse(g, 2.5e-15)
    delays = np.linspace(-15e-15, 15e-15, 87)
    return g, ew, delays


@pytest.mark.parametrize(
    "algorithm", ["copra", "copra-jax", "lbfgs", "lbfgs-hand", "lbfgs-ad", "lm"]
)
def test_retrieve_dispatch_uniform_result(problem, algorithm):
    g, ew, delays = problem
    trace = croak.maketrace(g.omega, delays, ew, "pg")
    near = croak.gaussian_pulse(g, 1.5e-15)
    res = croak.retrieve(
        trace, g.omega, delays, "pg", algorithm=algorithm, guess=near, maxiters=200
    )
    assert isinstance(res, RetrievalResult)
    assert res.algorithm == algorithm
    assert res.interaction == "pg"
    assert res.spectrum.shape == (g.n,)
    assert res.trace.shape == trace.shape
    assert res.field.shape == (g.n,)
    assert res.error < 5e-3


def test_registry_lists_all_algorithms():
    # dcopra has been removed (it was just COPRA with dispersion); the JAX
    # solvers are registered alongside the analytic ones, including the
    # Optimistix-backed ``-optx`` twins.
    assert set(croak.ALGORITHMS) == {
        "copra",
        "copra-jax",
        "lbfgs",
        "lbfgs-hand",
        "lbfgs-ad",
        "lbfgs-optx",
        "lm",
        "lm-optx",
        "cma-es",
        # COPRA's local sweep used as a starting point for lbfgs-ad.
        "warm-lbfgs",
    }


def test_retrieve_forwards_omega0_to_dispersive_constructor():
    """Regression: ``retrieve`` must pass ``omega0`` to the solver *constructor*.

    Dispersive solvers build their slab dispersion from the constructor's ``omega0``;
    ``run``'s ``omega0`` only records the carrier on the result. Previously ``retrieve``
    forwarded ``omega0`` to ``run`` only, so a dispersive ``retrieve(..., omega0=W)``
    silently built the dispersion at the wrong carrier (``omega0=0``). It must now match
    a solver constructed with ``omega0`` directly (the known-correct pipeline path).
    """
    g = croak.Grid(64, dt=0.5e-15)
    ew = croak.gaussian_pulse(g, 6e-15)
    delays = np.linspace(-40e-15, 40e-15, 30)
    omega0 = float(croak.maths.wlfreq(800e-9))  # carrier ~ 800 nm
    disp = {"material": "SiO2", "thickness": 100e-6, "npoints": 12}
    # PG (TG-FROG) admits a dispersive slab; the trace carries real dispersion.
    trace = croak.maketrace(g.omega, delays, ew, "pg", omega0=omega0, **disp)

    via_retrieve = croak.retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="copra",
        guess=ew,
        omega0=omega0,
        maxiters=15,
        progress=False,
        rng=np.random.default_rng(0),
        **disp,
    )
    # Known-correct path: omega0 in the constructor (what retrieve_from_tracedata does).
    direct = croak.COPRA(maxiters=15, omega0=omega0, **disp).run(
        trace,
        g.omega,
        delays,
        "pg",
        guess=ew,
        omega0=omega0,
        rng=np.random.default_rng(0),
    )
    assert np.all(np.isfinite(via_retrieve.spectrum))
    np.testing.assert_allclose(
        via_retrieve.spectrum, direct.spectrum, rtol=1e-9, atol=1e-12
    )


def test_retrieve_unknown_algorithm(problem):
    g, ew, delays = problem
    trace = croak.maketrace(g.omega, delays, ew, "pg")
    with pytest.raises(ValueError):
        croak.retrieve(trace, g.omega, delays, "pg", algorithm="nope")


def test_retrieve_trace_shape_mismatch(problem):
    g, ew, delays = problem
    trace = croak.maketrace(g.omega, delays, ew, "pg")
    with pytest.raises(ValueError):
        croak.retrieve(trace[:-1], g.omega, delays, "pg")


def test_result_derived_quantities(problem):
    g, ew, delays = problem
    trace = croak.maketrace(g.omega, delays, ew, "shg")
    res = croak.retrieve(trace, g.omega, delays, "shg", guess=ew, maxiters=20)
    assert np.allclose(res.intensity_omega, np.abs(res.spectrum) ** 2)
    assert res.intensity_t.shape == (g.n,)
    assert np.all(res.intensity_t >= 0)


@pytest.mark.parametrize("algorithm", ["copra", "copra-jax", "lbfgs-ad", "warm-lbfgs"])
def test_three_argument_callbacks_still_work(algorithm):
    """A callback without a ``snapshot`` parameter is not an error.

    Every solver can now offer an in-progress result, but the shorter three-argument
    form has always worked with the projection solvers (which had nothing to offer)
    and must keep working — otherwise enabling previews would break user code with a
    ``TypeError`` deep inside the iteration.
    """
    g = croak.Grid(64, dt=2e-15)
    ew = croak.gaussian_pulse(g, 20e-15)
    delays = np.linspace(-60e-15, 60e-15, 24)
    trace = croak.maketrace(g.omega, delays, ew, "pg")
    seen = []
    croak.retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm=algorithm,
        guess=croak.gaussian_pulse(g, 25e-15),
        maxiters=3,
        callback=lambda it, R, best: seen.append(it),
    )
    assert seen and seen == sorted(seen)
