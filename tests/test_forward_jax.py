"""Equivalence tests for the JAX forward model (:mod:`croak.forward_jax`).

The JAX twin must reproduce the numpy :class:`croak.forward.ForwardModel` trace to
near machine precision (it shares all conventions; only the array library
differs). These tests pin that equivalence and the dispersive-SHG guard. The
JAX model also underpins :mod:`test_ad_vs_analytic`, where it validates the
hand-derived adjoints by automatic differentiation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from croak.forward import maketrace
from croak.forward_jax import make_param_trace_fn, make_trace_fn, maketrace_jax
from croak.grid import Grid
from croak.maths import wlfreq
from croak.pulses import gaussian_pulse
from croak.smearing import square_boxcars_kernel


@pytest.fixture
def setup():
    g = Grid(96, dt=0.3e-15)
    ew = gaussian_pulse(g, 2.5e-15, phases=[3e-30])
    delays = np.linspace(-15e-15, 15e-15, 61)
    return g, ew, delays


@pytest.mark.parametrize("interaction", ["shg", "sd", "pg"])
def test_jax_trace_matches_numpy_thin(setup, interaction):
    g, ew, delays = setup
    a = maketrace(g.omega, delays, ew, interaction)
    b = maketrace_jax(g.omega, delays, ew, interaction)
    assert np.allclose(a, b, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("interaction", ["pg", "sd"])  # SHG invalid with dispersion
def test_jax_trace_matches_numpy_dispersive(setup, interaction):
    g, ew, delays = setup
    omega0 = wlfreq(800e-9)
    kw = dict(material="SiO2", thickness=15e-6, npoints=16, omega0=omega0)
    a = maketrace(g.omega, delays, ew, interaction, **kw)
    b = maketrace_jax(g.omega, delays, ew, interaction, **kw)
    assert np.allclose(a, b, atol=1e-10, rtol=1e-10)


def test_jax_unnormalised_matches_numpy(setup):
    """The (un-normalised) trace also matches, not just the peak-scaled one."""
    g, ew, delays = setup
    a = maketrace(g.omega, delays, ew, "pg", normalize=False)
    b = maketrace_jax(g.omega, delays, ew, "pg", normalize=False)
    assert np.allclose(a, b, atol=1e-12, rtol=1e-10)


def test_dispersive_shg_guard(setup):
    g, ew, delays = setup
    with pytest.raises(ValueError, match="dispersive"):
        make_trace_fn(
            g.omega,
            delays,
            "shg",
            material="SiO2",
            thickness=10e-6,
            npoints=8,
            omega0=wlfreq(400e-9),
        )


# ---------------------------------------------------------------------------
# Parametric forward (thickness / delay-zero as runtime arguments)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("interaction", ["shg", "sd", "pg"])
def test_param_trace_matches_standard_thin(setup, interaction):
    """With no extras fitted the parametric trace equals the standard one."""
    g, ew, delays = setup
    std = make_trace_fn(g.omega, delays, interaction)(jnp.asarray(ew))
    par = make_param_trace_fn(g.omega, delays, interaction)(jnp.asarray(ew), 0.0, 0.0)
    assert np.allclose(np.asarray(std), np.asarray(par), atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("interaction", ["pg", "sd"])
def test_param_trace_matches_standard_dispersive(setup, interaction):
    g, ew, delays = setup
    omega0 = wlfreq(800e-9)
    kw = dict(material="SiO2", thickness=15e-6, npoints=16, omega0=omega0)
    std = make_trace_fn(g.omega, delays, interaction, **kw)(jnp.asarray(ew))
    # fixed thickness (fit_thickness=False) must reproduce the baked-node path
    par = make_param_trace_fn(g.omega, delays, interaction, **kw)(
        jnp.asarray(ew), kw["thickness"], 0.0
    )
    assert np.allclose(np.asarray(std), np.asarray(par), atol=1e-12, rtol=1e-10)


def test_param_tau0_equals_shifted_delays(setup):
    """A delay-zero offset tau0 equals evaluating on a shifted delay axis."""
    g, ew, delays = setup
    tau0 = 1.3e-15
    shifted = make_param_trace_fn(g.omega, delays, "shg", fit_tau0=True)(
        jnp.asarray(ew), 0.0, tau0
    )
    ref = make_param_trace_fn(g.omega, delays - tau0, "shg")(jnp.asarray(ew), 0.0, 0.0)
    assert np.allclose(np.asarray(shifted), np.asarray(ref), atol=1e-12, rtol=1e-10)


def test_param_thickness_grad_matches_finite_difference(setup):
    """AD gradient wrt thickness matches a central finite difference."""
    g, ew, delays = setup
    omega0 = wlfreq(800e-9)
    trace = make_param_trace_fn(
        g.omega,
        delays,
        "pg",
        material="SiO2",
        thickness=15e-6,
        npoints=12,
        omega0=omega0,
        fit_thickness=True,
    )
    ewj = jnp.asarray(ew)

    def scalar(thk):
        return jnp.sum(trace(ewj, thk, 0.0) ** 2)

    L0 = 15e-6
    g_ad = float(jax.grad(scalar)(L0))
    h = L0 * 1e-4
    g_fd = (float(scalar(L0 + h)) - float(scalar(L0 - h))) / (2 * h)
    assert g_ad == pytest.approx(g_fd, rel=1e-5)


def test_param_smearing_grad_matches_finite_difference(setup):
    """AD gradient wrt the smearing scale matches a central finite difference.

    The Gauss-Hermite nodes are ``scale * sigma_p * x_k`` and the delay-blur width is
    ``scale * sigma_delta``, so the whole quadrature is smooth in the multiplier and
    reverse-mode AD needs no special handling.
    """
    g, ew, delays = setup
    kernel = square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=1e-3, wavelength=800e-9
    )
    trace = make_param_trace_fn(
        g.omega, delays, "pg", smearing=kernel, fit_smearing=True
    )
    ewj = jnp.asarray(ew)

    def scalar(scale):
        return jnp.sum(trace(ewj, 0.0, 0.0, scale) ** 2)

    s0 = 1.0
    g_ad = float(jax.grad(scalar)(s0))
    h = 1e-5
    g_fd = (float(scalar(s0 + h)) - float(scalar(s0 - h))) / (2 * h)
    assert g_ad == pytest.approx(g_fd, rel=1e-6)


def test_param_fit_smearing_requires_a_kernel(setup):
    g, _ew, delays = setup
    with pytest.raises(ValueError, match="requires a smearing kernel"):
        make_param_trace_fn(g.omega, delays, "pg", fit_smearing=True)


def test_param_fit_thickness_requires_dispersive(setup):
    """Fitting thickness needs a dispersive slab (material set, thickness > 0)."""
    g, _ew, delays = setup
    with pytest.raises(ValueError, match="dispersive slab"):
        make_param_trace_fn(g.omega, delays, "pg", fit_thickness=True)


def test_param_dispersive_shg_guard(setup):
    g, _ew, delays = setup
    with pytest.raises(ValueError, match="dispersive"):
        make_param_trace_fn(
            g.omega,
            delays,
            "shg",
            material="SiO2",
            thickness=10e-6,
            npoints=8,
            omega0=wlfreq(400e-9),
            fit_thickness=True,
        )
