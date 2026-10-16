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

from croak.collection import mask_hole_aperture
from croak.focal import focal_mixture
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


# ---------------------------------------------------------------------------
# Focal-mixture options: guards and the peak normalisation
# ---------------------------------------------------------------------------
LAMBDA_DUV = 260e-9
OMEGA_DUV = 2.0 * np.pi * 299_792_458.0 / LAMBDA_DUV
FOCAL_GEOM = dict(
    hole_diameter=1e-3, hole_spacing=1e-3, f_foc=0.1, wavelength=LAMBDA_DUV
)


@pytest.fixture
def duv():
    g = Grid(64, dt=0.5e-15)
    ew = gaussian_pulse(g, 2e-15)
    delays = np.linspace(-6e-15, 6e-15, 5)
    kw = dict(material="SiO2", thickness=10e-6, npoints=3, omega0=OMEGA_DUV)
    return g, ew, delays, kw


def _focal(collection=None):
    return focal_mixture(
        n_radial=3, n_azimuth=4, r_max_units=3.0, collection=collection, **FOCAL_GEOM
    )


def test_depth_node_phase_needs_the_focal_path_and_its_shape(duv):
    g, _, delays, kw = duv
    with pytest.raises(ValueError, match="applies only to the `focal` path"):
        make_param_trace_fn(
            g.omega, delays, "pg", depth_node_phase=np.ones((1, 3)), **kw
        )
    mix = _focal()
    with pytest.raises(ValueError, match=r"must have shape \(focal.nodes, npoints\)"):
        make_param_trace_fn(
            g.omega, delays, "pg", focal=mix, depth_node_phase=np.ones((1, 2)), **kw
        )


def test_focal_and_reduced_smearing_are_exclusive(duv):
    g, _, delays, kw = duv
    kernel = square_boxcars_kernel(
        "pg", hole_diameter=1e-3, hole_spacing=1e-3, wavelength=LAMBDA_DUV
    )
    with pytest.raises(ValueError, match="pass either `smearing`"):
        make_param_trace_fn(
            g.omega, delays, "pg", focal=_focal(), smearing=kernel, **kw
        )


def test_unit_depth_node_phase_is_the_identity(duv):
    """The diagnostic per-(node, depth) phase of exactly 1 changes nothing."""
    g, ew, delays, kw = duv
    mix = _focal()
    plain = make_param_trace_fn(g.omega, delays, "pg", focal=mix, **kw)
    ones = np.ones((int(mix.nodes), kw["npoints"]))
    phased = make_param_trace_fn(
        g.omega, delays, "pg", focal=mix, depth_node_phase=ones, **kw
    )
    a = np.asarray(plain(jnp.asarray(ew), 10e-6, 0.0))
    b = np.asarray(phased(jnp.asarray(ew), 10e-6, 0.0))
    np.testing.assert_allclose(b, a, rtol=1e-12)


@pytest.mark.parametrize("collected", [None, "integrated", "reimaged"])
def test_focal_trace_normalize_scales_to_unit_peak(duv, collected):
    """``normalize=True`` is the unnormalised focal trace divided by its peak."""
    g, ew, delays, kw = duv
    extra = {}
    if collected:
        aperture = mask_hole_aperture(
            mode=collected,
            hole_x=-1e-3,
            hole_y=-1e-3,
            hole_diameter=0.5e-3,
            z_mask=0.1,
            apod="tanh",
            apod_param=1e-4,
        )
        mix = _focal(collection=aperture)
        extra["depth_transverse"] = True
    else:
        mix = _focal()
    raw = make_param_trace_fn(g.omega, delays, "pg", focal=mix, **extra, **kw)
    unit = make_param_trace_fn(
        g.omega, delays, "pg", focal=mix, normalize=True, **extra, **kw
    )
    t_raw = np.asarray(raw(jnp.asarray(ew), 10e-6, 0.0))
    t_unit = np.asarray(unit(jnp.asarray(ew), 10e-6, 0.0))
    # one division by the same peak; rounding only
    np.testing.assert_allclose(t_unit, t_raw / t_raw.max(), rtol=1e-12)
    assert t_unit.max() == pytest.approx(1.0)


def test_param_trace_normalize_scales_to_unit_peak(setup):
    """The plain (no smearing, no focal) trace honours ``normalize`` too."""
    g, ew, delays = setup[:3]
    raw = make_param_trace_fn(g.omega, delays, "pg")
    unit = make_param_trace_fn(g.omega, delays, "pg", normalize=True)
    t_raw = np.asarray(raw(jnp.asarray(ew), 0.0, 0.0))
    t_unit = np.asarray(unit(jnp.asarray(ew), 0.0, 0.0))
    # one division by the same peak; rounding only
    np.testing.assert_allclose(t_unit, t_raw / t_raw.max(), rtol=1e-12)
