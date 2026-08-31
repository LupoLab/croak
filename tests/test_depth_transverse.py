"""Per-depth transverse dephasing (``depth_transverse``) of the collection model.

The load-bearing physics validation lives in
``tests/test_collection.py::test_matches_a_dense_fourier_reference_with_depth``
(the per-depth dense-FFT reference that pins the sign and the absolute-wavevector
convention). This file pins the contract around it: the flag off is bit-identical
to the standard model, the phase sits inside the coherent depth sum (the Q = 1
parity split), the magnitude at the reference geometry, differentiability, the
wide-window full-collection route, the guards and the solver plumbing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import croak.forward_jax as forward_jax
from croak import Grid
from croak.collection import CollectionAperture, mask_hole_aperture
from croak.focal import _ARMS_PG, focal_mixture, mixture_from_nodes
from croak.forward_jax import make_param_trace_fn
from croak.pulses import gaussian_pulse

C_LIGHT = 299_792_458.0
LAMBDA0 = 260e-9
OMEGA0 = 2.0 * np.pi * C_LIGHT / LAMBDA0

#: The reference instrument (as tests/test_collection.py): 1 mm mask holes at 1 mm
#: edge-to-edge gap, f = 100 mm, collected through a 0.5 mm tanh-apodised hole at
#: the signal corner (-d, -d) of a mask plane 100 mm from the focus; SiO2 slab.
MASK_D = 1.0e-3
MASK_GAP = 1.0e-3
F_FOC = 0.1
ARM_OFFSET = 0.5 * (MASK_GAP + MASK_D)
Z_MASK = 0.1
GEOM = dict(
    hole_diameter=MASK_D, hole_spacing=MASK_GAP, f_foc=F_FOC, wavelength=LAMBDA0
)
HOLE = dict(
    hole_x=-ARM_OFFSET,
    hole_y=-ARM_OFFSET,
    hole_diameter=0.5e-3,
    z_mask=Z_MASK,
    apod="tanh",
    apod_param=9.687034277198214e-05,
)


def _mixture(mode="integrated", n_radial=8, n_azimuth=6):
    """A small collected mixture at the reference geometry."""
    aperture = mask_hole_aperture(mode=mode, **HOLE)
    return focal_mixture(
        n_radial=n_radial,
        n_azimuth=n_azimuth,
        r_max_units=3.0,
        collection=aperture,
        **GEOM,
    )


@pytest.fixture
def setup():
    """A 1 fs DUV pulse, a short delay axis and the 40 um slab configuration."""
    g = Grid(128, dt=0.5e-15)
    ew = gaussian_pulse(g, 1.0e-15)
    delays = np.linspace(-8e-15, 8e-15, 9)
    kwargs = dict(material="SiO2", thickness=40e-6, npoints=6, omega0=OMEGA0)
    return g, ew, delays, kwargs


def _trace(fn, ew, thickness):
    return np.asarray(fn(jnp.asarray(ew), thickness, 0.0), dtype=float)


def test_flag_off_is_bit_identical(setup):
    """``depth_transverse=False`` takes the standard path, byte for byte."""
    g, ew, delays, kw = setup
    mix = _mixture()
    default = make_param_trace_fn(g.omega, delays, "pg", focal=mix, **kw)
    off = make_param_trace_fn(
        g.omega, delays, "pg", focal=mix, depth_transverse=False, **kw
    )
    a = _trace(default, ew, kw["thickness"])
    b = _trace(off, ew, kw["thickness"])
    assert np.array_equal(a, b)


def test_zero_rate_reproduces_the_standard_model(setup, monkeypatch):
    """With the dephasing rate forced to zero the flag adds nothing but a resum.

    Pins that the alternate path differs from the standard one ONLY by the
    phase: with g = 0 the two agree to machine precision (the scan reorders the
    floating-point sums, so bit identity is not expected), and with the real
    phase they measurably differ.
    """
    g, ew, delays, kw = setup
    mix = _mixture()
    base = make_param_trace_fn(g.omega, delays, "pg", focal=mix, **kw)
    a = _trace(base, ew, kw["thickness"])

    real_rate = forward_jax._depth_phase_rate
    monkeypatch.setattr(
        forward_jax,
        "_depth_phase_rate",
        lambda *args: np.zeros_like(real_rate(*args)),
    )
    zeroed = make_param_trace_fn(
        g.omega, delays, "pg", focal=mix, depth_transverse=True, **kw
    )
    b = _trace(zeroed, ew, kw["thickness"])
    np.testing.assert_allclose(b, a, rtol=0.0, atol=1e-13 * a.max())

    monkeypatch.setattr(forward_jax, "_depth_phase_rate", real_rate)
    on = make_param_trace_fn(
        g.omega, delays, "pg", focal=mix, depth_transverse=True, **kw
    )
    c = _trace(on, ew, kw["thickness"])
    assert not np.allclose(c, a, rtol=0.0, atol=1e-6 * a.max())


@pytest.mark.parametrize("mode", ["integrated", "reimaged"])
def test_single_depth_node_parity_split(setup, mode):
    """With Q = 1 the phase is a per-(node, frequency) unit-modulus constant.

    ``"integrated"`` takes ``|S_j|^2`` per aperture node, which kills a
    unit-modulus factor — the trace must be unchanged. ``"reimaged"`` sums the
    nodes coherently first, so the same factor changes the trace. Together the
    two pin where the phase sits relative to the modulus.
    """
    g, ew, delays, kw = setup
    kw = {**kw, "npoints": 1}
    mix = _mixture(mode=mode)
    off = make_param_trace_fn(g.omega, delays, "pg", focal=mix, **kw)
    on = make_param_trace_fn(
        g.omega, delays, "pg", focal=mix, depth_transverse=True, **kw
    )
    a = _trace(off, ew, kw["thickness"])
    b = _trace(on, ew, kw["thickness"])
    if mode == "integrated":
        np.testing.assert_allclose(b, a, rtol=0.0, atol=1e-13 * a.max())
    else:
        assert not np.allclose(b, a, rtol=0.0, atol=1e-7 * a.max())


@pytest.mark.parametrize("mode", ["integrated", "reimaged"])
def test_magnitude_at_the_reference_geometry(setup, mode):
    """The 40 um effect is real but small — a loose regression bracket.

    Order of the collection over-correction scale: the rms trace change must be
    neither zero (the phase reaching nothing) nor order unity (a sign/units
    blunder upstream of the dense-FFT reference test).
    """
    g, ew, delays, kw = setup
    mix = _mixture(mode=mode)
    off = make_param_trace_fn(g.omega, delays, "pg", focal=mix, **kw)
    on = make_param_trace_fn(
        g.omega, delays, "pg", focal=mix, depth_transverse=True, **kw
    )
    a = _trace(off, ew, kw["thickness"])
    b = _trace(on, ew, kw["thickness"])
    change = np.sqrt(np.mean((b - a) ** 2)) / a.max()
    assert 1e-5 < change < 5e-2


def test_full_k_grid_is_the_full_collection_route(setup):
    """Full collection with decoherence = a window much wider than the signal.

    There is deliberately no separate full-collection path: an ``"integrated"``
    aperture covering the whole reciprocal grid IS full collection (Parseval —
    with the flag off it reproduces ``collection=None`` exactly), and turning
    the flag on then gives the decohered full-collection model, which differs.
    """
    g, ew, delays, kw = setup
    omega, thickness = g.omega, kw["thickness"]
    n_grid, dr = 24, 6.0e-6
    axis = (np.arange(n_grid) - n_grid // 2) * dr
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    common = dict(
        x=gx.ravel(),
        y=gy.ravel(),
        area=np.full(gx.size, dr**2),
        arms=np.array(_ARMS_PG) * ARM_OFFSET,
        hole_diameter=MASK_D,
        f_foc=F_FOC,
    )
    kax = np.fft.fftfreq(n_grid, d=dr) * 2.0 * np.pi
    kx, ky = np.meshgrid(kax, kax, indexing="ij")
    full_k = CollectionAperture(
        x=kx.ravel(),
        y=ky.ravel(),
        weight=np.full(kx.size, (kax[1] - kax[0]) ** 2),
        transmission=np.ones(kx.size),
        z_mask=Z_MASK,
        chromatic=False,
    )
    incoherent = make_param_trace_fn(
        omega, delays, "pg", focal=mixture_from_nodes(**common), **kw
    )
    windowed = mixture_from_nodes(collection=full_k, **common)
    off = make_param_trace_fn(omega, delays, "pg", focal=windowed, **kw)
    on = make_param_trace_fn(
        omega, delays, "pg", focal=windowed, depth_transverse=True, **kw
    )
    t_none = _trace(incoherent, ew, thickness)
    t_off = _trace(off, ew, thickness)
    t_on = _trace(on, ew, thickness)
    assert t_off == pytest.approx(t_none, rel=1e-10)
    assert not np.allclose(t_on, t_off, rtol=0.0, atol=1e-6 * t_off.max())


def test_gradients_are_finite(setup):
    """The decohered trace stays usable inside AD — the point of the JAX model."""
    g, ew, delays, kw = setup
    mix = _mixture()
    fn = make_param_trace_fn(
        g.omega, delays, "pg", focal=mix, depth_transverse=True, **kw
    )

    def loss(scale):
        return jnp.sum(fn(jnp.asarray(ew) * scale, kw["thickness"], 0.0))

    grad = float(jax.grad(loss)(1.0))
    assert np.isfinite(grad)
    # Homogeneous of degree 6 in the field, so d/ds at s = 1 is 6 T.
    assert grad == pytest.approx(6.0 * float(loss(1.0)), rel=1e-6)

    fit = make_param_trace_fn(
        g.omega,
        delays,
        "pg",
        focal=mix,
        depth_transverse=True,
        fit_thickness=True,
        **kw,
    )

    def loss_l(thk):
        return jnp.sum(fit(jnp.asarray(ew), thk, 0.0))

    grad_l = float(jax.grad(loss_l)(kw["thickness"]))
    assert np.isfinite(grad_l) and grad_l != 0.0
    h = 1e-3 * kw["thickness"]
    fd = (float(loss_l(kw["thickness"] + h)) - float(loss_l(kw["thickness"] - h))) / (
        2.0 * h
    )
    assert grad_l == pytest.approx(fd, rel=1e-4)


def test_depth_transverse_validation(setup):
    g, ew, delays, kw = setup
    with pytest.raises(ValueError, match="collection"):
        make_param_trace_fn(g.omega, delays, "pg", depth_transverse=True, **kw)
    no_collection = focal_mixture(n_radial=8, n_azimuth=6, **GEOM)
    with pytest.raises(ValueError, match="collection"):
        make_param_trace_fn(
            g.omega, delays, "pg", focal=no_collection, depth_transverse=True, **kw
        )
    with pytest.raises(ValueError, match="dispersive slab"):
        make_param_trace_fn(
            g.omega,
            delays,
            "pg",
            focal=_mixture(),
            depth_transverse=True,
            omega0=OMEGA0,
        )


def test_solver_plumbing():
    """`lbfgs-ad` advertises the flag; the pipeline refuses solvers that cannot."""
    from croak.maths import wlfreq
    from croak.pipeline import retrieve_from_tracedata
    from croak.preprocess import TraceData
    from croak.retrieve import algorithm_params

    assert "depth_transverse" in algorithm_params("lbfgs-ad")
    assert "depth_transverse" not in algorithm_params("copra")

    g = Grid(64, dt=0.5e-15)
    delays = np.linspace(-5e-15, 5e-15, 5)
    trace = np.ones((g.omega.size, delays.size))
    td = TraceData(
        interaction="pg",
        grid=g,
        delays=delays,
        trace=trace,
        omega0_pulse=OMEGA0,
        omega0_trace=OMEGA0,
        Iomega=None,
        lam_min=200e-9,
        lam_max=320e-9,
        lamm_lims=None,
        tau_lims=None,
        lam_frog=wlfreq(g.omega + OMEGA0),
        tau_meas=delays,
        Ifrog_meas=trace.T,
        tau_filt=delays,
        Ifrog_filt=trace.T,
        lam_spec=None,
        Ilam_spec_meas=None,
        Ilam_spec_filt=None,
        scalecurve=np.ones(g.n),
    )
    with pytest.raises(ValueError, match="transverse decoherence"):
        retrieve_from_tracedata(td, algorithm="copra", depth_transverse=True)
