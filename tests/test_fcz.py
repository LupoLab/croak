"""fc-z: the depth-resolved chromatic focal mixture, tested in the brief's order.

1. Reduction: a zero-evolution table reproduces the entrance-face fc trace
   bit for bit (the difference-form build makes the z->0 table literally the
   closed-form entrance filter).
2. Propagator truth: the Hankel-evolved profile against an independent
   plane-wave-superposition propagator (trapezoid azimuth -- no shared Bessel
   code) at 1e-6 of peak, plus a 2-D FFT orientation/sign smoke check.
3. Moment pinning: the mixture's evolved |A1 A2 A3|^2-weighted moments
   reproduce the R19 beamlet-companion table (geometry wiring: walk sign,
   arm order, index convention) before any physics claim.
4. Gaussian null: on the Gaussian-control geometry (z_R >> slab) evolution
   changes the trace negligibly -- matching the measured absence of the
   arrival advance there.
5. Differentiability and guards.
"""

from __future__ import annotations

import numpy as np
import pytest

import croak.forward_jax as fj
from croak import materials
from croak.focal import (
    evolved_profile_table,
    focal_mixture,
)
from croak.forward_jax import make_param_trace_fn

C_LIGHT = 299_792_458.0
LAMBDA0 = 260e-9
OMEGA0 = 2.0 * np.pi * C_LIGHT / LAMBDA0
GEOM = dict(hole_diameter=1.0e-3, hole_spacing=1.0e-3, f_foc=0.1, wavelength=LAMBDA0)
SLAB = dict(material="SiO2", thickness=40e-6, npoints=6)


def _grid(n=48, dt=0.5e-15, n_delay=3, span=6e-15):
    omega = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, d=dt))
    delays = np.linspace(-span, span, n_delay)
    t = (np.arange(n) - n // 2) * dt
    ew = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(np.exp(-(t**2) / (2e-15) ** 2))))
    return omega, delays, ew.astype(complex)


def _mixture(evolve: bool, profile="airy", waist=None, **kw):
    args = dict(
        GEOM,
        n_radial=12,
        n_azimuth=8,
        r_max_units=4.0,
        profile=profile,
        waist=waist,
        **kw,
    )
    if evolve:
        args.update(evolve_profiles=True, **SLAB)
    return focal_mixture(**args)


def _trace(mix, omega, delays, ew):
    fn = make_param_trace_fn(
        omega, delays, "pg", omega0=OMEGA0, focal=mix, normalize=False, **SLAB
    )
    return np.asarray(fn(ew, SLAB["thickness"], 0.0), float)


# --- 1. reduction ---------------------------------------------------------


def test_zero_evolution_reduces_to_fc(monkeypatch):
    """The z->0 table IS the entrance filter; the trace matches at machine eps.

    The reduction is asserted at two levels. The filter TABLE is bit-identical
    to the entrance-face filter (the difference-form build cancels its own
    quadrature exactly at zero phase, and the zero-walk radii are written
    exactly). The TRACE is machine-precision equal rather than bit-equal: the
    evolved path materialises (N, Q) per-arm spectra where the entrance path
    broadcasts an (N, 1) shared one, and XLA fuses the two graphs differently
    at the last ulp even though every input bit agrees.
    """
    omega, delays, ew = _grid()
    mix0, mixz = _mixture(False), _mixture(True)
    zero_table = evolved_profile_table(mixz, omega, OMEGA0, np.zeros(SLAB["npoints"]))
    assert np.array_equal(zero_table.imag, np.zeros_like(zero_table.imag))
    entrance = mix0.spectral_filter(omega + OMEGA0)
    for arm in range(3):
        for q in range(SLAB["npoints"]):
            assert np.array_equal(zero_table[arm, :, q, :].real, entrance)
    base = _trace(mix0, omega, delays, ew)
    monkeypatch.setattr(
        fj,
        "evolved_arm_filters",
        lambda mix, om, om0: evolved_profile_table(
            mix, om, om0, np.zeros(mix.evolve.npoints)
        ),
    )
    frozen = _trace(mixz, omega, delays, ew)
    np.testing.assert_allclose(frozen, base, rtol=0.0, atol=1e-14 * base.max())
    # and the real evolution is not a no-op (the two-assert shape)
    monkeypatch.undo()
    evolved = _trace(mixz, omega, delays, ew)
    assert not np.allclose(evolved, base, rtol=0.0, atol=1e-12 * base.max())


# --- 2. propagator truth --------------------------------------------------


def _reference_profile(rho, omega_abs, z, n0, nk=1200, ntheta=512):
    """Plane-wave superposition over the disc: no Bessel functions anywhere."""
    k_edge = omega_abs * (0.5 * GEOM["hole_diameter"]) / (C_LIGHT * GEOM["f_foc"])
    x, w = np.polynomial.legendre.leggauss(nk)
    k = 0.5 * k_edge * (x + 1.0)
    wk = 0.5 * k_edge * w * k
    kz0 = n0 * omega_abs / C_LIGHT
    dkz = -(k**2) / (kz0 + np.sqrt(kz0**2 - k**2))
    th = 2.0 * np.pi * np.arange(ntheta) / ntheta
    # mean over azimuth of exp(i k rho cos theta): J0 from first principles
    ring = np.exp(1j * np.outer(np.outer(k, rho).ravel(), np.cos(th))).mean(axis=1)
    j0kr = ring.reshape(k.size, rho.size).real
    return (wk * np.exp(1j * dkz * z)) @ j0kr / wk.sum()


def test_hankel_matches_planewave_propagator():
    mix = _mixture(True)
    n0 = float(materials.refractive_index("SiO2")(LAMBDA0))
    z = 40e-6
    for lam in (200e-9, 260e-9, 350e-9):
        om_abs = 2.0 * np.pi * C_LIGHT / lam
        table = evolved_profile_table(
            mix, np.array([om_abs - OMEGA0]), OMEGA0, np.array([z])
        )
        walk = -np.asarray(mix.arms) / (mix.f_foc * n0)
        rx, ry = mix.r * np.cos(mix.phi), mix.r * np.sin(mix.phi)
        for j in range(3):
            rho = np.hypot(rx - z * walk[j, 0], ry - z * walk[j, 1])
            ref = _reference_profile(rho, om_abs, z, n0)
            assert np.max(np.abs(table[j, :, 0, 0] - ref)) < 1e-6


def test_fft2_smoke_pins_sign_and_orientation():
    """A genuinely 2-D FFT propagation agrees at its grid-limited accuracy.

    The jagged edge of a discretely sampled top-hat caps an FFT reference at
    the few-1e-3 level, which is why the 1e-6 propagator-truth test above
    uses the continuum plane-wave quadrature instead; this cross-check pins
    the overall sign and orientation through an independent 2-D transform.
    """
    from scipy.special import j0

    om_abs = OMEGA0
    n0 = float(materials.refractive_index("SiO2")(LAMBDA0))
    z = 40e-6
    k_edge = om_abs * (0.5 * GEOM["hole_diameter"]) / (C_LIGHT * GEOM["f_foc"])
    n = 2048
    kax = np.linspace(-1.5 * k_edge, 1.5 * k_edge, n, endpoint=False)
    kx, ky = np.meshgrid(kax, kax, indexing="ij")
    kk = np.hypot(kx, ky)
    disc = (kk <= k_edge).astype(float)
    kz0 = n0 * om_abs / C_LIGHT
    dkz = -(kk**2) / (kz0 + np.sqrt(np.maximum(kz0**2 - kk**2, 0.0)))
    field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(disc * np.exp(1j * dkz * z))))
    dx = 2.0 * np.pi / (kax[1] - kax[0]) / n
    rho = np.arange(40) * dx
    cut = field[n // 2, n // 2 : n // 2 + 40]
    cut = cut / cut[0]
    x_gl, w_gl = np.polynomial.legendre.leggauss(256)
    k = 0.5 * k_edge * (x_gl + 1.0)
    wk = 0.5 * k_edge * w_gl * k
    dkz1 = -(k**2) / (kz0 + np.sqrt(kz0**2 - k**2))
    mine = (wk * np.exp(1j * dkz1 * z)) @ j0(np.outer(k, rho)) / wk.sum()
    assert np.max(np.abs(mine / mine[0] - cut)) < 5e-3


# --- 3. moment pinning against the R19 companion --------------------------

#: Measured by the R19 beamlet analytic companion on the F66 profile
#: (FROG campaign, diagnostics_2026-08-31/r19_companion_log.txt, channel A
#: table at z = 40 um). Geometry-dominated numbers: the walked-product
#: centroid and its (theta, p) projections are profile-shape independent to
#: O(walk^3); only the overlap decay 1-g carries the (Airy here vs measured
#: there) profile width, hence its looser tolerance.
R19_G_RATIO_40 = 0.99850
R19_CENTROID_40 = -88.7e-9  # each of (x, y): the signal diagonal
R19_THETA_40 = +0.00296e-15
R19_P_40 = -0.00592e-15


def test_moments_reproduce_the_r19_companion_table():
    mix = focal_mixture(
        **GEOM, n_radial=32, n_azimuth=24, r_max_units=6.0, evolve_profiles=True, **SLAB
    )
    a = evolved_profile_table(mix, np.array([0.0]), OMEGA0, np.array([0.0, 40e-6]))
    w = mix.area[None, :] * np.abs(a[0, :, :, 0] * a[1, :, :, 0] * a[2, :, :, 0]).T ** 2
    g0, g1 = w.sum(axis=1)
    rx, ry = mix.r * np.cos(mix.phi), mix.r * np.sin(mix.phi)
    cx = float((w[1] * rx).sum() / g1)
    cy = float((w[1] * ry).sum() / g1)
    th = float((w[1] * mix.theta).sum() / g1)
    pp = float((w[1] * mix.p).sum() / g1)
    assert g1 / g0 == pytest.approx(R19_G_RATIO_40, abs=0.3 * (1.0 - R19_G_RATIO_40))
    assert cx == pytest.approx(R19_CENTROID_40, rel=0.05)
    assert cy == pytest.approx(R19_CENTROID_40, rel=0.05)
    assert th == pytest.approx(R19_THETA_40, rel=0.05)
    assert pp == pytest.approx(R19_P_40, rel=0.05)
    # exact geometric identity on the signal diagonal: <p> = -2 <theta>
    assert pp / th == pytest.approx(-2.0, rel=1e-3)


# --- 4. Gaussian null ------------------------------------------------------


def test_gaussian_control_is_a_null():
    """z_R ~ 3.3 mm >> 40 um: evolution must be negligible, as measured (R19)."""
    omega, delays, ew = _grid()
    base = _trace(_mixture(False, profile="gaussian", waist=13.5e-6), omega, delays, ew)
    evolved = _trace(
        _mixture(True, profile="gaussian", waist=13.5e-6), omega, delays, ew
    )
    change = np.sqrt(np.mean((evolved - base) ** 2)) / base.max()
    assert change < 1e-3
    assert change > 0.0


# --- 5. differentiability and guards ---------------------------------------


def test_fcz_is_differentiable():
    import jax
    import jax.numpy as jnp

    omega, delays, ew = _grid(n=32, n_delay=2)
    fn = make_param_trace_fn(
        omega, delays, "pg", omega0=OMEGA0, focal=_mixture(True), **SLAB
    )
    g = jax.grad(lambda e: jnp.sum(fn(e, SLAB["thickness"], 0.0)))(jnp.asarray(ew))
    assert np.all(np.isfinite(np.asarray(g)))


def test_fcz_guards():
    omega, delays, ew = _grid(n=32, n_delay=2)
    with pytest.raises(ValueError, match="needs the slab"):
        focal_mixture(**GEOM, evolve_profiles=True)
    mix = _mixture(True)
    with pytest.raises(ValueError, match="must match the trace map"):
        make_param_trace_fn(
            omega,
            delays,
            "pg",
            omega0=OMEGA0,
            focal=mix,
            material="SiO2",
            thickness=20e-6,
            npoints=SLAB["npoints"],
        )
    with pytest.raises(ValueError, match="fit_thickness"):
        make_param_trace_fn(
            omega,
            delays,
            "pg",
            omega0=OMEGA0,
            focal=mix,
            fit_thickness=True,
            **SLAB,
        )
    with pytest.raises(ValueError, match="PG"):
        make_param_trace_fn(omega, delays, "sd", omega0=OMEGA0, focal=mix, **SLAB)


def test_fcz_quadrature_convergence():
    """Evolved profiles oscillate more than the entrance Airy: check the grid."""
    omega, delays, ew = _grid()
    coarse = focal_mixture(
        **GEOM, n_radial=24, n_azimuth=16, r_max_units=6.0, evolve_profiles=True, **SLAB
    )
    dense = focal_mixture(
        **GEOM, n_radial=32, n_azimuth=24, r_max_units=6.0, evolve_profiles=True, **SLAB
    )
    a = _trace(coarse, omega, delays, ew)
    b = _trace(dense, omega, delays, ew)
    assert np.sqrt(np.mean((a / a.max() - b / b.max()) ** 2)) < 1e-3
