"""SHG in a Type I crystal through the forward models.

The reference throughout is the exact double sum over component pairs: the signal at
grid bin ``k`` is ``sum_j E(j) E_tau(k - j) * amp(k) * H_jk`` with the depth integral
``H_jk = exp(i beta_sig(k) P) * int_0^P exp(i a_jk z) dz`` done analytically
(``a_jk = beta_in(j) + beta_in(k - j) - beta_sig(k)``, ``P`` the path). The models
evaluate the same sum by FFTs and Gauss-Legendre quadrature, so they must agree to the
quadrature's accuracy (:func:`croak.crystals.depth_points`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.fft
from test_crystals import BBO, W0

from croak import crystals
from croak.constants import C
from croak.crystals import CrystalApproximationWarning, CrystalSetup
from croak.forward import ForwardModel
from croak.forward_jax import (
    make_forward_adjoint_fns,
    make_param_trace_fn,
    make_trace_fn,
)
from croak.geometry import TwoBeamGeometry
from croak.grid import Grid

GRID = Grid(64, dt=3.0e-15)  # about 560-1440 nm around 800 nm
GEOMETRY = TwoBeamGeometry(beam_separation=5e-3, focal_length=0.1)  # 1.43 degrees


def _pulse(g, fwhm=12e-15, gdd=30e-30):
    """A chirped Gaussian at the carrier, peak-normalised spectrum."""
    et = np.exp(-0.5 * (g.t / (fwhm / 2.3548)) ** 2).astype(complex)
    ew = g.fft(et) * np.exp(0.5j * gdd * g.omega**2)
    return ew / np.max(np.abs(ew))


def _setup(**kw) -> CrystalSetup:
    return CrystalSetup(BBO, math.radians(29.2), phase_match_wavelength=800e-9, **kw)


def _model(setup, thickness, delays, **kw):
    npoints = kw.pop("npoints", crystals.depth_points(setup, GRID.omega, W0, thickness))
    return ForwardModel(
        GRID.omega,
        delays,
        "shg",
        material=setup,
        thickness=thickness,
        npoints=npoints,
        omega0=W0,
        **kw,
    )


def _depth_integral(a, path):
    """int_0^path exp(i a z) dz, exactly."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (np.exp(1j * a * path) - 1.0) / (1j * a)
    return np.where(np.abs(a * path) < 1e-12, path, out)


def _pair_sum(setup, thickness, ew, tau, beta_sig_pair=None):
    """Exact double sum over pairs, centred order; ``beta_sig_pair(j, m, k)`` may
    replace the signal's propagation constant per pair."""
    prop = crystals.propagation(setup, GRID.omega, W0, thickness)
    path = prop.path_length
    n = GRID.n
    w = scipy.fft.ifftshift(GRID.omega)
    b_in = scipy.fft.ifftshift(prop.beta_in)
    b_sig = scipy.fft.ifftshift(prop.beta_sig)
    amp = scipy.fft.ifftshift(prop.sig_amp)
    e = scipy.fft.ifftshift(ew)
    e_tau = e * np.exp(1j * w * tau)
    j = np.arange(n)
    psi = np.zeros(n, dtype=complex)
    for k in range(n):
        m = (k - j) % n
        # the model: the signal along the bisector, plus the arms' delay drift
        model = b_sig[k] + prop.arm_shift * (w[j] - w[m])
        bs = model if beta_sig_pair is None else beta_sig_pair(j, m, k)
        a = b_in[j] + b_in[m] - bs
        h = np.exp(1j * bs * path) * _depth_integral(a, path)
        psi[k] = amp[k] * np.sum(e[j] * e_tau[m] * h)
    return scipy.fft.fftshift(psi)


# -- agreement with the exact pair sum ------------------------------------------
# The geometries: out of the principal plane, and in it with the arms' delay drift.
GEOMETRIES = {
    "parallel": dict(geometry=GEOMETRY, polarization="parallel"),
    "principal": dict(
        geometry=GEOMETRY, polarization="perpendicular", delayed_beam="toward_axis"
    ),
}


@pytest.mark.parametrize("plane", GEOMETRIES)
@pytest.mark.parametrize("thickness", [10e-6, 100e-6])
@pytest.mark.parametrize("tau", [-15e-15, 0.0, 8e-15])
def test_signal_matches_the_exact_pair_sum(thickness, tau, plane):
    setup = _setup(**GEOMETRIES[plane])
    ew = _pulse(GRID)
    model = _model(setup, thickness, np.array([tau]))
    psi = model.signal_single(ew, tau)
    exact = _pair_sum(setup, thickness, ew, tau)
    # depth_points promises 1e-6 of the path per pair; summed over at most 64 pairs
    # of unit-normalised amplitudes, against a peak of order 10 path lengths.
    np.testing.assert_allclose(psi, exact, atol=1e-5 * np.max(np.abs(exact)))


def test_too_few_depth_points_are_refused():
    setup = _setup()
    needed = crystals.depth_points(setup, GRID.omega, W0, 100e-6)
    with pytest.raises(ValueError, match=f"npoints >= {needed}"):
        _model(setup, 100e-6, np.array([0.0]), npoints=needed - 1)


def test_frequency_marginal_is_the_sum_of_pair_sinc_squares():
    # Over a full period of delays (the grid's own time axis), Parseval turns the
    # delay sum of |psi|^2 into N * sum_j |E_j E_m amp H_jk|^2 exactly: the
    # pair-resolved phase-matching response sinc^2(a P / 2) (P^2 sinc^2 = |H|^2).
    setup = _setup()
    thickness = 50e-6
    ew = _pulse(GRID)
    model = _model(setup, thickness, GRID.t)
    marginal = model.trace(ew, normalize=False).sum(axis=1)
    prop = crystals.propagation(setup, GRID.omega, W0, thickness)
    path = prop.path_length
    e = scipy.fft.ifftshift(ew)
    b_in = scipy.fft.ifftshift(prop.beta_in)
    b_sig = scipy.fft.ifftshift(prop.beta_sig)
    amp = scipy.fft.ifftshift(prop.sig_amp)
    j = np.arange(GRID.n)
    expected = np.zeros(GRID.n)
    for k in range(GRID.n):
        m = (k - j) % GRID.n
        a = b_in[j] + b_in[m] - b_sig[k]
        sinc = np.sinc(a * path / (2 * np.pi))  # np.sinc(x) = sin(pi x) / (pi x)
        expected[k] = (
            GRID.n * amp[k] ** 2 * np.sum(np.abs(e[j] * e[m]) ** 2 * (path * sinc) ** 2)
        )
    expected = scipy.fft.fftshift(expected)
    # quadrature accuracy as above, squared into intensity
    np.testing.assert_allclose(marginal, expected, rtol=0, atol=1e-5 * expected.max())


def test_zero_thickness_is_the_ideal_trace_times_the_prefactor():
    setup = _setup()
    ew = _pulse(GRID)
    delays = np.linspace(-30e-15, 30e-15, 11)
    crystal = _model(setup, 0.0, delays).trace(ew, normalize=False)
    ideal = ForwardModel(GRID.omega, delays, "shg").trace(ew, normalize=False)
    amp = crystals.propagation(setup, GRID.omega, W0, 0.0).sig_amp
    np.testing.assert_allclose(crystal, ideal * amp[:, None] ** 2, rtol=1e-12)


def test_phase_mismatch_narrows_the_trace_band():
    # A thick crystal tuned to 800 nm passes less of the wings than a thin one.
    setup = _setup()
    ew = _pulse(GRID, fwhm=6e-15, gdd=0.0)
    delays = np.linspace(-20e-15, 20e-15, 9)

    def width(thickness):
        m = _model(setup, thickness, delays).trace(ew, normalize=False).sum(axis=1)
        m = m / m.sum()
        mean = np.sum(GRID.omega * m)
        return np.sqrt(np.sum((GRID.omega - mean) ** 2 * m))

    assert width(200e-6) < 0.9 * width(1e-6)


# -- the neglected signal tilt ---------------------------------------------------
def _tilted_signal(setup, thickness):
    """Per-pair signal constant including its tilt in the principal plane."""
    prop = crystals.propagation(setup, GRID.omega, W0, thickness)
    theta = setup.internal_angle(W0)
    sin_a = math.sin(setup.crossing_half_angle)
    beta1 = crystals._group_delay(BBO, W0, sin_a)
    w = scipy.fft.ifftshift(GRID.omega)
    del prop

    def beta_sig_pair(j, m, k):
        w1, w2 = w[j] + W0, w[m] + W0
        ws = w[k] + 2 * W0
        lam = 2 * np.pi * C / ws
        n_o, n_e = BBO.n_o(np.array([lam])), BBO.n_e(np.array([lam]))
        k0 = ws / C
        sign = 1.0 if setup.delayed_beam == "toward_axis" else -1.0
        k_x = sign * (w1 - w2) * sin_a / C

        # Self-consistent direction psi: k0 n_e(theta + psi) sin(psi) = k_x, solved
        # by bisection over +-10 mrad (independently of the library's Newton).
        def f(p):
            return (
                k0 * crystals.extraordinary_index(n_o, n_e, theta + p) * np.sin(p) - k_x
            )

        lo, hi = np.full_like(k_x, -1e-2), np.full_like(k_x, 1e-2)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = f(mid) < 0
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        psi = 0.5 * (lo + hi)
        n_psi = crystals.extraordinary_index(n_o, n_e, theta + psi)
        return k0 * n_psi * np.cos(psi) - beta1 * w[k]

    return beta_sig_pair


DELAYS = np.linspace(-24e-15, 24e-15, 9)
PRINCIPAL = dict(geometry=GEOMETRY, polarization="perpendicular")


def _worst(setup, thickness, amp, beta_sig_pair):
    """Largest |model - reference| over DELAYS, relative to the reference's peak."""
    reference = _tilted_signal(setup, thickness)
    pairs = [
        (
            _pair_sum(setup, thickness, amp, t, beta_sig_pair),
            _pair_sum(setup, thickness, amp, t, reference),
        )
        for t in DELAYS
    ]
    worst = max(np.max(np.abs(a - b)) for a, b in pairs)
    return worst / max(np.max(np.abs(b)) for _, b in pairs)


@pytest.mark.parametrize("side", ["toward_axis", "away_from_axis"])
def test_arm_drift_captures_the_signal_tilt(side):
    # 12 fs in 100 um crossing at 1.43 degrees in the principal plane. Against each
    # pair's signal in its own self-consistent direction: with the arms' drift the
    # model is within 0.1 % (the residual is second order in the tilt); without it,
    # about 5 % off.
    setup = _setup(**PRINCIPAL, delayed_beam=side)
    amp = np.abs(_pulse(GRID, gdd=0.0))
    with_drift = _worst(setup, 100e-6, amp, None)
    prop = crystals.propagation(setup, GRID.omega, W0, 100e-6)
    b_sig = scipy.fft.ifftshift(prop.beta_sig)
    without = _worst(setup, 100e-6, amp, lambda j, m, k: b_sig[k])
    assert with_drift < 1e-3
    assert without > 0.03


def test_the_wrong_side_is_the_delay_mirror():
    # Flipping the side is the same as mirroring the delay axis (swapping the arms
    # of SHG); the two sides differ, so the data can tell. On delays that are whole
    # multiples of the grid step the identity is exact to rounding: off them, pairs
    # whose sum wraps round the grid pick up a phase that differs between tau and
    # -tau, which breaks it at the 5e-5 level.
    delays = GRID.dt * np.arange(-10, 11)
    ew = _pulse(GRID, fwhm=8e-15, gdd=40e-30)
    toward = _model(_setup(**PRINCIPAL, delayed_beam="toward_axis"), 100e-6, delays)
    away = _model(_setup(**PRINCIPAL, delayed_beam="away_from_axis"), 100e-6, delays)
    a = toward.trace(ew, normalize=False)
    b = away.trace(ew, normalize=False)
    np.testing.assert_allclose(a, b[:, ::-1], rtol=0, atol=1e-10 * a.max())
    assert np.max(np.abs(a - b)) > 0.03 * a.max()


def test_tilt_field_error_is_the_pair_sum_difference():
    # The library's residual estimate against this module's own reference.
    setup = _setup(**PRINCIPAL, delayed_beam="toward_axis")
    amp = np.abs(_pulse(GRID, gdd=0.0))
    expected = _worst(setup, 100e-6, amp, None)
    error = crystals.tilt_field_error(setup, GRID.omega, W0, 100e-6, amp, DELAYS)
    # two independent root solves (Newton here, bisection in the test) agree to
    # about 1e-15 rad; the differences they leave are far below the residual
    assert error == pytest.approx(expected, rel=1e-6)


def test_tilt_field_error_vanishes_out_of_the_principal_plane():
    amp = np.abs(_pulse(GRID))
    parallel = _setup(geometry=GEOMETRY, polarization="parallel")
    args = (GRID.omega, W0, 100e-6, amp, DELAYS)
    assert crystals.tilt_field_error(parallel, *args) == 0.0
    assert crystals.tilt_field_error(_setup(), *args) == 0.0


def test_check_tilt_warns_above_one_percent():
    # A 5 fs pulse, 300 um, crossing at 2.86 degrees: the second-order residual
    # reaches about 3 %.
    setup = _setup(
        geometry=TwoBeamGeometry(beam_separation=10e-3, focal_length=0.1),
        polarization="perpendicular",
        delayed_beam="toward_axis",
    )
    amp = np.abs(_pulse(GRID, fwhm=5e-15, gdd=0.0))
    with pytest.warns(CrystalApproximationWarning, match="first order only"):
        error = crystals.check_tilt(setup, GRID.omega, W0, 300e-6, amp, DELAYS)
    assert error > crystals.TILT_ERROR_WARN


def test_check_tilt_is_silent_below_one_percent():
    setup = _setup(**PRINCIPAL, delayed_beam="toward_axis")
    amp = np.abs(_pulse(GRID, gdd=0.0))
    assert crystals.check_tilt(setup, GRID.omega, W0, 100e-6, amp, DELAYS) < 0.01


def test_an_isotropic_slab_cannot_model_shg():
    with pytest.raises(ValueError, match="dispersive propagation in an isotropic"):
        ForwardModel(
            GRID.omega, [0.0], "shg", material="SiO2", thickness=10e-6, omega0=W0
        )


def test_a_crystal_models_shg_only():
    with pytest.raises(ValueError, match="Type I SHG only"):
        ForwardModel(GRID.omega, [0.0], "pg", material=_setup(), omega0=W0)


# -- the JAX models -----------------------------------------------------------------
@pytest.mark.parametrize("plane", GEOMETRIES)
def test_jax_trace_matches_numpy(plane):
    setup = _setup(**GEOMETRIES[plane])
    thickness = 50e-6
    ew = _pulse(GRID)
    delays = np.linspace(-20e-15, 20e-15, 7)
    npoints = crystals.depth_points(setup, GRID.omega, W0, thickness)
    expected = _model(setup, thickness, delays).trace(ew, normalize=False)
    kw = dict(material=setup, thickness=thickness, npoints=npoints, omega0=W0)
    jax_trace = np.asarray(make_trace_fn(GRID.omega, delays, "shg", **kw)(ew))
    param = make_param_trace_fn(GRID.omega, delays, "shg", **kw)
    param_trace = np.asarray(param(ew, thickness, 0.0))
    # both evaluate the same FFTs and quadrature in float64
    np.testing.assert_allclose(jax_trace, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(param_trace, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "option", [dict(fit_thickness=True), dict(depth_transverse=True)]
)
def test_jax_crystal_refuses_slab_only_options(option):
    setup = _setup()
    npoints = crystals.depth_points(setup, GRID.omega, W0, 50e-6)
    with pytest.raises(ValueError, match="does not support"):
        make_param_trace_fn(
            GRID.omega,
            [0.0],
            "shg",
            material=setup,
            thickness=50e-6,
            npoints=npoints,
            omega0=W0,
            **option,
        )


# -- the hand-derived adjoint ---------------------------------------------------
@pytest.mark.parametrize("plane", GEOMETRIES)
def test_adjoint_matches_finite_differences(plane):
    rng = np.random.default_rng(11)
    setup = _setup(**GEOMETRIES[plane])
    thickness = 50e-6
    ew = _pulse(GRID) + 0.05 * (
        rng.standard_normal(GRID.n) + 1j * rng.standard_normal(GRID.n)
    )
    delays = np.array([-8e-15, 0.0, 8e-15])
    model = _model(setup, thickness, delays)
    weights = rng.standard_normal((GRID.n, 3)) + 1j * rng.standard_normal((GRID.n, 3))

    ew_bar = np.zeros(GRID.n, dtype=complex)
    for col, tau in enumerate(delays):
        model.signal_single(ew, tau, record=True)
        ew_bar += model.adjoint_single(0.5 * weights[:, col], tau)

    def loss(z):
        return sum(
            float(np.real(np.sum(np.conj(weights[:, c]) * model.signal_single(z, t))))
            for c, t in enumerate(delays)
        )

    # central differences on a quadratic form: exact up to rounding (~1e-9 at h=1e-6
    # against losses of order 1e2)
    h = 1e-6
    for idx in rng.choice(GRID.n, size=6, replace=False):
        step = np.zeros(GRID.n, dtype=complex)
        step[idx] = h
        d_re = (loss(ew + step) - loss(ew - step)) / (2 * h)
        d_im = (loss(ew + 1j * step) - loss(ew - 1j * step)) / (2 * h)
        assert 2 * ew_bar[idx].real == pytest.approx(d_re, rel=1e-6, abs=1e-6)
        assert 2 * ew_bar[idx].imag == pytest.approx(d_im, rel=1e-6, abs=1e-6)


def test_a_tilted_crystal_integrates_over_its_effective_path():
    # A 23.2 degree cut tilted by 10 degrees: the path differs from the plate by
    # 1.3 %, a few tenths of a radian of the mismatch phase, so quadrature over the
    # plate instead would miss the pair sum.
    setup = CrystalSetup(BBO, math.radians(23.2), tilt=math.radians(10.0))
    thickness = 100e-6
    assert abs(setup.path_length(thickness, W0) / thickness - 1) > 0.01
    ew = _pulse(GRID)
    model = _model(setup, thickness, np.array([5e-15]))
    exact = _pair_sum(setup, thickness, ew, 5e-15)
    np.testing.assert_allclose(
        model.signal_single(ew, 5e-15), exact, atol=1e-5 * np.max(np.abs(exact))
    )


def test_tilt_field_error_skips_absorbed_signal_frequencies():
    # Signal bins outside the transparency range carry no signal in either sum.
    narrow = crystals.UniaxialCrystal("BBO-like", BBO.n_o, BBO.n_e, (380e-9, 1500e-9))
    setup = CrystalSetup(
        narrow,
        math.radians(29.2),
        phase_match_wavelength=800e-9,
        geometry=GEOMETRY,
        polarization="perpendicular",
        delayed_beam="toward_axis",
    )
    amp = np.abs(_pulse(GRID, gdd=0.0))
    full = crystals.tilt_field_error(
        _setup(
            geometry=GEOMETRY, polarization="perpendicular", delayed_beam="toward_axis"
        ),
        GRID.omega,
        W0,
        100e-6,
        amp,
        DELAYS,
    )
    error = crystals.tilt_field_error(setup, GRID.omega, W0, 100e-6, amp, DELAYS)
    # the absorbed bins lie in the far wings, so the error barely moves
    assert error == pytest.approx(full, rel=0.2)


@pytest.mark.parametrize("plane", GEOMETRIES)
def test_jax_hand_adjoint_matches_numpy(plane):
    # The JAX twin of the hand adjoint, against the NumPy one (which the finite
    # differences above pin), with and without the arms' delay drift.
    rng = np.random.default_rng(5)
    setup = _setup(**GEOMETRIES[plane])
    thickness = 50e-6
    npoints = crystals.depth_points(setup, GRID.omega, W0, thickness)
    ew = _pulse(GRID)
    psi_bar = rng.standard_normal(GRID.n) + 1j * rng.standard_normal(GRID.n)
    tau = 6e-15
    model = _model(setup, thickness, np.array([tau]))
    model.signal_single(ew, tau, record=True)
    expected = model.adjoint_single(psi_bar, tau)
    signal_tape, adjoint = make_forward_adjoint_fns(
        GRID.omega,
        [tau],
        "shg",
        material=setup,
        thickness=thickness,
        npoints=npoints,
        omega0=W0,
    )
    _, test, gate = signal_tape(ew, tau)
    got = np.asarray(adjoint(psi_bar, test, gate, tau))
    # the same FFTs and products in float64
    np.testing.assert_allclose(
        got, expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max()
    )
