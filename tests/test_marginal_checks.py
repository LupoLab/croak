"""Tests for :mod:`croak.marginal_checks`.

The predicted-marginal physics rests on two convolution identities that hold for
any pulse: marginal centroids and variances are the sums of the factor centroids
and variances. These tests assert the resulting hard anchors per geometry
(SHG: ``2ω̄_S`` / ``√2 σ_S``; PG/TG: ``ω̄_S`` exact, width ``> σ_S``; SD:
``≈ ω̄_S`` at TL) and that the centroid-matching exponent picker recovers a known
``(λ/µm)^n`` tilt.
"""

import numpy as np
import pytest

from croak import marginal_checks as mc
from croak.grid import Grid
from croak.maths import wlfreq


@pytest.fixture
def gaussian_spectrum():
    """A Gaussian fundamental spectrum on an absolute angular-frequency axis.

    Returns ``(omega, S, omega0, sigma_omega)`` for an ~800 nm carrier on a
    well-sampled grid wide enough that the doubled SHG autoconvolution does not
    run off the (extended) axis.
    """
    omega0 = float(wlfreq(800e-9))
    g = Grid(1024, dt=0.4e-15)
    omega = g.omega + omega0
    sigma_omega = 0.03 * omega0
    s = np.exp(-0.5 * ((omega - omega0) / sigma_omega) ** 2)
    return omega, s, omega0, sigma_omega


def test_centroid_and_rms_match_analytic(gaussian_spectrum):
    omega, s, omega0, sigma_omega = gaussian_spectrum
    np.testing.assert_allclose(mc.centroid(omega, s), omega0, rtol=1e-6)
    np.testing.assert_allclose(mc.rms_width(omega, s), sigma_omega, rtol=1e-3)


def test_shg_marginal_is_autoconvolution(gaussian_spectrum):
    """SHG: centroid doubles and width grows by √2 (both exact, hard checks)."""
    omega, s, omega0, sigma_omega = gaussian_spectrum
    pred = mc.predicted_marginal(omega, s, "shg")
    assert pred.centroid_is_hard and pred.width_is_hard
    assert pred.width_bracket is None
    # analytic anchors
    np.testing.assert_allclose(pred.centroid, 2.0 * omega0, rtol=1e-6)
    np.testing.assert_allclose(pred.rms_width, np.sqrt(2.0) * sigma_omega, rtol=1e-3)
    # the placed, peak-normalised shape carries the same moments
    assert pred.marginal.max() == pytest.approx(1.0)
    np.testing.assert_allclose(mc.centroid(pred.omega, pred.marginal), 2.0 * omega0)
    np.testing.assert_allclose(
        mc.rms_width(pred.omega, pred.marginal),
        np.sqrt(2.0) * sigma_omega,
        rtol=2e-2,
    )


def test_pg_centroid_is_exact_and_width_brackets(gaussian_spectrum):
    """PG: centroid stays at the fundamental (exact); width exceeds σ_S."""
    omega, s, omega0, sigma_omega = gaussian_spectrum
    pred = mc.predicted_marginal(omega, s, "pg")
    assert pred.centroid_is_hard and not pred.width_is_hard
    np.testing.assert_allclose(pred.centroid, omega0, rtol=1e-6)
    np.testing.assert_allclose(mc.centroid(pred.omega, pred.marginal), omega0)
    # the gate factor strictly broadens the marginal: σ_M in (σ_S, σ_TL]
    lower, upper = pred.width_bracket
    assert upper is not None
    np.testing.assert_allclose(lower, sigma_omega, rtol=1e-3)
    assert upper > lower
    assert lower < pred.rms_width <= upper * (1 + 1e-9)


def test_sd_centroid_at_tl_and_lower_bound_only(gaussian_spectrum):
    """SD: centroid ≈ fundamental at TL (soft); only the lower width bound holds."""
    omega, s, omega0, sigma_omega = gaussian_spectrum
    pred = mc.predicted_marginal(omega, s, "sd")
    assert not pred.centroid_is_hard and not pred.width_is_hard
    np.testing.assert_allclose(pred.centroid, omega0, rtol=1e-6)
    lower, upper = pred.width_bracket
    assert upper is None  # SD loses the rigorous upper reference
    np.testing.assert_allclose(lower, sigma_omega, rtol=1e-3)
    assert pred.rms_width > sigma_omega


def test_tg_is_identical_to_pg(gaussian_spectrum):
    omega, s, _, _ = gaussian_spectrum
    tg = mc.predicted_marginal(omega, s, "TG")
    pg = mc.predicted_marginal(omega, s, "pg")
    assert tg.interaction == "pg"
    np.testing.assert_allclose(tg.marginal, pg.marginal)
    np.testing.assert_allclose(tg.centroid, pg.centroid)


@pytest.mark.parametrize(
    ("interaction", "factor"),
    [("shg", 2.0), ("pg", 1.0), ("sd", 1.0), ("tg", 1.0)],
)
def test_predicted_centroid(gaussian_spectrum, interaction, factor):
    omega, s, omega0, _ = gaussian_spectrum
    np.testing.assert_allclose(
        mc.predicted_centroid(omega, s, interaction), factor * omega0, rtol=1e-6
    )


def test_autoconvolution_linear_vs_circular(gaussian_spectrum):
    omega, s, _, _ = gaussian_spectrum
    lin = mc.spectral_autoconvolution(s)
    circ = mc.spectral_autoconvolution(s, circular=True)
    assert lin.size == 2 * s.size - 1
    assert circ.size == s.size
    # both integrate to (sum S)^2 up to the grid spacing (linear: no wrap loss)
    np.testing.assert_allclose(lin.sum(), s.sum() ** 2, rtol=1e-6)


def test_apply_marginal_correction_matches_autoconvolution():
    """The corrected frequency marginal equals the predicted SHG autoconvolution."""
    g = Grid(128, dt=0.5e-15)
    # centred baseband spectrum (zero frequency in the middle)
    Iomega = np.exp(-0.5 * (g.omega / (0.05 * g.omega.max())) ** 2)
    rng = np.random.default_rng(0)
    # a trace with an arbitrary tilted, positive frequency marginal
    delay_profile = np.exp(-0.5 * ((np.arange(20) - 10) / 3.0) ** 2)
    tilt = (1.0 + 0.5 * np.linspace(-1.0, 1.0, g.n)) ** 2
    trace = np.outer(tilt, delay_profile) + 0.01 * rng.random((g.n, 20))
    orig = trace.copy()
    out = mc.apply_marginal_correction(trace, Iomega)
    np.testing.assert_array_equal(trace, orig)  # input not mutated
    IwSHG = mc.spectral_autoconvolution(Iomega, circular=True)
    np.testing.assert_allclose(
        out.sum(axis=1) / out.sum(axis=1).max(), IwSHG / IwSHG.max(), atol=1e-9
    )


def test_apply_marginal_correction_zeros_empty_rows():
    """Rows with no marginal power are zeroed rather than dividing by zero."""
    g = Grid(64, dt=0.5e-15)
    Iomega = np.exp(-0.5 * (g.omega / (0.05 * g.omega.max())) ** 2)
    trace = np.ones((g.n, 8))
    trace[3] = 0.0
    out = mc.apply_marginal_correction(trace, Iomega)
    assert np.all(out[3] == 0.0)


@pytest.mark.parametrize("interaction", ["pg", "sd", "tg"])
@pytest.mark.parametrize("n_true", [1.5, 3.0, 4.5])
def test_auto_exponent_recovers_known_tilt(interaction, n_true):
    """A base marginal = spectrum / (λ/µm)^n recovers n by centroid matching."""
    lam = np.linspace(700e-9, 900e-9, 400)
    spec = np.exp(-((lam - 800e-9) ** 2) / (2 * (25e-9) ** 2))
    base_marg = spec / (lam / 1e-6) ** n_true
    n = mc.auto_third_order_exponent(lam, base_marg, lam, spec, interaction)
    assert n == pytest.approx(n_true, abs=0.02)


def test_auto_exponent_target_centroid_override():
    """An explicit target centroid is matched instead of the spectrum's own."""
    lam = np.linspace(700e-9, 900e-9, 400)
    spec = np.exp(-((lam - 800e-9) ** 2) / (2 * (25e-9) ** 2))
    base_marg = spec / (lam / 1e-6) ** 2.0
    # target the spectrum's own omega-centroid -> recovers the true tilt (n = 2)
    omega = wlfreq(lam)
    target = mc.predicted_centroid(omega, spec * lam**2, "pg")
    n = mc.auto_third_order_exponent(
        lam, base_marg, lam, spec, "pg", target_centroid=target
    )
    assert n == pytest.approx(2.0, abs=0.05)


def test_auto_exponent_clamped_to_bounds():
    lam = np.linspace(700e-9, 900e-9, 400)
    spec = np.exp(-((lam - 800e-9) ** 2) / (2 * (25e-9) ** 2))
    base_marg = spec / (lam / 1e-6) ** 10  # steeper than the (0, 6) cap
    n = mc.auto_third_order_exponent(lam, base_marg, lam, spec, "pg", bounds=(0.0, 6.0))
    assert n == 6.0


def test_auto_exponent_no_signal_raises():
    lam = np.linspace(700e-9, 900e-9, 100)
    marg = np.exp(-((lam - 800e-9) ** 2) / (2 * (25e-9) ** 2))
    spec = np.ones_like(lam)
    # a trust window with no overlap leaves nothing to fit
    with pytest.raises(ValueError, match="no signal in the trust window"):
        mc.auto_third_order_exponent(
            lam, marg, lam, spec, "pg", window=(200e-9, 300e-9)
        )


def test_unknown_interaction_raises(gaussian_spectrum):
    omega, s, _, _ = gaussian_spectrum
    with pytest.raises(ValueError, match="unknown interaction"):
        mc.predicted_marginal(omega, s, "xpw")


def test_degenerate_axis_raises():
    with pytest.raises(ValueError, match="at least 4 samples"):
        mc.predicted_marginal(np.array([0.0, 1.0]), np.array([1.0, 1.0]), "shg")
