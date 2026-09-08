"""Tests for the linearised Gauss–Newton covariance (croak.covariance)."""

from __future__ import annotations

import numpy as np
import pytest

import croak
from croak.covariance import (
    CovarianceResult,
    covariance_uncertainty,
    parameter_covariance,
)
from croak.dispersion import dispersion_phase
from croak.uncertainty import _apply_propagation, _fwhm_and_profile


@pytest.fixture(scope="module")
def chirped():
    """A chirped synthetic PG pulse with a clean trace and an LM retrieval.

    Returns ``(grid, omega0, ew_true, delays, trace, result_pointwise,
    result_bspline)``; both retrievals are warm-started from the true spectrum so
    the base fit is accurate and the covariance probes a genuine optimum.
    """
    g = croak.Grid(64, dt=1.0e-15)
    omega0 = float(croak.maths.wlfreq(800e-9))
    ew = croak.apply_dispersion(
        g.omega, omega0, croak.gaussian_pulse(g, 20e-15), gdd=200e-30
    )
    delays = np.linspace(-80e-15, 80e-15, 48)
    trace = croak.maketrace(g.omega, delays, ew, "pg")
    res_pw = croak.retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="lm",
        guess=ew,
        omega0=omega0,
        maxiters=80,
        progress=False,
    )
    res_bs = croak.retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="lm",
        guess=ew,
        omega0=omega0,
        phase_basis="bspline",
        n_nodes=8,
        maxiters=150,
        progress=False,
    )
    return g, omega0, ew, delays, trace, res_pw, res_bs


def _gdd_transfer(grid, omega0: float, gdd: float) -> np.ndarray:
    return np.exp(1j * dispersion_phase(grid.omega, omega0, gdd=gdd))


# ---------------------------------------------------------------------------
# parameter_covariance
# ---------------------------------------------------------------------------
def test_parameter_covariance_shapes_and_psd(chirped):
    _g, _o, _ew, _d, trace, res_pw, _bs = chirped
    cov = parameter_covariance(res_pw, trace, noise=0.02)
    assert isinstance(cov, CovarianceResult)
    assert cov.cov.shape == (cov.n_params, cov.n_params)
    assert cov.n_residuals == trace.size
    # symmetric, positive semi-definite covariance
    np.testing.assert_allclose(cov.cov, cov.cov.T, atol=1e-20)
    evals = np.linalg.eigvalsh(cov.cov)
    assert evals.min() > -1e-12 * max(evals.max(), 1e-30)
    # rank is bounded and the gauge/soft null space is non-empty for a pointwise fit
    assert 0 < cov.rank <= cov.n_params
    assert cov.null_dim == cov.n_params - cov.rank >= 0
    assert np.all(cov.stderr >= 0.0)


def test_covariance_internal_scale_without_noise(chirped):
    """Without a noise model the covariance is scaled by the internal reduced χ²."""
    _g, _o, _ew, _d, trace, _pw, res_bs = chirped
    cov = parameter_covariance(res_bs, trace, phase_basis="bspline", n_nodes=8)
    assert cov.noise_scaled is True
    assert cov.reduced_chisq >= 0.0
    assert np.all(np.isfinite(cov.cov))


def test_bspline_covariance_is_well_conditioned(chirped):
    """The low-dimensional B-spline phase basis removes the soft-mode null space."""
    _g, _o, _ew, _d, trace, _pw, res_bs = chirped
    cov = parameter_covariance(
        res_bs, trace, noise=0.02, phase_basis="bspline", n_nodes=8
    )
    assert cov.n_params == 8
    assert cov.rank == cov.n_params  # all directions constrained
    assert cov.reduced_chisq >= 0.0


# ---------------------------------------------------------------------------
# covariance_uncertainty
# ---------------------------------------------------------------------------
def test_covariance_uncertainty_basic(chirped):
    """The well-conditioned B-spline basis gives a clean covariance distribution."""
    _g, _o, _ew, _d, trace, _pw, res_bs = chirped
    u = covariance_uncertainty(
        res_bs,
        trace,
        noise=0.02,
        n_samples=200,
        collect_profiles=True,
        phase_basis="bspline",
        n_nodes=8,
        rng=np.random.default_rng(0),
    )
    assert u.method == "covariance"
    assert u.n_converged == 200  # tame draws: none degenerate
    assert np.isfinite(u.plus_minus) and u.plus_minus > 0
    assert u.profiles is not None and u.profiles.shape[0] == u.n_converged
    # the point estimate is the FWHM of the retrieved pulse (no propagation)
    plain, _, _ = _fwhm_and_profile(res_bs.grid, res_bs.spectrum, 8)
    assert u.point_estimate == pytest.approx(plain, rel=0, abs=1e-30)
    assert "covariance" in u.summary()


def test_pointwise_covariance_draws_filter_degenerates(chirped):
    """Pointwise soft modes can yield degenerate draws; ``_assemble`` drops them."""
    _g, _o, _ew, _d, trace, res_pw, _bs = chirped
    u = covariance_uncertainty(
        res_pw, trace, noise=0.02, n_samples=120, rng=np.random.default_rng(0)
    )
    assert u.method == "covariance"
    assert 0 < u.n_converged <= 120
    assert np.all(np.isfinite(u.samples))


def test_covariance_uncertainty_propagation_identity(chirped):
    """An all-ones transfer function leaves the result unchanged."""
    _g, _o, _ew, _d, trace, res_pw, _bs = chirped
    ones = np.ones_like(res_pw.spectrum)
    a = covariance_uncertainty(
        res_pw, trace, noise=0.02, n_samples=80, rng=np.random.default_rng(1)
    )
    b = covariance_uncertainty(
        res_pw,
        trace,
        noise=0.02,
        n_samples=80,
        propagation=ones,
        rng=np.random.default_rng(1),
    )
    assert b.point_estimate == pytest.approx(a.point_estimate, rel=0, abs=1e-30)
    np.testing.assert_allclose(b.samples, a.samples)


def test_covariance_uncertainty_propagation_moves_point(chirped):
    _g, omega0, _ew, _d, trace, res_pw, _bs = chirped
    H = _gdd_transfer(res_pw.grid, omega0, gdd=-200e-30)  # compress the +200 fs² chirp
    u = covariance_uncertainty(
        res_pw,
        trace,
        noise=0.02,
        n_samples=100,
        propagation=H,
        propagation_label="-200 fs²",
        rng=np.random.default_rng(2),
    )
    expected, _, _ = _fwhm_and_profile(
        res_pw.grid, _apply_propagation(res_pw.spectrum, H), 8
    )
    assert u.point_estimate == pytest.approx(expected, rel=0, abs=1e-30)
    assert "-200 fs²" in u.summary()


def test_covariance_matches_bootstrap_order_of_magnitude(chirped):
    """Headline cross-check: analytic covariance ≈ parametric bootstrap (B-spline).

    In the well-conditioned B-spline basis the linearised covariance and the
    resampling bootstrap should agree to within a small factor; the Gaussian
    estimate typically runs a little tighter. Compare sample standard deviations:
    unlike a bias-corrected interval width, this direct measure of spread does not
    jump into a sparsely sampled tail when a platform's optimiser moves one or two
    of the 40 bootstrap samples across the full-data point estimate. A loose
    factor-of-three band keeps the test robust to Monte-Carlo scatter.
    """
    _g, omega0, _ew, _d, trace, _pw, res_bs = chirped
    uc = covariance_uncertainty(
        res_bs,
        trace,
        noise=0.02,
        n_samples=400,
        phase_basis="bspline",
        n_nodes=8,
        rng=np.random.default_rng(0),
    )
    ub = croak.parametric_bootstrap(
        res_bs,
        noise=0.02,
        n_resamples=40,
        algorithm="lm",
        omega0=omega0,
        phase_basis="bspline",
        n_nodes=8,
        maxiters=150,
        rng=np.random.default_rng(0),
    )
    assert uc.point_estimate == pytest.approx(ub.point_estimate, rel=0.02)
    ratio = uc.std / ub.std
    assert 1.0 / 3.0 < ratio < 3.0


def test_covariance_uncertainty_rejects_bad_n_samples(chirped):
    _g, _o, _ew, _d, trace, res_pw, _bs = chirped
    with pytest.raises(ValueError, match="n_samples"):
        covariance_uncertainty(res_pw, trace, noise=0.02, n_samples=0)


# ---------------------------------------------------------------------------
# Fitted extra parameters (thickness / delay-zero) in the covariance
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def thickness_fit():
    """A PG dispersive retrieval that also fits the slab thickness."""
    g = croak.Grid(128, dt=2e-15)
    omega0 = float(croak.maths.wlfreq(800e-9))
    ew = croak.gaussian_pulse(g, 8e-15)
    delays = np.linspace(-120e-15, 120e-15, 90)
    L_true = 400e-6
    trace = croak.maketrace(
        g.omega,
        delays,
        ew,
        "pg",
        material="SiO2",
        thickness=L_true,
        npoints=20,
        omega0=omega0,
    )
    res = croak.retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="lm",
        guess=ew,
        omega0=omega0,
        material="SiO2",
        thickness=250e-6,
        npoints=20,
        fit_thickness=True,
        maxiters=300,
        progress=False,
    )
    return g, omega0, delays, trace, res


def test_covariance_reports_sigma_thickness(thickness_fit):
    _g, omega0, _d, trace, res = thickness_fit
    cov = parameter_covariance(
        res,
        trace,
        noise=0.01,
        material="SiO2",
        thickness=res.thickness,
        npoints=20,
        omega0=omega0,
        fit_thickness=True,
    )
    # one extra parameter appended; its σ is finite, positive and physical (m)
    assert cov.sigma_thickness is not None
    assert cov.sigma_tau0 is None
    assert np.isfinite(cov.sigma_thickness)
    assert cov.sigma_thickness > 0.0
    # a sane fit pins the thickness far better than the 1 mm scale
    assert cov.sigma_thickness < 1e-3


def test_covariance_sigma_tau0(chirped):
    """Fitting tau0 adds a finite delay-zero standard error."""
    _g, omega0, ew, delays, _trace, _pw, _bs = chirped
    # rebuild a trace with a small delay-zero offset and fit it
    tau0_true = 1.0e-15
    g = _g
    trace = croak.maketrace(g.omega, delays - tau0_true, ew, "pg")
    res = croak.retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="lm",
        guess=ew,
        omega0=omega0,
        fit_tau0=True,
        maxiters=200,
        progress=False,
    )
    cov = parameter_covariance(res, trace, noise=0.01, fit_tau0=True)
    assert cov.sigma_tau0 is not None
    assert np.isfinite(cov.sigma_tau0)
    assert cov.sigma_tau0 > 0.0


def test_covariance_uncertainty_with_fitted_thickness(thickness_fit):
    """The FWHM covariance estimator runs with the thickness in the parameter set."""
    _g, omega0, _d, trace, res = thickness_fit
    u = covariance_uncertainty(
        res,
        trace,
        noise=0.01,
        n_samples=100,
        material="SiO2",
        thickness=res.thickness,
        npoints=20,
        omega0=omega0,
        fit_thickness=True,
        rng=np.random.default_rng(0),
    )
    assert u.method == "covariance"
    assert np.isfinite(u.point_estimate)
