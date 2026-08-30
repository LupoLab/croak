"""Tests for the FWHM-uncertainty bootstrap (croak.uncertainty)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import croak
from croak.dispersion import dispersion_phase
from croak.maths import wlfreq
from croak.result import RetrievalResult
from croak.uncertainty import (
    NoiseModel,
    UncertaintyResult,
    _apply_propagation,
    _fwhm_and_profile,
    bias_corrected_interval,
    combine_uncertainties,
    coverage_calibration,
    estimate_fwhm_uncertainty,
    noise_from_background,
    noise_from_residual,
    parametric_bootstrap,
    percentile_interval,
    resampling_bootstrap,
    synthetic_experiment,
    thickness_bootstrap,
)


@pytest.fixture(scope="module")
def problem():
    """A small synthetic experiment with a noisy trace and a converged retrieval."""
    exp = synthetic_experiment(
        fwhm=6e-15, phases=(15e-30,), n=64, dt=0.5e-15, ndelay=40
    )
    rng = np.random.default_rng(0)
    nm = NoiseModel(sigma=0.01)
    noisy = nm.draw(exp["trace"], rng)
    # Warm-start from the true spectrum so the base retrieval is fast and accurate;
    # the bootstrap then probes the noise around a good solution.
    res = croak.retrieve(
        noisy,
        exp["grid"].omega,
        exp["delays"],
        exp["interaction"],
        guess=exp["ew"],
        omega0=exp["omega0"],
        maxiters=60,
        progress=False,
    )
    return exp, noisy, res


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------
def test_noise_model_draw_is_nonnegative_and_scaled():
    rng = np.random.default_rng(1)
    clean = np.ones((40, 30))
    nm = NoiseModel(sigma=0.05)  # 5 % of peak (peak == 1 here)
    noisy = nm.draw(clean, rng)
    assert noisy.shape == clean.shape
    assert np.all(noisy >= 0.0)
    # standard deviation of the added noise ~= sigma * peak
    assert np.std(noisy - clean) == pytest.approx(0.05, rel=0.1)


def test_noise_from_residual_recovers_injected_sigma():
    g = croak.Grid(64, dt=0.5e-15)
    ew = croak.gaussian_pulse(g, 6e-15)
    delays = np.linspace(-40e-15, 40e-15, 40)
    clean = croak.maketrace(g.omega, delays, ew, "pg")
    rng = np.random.default_rng(2)
    sigma = 0.02
    # Pure Gaussian residual (no zero-clip) so the estimator can be checked cleanly;
    # clipping a sparse trace at zero would bias the robust MAD downward.
    measured = clean + sigma * clean.max() * rng.standard_normal(clean.shape)
    result = RetrievalResult(
        spectrum=ew,
        grid=g,
        delays=delays,
        interaction="pg",
        algorithm="copra",
        error=0.0,
        trace=clean,
    )
    nm = noise_from_residual(measured, result)
    assert float(nm.sigma) == pytest.approx(sigma, rel=0.15)


def test_noise_from_background_recovers_sigma():
    rng = np.random.default_rng(3)
    raw = np.zeros((50, 50))
    raw[20:30, 20:30] = 1.0  # signal island; peak == 1
    sigma_abs = 0.03
    raw[:10, :10] += sigma_abs * rng.standard_normal((10, 10))  # background patch
    nm = noise_from_background(raw, (slice(0, 10), slice(0, 10)))
    assert float(nm.sigma) == pytest.approx(sigma_abs, rel=0.3)


# ---------------------------------------------------------------------------
# Interval estimators
# ---------------------------------------------------------------------------
def test_percentile_interval_brackets_level():
    rng = np.random.default_rng(4)
    samples = rng.normal(10.0, 1.0, size=5000)
    lo, hi = percentile_interval(samples, 0.68)
    assert lo == pytest.approx(10.0 - 1.0, abs=0.1)
    assert hi == pytest.approx(10.0 + 1.0, abs=0.1)


def test_bias_corrected_matches_percentile_when_symmetric():
    rng = np.random.default_rng(5)
    samples = rng.normal(0.0, 1.0, size=5000)
    bc = bias_corrected_interval(samples, point_estimate=0.0, level=0.68)
    pc = percentile_interval(samples, 0.68)
    assert bc[0] == pytest.approx(pc[0], abs=0.1)
    assert bc[1] == pytest.approx(pc[1], abs=0.1)


def test_bias_corrected_falls_back_when_degenerate():
    samples = np.array([1.0, 2.0, 3.0, 4.0])
    # point estimate below all samples -> degenerate bias correction -> percentile
    assert bias_corrected_interval(samples, 0.0, 0.68) == percentile_interval(
        samples, 0.68
    )


# ---------------------------------------------------------------------------
# Method A — parametric bootstrap
# ---------------------------------------------------------------------------
def test_parametric_bootstrap_runs_and_brackets_truth(problem):
    _exp, _noisy, res = problem
    u = parametric_bootstrap(
        res,
        noise=NoiseModel(sigma=0.01),
        n_resamples=30,
        maxiters=60,
        collect_profiles=True,
        rng=np.random.default_rng(10),
    )
    assert isinstance(u, UncertaintyResult)
    assert u.method == "parametric"
    assert u.n_converged > 0
    assert u.point_estimate > 0
    assert u.interval_68[0] <= u.interval_68[1]
    assert u.interval_95[0] <= u.interval_68[0]
    assert u.interval_68[1] <= u.interval_95[1]
    assert u.profiles is not None and u.profiles.shape[0] == u.n_converged
    assert u.noise_floor is not None and u.noise_floor > 0


def test_parametric_spread_grows_from_noise(problem):
    # Noise is what produces the uncertainty: with ~zero noise every replicate
    # reproduces the full-data solution (zero spread); with finite noise the FWHM
    # samples scatter. (A robust, resolution-independent monotonicity check — the
    # absolute spread at a coarse grid is dominated by the FWHM estimator, not noise.)
    _exp, _noisy, res = problem
    quiet = parametric_bootstrap(
        res,
        noise=NoiseModel(sigma=1e-7),
        n_resamples=15,
        maxiters=60,
        rng=np.random.default_rng(11),
    )
    noisy = parametric_bootstrap(
        res,
        noise=NoiseModel(sigma=0.02),
        n_resamples=30,
        maxiters=60,
        rng=np.random.default_rng(11),
    )
    assert quiet.std == pytest.approx(0.0, abs=1e-17)
    assert noisy.std > quiet.std


# ---------------------------------------------------------------------------
# Method B — data-resampling bootstrap
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("unit", ["delay", "frequency"])
def test_resampling_bootstrap_runs(problem, unit):
    _exp, noisy, res = problem
    u = resampling_bootstrap(
        res,
        noisy,
        unit=unit,
        n_resamples=25,
        maxiters=60,
        rng=np.random.default_rng(12),
    )
    assert u.method == unit
    assert u.n_converged > 0
    assert np.isfinite(u.point_estimate)
    assert u.interval_68[0] <= u.interval_68[1]


def test_delay_resampling_tolerates_duplicate_delays(problem):
    """Regression: resampled delays contain duplicates; retrieval must still run."""
    _exp, noisy, res = problem
    u = resampling_bootstrap(
        res,
        noisy,
        unit="delay",
        n_resamples=10,
        maxiters=40,
        rng=np.random.default_rng(13),
    )
    assert u.n_converged > 0


def test_resampling_leave_out_runs(problem):
    _exp, noisy, res = problem
    u = resampling_bootstrap(
        res,
        noisy,
        unit="delay",
        leave_out=0.2,
        n_resamples=10,
        maxiters=40,
        rng=np.random.default_rng(14),
    )
    assert u.n_converged > 0


# ---------------------------------------------------------------------------
# Dispatch & errors
# ---------------------------------------------------------------------------
def test_estimate_dispatch_parametric_estimates_noise(problem):
    _exp, noisy, res = problem
    # no explicit noise -> estimated from the residual
    u = estimate_fwhm_uncertainty(
        res,
        noisy,
        method="parametric",
        n_resamples=15,
        maxiters=40,
        rng=np.random.default_rng(15),
    )
    assert u.method == "parametric"


def test_estimate_requires_measured_for_data_methods(problem):
    _exp, _noisy, res = problem
    with pytest.raises(ValueError, match="requires the measured trace"):
        estimate_fwhm_uncertainty(res, None, method="delay")


def test_estimate_rejects_unknown_method(problem):
    _exp, noisy, res = problem
    with pytest.raises(ValueError, match="unknown method"):
        estimate_fwhm_uncertainty(res, noisy, method="bogus")


def test_summary_has_value_units_and_caveat(problem):
    _exp, _noisy, res = problem
    u = parametric_bootstrap(
        res,
        noise=NoiseModel(sigma=0.01),
        n_resamples=15,
        maxiters=40,
        rng=np.random.default_rng(16),
    )
    s = u.summary()
    assert "fs" in s and "±" in s and "statistical" in s


# ---------------------------------------------------------------------------
# Coverage calibration & plotting
# ---------------------------------------------------------------------------
def test_coverage_calibration_runs_tiny():
    cov = coverage_calibration(
        noise=NoiseModel(sigma=0.01),
        n_trials=3,
        n_resamples=10,
        maxiters=40,
        rng=np.random.default_rng(20),
    )
    assert 0.0 <= cov.coverage <= 1.0
    assert cov.n_trials == 3
    assert cov.covered.shape == (3,)


def test_synthetic_experiment_reports_positive_true_fwhm():
    exp = synthetic_experiment(fwhm=6e-15, phases=(20e-30,))
    assert exp["true_fwhm"] > 6e-15  # chirp broadens the pulse
    assert exp["trace"].shape[1] == exp["delays"].size


def test_plot_uncertainty_smoke(problem):
    _exp, _noisy, res = problem
    u = parametric_bootstrap(
        res,
        noise=NoiseModel(sigma=0.01),
        n_resamples=15,
        maxiters=40,
        collect_profiles=True,
        rng=np.random.default_rng(21),
    )
    fig = croak.plot_uncertainty(u, truth=6e-15)
    assert len(fig.axes) == 2  # histogram + confidence band


# ---------------------------------------------------------------------------
# Substrate-thickness systematic
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dispersive_problem():
    """A synthetic substrate (dispersive) experiment + a converged retrieval at L0."""
    g = croak.Grid(64, dt=0.5e-15)
    omega0 = float(wlfreq(800e-9))
    ew = croak.gaussian_pulse(g, 6e-15)
    delays = np.linspace(-40e-15, 40e-15, 40)
    material, l0, npoints = "SiO2-Franta", 10e-6, 12
    trace = croak.maketrace(
        g.omega,
        delays,
        ew,
        "pg",
        material=material,
        thickness=l0,
        npoints=npoints,
        omega0=omega0,
    )
    # Warm-start from the true spectrum at the true thickness so the base solution is
    # accurate; the thickness Monte-Carlo then probes the substrate uncertainty.
    res = croak.retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        guess=ew,
        omega0=omega0,
        material=material,
        thickness=l0,
        npoints=npoints,
        maxiters=60,
        progress=False,
    )
    return {
        "grid": g,
        "omega0": omega0,
        "ew": ew,
        "delays": delays,
        "material": material,
        "l0": l0,
        "npoints": npoints,
        "trace": trace,
        "res": res,
    }


def test_thickness_bootstrap_runs_and_reports(dispersive_problem):
    dp = dispersive_problem
    u = thickness_bootstrap(
        dp["res"],
        dp["trace"],
        central_thickness=dp["l0"],
        thickness_sigma=0.5e-6,
        material=dp["material"],
        npoints=dp["npoints"],
        n_resamples=12,
        maxiters=40,
        collect_profiles=True,
        rng=np.random.default_rng(30),
    )
    assert isinstance(u, UncertaintyResult)
    assert u.method == "thickness"
    assert u.n_converged > 0
    assert np.isfinite(u.point_estimate) and u.point_estimate > 0
    assert u.thicknesses is not None and u.thicknesses.shape[0] == u.n_converged
    assert u.frog_errors.shape[0] == u.n_converged
    assert u.thickness_central == pytest.approx(dp["l0"])
    assert u.thickness_sigma == pytest.approx(0.5e-6)
    assert u.noise_floor is None
    assert u.interval_68[0] <= u.interval_68[1]
    assert u.interval_95[0] <= u.interval_68[0]
    assert u.interval_68[1] <= u.interval_95[1]
    assert u.profiles is not None and u.profiles.shape[0] == u.n_converged


def test_thickness_spread_grows_with_sigma(dispersive_problem):
    # With sigma == 0 every draw uses the same thickness and reproduces the central
    # solution (zero spread); a finite sigma scatters the FWHM.
    dp = dispersive_problem
    zero = thickness_bootstrap(
        dp["res"],
        dp["trace"],
        central_thickness=dp["l0"],
        thickness_sigma=0.0,
        material=dp["material"],
        npoints=dp["npoints"],
        n_resamples=8,
        maxiters=40,
        rng=np.random.default_rng(31),
    )
    wide = thickness_bootstrap(
        dp["res"],
        dp["trace"],
        central_thickness=dp["l0"],
        thickness_sigma=1.0e-6,
        material=dp["material"],
        npoints=dp["npoints"],
        n_resamples=20,
        maxiters=40,
        rng=np.random.default_rng(31),
    )
    assert zero.std == pytest.approx(0.0, abs=1e-17)
    assert wide.std > zero.std


def test_thickness_bootstrap_rejects_negative_sigma(dispersive_problem):
    dp = dispersive_problem
    with pytest.raises(ValueError, match="thickness_sigma"):
        thickness_bootstrap(
            dp["res"],
            dp["trace"],
            central_thickness=dp["l0"],
            thickness_sigma=-1e-6,
            material=dp["material"],
        )


def test_thickness_bootstrap_requires_material(dispersive_problem):
    dp = dispersive_problem
    with pytest.raises(ValueError, match="material"):
        thickness_bootstrap(
            dp["res"],
            dp["trace"],
            central_thickness=dp["l0"],
            thickness_sigma=0.5e-6,
            material="",
        )


def test_estimate_dispatch_thickness(dispersive_problem):
    dp = dispersive_problem
    u = estimate_fwhm_uncertainty(
        dp["res"],
        dp["trace"],
        method="thickness",
        central_thickness=dp["l0"],
        thickness_sigma=0.5e-6,
        material=dp["material"],
        npoints=dp["npoints"],
        n_resamples=8,
        maxiters=40,
        rng=np.random.default_rng(32),
    )
    assert u.method == "thickness"
    assert u.thicknesses is not None


def test_estimate_thickness_requires_measured(dispersive_problem):
    dp = dispersive_problem
    with pytest.raises(ValueError, match="requires the measured trace"):
        estimate_fwhm_uncertainty(
            dp["res"],
            None,
            method="thickness",
            central_thickness=dp["l0"],
            thickness_sigma=0.5e-6,
            material=dp["material"],
        )


def test_thickness_summary_reports_systematic(dispersive_problem):
    dp = dispersive_problem
    u = thickness_bootstrap(
        dp["res"],
        dp["trace"],
        central_thickness=dp["l0"],
        thickness_sigma=0.5e-6,
        material=dp["material"],
        npoints=dp["npoints"],
        n_resamples=8,
        maxiters=40,
        rng=np.random.default_rng(33),
    )
    s = u.summary()
    assert "fs" in s and "±" in s and "systematic" in s and "thickness" in s


def test_plot_thickness_sensitivity_smoke(dispersive_problem):
    dp = dispersive_problem
    u = thickness_bootstrap(
        dp["res"],
        dp["trace"],
        central_thickness=dp["l0"],
        thickness_sigma=0.5e-6,
        material=dp["material"],
        npoints=dp["npoints"],
        n_resamples=10,
        maxiters=40,
        collect_profiles=True,
        rng=np.random.default_rng(34),
    )
    fig = croak.plot_thickness_sensitivity(u, truth=6e-15)
    assert len(fig.axes) == 3  # FWHM vs L, R vs L, intensity band
    # plot_uncertainty delegates to the thickness figure for a thickness result.
    fig2 = croak.plot_uncertainty(u)
    assert len(fig2.axes) == 3


# ---------------------------------------------------------------------------
# Combining independent contributions
# ---------------------------------------------------------------------------
def _synthetic_uresult(method, pe, sd, *, n=3000, profiles=False, rng=None):
    """A lightweight UncertaintyResult with Gaussian FWHM samples (no retrieval)."""
    rng = np.random.default_rng() if rng is None else rng
    samples = pe + sd * rng.standard_normal(n)
    prof = t = None
    if profiles:
        t = np.linspace(-15e-15, 15e-15, 40)
        base = np.exp(-((t / pe) ** 2))
        prof = np.clip(base + 0.02 * rng.standard_normal((n, t.size)), 0.0, None)
    return UncertaintyResult(
        statistic="fwhm",
        point_estimate=pe,
        samples=samples,
        interval_68=percentile_interval(samples, 0.68),
        interval_95=percentile_interval(samples, 0.95),
        interval_method="percentile",
        method=method,
        n_resamples=n,
        n_converged=n,
        frog_errors=np.empty(0),
        base_error=0.01,
        profiles=prof,
        t_profile=t,
    )


def test_combine_uncertainties_quadrature_and_band():
    rng = np.random.default_rng(40)
    a = _synthetic_uresult("parametric", 1.5e-15, 0.05e-15, profiles=True, rng=rng)
    b = _synthetic_uresult("thickness", 1.5e-15, 0.12e-15, profiles=True, rng=rng)
    c = combine_uncertainties(
        [a, b], interval="percentile", rng=np.random.default_rng(1)
    )
    assert c.method == "combined"
    assert c.components == ("parametric", "thickness")
    # independent sources add in quadrature
    assert c.std == pytest.approx(np.hypot(a.std, b.std), rel=0.1)
    # the combined spread exceeds either alone
    assert c.std > a.std and c.std > b.std
    # the temporal band is combined too, on the shared time axis
    assert c.profiles is not None and c.t_profile is not None
    assert c.t_profile.shape == (40,)
    s = c.summary()
    assert "combined" in s and "parametric" in s and "thickness" in s


def test_combine_uncertainties_no_band_when_missing():
    rng = np.random.default_rng(41)
    a = _synthetic_uresult("parametric", 1.5e-15, 0.05e-15, profiles=True, rng=rng)
    b = _synthetic_uresult("thickness", 1.5e-15, 0.12e-15, profiles=False, rng=rng)
    c = combine_uncertainties(
        [a, b], interval="percentile", rng=np.random.default_rng(2)
    )
    assert c.profiles is None  # one input lacked a band


def test_combine_uncertainties_needs_two():
    a = _synthetic_uresult("parametric", 1.5e-15, 0.05e-15)
    with pytest.raises(ValueError, match="at least two"):
        combine_uncertainties([a])


# ---------------------------------------------------------------------------
# Propagation to a different beamline point (apply H(ω) before the FWHM/band)
# ---------------------------------------------------------------------------
def _gdd_transfer(grid, omega0: float, gdd: float) -> np.ndarray:
    """Pure-GDD spectral transfer function H(ω) = exp(i·GDD·ω²/2)."""
    return np.exp(1j * dispersion_phase(grid.omega, omega0, gdd=gdd))


def test_apply_propagation_none_is_identity():
    rng = np.random.default_rng(0)
    ew = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    out = _apply_propagation(ew, None)
    assert out is ew or np.array_equal(out, ew)


def test_propagation_identity_matches_unpropagated(problem):
    """An all-ones transfer function reproduces the measurement-plane result."""
    exp, _noisy, res = problem
    ones = np.ones_like(res.spectrum)
    base = parametric_bootstrap(
        res, noise=0.01, n_resamples=8, maxiters=40, rng=np.random.default_rng(7)
    )
    same = parametric_bootstrap(
        res,
        noise=0.01,
        n_resamples=8,
        maxiters=40,
        propagation=ones,
        rng=np.random.default_rng(7),
    )
    assert same.point_estimate == pytest.approx(base.point_estimate, rel=0, abs=1e-30)
    np.testing.assert_allclose(same.samples, base.samples)


def test_propagation_point_estimate_matches_applied_dispersion(problem):
    """The reported point estimate is the FWHM of the propagated retrieved pulse."""
    exp, _noisy, res = problem
    H = _gdd_transfer(res.grid, res.omega0, gdd=120e-30)
    u = parametric_bootstrap(
        res,
        noise=0.01,
        n_resamples=4,
        maxiters=30,
        propagation=H,
        propagation_label="+120 fs²",
        rng=np.random.default_rng(3),
    )
    expected, _, _ = _fwhm_and_profile(res.grid, _apply_propagation(res.spectrum, H), 8)
    assert u.point_estimate == pytest.approx(expected, rel=0, abs=1e-30)
    # the GDD changes the temporal width (direction depends on the retrieved chirp)
    plain, _, _ = _fwhm_and_profile(res.grid, res.spectrum, 8)
    assert not np.isclose(u.point_estimate, plain, rtol=0.05, atol=0.0)
    assert "+120 fs²" in u.summary()


def test_resampling_bootstrap_accepts_propagation(problem):
    """Delay resampling threads the transfer function without error."""
    exp, noisy, res = problem
    H = _gdd_transfer(res.grid, res.omega0, gdd=80e-30)
    u = resampling_bootstrap(
        res,
        noisy,
        unit="delay",
        n_resamples=6,
        maxiters=30,
        propagation=H,
        propagation_label="+80 fs²",
        rng=np.random.default_rng(9),
    )
    assert u.method == "delay"
    assert u.propagation_label == "+80 fs²"
    assert u.point_estimate > 0
