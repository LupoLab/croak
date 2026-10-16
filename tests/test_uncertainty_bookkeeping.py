"""Bookkeeping and validation tests for :mod:`croak.uncertainty`.

The statistical behaviour of the bootstraps is tested against real retrievals in
``test_uncertainty.py``. Here the solver is replaced by a deterministic fake so
the resampling, outlier rejection, intervals, callbacks and argument checks can
be tested exactly and quickly.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from croak import uncertainty
from croak.forward import maketrace
from croak.grid import Grid
from croak.maths import wlfreq
from croak.pulses import gaussian_pulse
from croak.result import RetrievalResult
from croak.uncertainty import (
    CoverageResult,
    NoiseModel,
    UncertaintyResult,
    _assemble,
    _build_runner,
    bias_corrected_interval,
    combine_uncertainties,
    coverage_calibration,
    estimate_fwhm_uncertainty,
    noise_from_background,
    noise_from_residual,
    parametric_bootstrap,
    resampling_bootstrap,
    thickness_bootstrap,
)

GRID = Grid(64, dt=1e-15)
DELAYS = np.linspace(-20e-15, 20e-15, 11)


def _result(trace=True) -> RetrievalResult:
    ew = gaussian_pulse(GRID, 8e-15)
    tr = maketrace(GRID.omega, DELAYS, ew, "pg")
    return RetrievalResult(
        spectrum=ew,
        grid=GRID,
        delays=DELAYS,
        interaction="pg",
        algorithm="copra",
        error=0.0,
        trace=tr / tr.max() if trace else None,
        omega0=wlfreq(800e-9),
    )


@pytest.fixture
def fake_runner(monkeypatch):
    """Replace the solver: replicate b returns the input pulse stretched by (1+b/100).

    Records each call's trace shape, delays, weights and the kwargs the runner
    was built with, so the resampling schemes can be checked directly.
    """
    calls: list[dict] = []
    built: list[dict] = []

    def build(result, retrieve_kwargs):
        built.append(dict(retrieve_kwargs))

        def run(trace, delays, *, guess, weights=None, rng=None):
            b = len(calls)
            calls.append(
                dict(shape=np.shape(trace), delays=delays, weights=weights, guess=guess)
            )
            stretched = gaussian_pulse(GRID, 8e-15 * (1.0 + b / 100.0))
            return SimpleNamespace(grid=GRID, spectrum=stretched, error=0.01 + b * 1e-4)

        return run

    monkeypatch.setattr(uncertainty, "_build_runner", build)
    return SimpleNamespace(calls=calls, built=built)


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------
def test_noise_model_per_pixel_sigma_array():
    clean = np.array([[0.0, 1.0], [4.0, 9.0]])
    nm = NoiseModel(sigma=np.full((2, 2), 0.1), gain=1.0)
    np.testing.assert_allclose(nm.std(clean), np.sqrt(0.01 + clean))


def test_noise_from_residual_validates_and_supports_plain_std():
    measured = np.array([[1.0, 2.0], [3.0, 4.0]])
    fit = SimpleNamespace(trace=measured - np.array([[0.1, -0.1], [0.1, -0.1]]))
    nm = noise_from_residual(measured, fit, robust=False)
    assert nm.sigma == pytest.approx(0.1 / 4.0)
    with pytest.raises(ValueError, match="result.trace is None"):
        noise_from_residual(measured, SimpleNamespace(trace=None))
    with pytest.raises(ValueError, match="non-positive peak"):
        noise_from_residual(-measured, SimpleNamespace(trace=-measured))


def test_noise_from_background_mask_and_validation():
    raw = np.array([[10.0, 0.1], [0.3, 0.2]])
    mask = raw < 1.0
    nm = noise_from_background(raw, mask)
    assert nm.sigma == pytest.approx(np.std([0.1, 0.3, 0.2]) / 10.0)
    with pytest.raises(ValueError, match="at least two pixels"):
        noise_from_background(raw, (slice(0, 1), slice(0, 1)))
    with pytest.raises(ValueError, match="non-positive peak"):
        noise_from_background(-raw, mask)


# ---------------------------------------------------------------------------
# Intervals and assembly
# ---------------------------------------------------------------------------
def test_bias_corrected_interval_needs_two_samples():
    lo, hi = bias_corrected_interval([1.0], 1.0)
    assert np.isnan(lo) and np.isnan(hi)


def _assemble_kw(**kw):
    base = dict(
        statistic="fwhm",
        point_estimate=1.0,
        fwhms=[1.0, 1.1, 0.9],
        errors=[0.1, 0.1, 0.1],
        profiles=[],
        t_profile=None,
        method="parametric",
        n_resamples=3,
        base_error=0.1,
        noise_floor=None,
        warm_start=True,
        convergence_tol=3.0,
        interval="percentile",
    )
    base.update(kw)
    return base


def test_assemble_rejects_all_non_finite_replicates():
    with pytest.raises(RuntimeError, match="no replicate produced a finite FWHM"):
        _assemble(**_assemble_kw(fwhms=[np.nan, np.nan, np.inf]))


def test_assemble_keeps_all_finite_replicates_when_tolerance_rejects_everything():
    """A tolerance below one can reject even the median; fall back to all finite."""
    u = _assemble(**_assemble_kw(convergence_tol=0.5))
    assert u.n_converged == 3


def test_assemble_percentile_interval_and_unknown_interval():
    u = _assemble(**_assemble_kw())
    assert u.interval_68 == (
        pytest.approx(np.percentile([1.0, 1.1, 0.9], 16)),
        pytest.approx(np.percentile([1.0, 1.1, 0.9], 84)),
    )
    with pytest.raises(ValueError, match="unknown interval 'hdi'"):
        _assemble(**_assemble_kw(interval="hdi"))


def test_uncertainty_result_median_of_no_samples_is_nan():
    u = UncertaintyResult(
        statistic="fwhm",
        point_estimate=1.0,
        samples=np.empty(0),
        interval_68=(np.nan, np.nan),
        interval_95=(np.nan, np.nan),
        interval_method="percentile",
        method="parametric",
        n_resamples=0,
        n_converged=0,
        frog_errors=np.empty(0),
        base_error=0.0,
    )
    assert np.isnan(u.median)


def test_build_runner_rejects_an_unknown_algorithm():
    with pytest.raises(ValueError, match="unknown algorithm 'gerchberg'"):
        _build_runner(_result(), {"algorithm": "gerchberg"})


# ---------------------------------------------------------------------------
# Bootstraps on a fake solver
# ---------------------------------------------------------------------------
def test_parametric_bootstrap_validates_inputs():
    with pytest.raises(ValueError, match="result.trace is None"):
        parametric_bootstrap(_result(trace=False), noise=0.01)
    with pytest.raises(ValueError, match="n_resamples must be >= 1"):
        parametric_bootstrap(_result(), noise=0.01, n_resamples=0)


def test_parametric_bootstrap_reports_progress_and_profiles(fake_runner):
    progress = []
    u = parametric_bootstrap(
        _result(),
        noise=0.01,
        n_resamples=3,
        collect_profiles=True,
        rng=np.random.default_rng(0),
        callback=lambda done, total, err: progress.append((done, total, err)),
    )
    assert [p[:2] for p in progress] == [(1, 3), (2, 3), (3, 3)]
    assert progress[-1][2] == pytest.approx(0.0102)
    assert u.profiles.shape[0] == 3 and u.t_profile is not None
    # the fake solver stretched replicate b by (1 + b/100): samples grow with b
    assert np.all(np.diff(u.samples) > 0)


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"n_resamples": 0}, "n_resamples must be >= 1"),
        ({"unit": "pixel"}, "unknown unit 'pixel'"),
    ],
)
def test_resampling_bootstrap_validates_inputs(kw, message):
    r = _result()
    with pytest.raises(ValueError, match=message):
        resampling_bootstrap(r, r.trace, **kw)


def test_resampling_bootstrap_delay_leave_out(fake_runner):
    """Leave-out keeps a sorted subset of 70% of the delay columns, without repeats."""
    r = _result()
    progress = []
    u = resampling_bootstrap(
        r,
        r.trace,
        unit="delay",
        leave_out=0.3,
        n_resamples=2,
        collect_profiles=True,
        rng=np.random.default_rng(1),
        callback=lambda *a: progress.append(a),
    )
    for call in fake_runner.calls:
        kept = call["delays"]
        assert kept.size == round(0.7 * DELAYS.size)
        assert np.all(np.diff(kept) > 0)  # sorted, no duplicates
        assert call["shape"] == (GRID.n, kept.size)
    assert len(progress) == 2 and u.profiles.shape[0] == 2


def test_resampling_bootstrap_frequency_leave_out_zeroes_rows(fake_runner):
    r = _result()
    resampling_bootstrap(
        r,
        r.trace,
        unit="frequency",
        leave_out=0.25,
        n_resamples=2,
        rng=np.random.default_rng(2),
    )
    for call in fake_runner.calls:
        w = call["weights"]
        assert np.count_nonzero(w == 0.0) == round(0.25 * GRID.n)
        assert set(np.unique(w)) == {0.0, 1.0}


def test_resampling_bootstrap_frequency_multinomial_weights(fake_runner):
    """Without leave-out the rows are weighted by sqrt of multinomial counts."""
    r = _result()
    resampling_bootstrap(
        r, r.trace, unit="frequency", n_resamples=1, rng=np.random.default_rng(3)
    )
    w = fake_runner.calls[0]["weights"]
    # squared weights are integer counts summing to the number of rows
    assert np.sum(w**2) == pytest.approx(GRID.n)
    np.testing.assert_allclose(w**2, np.round(w**2))


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"thickness_sigma": -1e-6}, "thickness_sigma must be >= 0"),
        ({"n_resamples": 0}, "n_resamples must be >= 1"),
        ({"material": ""}, "requires a dispersive `material`"),
    ],
)
def test_thickness_bootstrap_validates_inputs(kw, message):
    r = _result()
    args = dict(central_thickness=10e-6, thickness_sigma=1e-6, material="SiO2")
    args.update(kw)
    with pytest.raises(ValueError, match=message):
        thickness_bootstrap(r, r.trace, **args)


def test_thickness_bootstrap_draws_clipped_thicknesses(fake_runner):
    """Each draw rebuilds the solver at its own thickness; negatives are clipped."""
    r = _result()
    progress = []
    u = thickness_bootstrap(
        r,
        r.trace,
        central_thickness=1e-6,
        thickness_sigma=5e-6,
        material="SiO2",
        n_resamples=6,
        collect_profiles=True,
        rng=np.random.default_rng(4),
        callback=lambda *a: progress.append(a),
        thickness=99.0,  # a stale retrieval setting: replaced per draw
    )
    built = [b["thickness"] for b in fake_runner.built]
    assert built[0] == 1e-6  # the central retrieval
    assert min(built[1:]) >= 0.0 and 0.0 in built[1:]  # sigma ≫ L0 forces clipping
    np.testing.assert_allclose(u.thicknesses, built[1:])
    assert len(progress) == 6 and u.profiles.shape[0] == 6


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def test_estimate_dispatch_requires_data_for_each_method():
    r = _result()
    with pytest.raises(ValueError, match="needs a `noise` model"):
        estimate_fwhm_uncertainty(r, method="parametric")
    with pytest.raises(ValueError, match="method 'delay' requires the measured trace"):
        estimate_fwhm_uncertainty(r, method="delay")
    with pytest.raises(ValueError, match="'thickness' requires the measured trace"):
        estimate_fwhm_uncertainty(r, method="thickness")


def test_estimate_dispatch_estimates_parametric_noise_from_the_residual(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        uncertainty,
        "parametric_bootstrap",
        lambda result, *, noise, **kw: seen.setdefault("noise", noise),
    )
    r = _result()
    measured = r.trace + 0.01 * np.random.default_rng(5).standard_normal(r.trace.shape)
    estimate_fwhm_uncertainty(r, measured, method="parametric")
    assert seen["noise"].sigma == pytest.approx(0.01, rel=0.2)  # MAD of 704 draws


# ---------------------------------------------------------------------------
# Combining independent sources
# ---------------------------------------------------------------------------
def _u(samples, point=1.0, profiles=None, t_profile=None) -> UncertaintyResult:
    samples = np.asarray(samples, dtype=float)
    return UncertaintyResult(
        statistic="fwhm",
        point_estimate=point,
        samples=samples,
        interval_68=(np.nan, np.nan),
        interval_95=(np.nan, np.nan),
        interval_method="percentile",
        method="parametric",
        n_resamples=samples.size,
        n_converged=samples.size,
        frog_errors=np.empty(0),
        base_error=0.0,
        profiles=profiles,
        t_profile=t_profile,
    )


def test_combine_skips_sources_without_samples():
    rng = np.random.default_rng(6)
    combined = combine_uncertainties(
        [_u([0.9, 1.1]), _u([])], interval="percentile", n_samples=50, rng=rng
    )
    # only the first source contributes: every draw is the point ± 0.1
    assert set(np.round(combined.samples, 12)) <= {0.9, 1.1}


def test_combine_rejects_inputs_without_samples_and_unknown_intervals():
    with pytest.raises(ValueError, match="no input carried any samples"):
        combine_uncertainties([_u([]), _u([])])
    with pytest.raises(ValueError, match="unknown interval 'hdi'"):
        combine_uncertainties([_u([1.0, 1.2]), _u([0.8, 1.0])], interval="hdi")


def test_combine_drops_the_band_when_time_axes_differ():
    t1, t2 = np.linspace(0, 1, 4), np.linspace(0, 2, 4)
    prof = np.ones((2, 4))
    combined = combine_uncertainties(
        [
            _u([1.0, 1.1], profiles=prof, t_profile=t1),
            _u([1.0, 0.9], profiles=prof, t_profile=t2),
        ],
        n_samples=10,
        rng=np.random.default_rng(7),
    )
    assert combined.profiles is None and combined.t_profile is None


# ---------------------------------------------------------------------------
# Coverage calibration
# ---------------------------------------------------------------------------
def test_coverage_result_summary():
    c = CoverageResult(
        level=0.68,
        n_trials=10,
        n_resamples=50,
        method="parametric",
        coverage=0.7,
        mean_bias=-0.25e-15,
        biases=np.zeros(10),
        covered=np.ones(10, bool),
    )
    assert c.summary() == (
        "coverage 70% of nominal 68% (parametric, 10 trials, B=50); "
        "median-bias -0.25 fs"
    )


def test_coverage_calibration_percentile_and_empty_trials(monkeypatch):
    """Percentile intervals are used when asked; a trial with no samples misses."""
    # an interval spanning 0.1-100 fs certainly holds the ~2 fs true FWHM
    outcomes = iter([np.array([1e-16, 2e-15, 1e-13]), np.empty(0)])

    def fake_estimate(base, noisy, **kw):
        samples = next(outcomes)
        return SimpleNamespace(samples=samples, point_estimate=2e-15)

    monkeypatch.setattr(uncertainty, "estimate_fwhm_uncertainty", fake_estimate)
    monkeypatch.setattr(
        uncertainty, "retrieve", lambda *a, **k: SimpleNamespace(error=0.0)
    )
    progress = []
    c = coverage_calibration(
        noise=0.01,
        n_trials=2,
        n_resamples=3,
        interval="percentile",
        fwhm_range=(2e-15, 2e-15),
        rng=np.random.default_rng(8),
        callback=lambda *a: progress.append(a),
    )
    assert progress == [(1, 2), (2, 2)]
    assert list(c.covered) == [True, False]
    assert np.isnan(c.biases[1])


def test_estimate_dispatch_resampling_passes_the_unit(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        uncertainty,
        "resampling_bootstrap",
        lambda result, measured, *, unit, **kw: seen.setdefault("unit", unit),
    )
    r = _result()
    estimate_fwhm_uncertainty(r, r.trace, method="frequency")
    assert seen["unit"] == "frequency"


def test_covariance_summary_names_the_approximation():
    u = _u([1e-15, 1.2e-15], point=1.1e-15)
    u.method = "covariance"
    text = u.summary()
    assert text.startswith("FWHM = 1.10 ± ")
    assert "linearised covariance" in text and "systematics excluded" in text
