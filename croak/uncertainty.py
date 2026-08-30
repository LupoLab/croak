r"""Statistical uncertainty of a retrieved pulse from a single trace.

This module estimates a *statistical* (precision) error bar on the retrieved FWHM
intensity duration by resampling, following the analysis in
``docs/explanation/uncertainty_estimation.md``. Two estimators are provided:

* **Method A — parametric Monte-Carlo bootstrap** (:func:`parametric_bootstrap`):
  simulate synthetic traces from the best-fit clean trace plus a :class:`NoiseModel`,
  re-retrieve each (warm-started), and take the spread of the per-replicate FWHM.
* **Method B — data-resampling bootstrap** (:func:`resampling_bootstrap`): resample the
  *measured* trace over delay columns (a block bootstrap, recommended for delay-scanned
  data) or over frequency rows, and re-retrieve. Needs no noise model.

The FWHM is gauge-invariant under all the trivial FROG ambiguities (translation, CEP,
and — for the TG/PG kernel — there is no direction-of-time ambiguity), so it is a
well-defined scalar to bootstrap with no gauge alignment required.

The interval is a **statistical** bound *conditional on the forward and noise
models*; it excludes systematic error (calibration, geometry, model mismatch),
which often dominates.
:func:`coverage_calibration` validates, on synthetic ground truth, that the nominal
interval has the coverage it claims.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike, NDArray

from .forward import maketrace
from .grid import Grid
from .maths import fwhm as _fwhm
from .maths import wlfreq
from .metrics import frog_error
from .processing import oversample, shiftnorm
from .pulses import gaussian_pulse, sech_pulse
from .result import RetrievalResult
from .retrieve import ALGORITHMS, algorithm_params, retrieve

__all__ = [
    "NoiseModel",
    "noise_from_residual",
    "noise_from_background",
    "UncertaintyResult",
    "parametric_bootstrap",
    "resampling_bootstrap",
    "thickness_bootstrap",
    "combine_uncertainties",
    "estimate_fwhm_uncertainty",
    "percentile_interval",
    "bias_corrected_interval",
    "synthetic_experiment",
    "CoverageResult",
    "coverage_calibration",
]


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NoiseModel:
    r"""Additive (and optionally shot-dependent) Gaussian trace noise.

    The per-pixel variance is :math:`\sigma_0^2 + g\,T`, where :math:`\sigma_0` is the
    additive (read) noise standard deviation, :math:`g` the shot-noise coefficient and
    :math:`T` the (clipped, non-negative) clean trace intensity. Samples are clipped at
    zero, matching a detector that cannot record negative counts.

    Parameters
    ----------
    sigma : float or numpy.ndarray
        Additive noise standard deviation. A scalar is interpreted as a
        **fraction of the clean-trace peak** (so ``sigma=0.01`` is 1 % of peak,
        matching the convention used in the test suite); an array is taken as
        **absolute** per-pixel standard deviation
        with the same shape as the trace.
    gain : float, optional
        Shot-noise coefficient :math:`g` in trace-intensity units (``0`` ⇒
        purely additive noise). Adds a signal-dependent variance term
        ``gain * intensity``.
    """

    sigma: float | NDArray[np.float64]
    gain: float = 0.0

    def std(self, clean: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the per-pixel standard deviation for a given clean trace."""
        clean = np.asarray(clean, dtype=float)
        if np.ndim(self.sigma) == 0:
            base = float(self.sigma) * float(clean.max())
            var = base**2 + self.gain * np.clip(clean, 0.0, None)
        else:
            base = np.asarray(self.sigma, dtype=float)
            var = base**2 + self.gain * np.clip(clean, 0.0, None)
        return np.sqrt(var)

    def draw(
        self, clean: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Draw one noisy trace ``clip(clean + noise, 0, None)``."""
        clean = np.asarray(clean, dtype=float)
        noise = self.std(clean) * rng.standard_normal(clean.shape)
        return np.clip(clean + noise, 0.0, None)


def noise_from_residual(
    measured: ArrayLike, result: RetrievalResult, *, robust: bool = True
) -> NoiseModel:
    r"""Estimate a :class:`NoiseModel` from the post-fit residual.

    Uses the residual ``measured - result.trace`` after a converged retrieval. The
    residual is noise plus any model misfit, so this tends to *over*-estimate the noise
    when the forward model is imperfect (a conservative bias). Needs no extra data.

    Parameters
    ----------
    measured : array_like
        The measured trace ``(Nomega, Ndelay)`` (same normalisation as
        ``result.trace``).
    result : RetrievalResult
        A completed retrieval; ``result.trace`` is the scaled clean model trace.
    robust : bool, optional
        If true (default), estimate the standard deviation from the median absolute
        deviation (MAD × 1.4826), which resists the structured misfit near the signal;
        otherwise use the plain sample standard deviation.

    Returns
    -------
    NoiseModel
        An additive model whose scalar ``sigma`` is the residual std as a
        fraction of the measured peak.
    """
    if result.trace is None:
        raise ValueError("result.trace is None; cannot form a residual")
    measured = np.asarray(measured, dtype=float)
    resid = measured - np.asarray(result.trace, dtype=float)
    if robust:
        mad = np.median(np.abs(resid - np.median(resid)))
        sigma_abs = 1.4826 * float(mad)
    else:
        sigma_abs = float(np.std(resid))
    peak = float(measured.max())
    if peak <= 0:
        raise ValueError("measured trace has non-positive peak")
    return NoiseModel(sigma=sigma_abs / peak)


def noise_from_background(
    raw_trace: ArrayLike, region: NDArray[np.bool_] | tuple[slice, slice]
) -> NoiseModel:
    r"""Estimate a read-noise :class:`NoiseModel` from a signal-free region.

    Computes the standard deviation of a background patch of the *raw* trace. Captures
    only additive read noise (not shot noise on the signal), and the patch must be
    genuinely signal-free.

    Parameters
    ----------
    raw_trace : array_like
        The trace to take the background from (ideally the unfiltered measurement).
    region : numpy.ndarray of bool, or tuple of slice
        A boolean mask selecting background pixels, or a ``(rows, cols)`` slice tuple.

    Returns
    -------
    NoiseModel
        An additive model whose scalar ``sigma`` is the background std as a fraction of
        the trace peak.
    """
    raw_trace = np.asarray(raw_trace, dtype=float)
    if isinstance(region, tuple):
        patch = raw_trace[region[0], region[1]]
    else:
        patch = raw_trace[np.asarray(region, dtype=bool)]
    if patch.size < 2:
        raise ValueError("background region must contain at least two pixels")
    peak = float(raw_trace.max())
    if peak <= 0:
        raise ValueError("raw trace has non-positive peak")
    return NoiseModel(sigma=float(np.std(patch)) / peak)


def _resolve_noise(noise: NoiseModel | float) -> NoiseModel:
    """Coerce a bare float into a :class:`NoiseModel` (fraction-of-peak additive)."""
    if isinstance(noise, NoiseModel):
        return noise
    return NoiseModel(sigma=float(noise))


# ---------------------------------------------------------------------------
# FWHM extraction
# ---------------------------------------------------------------------------
def _fwhm_and_profile(
    grid: Grid, spectrum: NDArray[np.complex128], oversampling: int
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(fwhm, t_over, intensity)`` for a spectrum, peak-centred & oversampled.

    Reuses :func:`croak.processing.shiftnorm`/:func:`croak.processing.oversample`
    and :func:`croak.maths.fwhm`, exactly as
    :func:`croak.processing.process_result` does for
    ``fwhm_retr`` — so the bootstrap statistic matches the headline number. The
    oversampled time axis depends only on the grid and factor, so it is identical
    across replicates and the intensity profiles can be stacked into a band.
    """
    field = shiftnorm(grid, grid.ifft(spectrum))
    t_over, e_over = oversample(grid.t, field, oversampling)
    intensity = np.abs(e_over) ** 2
    peak = intensity.max()
    if peak > 0:
        intensity = intensity / peak
    return _fwhm(t_over, intensity), t_over, intensity


def _apply_propagation(
    spectrum: NDArray[np.complex128],
    propagation: NDArray[np.complex128] | None,
) -> NDArray[np.complex128]:
    r"""Propagate a retrieved spectrum to a different beamline point.

    Propagation between two beamline points is the deterministic linear map
    :math:`\tilde E'(\omega) = H(\omega)\,\tilde E(\omega)`, with the spectral
    transfer function :math:`H(\omega) = a(\omega)\,e^{i\varphi_\mathrm{prop}(\omega)}`
    (a phase from Taylor/material/gas/free-space dispersion and, optionally, a
    mirror-reflectivity amplitude). Build ``H`` once with
    :func:`croak.session.dispersion.transfer_function` and pass it to the bootstrap
    estimators so every replicate's spectrum is propagated identically before the
    FWHM/band is taken — pushing the *joint* retrieval uncertainty through the same
    map preserves the amplitude↔phase↔frequency correlations a naive ±-bound
    propagation would lose. Returns the spectrum unchanged when ``propagation`` is
    ``None``.
    """
    if propagation is None:
        return spectrum
    return np.asarray(spectrum, dtype=np.complex128) * np.asarray(
        propagation, dtype=np.complex128
    )


# ---------------------------------------------------------------------------
# Interval estimators
# ---------------------------------------------------------------------------
def percentile_interval(samples: ArrayLike, level: float = 0.68) -> tuple[float, float]:
    """Return the central ``level`` percentile interval of ``samples``."""
    samples = np.asarray(samples, dtype=float)
    lo = 100.0 * (1.0 - level) / 2.0
    hi = 100.0 * (1.0 + level) / 2.0
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def bias_corrected_interval(
    samples: ArrayLike, point_estimate: float, level: float = 0.68
) -> tuple[float, float]:
    r"""Return a bias-corrected (BC) bootstrap interval.

    The BC interval shifts the percentile cut-offs by a bias correction
    :math:`z_0 = \Phi^{-1}(\#\{b: \theta^*_b < \hat\theta\}/B)` to account for
    median bias in a skewed bootstrap distribution. The acceleration term of the
    full BCa interval is omitted (``a = 0``) because it requires a jackknife over
    the original data — each point of which is a full retrieval. Falls back to
    :func:`percentile_interval` when the bias
    correction is degenerate (all replicates on one side of the point estimate).

    Parameters
    ----------
    samples : array_like
        Bootstrap replicates of the statistic.
    point_estimate : float
        The full-data estimate :math:`\hat\theta` (the bias-correction reference).
    level : float, optional
        Central probability of the interval (``0.68`` ≈ 1σ).
    """
    samples = np.asarray(samples, dtype=float)
    if samples.size < 2:
        return float("nan"), float("nan")
    prop = float(np.mean(samples < point_estimate))
    if prop <= 0.0 or prop >= 1.0:
        return percentile_interval(samples, level)
    z0 = scipy.stats.norm.ppf(prop)
    alpha = (1.0 - level) / 2.0
    z_lo, z_hi = scipy.stats.norm.ppf(alpha), scipy.stats.norm.ppf(1.0 - alpha)
    p_lo = scipy.stats.norm.cdf(2.0 * z0 + z_lo)
    p_hi = scipy.stats.norm.cdf(2.0 * z0 + z_hi)
    return (
        float(np.percentile(samples, 100.0 * p_lo)),
        float(np.percentile(samples, 100.0 * p_hi)),
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class UncertaintyResult:
    r"""Outcome of an uncertainty estimation: the FWHM distribution and its intervals.

    Attributes
    ----------
    statistic : str
        The quantity bootstrapped (currently always ``"fwhm"``, the temporal intensity
        FWHM in seconds).
    point_estimate : float
        Full-data best-fit FWHM (s) — the value the interval is centred
        reporting around.
    samples : numpy.ndarray
        Per-replicate FWHM (s), converged replicates only.
    interval_68, interval_95 : tuple of float
        Lower/upper bounds (s) of the 68 % and 95 % intervals.
    interval_method : str
        ``"bias-corrected"`` or ``"percentile"``.
    method : str
        ``"parametric"``, ``"delay"``, ``"frequency"`` (statistical bootstrap) or
        ``"thickness"`` (substrate-thickness systematic propagation).
    n_resamples : int
        Number of replicates attempted.
    n_converged : int
        Number kept after the convergence filter (the effective ``B``).
    frog_errors : numpy.ndarray
        Per-replicate trace error ``R`` (converged replicates).
    base_error : float
        Full-data trace error ``R`` (the convergence reference).
    noise_floor : float or None
        Median per-replicate noise floor ``R0`` (Method A only; ``None`` otherwise).
    warm_start : bool
        Whether replicates were warm-started from the full-data solution.
    profiles : numpy.ndarray or None
        ``(n_converged, n_t)`` peak-aligned, oversampled temporal intensities,
        if collected.
    t_profile : numpy.ndarray or None
        The shared oversampled time axis (s) for ``profiles``.
    components : tuple of str or None
        For ``method="combined"``, the methods that were combined (e.g.
        ``("parametric", "thickness")``); ``None`` otherwise.
    propagation_label : str or None
        Human-readable description of the beamline point the FWHM/band was
        evaluated at (e.g. ``"+1.0 mm SiO2"``), when a propagation transfer
        function was applied; ``None`` for the measurement plane.
    thicknesses : numpy.ndarray or None
        Per-replicate substrate thickness drawn from the prior (m), converged
        replicates only (``method="thickness"`` only; ``None`` otherwise). Pairs
        element-wise with ``samples``/``frog_errors`` for the FWHM-vs-thickness and
        error-vs-thickness diagnostics.
    thickness_central : float or None
        Central (measured) substrate thickness :math:`L_0` (m) of the thickness
        prior (``method="thickness"`` only).
    thickness_sigma : float or None
        Standard deviation :math:`\sigma_L` (m) of the measured thickness prior
        (``method="thickness"`` only).
    """

    statistic: str
    point_estimate: float
    samples: NDArray[np.float64]
    interval_68: tuple[float, float]
    interval_95: tuple[float, float]
    interval_method: str
    method: str
    n_resamples: int
    n_converged: int
    frog_errors: NDArray[np.float64]
    base_error: float
    noise_floor: float | None = None
    warm_start: bool = True
    profiles: NDArray[np.float64] | None = None
    t_profile: NDArray[np.float64] | None = None
    thicknesses: NDArray[np.float64] | None = None
    thickness_central: float | None = None
    thickness_sigma: float | None = None
    components: tuple[str, ...] | None = None
    propagation_label: str | None = None

    @property
    def median(self) -> float:
        """Median of the bootstrap FWHM samples (s)."""
        return float(np.median(self.samples)) if self.samples.size else float("nan")

    @property
    def std(self) -> float:
        """Standard deviation of the bootstrap FWHM samples (s)."""
        return float(np.std(self.samples)) if self.samples.size else float("nan")

    @property
    def plus_minus(self) -> float:
        """Symmetric half-width of the 68 % interval (s)."""
        lo, hi = self.interval_68
        return 0.5 * (hi - lo)

    def summary(self) -> str:
        """Return a one-line ``"x ± y fs (…)"`` summary tagged with the error class.

        The caveat differs by ``method``: the statistical bootstraps
        (``parametric``/``delay``/``frequency``) flag that systematics are *excluded*,
        whereas ``thickness`` reports a *systematic* error bar from the measured
        substrate-thickness prior and names that prior.
        """
        fwhm_fs = self.point_estimate / 1e-15
        pm_fs = self.plus_minus / 1e-15
        at = f" at {self.propagation_label}" if self.propagation_label else ""
        if self.method == "combined":
            parts = " ⊕ ".join(self.components) if self.components else "sources"
            return (
                f"FWHM = {fwhm_fs:.2f} ± {pm_fs:.2f} fs{at} "
                f"(1σ combined: {parts}, B={self.n_converged})"
            )
        if self.method == "thickness":
            central_um = (self.thickness_central or 0.0) / 1e-6
            sigma_um = (self.thickness_sigma or 0.0) / 1e-6
            return (
                f"FWHM = {fwhm_fs:.2f} ± {pm_fs:.2f} fs{at} "
                f"(1σ from substrate thickness {central_um:.3f} ± {sigma_um:.3f} µm, "
                f"systematic, B={self.n_converged})"
            )
        if self.method == "covariance":
            return (
                f"FWHM = {fwhm_fs:.2f} ± {pm_fs:.2f} fs{at} "
                f"(1σ linearised covariance, B={self.n_converged} Gaussian draws; "
                "first-order/Laplace approximation, systematics excluded)"
            )
        started = "warm" if self.warm_start else "cold"
        return (
            f"FWHM = {fwhm_fs:.2f} ± {pm_fs:.2f} fs{at} "
            f"(1σ statistical, {self.method}, B={self.n_converged}, {started}-started; "
            f"systematic uncertainty not included)"
        )


# ---------------------------------------------------------------------------
# Assembly (shared by both methods)
# ---------------------------------------------------------------------------
def _assemble(
    *,
    statistic: str,
    point_estimate: float,
    fwhms: list[float],
    errors: list[float],
    profiles: list[NDArray[np.float64]],
    t_profile: NDArray[np.float64] | None,
    method: str,
    n_resamples: int,
    base_error: float,
    noise_floor: float | None,
    warm_start: bool,
    convergence_tol: float,
    interval: str,
    thicknesses: list[float] | None = None,
    propagation_label: str | None = None,
) -> UncertaintyResult:
    """Filter non-converged/NaN replicates and build an :class:`UncertaintyResult`.

    Convergence is judged relative to the *median* replicate error: a replicate
    whose error
    exceeds ``convergence_tol × median`` is treated as a stagnation outlier and dropped.
    This is method-agnostic and robust to the differing error scales of Methods A and B.
    """
    fwhm_arr = np.asarray(fwhms, dtype=float)
    err_arr = np.asarray(errors, dtype=float)
    finite = np.isfinite(fwhm_arr) & np.isfinite(err_arr)
    if not np.any(finite):
        raise RuntimeError(
            "no replicate produced a finite FWHM; check the retrieval setup"
        )
    median_err = float(np.median(err_arr[finite]))
    keep = finite & (err_arr <= convergence_tol * median_err)
    if not np.any(keep):
        keep = finite

    samples = fwhm_arr[keep]
    kept_errors = err_arr[keep]
    kept_profiles = (
        np.stack([profiles[i] for i in np.flatnonzero(keep)])
        if profiles and t_profile is not None
        else None
    )
    kept_thicknesses = (
        np.asarray(thicknesses, dtype=float)[keep] if thicknesses is not None else None
    )

    if interval == "bias-corrected":
        i68 = bias_corrected_interval(samples, point_estimate, 0.68)
        i95 = bias_corrected_interval(samples, point_estimate, 0.95)
    elif interval == "percentile":
        i68 = percentile_interval(samples, 0.68)
        i95 = percentile_interval(samples, 0.95)
    else:
        raise ValueError(
            f"unknown interval {interval!r}; use 'bias-corrected' or 'percentile'"
        )

    return UncertaintyResult(
        statistic=statistic,
        point_estimate=point_estimate,
        samples=samples,
        interval_68=i68,
        interval_95=i95,
        interval_method=interval,
        method=method,
        n_resamples=n_resamples,
        n_converged=int(samples.size),
        frog_errors=kept_errors,
        base_error=base_error,
        noise_floor=noise_floor,
        warm_start=warm_start,
        profiles=kept_profiles,
        t_profile=t_profile,
        thicknesses=kept_thicknesses,
        propagation_label=propagation_label,
    )


def _build_runner(result: RetrievalResult, retrieve_kwargs: dict):
    """Construct the solver once and return a per-replicate ``run`` closure.

    The solver is built directly (mirroring
    :func:`croak.pipeline.retrieve_from_tracedata`) rather than via
    :func:`croak.retrieve`, because the dispersive carrier ``omega0`` must reach the
    solver *constructor* (where the dispersion model is built), not only ``run``.
    ``retrieve_kwargs`` are candidate constructor options (``maxiters``,
    ``material``/``thickness``, ``reg_*``, ``phase_basis`` …); only those the chosen
    solver accepts are forwarded, exactly as the pipeline does. Reusing one solver
    across replicates keeps any per-construction setup out of the loop.
    """
    algo = str(retrieve_kwargs.get("algorithm", result.algorithm)).lower()
    if algo not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algo!r}")
    candidate = {k: v for k, v in retrieve_kwargs.items() if k != "algorithm"}
    candidate.setdefault("omega0", result.omega0)
    accepted = algorithm_params(algo)
    solver = ALGORITHMS[algo](**{k: v for k, v in candidate.items() if k in accepted})
    omega, interaction, omega0 = result.omega, result.interaction, result.omega0

    def run(trace, delays, *, guess, weights=None, rng=None) -> RetrievalResult:
        return solver.run(
            trace,
            omega,
            delays,
            interaction,
            guess=guess,
            weights=weights,
            rng=rng,
            omega0=omega0,
        )

    return run


# ---------------------------------------------------------------------------
# Method A — parametric Monte-Carlo bootstrap
# ---------------------------------------------------------------------------
def parametric_bootstrap(
    result: RetrievalResult,
    *,
    noise: NoiseModel | float,
    n_resamples: int = 300,
    warm_start: bool = True,
    rng: np.random.Generator | None = None,
    oversampling: int = 8,
    convergence_tol: float = 3.0,
    interval: str = "bias-corrected",
    collect_profiles: bool = False,
    propagation: NDArray[np.complex128] | None = None,
    propagation_label: str | None = None,
    callback=None,
    **retrieve_kwargs,
) -> UncertaintyResult:
    r"""Estimate FWHM uncertainty by the parametric Monte-Carlo bootstrap (Method A).

    Simulates ``n_resamples`` synthetic traces from the best-fit clean trace
    (``result.trace``) plus draws from ``noise``, re-retrieves each (warm-started by
    default), and returns the spread of the per-replicate FWHM. Conditions on the fitted
    model as truth, so it yields a *statistical* interval only.

    Parameters
    ----------
    result : RetrievalResult
        A completed retrieval (must carry ``result.trace``).
    noise : NoiseModel or float
        The measurement noise model; a bare float is taken as a
        fraction-of-peak additive
        ``NoiseModel``.
    n_resamples : int, optional
        Number of bootstrap replicates ``B`` (≈200–300 for a 1σ interval).
    warm_start : bool, optional
        Warm-start each replicate from ``result.spectrum`` (recommended — isolates
        measurement variance from optimiser basin-hopping). If false, each
        replicate starts
        from a random guess.
    rng : numpy.random.Generator, optional
        Random generator (defaults to a fresh one).
    oversampling : int, optional
        Temporal oversampling for the FWHM (default 8, matching ``process_result``).
    convergence_tol : float, optional
        Drop replicates whose error exceeds this multiple of the median replicate error.
    interval : {"bias-corrected", "percentile"}, optional
        Interval estimator.
    collect_profiles : bool, optional
        Also collect peak-aligned oversampled intensity profiles (for a
        confidence band).
    propagation : numpy.ndarray of complex, optional
        Spectral transfer function :math:`H(\omega)` (same length as
        ``result.spectrum``) propagating the pulse to a different beamline point;
        each replicate's spectrum is multiplied by it before the FWHM/band is taken
        (see :func:`_apply_propagation`). ``None`` (default) reports at the
        measurement plane.
    propagation_label : str, optional
        Description of the propagation point recorded on the result (e.g.
        ``"+1.0 mm SiO2"``).
    callback : callable, optional
        ``callback(done, total, R)`` after each replicate; raising aborts the run.
    **retrieve_kwargs
        Forwarded to :func:`croak.retrieve` — must reproduce the original
        retrieval settings (``maxiters``, ``material``/``thickness``, ``reg_*``,
        ``phase_basis`` …).

    Returns
    -------
    UncertaintyResult
    """
    if result.trace is None:
        raise ValueError("result.trace is None; run a retrieval that records its trace")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    rng = np.random.default_rng() if rng is None else rng
    nm = _resolve_noise(noise)
    clean = np.asarray(result.trace, dtype=float)
    runner = _build_runner(result, retrieve_kwargs)
    guess = result.spectrum if warm_start else None

    point_estimate, _, _ = _fwhm_and_profile(
        result.grid, _apply_propagation(result.spectrum, propagation), oversampling
    )
    fwhms: list[float] = []
    errors: list[float] = []
    floors: list[float] = []
    profiles: list[NDArray[np.float64]] = []
    t_profile: NDArray[np.float64] | None = None

    for b in range(n_resamples):
        noisy = nm.draw(clean, rng)
        floors.append(frog_error(noisy, clean))
        rb = runner(noisy, result.delays, guess=guess, rng=rng)
        fw, t_over, prof = _fwhm_and_profile(
            rb.grid, _apply_propagation(rb.spectrum, propagation), oversampling
        )
        fwhms.append(fw)
        errors.append(rb.error)
        if collect_profiles:
            profiles.append(prof)
            t_profile = t_over
        if callback is not None:
            callback(b + 1, n_resamples, rb.error)

    return _assemble(
        statistic="fwhm",
        point_estimate=point_estimate,
        fwhms=fwhms,
        errors=errors,
        profiles=profiles,
        t_profile=t_profile,
        method="parametric",
        n_resamples=n_resamples,
        base_error=result.error,
        noise_floor=float(np.median(floors)) if floors else None,
        warm_start=warm_start,
        convergence_tol=convergence_tol,
        interval=interval,
        propagation_label=propagation_label,
    )


# ---------------------------------------------------------------------------
# Method B — data-resampling bootstrap
# ---------------------------------------------------------------------------
def resampling_bootstrap(
    result: RetrievalResult,
    measured: ArrayLike,
    *,
    unit: str = "delay",
    n_resamples: int = 300,
    leave_out: float | None = None,
    warm_start: bool = True,
    rng: np.random.Generator | None = None,
    oversampling: int = 8,
    convergence_tol: float = 3.0,
    interval: str = "bias-corrected",
    collect_profiles: bool = False,
    propagation: NDArray[np.complex128] | None = None,
    propagation_label: str | None = None,
    callback=None,
    **retrieve_kwargs,
) -> UncertaintyResult:
    r"""Estimate FWHM uncertainty by resampling the measured trace (Method B).

    Needs no noise model. Two resampling units are supported:

    * ``unit="delay"`` — resample whole **delay columns** with replacement (a block
      bootstrap; the natural unit for delay-scanned data, where each column is one laser
      exposure). The resampled delays may contain duplicates; they are sorted so
      the delay axis stays monotone.
    * ``unit="frequency"`` — resample **frequency rows** via multinomial pixel counts,
      passed to the weighted retrieval as ``weights = √counts`` (weight √k reproduces a
      multiplicity-``k`` row in the weighted least-squares objective).

    Parameters
    ----------
    result : RetrievalResult
        A completed retrieval (provides the grid, delays, interaction and
        warm-start guess).
    measured : array_like
        The measured trace ``(Nomega, Ndelay)`` to resample.
    unit : {"delay", "frequency"}, optional
        The resampling unit.
    n_resamples : int, optional
        Number of bootstrap replicates ``B``.
    leave_out : float, optional
        If given, instead of resampling with replacement, drop this fraction of columns
        (``unit="delay"``) or rows (``unit="frequency"``) each replicate.
    warm_start, rng, oversampling, convergence_tol, interval, collect_profiles, callback
        As in :func:`parametric_bootstrap`.
    propagation, propagation_label
        As in :func:`parametric_bootstrap`: a spectral transfer function
        :math:`H(\omega)` propagating each replicate to a different beamline point,
        and its description.
    **retrieve_kwargs
        Forwarded to :func:`croak.retrieve` (must reproduce the original settings).

    Returns
    -------
    UncertaintyResult
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if unit not in ("delay", "frequency"):
        raise ValueError(f"unknown unit {unit!r}; use 'delay' or 'frequency'")
    rng = np.random.default_rng() if rng is None else rng
    measured = np.asarray(measured, dtype=float)
    nomega, ndelay = measured.shape
    delays = np.asarray(result.delays, dtype=float)
    runner = _build_runner(result, retrieve_kwargs)
    guess = result.spectrum if warm_start else None

    point_estimate, _, _ = _fwhm_and_profile(
        result.grid, _apply_propagation(result.spectrum, propagation), oversampling
    )
    fwhms: list[float] = []
    errors: list[float] = []
    profiles: list[NDArray[np.float64]] = []
    t_profile: NDArray[np.float64] | None = None

    for b in range(n_resamples):
        if unit == "delay":
            if leave_out is not None:
                n_keep = max(2, int(round((1.0 - leave_out) * ndelay)))
                idx = np.sort(rng.choice(ndelay, size=n_keep, replace=False))
            else:
                idx = np.sort(rng.integers(0, ndelay, size=ndelay))
            rb = runner(measured[:, idx], delays[idx], guess=guess, rng=rng)
        else:  # frequency
            if leave_out is not None:
                weights = np.ones(nomega)
                drop = rng.choice(
                    nomega, size=int(round(leave_out * nomega)), replace=False
                )
                weights[drop] = 0.0
            else:
                counts = rng.multinomial(nomega, np.full(nomega, 1.0 / nomega))
                weights = np.sqrt(counts.astype(float))
            rb = runner(measured, delays, guess=guess, weights=weights, rng=rng)
        fw, t_over, prof = _fwhm_and_profile(
            rb.grid, _apply_propagation(rb.spectrum, propagation), oversampling
        )
        fwhms.append(fw)
        errors.append(rb.error)
        if collect_profiles:
            profiles.append(prof)
            t_profile = t_over
        if callback is not None:
            callback(b + 1, n_resamples, rb.error)

    return _assemble(
        statistic="fwhm",
        point_estimate=point_estimate,
        fwhms=fwhms,
        errors=errors,
        profiles=profiles,
        t_profile=t_profile,
        method=unit,
        n_resamples=n_resamples,
        base_error=result.error,
        noise_floor=None,
        warm_start=warm_start,
        convergence_tol=convergence_tol,
        interval=interval,
        propagation_label=propagation_label,
    )


# ---------------------------------------------------------------------------
# Substrate-thickness systematic — brute-force Monte-Carlo over the prior
# ---------------------------------------------------------------------------
def thickness_bootstrap(
    result: RetrievalResult,
    measured: ArrayLike,
    *,
    central_thickness: float,
    thickness_sigma: float,
    material: str,
    n_resamples: int = 300,
    warm_start: bool = True,
    rng: np.random.Generator | None = None,
    oversampling: int = 8,
    convergence_tol: float = 1e9,
    interval: str = "bias-corrected",
    collect_profiles: bool = False,
    clip_min: float = 0.0,
    propagation: NDArray[np.complex128] | None = None,
    propagation_label: str | None = None,
    callback=None,
    **retrieve_kwargs,
) -> UncertaintyResult:
    r"""Propagate the measured substrate-thickness uncertainty onto the FWHM.

    A dispersive retrieval fits the trace through a forward model that propagates the
    field through a substrate of an *assumed* thickness :math:`L`; the retrieved field
    — and hence its temporal-intensity FWHM — depends on :math:`L`. When the substrate
    thickness is an independently measured input with a Gaussian prior
    :math:`L \sim \mathcal N(L_0, \sigma_L)` (e.g. ``9.952 ± 0.539`` µm), the FWHM
    inherits a *systematic* uncertainty that the statistical bootstraps
    (:func:`parametric_bootstrap`, :func:`resampling_bootstrap`) cannot see — every one
    of their replicates reuses the *same* fixed thickness.

    This estimator marginalises that prior by brute-force Monte-Carlo: it draws
    ``n_resamples`` thicknesses from :math:`\mathcal N(L_0, \sigma_L)` (clipped at
    ``clip_min``), rebuilds the dispersive forward model at each, **re-retrieves the
    same measured trace** (warm-started from the central-thickness solution), and
    returns the spread of the per-draw FWHM. It is the systematic complement to the
    statistical bootstrap; the two are independent and combine in quadrature.

    Because thickness is a *solver-constructor* parameter (the dispersion model is
    built in :class:`~croak.forward.ForwardModel`), the solver is rebuilt for each draw
    via :func:`_build_runner` — cheap next to the retrieval iterations.

    Parameters
    ----------
    result : RetrievalResult
        A completed dispersive retrieval (provides the grid, delays, interaction,
        carrier ``omega0`` and the warm-start guess). Its ``algorithm`` and the
        forwarded ``retrieve_kwargs`` must reproduce the original settings.
    measured : array_like
        The measured trace ``(Nomega, Ndelay)`` to re-fit at each drawn thickness
        (the data are held fixed; only the assumed thickness varies).
    central_thickness : float
        Central (measured) substrate thickness :math:`L_0` in **metres**.
    thickness_sigma : float
        Standard deviation :math:`\sigma_L` of the thickness prior in **metres**
        (``>= 0``; ``0`` gives a degenerate, zero-width distribution).
    material : str
        Substrate material name (e.g. ``"SiO2-Franta"``), as accepted by
        :func:`croak.materials.refractive_index`.
    n_resamples : int, optional
        Number of Monte-Carlo draws ``B`` (≈200–300 for a 1σ interval).
    warm_start : bool, optional
        Warm-start each draw from the central-thickness solution (recommended:
        isolates the thickness-induced movement from optimiser basin-hopping). If
        false, each draw starts from a random guess.
    rng : numpy.random.Generator, optional
        Random generator (defaults to a fresh one).
    oversampling : int, optional
        Temporal oversampling for the FWHM (default 8, matching ``process_result``).
    convergence_tol : float, optional
        Drop draws whose error exceeds this multiple of the median draw error.
        Defaults to a large value (effectively *disabled*): unlike the statistical
        bootstrap, an off-nominal thickness fits the trace *genuinely* worse, so a
        raised :math:`R` in the tails is physical misfit — not stagnation — and must
        not be filtered out. Only non-finite (failed) draws are dropped.
    interval : {"bias-corrected", "percentile"}, optional
        Interval estimator.
    collect_profiles : bool, optional
        Also collect peak-aligned oversampled intensity profiles (for a temporal band).
    clip_min : float, optional
        Lower clip (m) for drawn thicknesses (default ``0.0``; thickness cannot be
        negative).
    propagation, propagation_label
        As in :func:`parametric_bootstrap`: a spectral transfer function
        :math:`H(\omega)` propagating each draw to a different beamline point, and
        its description. Stacks with the substrate dispersion the retrieval already
        models (the draw is fit at the measurement plane, then propagated).
    callback : callable, optional
        ``callback(done, total, R)`` after each draw; raising aborts the run.
    **retrieve_kwargs
        Forwarded to the solver constructor (``maxiters``, ``npoints``, ``reg_*``,
        ``phase_basis`` …), exactly reproducing the original retrieval. Any
        ``material``/``thickness`` here are ignored — they are injected per draw.

    Returns
    -------
    UncertaintyResult
        With ``method="thickness"``, ``thicknesses`` the per-draw thickness (m), and
        ``thickness_central``/``thickness_sigma`` recording the prior.

    Raises
    ------
    ValueError
        If ``thickness_sigma < 0``, ``n_resamples < 1`` or ``material`` is falsy.
    """
    if thickness_sigma < 0:
        raise ValueError("thickness_sigma must be >= 0")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not material:
        raise ValueError("thickness_bootstrap requires a dispersive `material`")
    rng = np.random.default_rng() if rng is None else rng
    measured = np.asarray(measured, dtype=float)
    delays = np.asarray(result.delays, dtype=float)
    # Thickness/material are injected per draw, so drop any that came in via the
    # reproduced retrieval settings to avoid duplicate constructor keywords.
    base_kwargs = {
        k: v for k, v in retrieve_kwargs.items() if k not in ("material", "thickness")
    }

    # Establish the centre: retrieve once at L0 to define both the reported point
    # estimate and the warm-start guess shared by every draw.
    runner0 = _build_runner(
        result, {**base_kwargs, "material": material, "thickness": central_thickness}
    )
    central_guess = result.spectrum if warm_start else None
    base = runner0(measured, delays, guess=central_guess, rng=rng)
    point_estimate, _, _ = _fwhm_and_profile(
        base.grid, _apply_propagation(base.spectrum, propagation), oversampling
    )
    guess = base.spectrum if warm_start else None

    fwhms: list[float] = []
    errors: list[float] = []
    thicknesses: list[float] = []
    profiles: list[NDArray[np.float64]] = []
    t_profile: NDArray[np.float64] | None = None

    for b in range(n_resamples):
        thickness_b = float(
            max(clip_min, rng.normal(central_thickness, thickness_sigma))
        )
        runner = _build_runner(
            result, {**base_kwargs, "material": material, "thickness": thickness_b}
        )
        rb = runner(measured, delays, guess=guess, rng=rng)
        fw, t_over, prof = _fwhm_and_profile(
            rb.grid, _apply_propagation(rb.spectrum, propagation), oversampling
        )
        fwhms.append(fw)
        errors.append(rb.error)
        thicknesses.append(thickness_b)
        if collect_profiles:
            profiles.append(prof)
            t_profile = t_over
        if callback is not None:
            callback(b + 1, n_resamples, rb.error)

    uresult = _assemble(
        statistic="fwhm",
        point_estimate=point_estimate,
        fwhms=fwhms,
        errors=errors,
        profiles=profiles,
        t_profile=t_profile,
        method="thickness",
        n_resamples=n_resamples,
        base_error=base.error,
        noise_floor=None,
        warm_start=warm_start,
        convergence_tol=convergence_tol,
        interval=interval,
        thicknesses=thicknesses,
        propagation_label=propagation_label,
    )
    uresult.thickness_central = float(central_thickness)
    uresult.thickness_sigma = float(thickness_sigma)
    return uresult


def estimate_fwhm_uncertainty(
    result: RetrievalResult,
    measured: ArrayLike | None = None,
    *,
    method: str = "parametric",
    noise: NoiseModel | float | None = None,
    **kwargs,
) -> UncertaintyResult:
    """Estimate FWHM uncertainty, dispatching to Method A or Method B.

    Parameters
    ----------
    result : RetrievalResult
        A completed retrieval.
    measured : array_like, optional
        The measured trace. Required for ``method`` in ``{"delay", "frequency"}``;
        ignored by ``"parametric"`` (which uses ``result.trace``).
    method : {"parametric", "delay", "frequency", "thickness"}, optional
        ``"parametric"`` → :func:`parametric_bootstrap`; ``"delay"``/``"frequency"`` →
        :func:`resampling_bootstrap`; ``"thickness"`` → :func:`thickness_bootstrap`
        (the substrate-thickness *systematic*, not a statistical bootstrap).
    noise : NoiseModel or float, optional
        Noise model for the parametric method. If omitted there and ``measured``
        is given, it is estimated from the residual via :func:`noise_from_residual`.
        Ignored by the other methods.
    **kwargs
        Forwarded to the chosen function. For ``method="thickness"`` this must include
        ``central_thickness``, ``thickness_sigma`` and ``material``. Pass
        ``propagation`` (a spectral transfer function ``H(ω)``) and
        ``propagation_label`` to report the FWHM/band at a different beamline point
        (see :func:`_apply_propagation`).

    Returns
    -------
    UncertaintyResult

    Examples
    --------
    >>> import numpy as np, croak
    >>> g = croak.Grid(64, dt=0.5e-15)
    >>> ew = croak.gaussian_pulse(g, 6e-15)
    >>> delays = np.linspace(-40e-15, 40e-15, 40)
    >>> trace = croak.maketrace(g.omega, delays, ew, "pg")
    >>> res = croak.retrieve(trace, g.omega, delays, "pg", guess=ew, progress=False)
    >>> u = croak.estimate_fwhm_uncertainty(
    ...     res, trace, method="parametric", noise=0.01,
    ...     n_resamples=20, maxiters=30, rng=np.random.default_rng(0))
    >>> u.point_estimate > 0
    True
    """
    if method == "parametric":
        if noise is None:
            if measured is None:
                raise ValueError(
                    "parametric method needs a `noise` model, "
                    "or `measured` to estimate one"
                )
            noise = noise_from_residual(measured, result)
        return parametric_bootstrap(result, noise=noise, **kwargs)
    if method in ("delay", "frequency"):
        if measured is None:
            raise ValueError(f"method {method!r} requires the measured trace")
        return resampling_bootstrap(result, measured, unit=method, **kwargs)
    if method == "thickness":
        if measured is None:
            raise ValueError("method 'thickness' requires the measured trace")
        return thickness_bootstrap(result, measured, **kwargs)
    raise ValueError(
        f"unknown method {method!r}; "
        "use 'parametric', 'delay', 'frequency' or 'thickness'"
    )


# ---------------------------------------------------------------------------
# Combining independent uncertainty contributions
# ---------------------------------------------------------------------------
def combine_uncertainties(
    results: Sequence[UncertaintyResult],
    *,
    interval: str = "bias-corrected",
    n_samples: int = 2000,
    rng: np.random.Generator | None = None,
) -> UncertaintyResult:
    r"""Combine *independent* FWHM uncertainty contributions into one error bar.

    Treats each input as an independent source of FWHM deviation (e.g. a statistical
    bootstrap **and** the substrate-thickness systematic) and forms the total by
    **convolving** their centred sample distributions: a Monte-Carlo draw of the total
    deviation is the sum of one independently-drawn deviation from each source. This
    preserves skew and reduces to the familiar quadrature
    :math:`\sigma_\text{tot}=\sqrt{\sum_i\sigma_i^2}` in the Gaussian limit. If every
    input carries a temporal-intensity band (``profiles`` on a common time axis), the
    bands are combined the same way, per time sample.

    .. warning::
       Only combine sources that estimate *different*, independent contributions.
       ``parametric``/``delay``/``frequency`` all estimate the **same** statistical
       precision — combining them double-counts it. A sound budget combines *one*
       statistical estimate with each independent *systematic* (e.g. ``thickness``).

    Parameters
    ----------
    results : sequence of UncertaintyResult
        Two or more independent contributions (each must carry ``samples``).
    interval : {"bias-corrected", "percentile"}, optional
        Interval estimator applied to the combined samples.
    n_samples : int, optional
        Number of Monte-Carlo draws of the combined distribution.
    rng : numpy.random.Generator, optional
        Random generator (defaults to a fresh one).

    Returns
    -------
    UncertaintyResult
        With ``method="combined"``, ``components`` listing the input methods, the
        combined ``samples`` and 68 %/95 % intervals, and the combined temporal band
        in ``profiles``/``t_profile`` when all inputs carried one.

    Raises
    ------
    ValueError
        If fewer than two results are given, or none carries any samples.
    """
    results = list(results)
    if len(results) < 2:
        raise ValueError("combine_uncertainties needs at least two results")
    rng = np.random.default_rng() if rng is None else rng

    point_estimate = float(np.mean([r.point_estimate for r in results]))

    # FWHM: total deviation = sum of one independent draw from each source's deviations.
    combined = np.full(n_samples, point_estimate, dtype=float)
    used = 0
    for r in results:
        samples = np.asarray(r.samples, dtype=float)
        if samples.size == 0:
            continue
        combined += rng.choice(samples - r.point_estimate, size=n_samples, replace=True)
        used += 1
    if used == 0:
        raise ValueError("no input carried any samples to combine")

    if interval == "bias-corrected":
        i68 = bias_corrected_interval(combined, point_estimate, 0.68)
        i95 = bias_corrected_interval(combined, point_estimate, 0.95)
    elif interval == "percentile":
        i68 = percentile_interval(combined, 0.68)
        i95 = percentile_interval(combined, 0.95)
    else:
        raise ValueError(
            f"unknown interval {interval!r}; use 'bias-corrected' or 'percentile'"
        )

    # Temporal band: combine per-time deviations when every input has a band on the
    # same oversampled time axis (they are peak-aligned, so the medians coincide).
    profiles: NDArray[np.float64] | None = None
    t_profile: NDArray[np.float64] | None = None
    have_bands = all(
        r.profiles is not None and r.t_profile is not None for r in results
    )
    if have_bands:
        t0 = np.asarray(results[0].t_profile, dtype=float)
        if all(
            r.t_profile is not None
            and np.array_equal(np.asarray(r.t_profile, dtype=float), t0)
            for r in results
        ):
            t_profile = t0
            medians = [np.median(np.asarray(r.profiles), axis=0) for r in results]
            base = np.mean(np.stack(medians), axis=0)
            stack = np.tile(base, (n_samples, 1))
            for r, med in zip(results, medians, strict=True):
                prof = np.asarray(r.profiles, dtype=float)
                idx = rng.integers(0, prof.shape[0], size=n_samples)
                stack += prof[idx] - med
            profiles = np.clip(stack, 0.0, None)

    return UncertaintyResult(
        statistic="fwhm",
        point_estimate=point_estimate,
        samples=combined,
        interval_68=i68,
        interval_95=i95,
        interval_method=interval,
        method="combined",
        n_resamples=n_samples,
        n_converged=int(combined.size),
        frog_errors=np.empty(0, dtype=float),
        base_error=float(np.mean([r.base_error for r in results])),
        warm_start=all(r.warm_start for r in results),
        profiles=profiles,
        t_profile=t_profile,
        components=tuple(r.method for r in results),
    )


# ---------------------------------------------------------------------------
# Coverage calibration
# ---------------------------------------------------------------------------
def synthetic_experiment(
    *,
    fwhm: float,
    phases: ArrayLike = (),
    shape: str = "gaussian",
    n: int = 128,
    dt: float = 0.4e-15,
    lambda0: float = 800e-9,
    ndelay: int = 80,
    delay_max: float = 60e-15,
    interaction: str = "pg",
    oversampling: int = 8,
) -> dict:
    """Build a synthetic experiment with a known true FWHM, from public primitives.

    Returns a dict with ``grid``, ``ew`` (true spectrum), ``omega0``, ``delays``,
    ``trace``
    (clean), ``interaction`` and ``true_fwhm`` (s). Mirrors the test fixture
    ``make_synthetic_experiment`` but uses only the public API so it can drive
    :func:`coverage_calibration`.

    Parameters
    ----------
    fwhm : float
        Intensity FWHM of the transform-limited pulse (s).
    phases : sequence of float, optional
        Taylor spectral-phase coefficients ``[GDD, TOD, …]`` (s², s³, …).
    shape : {"gaussian", "sech"}, optional
        Pulse envelope shape.
    n, dt, lambda0, ndelay, delay_max, interaction, oversampling
        Grid, axis and forward-model parameters.
    """
    grid = Grid(n, dt=dt)
    omega0 = float(wlfreq(lambda0))
    builder = {"gaussian": gaussian_pulse, "sech": sech_pulse}[shape]
    ew = builder(grid, fwhm, phases=phases)
    delays = np.linspace(-delay_max, delay_max, ndelay)
    trace = maketrace(grid.omega, delays, ew, interaction)
    true_fwhm, _, _ = _fwhm_and_profile(grid, ew, oversampling)
    return {
        "grid": grid,
        "ew": ew,
        "omega0": omega0,
        "delays": delays,
        "trace": trace,
        "interaction": interaction,
        "true_fwhm": true_fwhm,
    }


@dataclass
class CoverageResult:
    """Empirical coverage of a nominal bootstrap interval on synthetic ground truth."""

    level: float
    n_trials: int
    n_resamples: int
    method: str
    coverage: float
    mean_bias: float
    biases: NDArray[np.float64]
    covered: NDArray[np.bool_]
    true_fwhms: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))

    def summary(self) -> str:
        """Return a one-line coverage summary."""
        return (
            f"coverage {self.coverage:.0%} of nominal {self.level:.0%} "
            f"({self.method}, {self.n_trials} trials, B={self.n_resamples}); "
            f"median-bias {self.mean_bias / 1e-15:+.2f} fs"
        )


def coverage_calibration(
    *,
    noise: NoiseModel | float,
    n_trials: int = 50,
    n_resamples: int = 200,
    level: float = 0.68,
    method: str = "parametric",
    interval: str = "bias-corrected",
    fwhm_range: tuple[float, float] = (4e-15, 10e-15),
    gdd_range: tuple[float, float] = (0.0, 40e-30),
    shapes: tuple[str, ...] = ("gaussian",),
    algorithm: str = "warm-lbfgs",
    guess_fwhm: float | None = None,
    rng: np.random.Generator | None = None,
    callback=None,
    **retrieve_kwargs,
) -> CoverageResult:
    r"""Validate interval coverage on synthetic experiments with known FWHM.

    For each trial, draws a random true pulse (FWHM, GDD, shape), synthesises
    and noises its
    trace, retrieves it, runs the bootstrap, and records whether the nominal-``level``
    interval contains the true FWHM. Converts a "spread" into a calibrated
    interval: if the
    nominal 68 % interval covers ≈68 % of trials, it is calibrated.

    Parameters
    ----------
    noise : NoiseModel or float
        Noise injected into each synthetic trace (and used by the parametric method).
    n_trials : int, optional
        Number of synthetic experiments.
    n_resamples : int, optional
        Bootstrap replicates per trial.
    level : float, optional
        Nominal interval probability to test (e.g. ``0.68``).
    method : {"parametric", "delay", "frequency"}, optional
        Bootstrap method to validate.
    interval : {"bias-corrected", "percentile"}, optional
        Interval estimator (applied to the bootstrap samples at the requested
        ``level``).
    fwhm_range, gdd_range : tuple of float, optional
        Uniform sampling ranges for the true FWHM (s) and GDD (s²).
    shapes : tuple of str, optional
        Pulse shapes to sample from.
    algorithm : str, optional
        Retrieval algorithm for each trial.
    guess_fwhm : float, optional
        FWHM of the transform-limited warm-start guess (defaults to the midpoint of
        ``fwhm_range``).
    rng : numpy.random.Generator, optional
        Random generator.
    callback : callable, optional
        ``callback(trial, n_trials)`` after each trial.
    **retrieve_kwargs
        Forwarded to :func:`croak.retrieve` and the bootstrap (e.g. ``maxiters``).

    Returns
    -------
    CoverageResult
    """
    rng = np.random.default_rng() if rng is None else rng
    nm = _resolve_noise(noise)
    guess_fwhm = guess_fwhm if guess_fwhm is not None else float(np.mean(fwhm_range))

    covered: list[bool] = []
    biases: list[float] = []
    truths: list[float] = []
    for trial in range(n_trials):
        fwhm = float(rng.uniform(*fwhm_range))
        gdd = float(rng.uniform(*gdd_range))
        shape = str(rng.choice(shapes))
        exp = synthetic_experiment(fwhm=fwhm, phases=(gdd,), shape=shape)
        noisy = nm.draw(exp["trace"], rng)

        guess = gaussian_pulse(exp["grid"], guess_fwhm)
        base = retrieve(
            noisy,
            exp["grid"].omega,
            exp["delays"],
            exp["interaction"],
            algorithm=algorithm,
            guess=guess,
            omega0=exp["omega0"],
            rng=rng,
            progress=False,
            **retrieve_kwargs,
        )
        u = estimate_fwhm_uncertainty(
            base,
            noisy,
            method=method,
            noise=nm,
            n_resamples=n_resamples,
            interval=interval,
            rng=rng,
            **retrieve_kwargs,
        )
        if u.samples.size:
            if interval == "bias-corrected":
                lo, hi = bias_corrected_interval(u.samples, u.point_estimate, level)
            else:
                lo, hi = percentile_interval(u.samples, level)
            covered.append(bool(lo <= exp["true_fwhm"] <= hi))
            biases.append(float(np.median(u.samples)) - exp["true_fwhm"])
        else:
            covered.append(False)
            biases.append(float("nan"))
        truths.append(exp["true_fwhm"])
        if callback is not None:
            callback(trial + 1, n_trials)

    covered_arr = np.asarray(covered, dtype=bool)
    biases_arr = np.asarray(biases, dtype=float)
    return CoverageResult(
        level=level,
        n_trials=n_trials,
        n_resamples=n_resamples,
        method=method,
        coverage=float(np.mean(covered_arr)) if covered_arr.size else float("nan"),
        mean_bias=float(np.nanmedian(biases_arr)) if biases_arr.size else float("nan"),
        biases=biases_arr,
        covered=covered_arr,
        true_fwhms=np.asarray(truths, dtype=float),
    )
