r"""Linearised (Gauss–Newton/Laplace) covariance of a retrieved pulse.

Pulse retrieval is a nonlinear least-squares fit: a damped Gauss–Newton /
Levenberg–Marquardt solver (:class:`croak.lm.LM`, ``lm-optx``) drives the flattened
FROG residual :math:`r(u)` to a minimum :math:`u^\*`, where ``u`` is the real
parameter vector (spectral amplitude+phase, or B-spline phase coefficients) of
:mod:`croak._jax_pulse`. At that optimum the standard nonlinear-least-squares
covariance applies:

.. math::

    \mathrm{cov}(u) \approx \hat\sigma^2\,(J^\mathsf{T}J)^{+},
    \qquad J = \left.\frac{\partial r}{\partial u}\right|_{u^\*},

the Gauss–Newton approximation to the inverse Hessian. With the residual *whitened*
by a :class:`~croak.uncertainty.NoiseModel` (each pixel divided by its noise σ) the
scale :math:`\hat\sigma^2 = 1` and ``cov(u) = (JᵀJ)⁺`` directly; without a noise
model it falls back to the internal reduced-χ² estimate
:math:`\hat\sigma^2 = \lVert r(u^\*)\rVert^2/(m-n)`.

This is the **fast, analytic** complement to the resampling bootstrap in
:mod:`croak.uncertainty`: one Jacobian and one eigensolve instead of hundreds of
re-retrievals. Two subtleties make it rigorous:

* **Gauge degeneracy.** FROG's continuous trivial ambiguities — the absolute phase
  (CEP) and the absolute timing (a linear spectral phase) — are *flat directions*
  of the objective, so :math:`J^\mathsf{T}J` is rank-deficient. We use the
  pseudo-inverse (eigenvalues below ``rcond`` × the largest are dropped); the null
  space carries no uncertainty for any **gauge-invariant** quantity (FWHM,
  :math:`|\tilde E(\omega)|^2`, the temporal intensity), whose gradient is
  orthogonal to it.
* **Linearisation.** It is a first-order (Laplace) Gaussian approximation about
  :math:`u^\*`. We propagate it by drawing parameter samples
  :math:`u_k\sim\mathcal N(u^\*,\mathrm{cov}(u))` and pushing each through the exact
  (nonlinear) field → FWHM map, which captures the curvature of that map under the
  Gaussian parameter posterior. It is fast and excellent for well-constrained,
  high-SNR retrievals; near the transform limit or at low SNR the full
  :mod:`croak.uncertainty` bootstrap is the reference and this is the cross-check.

Because the covariance is rebuilt from the retrieved spectrum and the JAX forward
model, it works for *any* result, not only one produced by an LM solver — the
least-squares algorithm only matters for whether the Jacobian is exposed natively.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# forward_jax has now set float64; safe to bind the JAX namespaces.
import jax  # noqa: E402 - deliberately after the forward_jax import that sets x64
import jax.numpy as jnp  # noqa: E402 - deliberately after that same import
import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import metrics_jax as mj
from ._jax_pulse import ExtraParamSpec, augment, ls_parameterisation
from .forward_jax import make_param_trace_fn  # enables jax_enable_x64 at import
from .result import RetrievalResult
from .smearing import SmearingKernel
from .uncertainty import (
    NoiseModel,
    UncertaintyResult,
    _apply_propagation,
    _assemble,
    _fwhm_and_profile,
    _resolve_noise,
)

__all__ = [
    "CovarianceResult",
    "parameter_covariance",
    "covariance_uncertainty",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class CovarianceResult:
    r"""Linearised parameter covariance of a least-squares retrieval.

    Attributes
    ----------
    cov : numpy.ndarray
        Parameter covariance :math:`\mathrm{cov}(u)` (``n_params × n_params``),
        the pseudo-inverse Gauss–Newton estimate :math:`\hat\sigma^2 (J^\mathsf{T}J)^+`.
    u : numpy.ndarray
        The parameters :math:`u^\*` (the retrieved spectrum in the solver's
        parameterisation) the covariance is taken about.
    n_params : int
        Number of free parameters ``n`` (length of ``u``).
    n_residuals : int
        Number of residual rows ``m`` (the FROG block, ``Nomega × Ndelay``).
    rank : int
        Numerical rank of :math:`J^\mathsf{T}J` (eigenvalues above ``rcond`` × max).
    null_dim : int
        Dimension of the gauge/flat null space, ``n_params - rank`` (≈ 2 for a
        pointwise full/phase parameterisation: CEP + absolute timing).
    reduced_chisq : float
        :math:`\lVert r(u^\*)\rVert^2/(m-n)`. When a noise model whitened the
        residual this is the reduced χ² (≈ 1 for a good fit and correct noise);
        otherwise it is the internal variance scale applied to the covariance.
    noise_scaled : bool
        ``True`` when ``cov`` was scaled by the internal reduced-χ² estimate (no
        noise model given); ``False`` when an absolute noise model set
        :math:`\hat\sigma^2 = 1`.
    sigma_thickness : float or None
        Standard error of the fitted dispersive-slab thickness (m), when
        ``fit_thickness`` was set; ``None`` otherwise. Already converted from the
        internal dimensionless offset to physical metres.
    sigma_tau0 : float or None
        Standard error of the fitted delay-zero offset (s), when ``fit_tau0`` was
        set; ``None`` otherwise.
    sigma_smear : float or None
        Standard error of the fitted geometric-smearing multiplier (dimensionless),
        when ``fit_smearing`` was set; ``None`` otherwise. For a split fit
        (``fit_smearing_split``) this is the gate-shape (``p``) channel.
    sigma_smear_delta : float or None
        Standard error of the delay-channel multiplier of a split smearing fit,
        when ``fit_smearing_split`` was set; ``None`` otherwise.
    """

    cov: NDArray[np.float64]
    u: NDArray[np.float64]
    n_params: int
    n_residuals: int
    rank: int
    null_dim: int
    reduced_chisq: float
    noise_scaled: bool
    sigma_thickness: float | None = None
    sigma_tau0: float | None = None
    sigma_smear: float | None = None
    sigma_smear_delta: float | None = None

    @property
    def stderr(self) -> NDArray[np.float64]:
        r"""Per-parameter standard error :math:`\sqrt{\mathrm{diag}\,\mathrm{cov}}`."""
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, None))


# ---------------------------------------------------------------------------
# Eigen-decomposed covariance core (shared by the two public entry points)
# ---------------------------------------------------------------------------
@dataclass
class _CovCore:
    """Internal covariance bundle: the eigenfactorisation plus the spectrum map."""

    u0: NDArray[np.float64]
    spectrum_fn: Callable[[jnp.ndarray], jnp.ndarray]  # u_pulse -> complex spectrum
    evecs: NDArray[np.float64]
    mode_var: NDArray[np.float64]  # per-eigenmode variance (0 on dropped gauge modes)
    rank: int
    n_params: int
    n_residuals: int
    reduced_chisq: float
    noise_scaled: bool
    #: Length of the pulse block (``u0[:n_pulse]`` feeds ``spectrum_fn``).
    n_pulse: int = 0
    #: Index / physical scale of the thickness offset (``None`` when not fitted).
    idx_thickness: int | None = None
    scale_thickness: float = 0.0
    #: Index / physical scale of the delay-zero offset (``None`` when not fitted).
    idx_tau0: int | None = None
    scale_tau: float = 0.0
    #: Index / scale of the smearing-multiplier offset (``None`` when not fitted).
    idx_smear: int | None = None
    scale_smear: float = 0.0
    #: Index of the delay-channel scale offset (``None`` unless split-fitted).
    idx_smear_delta: int | None = None

    def covariance(self) -> NDArray[np.float64]:
        r"""Assemble :math:`\mathrm{cov}(u) = V\,\mathrm{diag}(\text{mode\_var})\,V^\mathsf{T}`."""  # noqa: E501 — single LaTeX expression
        return (self.evecs * self.mode_var) @ self.evecs.T

    def _sigma(self, idx: int | None, scale: float) -> float | None:
        """Physical σ of an extra parameter from the covariance diagonal."""
        if idx is None:
            return None
        cov = self.covariance()
        return float(np.sqrt(max(cov[idx, idx], 0.0)) * scale)

    def sigma_thickness(self) -> float | None:
        """Return the fitted thickness standard error (m), or ``None``."""
        return self._sigma(self.idx_thickness, self.scale_thickness)

    def sigma_tau0(self) -> float | None:
        """Return the fitted delay-zero standard error (s), or ``None``."""
        return self._sigma(self.idx_tau0, self.scale_tau)

    def sigma_smear(self) -> float | None:
        """Return the fitted smearing-multiplier standard error, or ``None``."""
        return self._sigma(self.idx_smear, self.scale_smear)

    def sigma_smear_delta(self) -> float | None:
        """Return the split delay-channel multiplier standard error, or ``None``."""
        return self._sigma(self.idx_smear_delta, self.scale_smear)


def _compute_core(
    result: RetrievalResult,
    measured: ArrayLike,
    *,
    noise: NoiseModel | float | None,
    weights: NDArray[np.float64] | None,
    material: str | None,
    thickness: float,
    npoints: int,
    omega0: float | None,
    quadrature: str,
    phase_only: bool,
    phase_basis: str,
    n_nodes: int,
    R_omega: bool,
    rcond: float,
    fit_thickness: bool = False,
    fit_tau0: bool = False,
    smearing: SmearingKernel | None = None,
    fit_smearing: bool = False,
    fit_smearing_split: bool = False,
) -> _CovCore:
    """Build the Jacobian, eigen-decompose ``JᵀJ`` and form the covariance modes.

    When ``fit_thickness``/``fit_tau0``/``fit_smearing`` are set the parameter vector
    is augmented with the same dimensionless offsets the AD retrievers
    use (:func:`croak._jax_pulse.augment`), re-linearised about the *fitted*
    values on ``result``, so the covariance includes their uncertainty and their
    correlation with the pulse.
    """
    grid = result.grid
    t_meas = np.asarray(measured, dtype=float)
    nomega = t_meas.shape[0]
    w = np.ones(nomega) if weights is None else np.asarray(weights, dtype=float)
    car = result.omega0 if omega0 is None else float(omega0)

    # Linearise about the fitted extras (fall back to the passed central values).
    central_thickness = (
        float(result.thickness)
        if fit_thickness and result.thickness is not None
        else float(thickness)
    )
    trace_param = make_param_trace_fn(
        grid.omega,
        result.delays,
        result.interaction,
        material=material,
        thickness=central_thickness,
        npoints=npoints,
        omega0=car,
        quadrature=quadrature,
        smearing=smearing,
        fit_thickness=fit_thickness,
        fit_tau0=fit_tau0,
        fit_smearing=fit_smearing,
    )
    spectrum_fn, u0_pulse, _nidcs, _phase_only = ls_parameterisation(
        result.spectrum,
        grid.omega,
        phase_only=phase_only,
        phase_basis=phase_basis,
        n_nodes=n_nodes,
    )
    delays = np.asarray(result.delays, dtype=float)
    scale_tau = float(np.median(np.abs(np.diff(delays)))) if delays.size > 1 else 1.0
    spec = ExtraParamSpec(
        fit_thickness=fit_thickness,
        thickness0=central_thickness,
        fit_tau0=fit_tau0,
        tau0_0=float(result.tau0),
        scale_tau=scale_tau,
        fit_smearing=fit_smearing,
        smear_scale0=(
            float(result.smear_scale)
            if fit_smearing and result.smear_scale is not None
            else 1.0
        ),
        fit_smearing_split=fit_smearing_split,
        smear_delta0=(
            float(result.smear_scale_delta)
            if fit_smearing_split and result.smear_scale_delta is not None
            else 1.0
        ),
    )
    aug = augment(u0_pulse, spec)
    u0 = aug.u0

    tm = jnp.asarray(t_meas)
    wj = jnp.asarray(w)

    # Whiten by the per-pixel noise σ when a noise model is given, so that
    # cov = (JᵀJ)⁺ needs no extra scale (absolute_sigma); else use the solver's
    # normalised FROG residual (‖r‖ = R) and scale by the internal reduced χ².
    if noise is not None:
        nm = _resolve_noise(noise)
        clean = np.asarray(
            result.trace if result.trace is not None else t_meas, dtype=float
        )
        sigma = nm.std(clean)
        floor = 1e-12 * float(np.max(sigma)) if np.max(sigma) > 0 else 1e-30
        sig = jnp.asarray(np.maximum(sigma, floor))

        def residual(u: jnp.ndarray) -> jnp.ndarray:
            u_pulse, thickness_v, tau0_v, smear_v = aug.unpack(u)
            ew = spectrum_fn(u_pulse)
            t_sim = trace_param(ew, thickness_v, tau0_v, smear_v)
            mu = (
                mj.mu_per_freq(tm, t_sim, wj)[:, None]
                if R_omega
                else mj.mu_global(tm, t_sim, wj)
            )
            return ((tm - mu * t_sim) / sig).ravel()

        noise_scaled = False
    else:

        def residual(u: jnp.ndarray) -> jnp.ndarray:
            u_pulse, thickness_v, tau0_v, smear_v = aug.unpack(u)
            ew = spectrum_fn(u_pulse)
            t_sim = trace_param(ew, thickness_v, tau0_v, smear_v)
            return mj.residual_vector(tm, t_sim, wj, R_omega=R_omega)

        noise_scaled = True

    r0 = np.asarray(residual(jnp.asarray(u0)), dtype=float)
    jac = np.asarray(jax.jacfwd(residual)(jnp.asarray(u0)), dtype=float)

    n_params = int(u0.size)
    n_residuals = int(r0.size)
    dof = max(n_residuals - n_params, 1)
    ssr = float(r0 @ r0)
    reduced_chisq = ssr / dof
    scale = reduced_chisq if noise_scaled else 1.0

    # Gauss–Newton Hessian and its pseudo-inverse via a symmetric eigensolve.
    # Flat (gauge) directions show up as eigenvalues far below the largest; drop
    # them so they neither blow up the covariance nor inject CEP/timing jitter.
    jtj = jac.T @ jac
    evals, evecs = np.linalg.eigh(jtj)
    max_ev = float(evals[-1]) if evals.size else 0.0
    thr = rcond * max_ev
    keep = evals > thr
    rank = int(np.count_nonzero(keep))
    mode_var = np.where(keep, scale / np.where(keep, evals, 1.0), 0.0)

    return _CovCore(
        u0=np.asarray(u0, dtype=float),
        spectrum_fn=spectrum_fn,
        evecs=evecs,
        mode_var=mode_var,
        rank=rank,
        n_params=n_params,
        n_residuals=n_residuals,
        reduced_chisq=reduced_chisq,
        noise_scaled=noise_scaled,
        n_pulse=aug.n_pulse,
        idx_thickness=aug.idx_thickness,
        scale_thickness=aug.scales[0],
        idx_tau0=aug.idx_tau0,
        scale_tau=aug.scales[1],
        idx_smear=aug.idx_smear,
        scale_smear=aug.scales[2],
        idx_smear_delta=aug.idx_smear_delta,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def parameter_covariance(
    result: RetrievalResult,
    measured: ArrayLike,
    *,
    noise: NoiseModel | float | None = None,
    weights: NDArray[np.float64] | None = None,
    material: str | None = None,
    thickness: float = 0.0,
    npoints: int = 20,
    omega0: float | None = None,
    quadrature: str = "gausslegendre",
    phase_only: bool = False,
    phase_basis: str = "pointwise",
    n_nodes: int = 20,
    R_omega: bool = False,
    fit_thickness: bool = False,
    fit_tau0: bool = False,
    smearing: SmearingKernel | None = None,
    fit_smearing: bool = False,
    fit_smearing_split: bool = False,
    rcond: float = 1e-10,
) -> CovarianceResult:
    r"""Gauss–Newton parameter covariance of a least-squares retrieval.

    Rebuilds the JAX forward model and the solver parameterisation at the retrieved
    spectrum, forms the residual Jacobian :math:`J = \partial r/\partial u` by
    automatic differentiation, and returns
    :math:`\mathrm{cov}(u) = \hat\sigma^2 (J^\mathsf{T}J)^{+}` (see the module
    docstring). The forward-model and parameterisation keywords **must reproduce the
    original retrieval** (the same ``material``/``thickness``/``phase_basis`` …),
    exactly as the uncertainty bootstraps require their ``retrieve_kwargs``.

    Parameters
    ----------
    result : RetrievalResult
        A completed retrieval (provides the spectrum, grid, delays, interaction and
        carrier ``omega0``).
    measured : array_like
        The measured trace ``(Nomega, Ndelay)`` that was fit.
    noise : NoiseModel or float, optional
        Measurement-noise model used to **whiten** the residual (each pixel divided
        by its σ), giving an absolutely scaled covariance ``(JᵀJ)⁺``. A bare float
        is a fraction-of-peak additive ``NoiseModel``. When omitted the covariance is
        scaled by the internal reduced-χ² estimate instead.
    weights : numpy.ndarray, optional
        Per-frequency weight vector (length ``Nomega``); defaults to ones.
    material, thickness, npoints, omega0, quadrature
        Dispersive forward-model parameters, as in :class:`croak.lm.LM`
        (``omega0`` defaults to ``result.omega0``).
    phase_only, phase_basis, n_nodes
        Pulse parameterisation, as in :func:`croak._jax_pulse.ls_parameterisation`.
    R_omega : bool, optional
        Per-frequency adaptive scaling (must match the retrieval).
    smearing : SmearingKernel or None, optional
        Geometric time-smearing kernel used by the forward model; must match the
        one the retrieval ran with, or the covariance describes a different model.
    fit_smearing : bool, optional
        Include the smearing multiplier in the augmented parameter vector, giving
        ``sigma_smear``. Requires ``smearing``.
    fit_smearing_split : bool, optional
        The retrieval fitted the two smearing channels separately
        (``fit_smearing_split``); adds the delay-channel multiplier to the
        vector, giving ``sigma_smear_delta``. Requires ``fit_smearing``.
    fit_thickness, fit_tau0 : bool, optional
        Include the dispersive-slab thickness and/or delay-zero offset in the
        covariance (must match the retrieval that fitted them). The fitted values
        are read from ``result``; their standard errors are returned as
        ``sigma_thickness`` / ``sigma_tau0``.
    rcond : float, optional
        Relative eigenvalue floor for the pseudo-inverse: eigenvalues of
        :math:`J^\mathsf{T}J` below ``rcond`` × the largest are treated as flat
        (gauge) directions and dropped (default ``1e-10``).

    Returns
    -------
    CovarianceResult
    """
    core = _compute_core(
        result,
        measured,
        noise=noise,
        weights=weights,
        material=material,
        thickness=thickness,
        npoints=npoints,
        omega0=omega0,
        quadrature=quadrature,
        phase_only=phase_only,
        phase_basis=phase_basis,
        n_nodes=n_nodes,
        R_omega=R_omega,
        rcond=rcond,
        fit_thickness=fit_thickness,
        fit_tau0=fit_tau0,
        smearing=smearing,
        fit_smearing=fit_smearing,
        fit_smearing_split=fit_smearing_split,
    )
    return CovarianceResult(
        cov=core.covariance(),
        u=core.u0,
        n_params=core.n_params,
        n_residuals=core.n_residuals,
        rank=core.rank,
        null_dim=core.n_params - core.rank,
        reduced_chisq=core.reduced_chisq,
        noise_scaled=core.noise_scaled,
        sigma_thickness=core.sigma_thickness(),
        sigma_tau0=core.sigma_tau0(),
        sigma_smear=core.sigma_smear(),
        sigma_smear_delta=core.sigma_smear_delta(),
    )


def covariance_uncertainty(
    result: RetrievalResult,
    measured: ArrayLike,
    *,
    noise: NoiseModel | float | None = None,
    propagation: NDArray[np.complex128] | None = None,
    propagation_label: str | None = None,
    n_samples: int = 500,
    oversampling: int = 8,
    collect_profiles: bool = False,
    interval: str = "bias-corrected",
    rng: np.random.Generator | None = None,
    weights: NDArray[np.float64] | None = None,
    material: str | None = None,
    thickness: float = 0.0,
    npoints: int = 20,
    omega0: float | None = None,
    quadrature: str = "gausslegendre",
    phase_only: bool = False,
    phase_basis: str = "pointwise",
    n_nodes: int = 20,
    R_omega: bool = False,
    fit_thickness: bool = False,
    fit_tau0: bool = False,
    smearing: SmearingKernel | None = None,
    fit_smearing: bool = False,
    fit_smearing_split: bool = False,
    rcond: float = 1e-10,
    callback=None,
) -> UncertaintyResult:
    r"""FWHM uncertainty from the linearised covariance (the analytic estimator).

    Builds the Gauss–Newton parameter covariance (:func:`parameter_covariance`),
    draws ``n_samples`` parameter vectors :math:`u_k \sim \mathcal N(u^\*,
    \mathrm{cov}(u))`, optionally **propagates** each to a different beamline point
    with the transfer function ``propagation`` (see
    :func:`croak.session.dispersion.transfer_function`), and returns the spread of the
    per-sample temporal-intensity FWHM as an
    :class:`~croak.uncertainty.UncertaintyResult` (``method="covariance"``) — drop-in
    compatible with the bootstrap output, so the same intervals, temporal band and
    plotting apply.

    Sampling from the Gaussian parameter posterior and pushing each draw through the
    exact field → FWHM map is the delta method carried to the nonlinearity of that
    map: fast (no re-retrieval), and the natural analytic cross-check on the
    resampling bootstraps. Gauge null directions carry zero variance, so the draws do
    not jitter the CEP or timing.

    ``result``, ``measured``, ``noise``, ``weights`` and the forward-model /
    parameterisation keywords (``material``, ``thickness``, ``npoints``, ``omega0``,
    ``quadrature``, ``phase_only``, ``phase_basis``, ``n_nodes``, ``R_omega``,
    ``rcond``) are as in :func:`parameter_covariance` and must reproduce the
    retrieval.

    Parameters
    ----------
    propagation : numpy.ndarray of complex, optional
        Spectral transfer function :math:`H(\omega)` propagating each draw to a
        different beamline point; ``None`` reports at the measurement plane.
    propagation_label : str, optional
        Description of the propagation point recorded on the result.
    n_samples : int, optional
        Number of Gaussian parameter draws ``B`` (default 500; cheap — each is a
        spectrum evaluation, no re-retrieval).
    oversampling : int, optional
        Temporal oversampling for the FWHM (default 8, matching ``process_result``).
    collect_profiles : bool, optional
        Also collect peak-aligned oversampled intensity profiles (for a band).
    interval : {"bias-corrected", "percentile"}, optional
        Interval estimator.
    rng : numpy.random.Generator, optional
        Random generator (defaults to a fresh one).
    callback : callable, optional
        ``callback(done, total, R)`` after the draws are evaluated; for parity with
        the bootstrap API (called once here, the draws being a single vectorised step).

    Returns
    -------
    UncertaintyResult
        With ``method="covariance"``.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    rng = np.random.default_rng() if rng is None else rng
    core = _compute_core(
        result,
        measured,
        noise=noise,
        weights=weights,
        material=material,
        thickness=thickness,
        npoints=npoints,
        omega0=omega0,
        quadrature=quadrature,
        phase_only=phase_only,
        phase_basis=phase_basis,
        n_nodes=n_nodes,
        R_omega=R_omega,
        rcond=rcond,
        fit_thickness=fit_thickness,
        fit_tau0=fit_tau0,
        smearing=smearing,
        fit_smearing=fit_smearing,
        fit_smearing_split=fit_smearing_split,
    )

    # Draw u_k = u* + V diag(sqrt(mode_var)) z, z ~ N(0, I). Dropped gauge modes
    # have zero variance, so the draws never move along CEP/timing directions.
    # Only the pulse block feeds the field → FWHM map; sampling the *joint*
    # vector (incl. any fitted thickness/tau0) and marginalising to the pulse
    # block inflates the FWHM spread by the extra parameters' uncertainty.
    std_modes = np.sqrt(core.mode_var)
    z = rng.standard_normal((n_samples, core.n_params))
    samples_u = core.u0[None, :] + (z * std_modes[None, :]) @ core.evecs.T
    samples_pulse = samples_u[:, : core.n_pulse]

    spectra = np.asarray(
        jax.vmap(core.spectrum_fn)(jnp.asarray(samples_pulse)), dtype=complex
    )

    point_estimate, _, _ = _fwhm_and_profile(
        result.grid, _apply_propagation(result.spectrum, propagation), oversampling
    )
    fwhms: list[float] = []
    profiles: list[NDArray[np.float64]] = []
    t_profile: NDArray[np.float64] | None = None
    for k in range(n_samples):
        fw, t_over, prof = _fwhm_and_profile(
            result.grid, _apply_propagation(spectra[k], propagation), oversampling
        )
        fwhms.append(fw)
        if collect_profiles:
            profiles.append(prof)
            t_profile = t_over
    if callback is not None:
        callback(n_samples, n_samples, result.error)

    return _assemble(
        statistic="fwhm",
        point_estimate=point_estimate,
        fwhms=fwhms,
        errors=[result.error] * n_samples,
        profiles=profiles,
        t_profile=t_profile,
        method="covariance",
        n_resamples=n_samples,
        base_error=result.error,
        noise_floor=None,
        warm_start=True,
        convergence_tol=float("inf"),
        interval=interval,
        propagation_label=propagation_label,
    )
