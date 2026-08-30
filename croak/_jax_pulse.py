"""Shared parameterisation helper for the JAX retrievers.

Both :class:`~croak.lbfgs_ad.LBFGSAD` and :class:`~croak.lm.LM` optimise the same
real parameter vector as :class:`croak.lbfgs.LBFGS`: the spectral amplitude and
phase over an optional support window (``u = [A; phi]``), or just the phase when
``phase_only`` holds the amplitude fixed. This module provides the single
differentiable map from those parameters to a complex spectrum, mirroring
:meth:`croak.pulses.ArrayPulse.spectrum`/``set_params`` so the AD solvers share
the analytic solver's exact parameterisation (and are directly comparable to it).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline

from . import metrics_jax as mj
from .maths import planck_taper

__all__ = [
    "make_spectrum_fn",
    "make_param_grad_fns",
    "ParamGradFns",
    "make_spline_phase_basis",
    "build_parameterisation",
    "ls_parameterisation",
    "Parameterisation",
    "Penalties",
    "time_window_mask",
    "ExtraParamSpec",
    "Augmentation",
    "augment",
]


def time_window_mask(
    t: ArrayLike, win_lo: float, win_hi: float, dt: float
) -> NDArray[np.float64]:
    """Weight (in ``[0, 1]``) that penalises field energy **outside** a window.

    The complement of a Planck taper over ``[win_lo, win_hi]`` (roll-off ``20*dt``
    each side, mirroring :func:`croak.processing.post_filter`): ``0`` inside the
    window, rising smoothly to ``1`` outside it. Multiplied against ``|E(t)|^2``
    by the temporal regularisation penalty.
    """
    r = 20.0 * dt
    keep = planck_taper(t, win_lo - r, win_lo, win_hi, win_hi + r)
    return 1.0 - np.asarray(keep, dtype=float)


def make_spectrum_fn(
    amplitude: ArrayLike,
    phase: ArrayLike,
    *,
    support: tuple[int, int] | None = None,
    phase_only: bool = False,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], NDArray[np.float64], int]:
    r"""Build the differentiable map ``u -> spectrum`` and its initial parameters.

    JAX twin of :class:`croak.pulses.ArrayPulse`. Parameters outside the support
    window keep the seed ``amplitude``/``phase`` and are not optimised.

    Parameters
    ----------
    amplitude, phase : array_like
        Seed spectral amplitude :math:`|\tilde E|` and phase (length ``N``).
    support : tuple of int, optional
        Half-open ``(start, stop)`` index range of the optimised window; the
        full grid by default.
    phase_only : bool, optional
        Optimise only the phase, holding ``amplitude`` fixed.

    Returns
    -------
    spectrum_fn : callable
        ``spectrum_fn(u) -> ew`` returning the centred complex spectrum
        :math:`A\,e^{i\phi}`.
    u0 : numpy.ndarray
        Initial parameter vector (``[A[support]; phi[support]]`` or, for
        ``phase_only``, ``phi[support]``).
    nidcs : int
        Number of optimised frequency indices — the split point between the
        amplitude and phase blocks of ``u`` (for the regularisation penalties).
    """
    amp0 = np.asarray(amplitude, dtype=float).copy()
    phase0 = np.asarray(phase, dtype=float).copy()
    n = amp0.size
    start, stop = (0, n) if support is None else support
    nidcs = stop - start

    amp0_j = jnp.asarray(amp0)
    phase0_j = jnp.asarray(phase0)

    def spectrum_fn(u: jnp.ndarray) -> jnp.ndarray:
        if phase_only:
            amp = amp0_j
            phi = phase0_j.at[start:stop].set(u)
        else:
            amp = amp0_j.at[start:stop].set(u[:nidcs])
            phi = phase0_j.at[start:stop].set(u[nidcs:])
        return amp * jnp.exp(1j * phi)

    if phase_only:
        u0 = phase0[start:stop].copy()
    else:
        u0 = np.concatenate([amp0[start:stop], phase0[start:stop]])

    return spectrum_fn, u0, nidcs


class ParamGradFns(NamedTuple):
    """Hand-derived parameter-space adjoints, the JAX twin of ``ArrayPulse``.

    These mirror :meth:`croak.pulses.ArrayPulse.ew_vjp` and the regularisation
    gradients, expressed in ``jnp`` (no autodiff). They let the non-AD retriever
    map a spectral cotangent ``ew_bar`` and the parameters ``u`` to ``dL/du``.
    """

    #: ``ew_vjp(u, ew_bar) -> du``: spectral cotangent → parameter gradient.
    ew_vjp: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    #: ``amplitude_penalty_grad(u) -> du`` (full-length; phase block zero).
    amplitude_penalty_grad: Callable[[jnp.ndarray], jnp.ndarray]
    #: ``phase_penalty_grad(u) -> du`` (full-length; amplitude block zero).
    phase_penalty_grad: Callable[[jnp.ndarray], jnp.ndarray]
    #: ``spectral_penalty_grad(u, target_support) -> du`` (amplitude block; zero
    #: in phase-only mode). ``target_support`` is the target amplitude over the
    #: optimised support (length ``nidcs``).
    spectral_penalty_grad: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def _second_diff_penalty_grad(a: jnp.ndarray, *, normalize_by_max: bool) -> jnp.ndarray:
    """Gradient of ``pulses._second_diff_penalty`` (max(a) treated as constant)."""
    n = a.size
    if n < 3:
        return jnp.zeros(n)
    d = a[2:] - 2.0 * a[1:-1] + a[:-2]
    # 5-point stencil: dP/da[k] = 2 (d[k-1] - 2 d[k] + d[k+1]) with d=0 outside.
    grad = jnp.zeros(n)
    grad = grad.at[2:].add(d)
    grad = grad.at[1:-1].add(-2.0 * d)
    grad = grad.at[:-2].add(d)
    if not normalize_by_max:
        return 2.0 / (n - 2) * grad
    eps = jnp.finfo(jnp.float64).eps
    amax = jnp.max(a)
    amax2 = jnp.maximum(amax**2, eps)
    out = 2.0 / ((n - 2) * amax2) * grad
    # Extra term from differentiating the max(a)**2 in the denominator.
    sumd2 = jnp.sum(d * d)
    extra = -2.0 * sumd2 / ((n - 2) * amax**3)
    cond = (amax**2 > eps) & (amax > 0)
    return out.at[jnp.argmax(a)].add(jnp.where(cond, extra, 0.0))


def make_param_grad_fns(
    amplitude: ArrayLike,
    phase: ArrayLike,
    *,
    support: tuple[int, int] | None = None,
    phase_only: bool = False,
) -> ParamGradFns:
    r"""Build the hand-derived parameter-space adjoints for ``u``.

    Same ``u`` as :func:`make_spectrum_fn`. JAX twin of the adjoint side of
    :class:`croak.pulses.ArrayPulse`
    (:meth:`~croak.pulses.ArrayPulse.ew_vjp` and the ``*_penalty_grad`` methods).
    The parameterisation (support window, ``phase_only``) matches
    :func:`make_spectrum_fn` exactly so the two are used together.
    """
    amp0 = np.asarray(amplitude, dtype=float)
    n = amp0.size
    start, stop = (0, n) if support is None else support
    nidcs = stop - start
    amp0_j = jnp.asarray(amp0)

    def ew_vjp(u: jnp.ndarray, ew_bar: jnp.ndarray) -> jnp.ndarray:
        d = ew_bar[start:stop]
        if phase_only:
            a = amp0_j[start:stop]
            phi = u
        else:
            a = u[:nidcs]
            phi = u[nidcs:]
        cos, sin = jnp.cos(phi), jnp.sin(phi)
        du_phase = 2.0 * a * (cos * jnp.imag(d) - sin * jnp.real(d))
        if phase_only:
            return du_phase
        du_amp = 2.0 * (cos * jnp.real(d) + sin * jnp.imag(d))
        return jnp.concatenate([du_amp, du_phase])

    def amplitude_penalty_grad(u: jnp.ndarray) -> jnp.ndarray:
        if phase_only:
            return jnp.zeros_like(u)
        g_amp = _second_diff_penalty_grad(u[:nidcs], normalize_by_max=True)
        return jnp.concatenate([g_amp, jnp.zeros(u.size - nidcs)])

    def phase_penalty_grad(u: jnp.ndarray) -> jnp.ndarray:
        if phase_only:
            return _second_diff_penalty_grad(u, normalize_by_max=False)
        g_phase = _second_diff_penalty_grad(u[nidcs:], normalize_by_max=False)
        return jnp.concatenate([jnp.zeros(nidcs), g_phase])

    def spectral_penalty_grad(u: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        if phase_only:
            return jnp.zeros_like(u)
        a = u[:nidcs]
        eps = jnp.finfo(jnp.float64).eps
        denom = jnp.maximum(jnp.sum(target * target), eps)
        g_amp = 2.0 * (a - target) / denom
        return jnp.concatenate([g_amp, jnp.zeros(u.size - nidcs)])

    return ParamGradFns(
        ew_vjp, amplitude_penalty_grad, phase_penalty_grad, spectral_penalty_grad
    )


# ---------------------------------------------------------------------------
# Reduced phase basis (cubic B-spline) + the unified parameterisation the AD
# retrievers consume.
# ---------------------------------------------------------------------------
def _spectrum_support(
    amplitude: NDArray[np.float64], threshold: float
) -> tuple[int, int]:
    """Half-open index window where the (peak-normalised) amplitude is non-zero."""
    amp = np.asarray(amplitude, dtype=float)
    peak = float(np.max(amp)) if amp.size else 0.0
    if peak <= 0:
        return 0, amp.size
    idx = np.nonzero(amp > threshold * peak)[0]
    if idx.size == 0:
        return 0, amp.size
    return int(idx[0]), int(idx[-1]) + 1


def _clamped_knots(
    x0: float, x1: float, n_coef: int, degree: int
) -> NDArray[np.float64]:
    """Clamped knot vector for ``n_coef`` B-spline control points of ``degree``."""
    n_interior = n_coef - degree - 1
    interior = (
        np.linspace(x0, x1, n_interior + 2)[1:-1] if n_interior > 0 else np.empty(0)
    )
    return np.concatenate([np.full(degree + 1, x0), interior, np.full(degree + 1, x1)])


def make_spline_phase_basis(
    amplitude: ArrayLike,
    phase: ArrayLike,
    omega: ArrayLike,
    *,
    n_nodes: int,
    degree: int = 3,
    support: tuple[int, int] | None = None,
    support_threshold: float = 1e-3,
) -> tuple[
    Callable[[jnp.ndarray], jnp.ndarray], NDArray[np.float64], Callable, tuple[int, int]
]:
    r"""Phase-only basis whose phase is a cubic B-spline over the spectrum support.

    The spectral phase on the support window is a fixed **linear** map
    :math:`\phi = B\,c` of ``n_nodes`` control points ``c`` (the optimised
    parameters), with the amplitude held at the seed ``amplitude``. Because
    ``B`` (the :func:`scipy.interpolate.BSpline.design_matrix`) is a constant, the
    map is exactly :math:`C^2` (cubic) and trivially AD-differentiable. Knots are
    clamped over the non-zero-spectrum window only, so nodes don't waste degrees
    of freedom where there is no signal.

    Returns ``(spectrum_fn, c0, phase_penalty, support)`` where ``c0`` fits the
    seed phase and ``phase_penalty(c)`` is the P-spline roughness penalty
    :math:`\mathrm{mean}((D_2 c)^2)` (a 2nd-difference on the coefficients).
    """
    amp0 = np.asarray(amplitude, dtype=float).copy()
    phase0 = np.asarray(phase, dtype=float).copy()
    omega = np.asarray(omega, dtype=float)
    n_coef = int(n_nodes)
    if n_coef < degree + 1:
        raise ValueError(f"n_nodes must be >= degree+1 ({degree + 1}); got {n_coef}")
    start, stop = support or _spectrum_support(amp0, support_threshold)
    if stop - start < degree + 1:
        raise ValueError("spectrum support is too narrow for the requested spline")

    x = omega[start:stop]
    knots = _clamped_knots(float(x[0]), float(x[-1]), n_coef, degree)
    B_np = np.asarray(
        BSpline.design_matrix(x, knots, degree).todense()
    )  # (n_sup, n_coef)
    B = jnp.asarray(B_np)
    amp0_j, phase0_j = jnp.asarray(amp0), jnp.asarray(phase0)

    def spectrum_fn(c: jnp.ndarray) -> jnp.ndarray:
        phi = phase0_j.at[start:stop].set(B @ c)
        return amp0_j * jnp.exp(1j * phi)

    # Seed coefficients fit the guess phase over the support (least squares).
    # Unwrap first so the smooth spline tracks the seed (exp(iφ) is 2π-periodic,
    # so a wrapped seed would force spurious 2π jumps the spline cannot follow).
    c0 = np.linalg.lstsq(B_np, np.unwrap(phase0[start:stop]), rcond=None)[0]

    # P-spline roughness: 2nd-difference matrix on the coefficients.
    if n_coef >= 3:
        D2 = np.eye(n_coef)[2:] - 2 * np.eye(n_coef)[1:-1] + np.eye(n_coef)[:-2]
        D2_j = jnp.asarray(D2)

        def phase_penalty(c: jnp.ndarray) -> jnp.ndarray:
            d = D2_j @ c
            return jnp.mean(d * d)
    else:  # pragma: no cover - n_nodes>=4 enforced above

        def phase_penalty(c: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(0.0)

    return spectrum_fn, c0.astype(float), phase_penalty, (start, stop)


class Penalties(NamedTuple):
    """Regularisation terms for a parameterisation (each maps ``u -> scalar``)."""

    phase: Callable[[jnp.ndarray], jnp.ndarray]
    amplitude: Callable[[jnp.ndarray], jnp.ndarray]
    spectral: Callable[[jnp.ndarray], jnp.ndarray]
    temporal: Callable[[jnp.ndarray], jnp.ndarray]


class Parameterisation(NamedTuple):
    """A differentiable ``u -> spectrum`` map plus its initial parameters/penalties."""

    spectrum_fn: Callable[[jnp.ndarray], jnp.ndarray]
    u0: NDArray[np.float64]
    penalties: Penalties


def build_parameterisation(
    ew0: NDArray[np.complex128],
    omega: ArrayLike,
    *,
    phase_only: bool,
    phase_basis: str = "pointwise",
    n_nodes: int = 20,
    support_threshold: float = 1e-3,
    spectral_target: NDArray[np.float64] | None = None,
    time_mask: NDArray[np.float64] | None = None,
) -> Parameterisation:
    """Build the ``u -> spectrum`` map shared by the AD retrievers.

    ``phase_basis="pointwise"`` is the existing per-frequency parameterisation
    (amplitude+phase, or phase-only); ``"bspline"`` is the reduced cubic B-spline
    phase basis (implies phase-only). The returned :class:`Penalties` already
    handle slicing/mode so each solver just weights them by its ``reg_*``.

    ``time_mask`` (length ``n``, on the centred ``grid.t`` axis) enables the
    temporal penalty: the energy fraction of ``E(t) = ifft(spectrum)`` falling
    where the mask is non-zero — a self-consistent, mode-agnostic alternative to
    the post-retrieval temporal filter. See :func:`time_window_mask`.
    """
    amp0 = np.abs(ew0)
    phase0 = np.angle(ew0)
    temporal_pen = _make_temporal_penalty(time_mask)
    if phase_basis == "bspline":
        spectrum_fn, u0, phase_pen, _support = make_spline_phase_basis(
            amp0, phase0, omega, n_nodes=n_nodes, support_threshold=support_threshold
        )
        zero = lambda u: jnp.asarray(0.0)  # noqa: E731 - amplitude fixed
        temporal = _bind_temporal(temporal_pen, spectrum_fn)
        return Parameterisation(
            spectrum_fn, u0, Penalties(phase_pen, zero, zero, temporal)
        )
    if phase_basis != "pointwise":
        raise ValueError(f"unknown phase_basis {phase_basis!r}")

    spectrum_fn, u0, nidcs = make_spectrum_fn(amp0, phase0, phase_only=phase_only)
    target = None if spectral_target is None else jnp.asarray(spectral_target)

    def phase_pen(u: jnp.ndarray) -> jnp.ndarray:
        return mj.phase_penalty(u if phase_only else u[nidcs:])

    def amp_pen(u: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(0.0) if phase_only else mj.amplitude_penalty(u[:nidcs])

    def spec_pen(u: jnp.ndarray) -> jnp.ndarray:
        if phase_only or target is None:
            return jnp.asarray(0.0)
        return mj.spectral_penalty(u[:nidcs], target)

    temporal = _bind_temporal(temporal_pen, spectrum_fn)
    return Parameterisation(
        spectrum_fn, u0, Penalties(phase_pen, amp_pen, spec_pen, temporal)
    )


def _make_temporal_penalty(time_mask):
    """Return a ``spectrum -> scalar`` temporal penalty, or ``None`` if disabled."""
    if time_mask is None:
        return None
    m = jnp.asarray(time_mask, dtype=float)

    def penalty(ew: jnp.ndarray) -> jnp.ndarray:
        # E(t) on the centred grid.t axis (matches grid.ifft ordering; the df
        # scale cancels in the energy ratio below).
        et = jnp.fft.fftshift(jnp.fft.fft(jnp.fft.ifftshift(ew)))
        power = jnp.abs(et) ** 2
        return jnp.sum(m * power) / jnp.maximum(jnp.sum(power), 1e-30)

    return penalty


def _bind_temporal(penalty, spectrum_fn):
    """Compose the temporal penalty with ``spectrum_fn`` (``u -> scalar``)."""
    if penalty is None:
        return lambda u: jnp.asarray(0.0)
    return lambda u: penalty(spectrum_fn(u))


def ls_parameterisation(
    ew0: NDArray[np.complex128],
    omega: ArrayLike,
    *,
    phase_only: bool,
    phase_basis: str = "pointwise",
    n_nodes: int = 20,
    support_threshold: float = 1e-3,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], NDArray[np.float64], int, bool]:
    """Build the least-squares (LM) solver parameterisation.

    The LM solvers add penalties as residual rows rather than scalar terms.
    Returns ``(spectrum_fn, u0, nidcs, phase_only_eff)``. For the B-spline basis
    ``nidcs=0`` and ``phase_only_eff=True`` (no amplitude block) — and the LM
    solvers' 2nd-difference phase rows on ``u`` then *are* the P-spline penalty on
    the spline coefficients.
    """
    amp0, phase0 = np.abs(ew0), np.angle(ew0)
    if phase_basis == "bspline":
        spectrum_fn, u0, _pen, _sup = make_spline_phase_basis(
            amp0, phase0, omega, n_nodes=n_nodes, support_threshold=support_threshold
        )
        return spectrum_fn, u0, 0, True
    if phase_basis != "pointwise":
        raise ValueError(f"unknown phase_basis {phase_basis!r}")
    spectrum_fn, u0, nidcs = make_spectrum_fn(amp0, phase0, phase_only=phase_only)
    return spectrum_fn, u0, nidcs, phase_only


# ---------------------------------------------------------------------------
# Optional extra scalar parameters (slab thickness, delay-zero offset) appended
# to the pulse vector. Shared by the AD retrievers and the covariance estimator.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtraParamSpec:
    r"""Specification of the optional scalar parameters fitted with the pulse.

    The slab **thickness** ``L`` and a **delay-zero offset** :math:`\tau_0` are
    optimised as *dimensionless, O(1)-scaled offsets* appended to the pulse
    parameter vector, so the augmented problem stays well-conditioned for NLopt /
    SciPy ``trf`` despite the tiny physical scales (``L ~ 1e-5 m``,
    :math:`\tau_0 \sim 10^{-15}` s):

    .. math::

        L = L_0 + \delta_L\,s_L, \qquad \tau_0 = \tau_{0,0} + \delta_\tau\,s_\tau

    with the optimised variables :math:`\delta_L, \delta_\tau` initialised at
    zero. :math:`\delta_L` is bounded below at :math:`-L_0/s_L` so ``L`` stays
    non-negative. The centres :math:`L_0`, :math:`\tau_{0,0}` are the priors: the
    nominal values for a cold-started retrieval, the *previous* fit's values when
    warm-starting (see :func:`croak.pipeline.retrieve_from_tracedata`), and the
    fitted values when re-linearising for covariance. An extra whose ``fit_*``
    flag is off is **held** at its centre.

    Parameters
    ----------
    fit_thickness : bool, optional
        Fit the slab thickness (requires a dispersive slab; ``thickness0 > 0``).
    thickness0 : float, optional
        Centre thickness :math:`L_0` (m) — the prior/held value.
    scale_L : float, optional
        Normalisation :math:`s_L` (m) for :math:`\delta_L`; defaults to
        ``thickness0`` (so :math:`\delta_L` is the fractional change).
    fit_tau0 : bool, optional
        Fit the delay-zero offset.
    tau0_0 : float, optional
        Centre delay-zero offset :math:`\tau_{0,0}` (s); ``0`` for retrieval.
    scale_tau : float, optional
        Normalisation :math:`s_\tau` (s) for :math:`\delta_\tau`; typically one
        delay step. Required when ``fit_tau0`` is set.
    fit_smearing : bool, optional
        Fit the geometric-smearing **strength** as a dimensionless multiplier on
        both kernel widths — equivalently, the mask ratio ``d/D``, which is the
        only physical lever on the effect. Requires a smearing kernel.
    smear_scale0 : float, optional
        Centre multiplier (``1.0`` = the kernel as supplied). For a split fit
        this is the ``p``-channel centre.
    scale_smear : float, optional
        Normalisation for the multiplier offset. It is already ``O(1)``, so the
        default ``1.0`` is almost always right.
    fit_smearing_split : bool, optional
        Fit the gate-shape (``p``) and delay (``delta``) kernel widths as two
        independent multipliers instead of one joint one — the diagnostic for
        *which* channel deviates from the geometric prediction. Requires
        ``fit_smearing``.
    smear_delta0 : float, optional
        Centre multiplier of the delay channel for a split fit (``1.0`` = the
        kernel as supplied).
    """

    fit_thickness: bool = False
    thickness0: float = 0.0
    scale_L: float = 0.0
    fit_tau0: bool = False
    tau0_0: float = 0.0
    scale_tau: float = 0.0
    fit_smearing: bool = False
    smear_scale0: float = 1.0
    scale_smear: float = 1.0
    fit_smearing_split: bool = False
    smear_delta0: float = 1.0

    @property
    def any(self) -> bool:
        """Whether any extra parameter is fitted."""
        return self.fit_thickness or self.fit_tau0 or self.fit_smearing


class Augmentation(NamedTuple):
    r"""The augmented parameter vector plus its (un)packing maps.

    See :func:`augment`. ``unpack`` is differentiable (used inside the JAX
    objective/Jacobian); ``physical`` returns plain floats for reporting; the
    scales convert a covariance σ on :math:`\delta` back to physical units.
    """

    n_pulse: int
    #: Augmented vector
    #: ``[u_pulse; delta_L?; delta_tau?; delta_smear?; delta_smear_d?]``.
    u0: NDArray[np.float64]
    #: ``unpack(u_ext) -> (u_pulse, thickness, tau0, smear_scale)`` (JAX-friendly).
    #: ``smear_scale`` is a scalar for a joint fit and a length-2 vector
    #: ``(scale_p, scale_delta)`` when the channels are fitted separately.
    unpack: Callable[
        [jnp.ndarray],
        tuple[
            jnp.ndarray,
            jnp.ndarray | float,
            jnp.ndarray | float,
            jnp.ndarray | float,
        ],
    ]
    #: ``physical(u_ext) -> (thickness, tau0, smear_scale)`` as floats;
    #: ``smear_scale`` is a ``(scale_p, scale_delta)`` tuple for a split fit.
    physical: Callable[[ArrayLike], tuple[float, float, float | tuple[float, float]]]
    #: Per-component lower bounds (``-inf`` for the pulse block and ``tau0``).
    lower_bounds: NDArray[np.float64]
    #: Index of the thickness offset in ``u_ext`` (``None`` when not fitted).
    idx_thickness: int | None
    #: Index of the delay-zero offset in ``u_ext`` (``None`` when not fitted).
    idx_tau0: int | None
    #: Index of the smearing-scale offset in ``u_ext`` (``None`` when not fitted).
    #: For a split fit this is the ``p``-channel scale.
    idx_smear: int | None
    #: Index of the delay-channel scale offset (``None`` unless split-fitted).
    idx_smear_delta: int | None
    #: Resolved scales ``(scale_L, scale_tau, scale_smear)`` (0 where not fitted).
    #: The split delay-channel offset shares ``scale_smear``.
    scales: tuple[float, float, float]


def augment(u0_pulse: NDArray[np.float64], spec: ExtraParamSpec) -> Augmentation:
    """Append the fitted extra scalars to a pulse parameter vector.

    Parameters
    ----------
    u0_pulse : numpy.ndarray
        The pulse-only initial parameter vector (from
        :func:`build_parameterisation` / :func:`ls_parameterisation`).
    spec : ExtraParamSpec
        Which extras to fit and their scales.

    Returns
    -------
    Augmentation
        The augmented vector and its packing maps (see the class docstring).

    Raises
    ------
    ValueError
        If a fitted extra has a non-positive scale.
    """
    u0_pulse = np.asarray(u0_pulse, dtype=float)
    n_pulse = u0_pulse.size
    thickness0 = float(spec.thickness0)
    tau0_0 = float(spec.tau0_0)

    scale_L = 0.0
    idx_thickness: int | None = None
    lowers = list(np.full(n_pulse, -np.inf))
    if spec.fit_thickness:
        scale_L = spec.scale_L if spec.scale_L > 0 else thickness0
        if scale_L <= 0:
            raise ValueError("fitting thickness requires thickness0 > 0")
        idx_thickness = len(lowers)
        # delta_L >= -L0/scale_L keeps L = L0 + delta_L*scale_L >= 0.
        lowers.append(-thickness0 / scale_L)

    scale_tau = 0.0
    idx_tau0: int | None = None
    if spec.fit_tau0:
        scale_tau = float(spec.scale_tau)
        if scale_tau <= 0:
            raise ValueError("fitting tau0 requires scale_tau > 0")
        idx_tau0 = len(lowers)
        lowers.append(-np.inf)

    scale_smear = 0.0
    idx_smear: int | None = None
    idx_smear_delta: int | None = None
    smear_scale0 = float(spec.smear_scale0)
    smear_delta0 = float(spec.smear_delta0)
    if spec.fit_smearing_split and not spec.fit_smearing:
        raise ValueError("fit_smearing_split requires fit_smearing")
    if spec.fit_smearing:
        scale_smear = float(spec.scale_smear)
        if scale_smear <= 0:
            raise ValueError("fitting the smearing scale requires scale_smear > 0")
        if smear_scale0 <= 0:
            raise ValueError("fitting the smearing scale requires smear_scale0 > 0")
        idx_smear = len(lowers)
        # Same construction as the thickness: bound the offset so the kernel widths,
        # which are proportional to the multiplier, can never go negative.
        lowers.append(-smear_scale0 / scale_smear)
        if spec.fit_smearing_split:
            if smear_delta0 <= 0:
                raise ValueError("splitting the smearing fit requires smear_delta0 > 0")
            idx_smear_delta = len(lowers)
            lowers.append(-smear_delta0 / scale_smear)

    n_extra = len(lowers) - n_pulse
    u0 = np.concatenate([u0_pulse, np.zeros(n_extra)])
    lower_bounds = np.asarray(lowers, dtype=float)

    def unpack(u_ext):
        u_pulse = u_ext[:n_pulse]
        thickness = (
            thickness0 + u_ext[idx_thickness] * scale_L
            if idx_thickness is not None
            else thickness0
        )
        # A non-fitted extra is *held* at its centre — the same rule for all
        # three, so a fixed non-zero tau0_0 (a known delay-zero offset, e.g. the
        # one covariance re-linearises about) shifts the model rather than being
        # silently dropped.
        tau0 = tau0_0 + u_ext[idx_tau0] * scale_tau if idx_tau0 is not None else tau0_0
        smear = (
            smear_scale0 + u_ext[idx_smear] * scale_smear
            if idx_smear is not None
            else smear_scale0
        )
        if idx_smear_delta is not None:
            # split fit: pass the channels as a pair (p, delta); the trace
            # functions accept a scalar or a length-2 vector interchangeably
            smear = jnp.stack(
                [smear, smear_delta0 + u_ext[idx_smear_delta] * scale_smear]
            )
        return u_pulse, thickness, tau0, smear

    def physical(u_ext) -> tuple[float, float, float | tuple[float, float]]:
        u_ext = np.asarray(u_ext, dtype=float)
        thickness = (
            thickness0 + float(u_ext[idx_thickness]) * scale_L
            if idx_thickness is not None
            else thickness0
        )
        tau0 = (
            tau0_0 + float(u_ext[idx_tau0]) * scale_tau
            if idx_tau0 is not None
            else tau0_0
        )
        smear_p = (
            smear_scale0 + float(u_ext[idx_smear]) * scale_smear
            if idx_smear is not None
            else smear_scale0
        )
        smear: float | tuple[float, float] = smear_p
        if idx_smear_delta is not None:
            smear = (
                smear_p,
                smear_delta0 + float(u_ext[idx_smear_delta]) * scale_smear,
            )
        return thickness, tau0, smear

    return Augmentation(
        n_pulse=n_pulse,
        u0=u0,
        unpack=unpack,
        physical=physical,
        lower_bounds=lower_bounds,
        idx_thickness=idx_thickness,
        idx_tau0=idx_tau0,
        idx_smear=idx_smear,
        idx_smear_delta=idx_smear_delta,
        scales=(scale_L, scale_tau, scale_smear),
    )
