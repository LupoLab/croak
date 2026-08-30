r"""Pre-retrieval consistency checks from FROG trace frequency marginals.

The **frequency marginal** of a FROG-type trace is its integral over delay,

.. math::
    M(\omega) = \int T(\omega, \tau)\,\mathrm{d}\tau ,

a one-dimensional function of optical frequency. Every FROG signal factorises
into an undelayed part :math:`P(t)` and a delayed gate :math:`g(t-\tau)`, and
integrating over delay collapses the delay phase to a delta function, leaving a
**convolution of spectral intensities**

.. math::
    M(\omega) \;\propto\; \bigl(|\tilde P|^2 * |\tilde g|^2\bigr)(\omega).

Two consequences hold for *any* pulse (intensity-weighted moments add under
convolution): the marginal centroid is the sum of the factor centroids and its
variance is the sum of the factor variances. From an independently measured
fundamental spectrum :math:`S(\omega) = |\tilde E(\omega)|^2` this gives, per
geometry:

- **SHG** — marginal :math:`S * S` (exact); centroid :math:`2\bar\omega_S`;
  RMS width :math:`\sqrt2\,\sigma_S`. Both anchors hard.
- **PG / TG** — marginal :math:`S * |\widehat{|E|^2}|^2` (transform-limited);
  centroid :math:`\bar\omega_S` (exact); width in :math:`[\sigma_S, \sigma_{TL}]`.
- **SD** — marginal :math:`|\widehat{E^2}|^2 * S(-\omega)` (transform-limited);
  centroid :math:`\approx\bar\omega_S` (TL/symmetric only); width :math:`>\sigma_S`.

SHG is the gold standard: its marginal is fully phase-independent, so any
mismatch is unambiguously a *trace* problem (calibration, spectral response,
detector nonlinearity, scatter). PG/TG keep one rigorous anchor (the centroid)
and one rigorous lower bound (the width); SD has only soft, transform-limited
(TL) anchors because its phase-dependent factor is the SHG spectrum.

This module computes those predictions (:func:`predicted_marginal`,
:func:`predicted_centroid`), the moments of a measured marginal
(:func:`centroid`, :func:`rms_width`), and an exponent picker that tilts a trace
marginal onto the predicted centroid (:func:`auto_third_order_exponent`). The
full physics is in the manual's *FROG trace marginals* explanation page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy.fft
import scipy.optimize
from numpy.typing import ArrayLike, NDArray

from .grid import Grid
from .interactions import get_interaction
from .maths import moment, wlfreq

__all__ = [
    "MarginalPrediction",
    "centroid",
    "rms_width",
    "spectral_autoconvolution",
    "apply_marginal_correction",
    "predicted_centroid",
    "predicted_marginal",
    "auto_third_order_exponent",
]


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------
def centroid(omega: ArrayLike, marginal: ArrayLike) -> float:
    r"""Intensity-weighted centroid :math:`\bar\omega = \sum\omega M/\sum M`.

    Parameters
    ----------
    omega : array_like
        Coordinate axis (typically angular frequency, rad/s).
    marginal : array_like
        Non-negative weights (the marginal intensity).

    Returns
    -------
    float
        The first moment of ``marginal`` over ``omega``.
    """
    return moment(omega, marginal, 1)


def rms_width(omega: ArrayLike, marginal: ArrayLike) -> float:
    r"""Root-mean-square width of a distribution about its centroid.

    Computes :math:`\sigma = \sqrt{\langle\omega^2\rangle - \bar\omega^2}`.

    Parameters
    ----------
    omega : array_like
        Coordinate axis (typically angular frequency, rad/s).
    marginal : array_like
        Non-negative weights (the marginal intensity).

    Returns
    -------
    float
        The standard deviation of ``marginal`` over ``omega`` (floored at zero
        before the square root to absorb discretisation noise in the tails).
    """
    c = moment(omega, marginal, 1)
    var = moment(omega, marginal, 2) - c * c
    return float(np.sqrt(max(var, 0.0)))


# ---------------------------------------------------------------------------
# Spectral convolution helpers
# ---------------------------------------------------------------------------
def spectral_autoconvolution(
    spectrum: ArrayLike, *, circular: bool = False
) -> NDArray[np.float64]:
    r"""Autoconvolution :math:`S * S` of a spectral intensity.

    This is the exact SHG-FROG marginal shape (phase-independent). Two boundary
    conventions are offered:

    * ``circular=False`` (default): the **linear** convolution, which doubles the
      support to ``2N - 1`` samples so there is no wrap-around. Use this for the
      true predicted marginal shape (it lands on a doubled, twice-as-wide axis).
    * ``circular=True``: the **circular** convolution via the convolution theorem
      (:math:`S*S = \mathcal F\{(\mathcal F^{-1}S)^2\}` on a centred array),
      returning ``N`` samples on the **same grid** as ``spectrum``. Use this to
      scale a trace's frequency rows onto the predicted SHG marginal in place
      (see :func:`croak.preprocess.load_and_clean`'s ``marginal_correct``); it
      aliases if the autoconvolution support exceeds the grid.

    Parameters
    ----------
    spectrum : array_like
        Spectral intensity :math:`S(\omega)` on a uniform grid (``>= 0``). For
        ``circular=True`` it must be a centred array (zero frequency in the
        middle), matching croak's grid convention.
    circular : bool, optional
        Select the circular, grid-matched convolution instead of the linear one.

    Returns
    -------
    numpy.ndarray
        The autoconvolution: length ``2N - 1`` (linear) or ``N`` (circular).
    """
    s = np.asarray(spectrum, dtype=float)
    if not circular:
        return np.convolve(s, s)
    # Convolution theorem on a centred array: multiplication in the conjugate
    # (time) domain is convolution in frequency. real(): S*S is real for real S.
    fund = np.asarray(scipy.fft.ifft(scipy.fft.ifftshift(s)))
    return np.real(scipy.fft.fftshift(scipy.fft.fft(fund**2)))


def apply_marginal_correction(
    trace: ArrayLike, Iomega: ArrayLike
) -> NDArray[np.float64]:
    r"""Rescale a SHG trace's frequency rows onto the spectrum-predicted marginal.

    The exact SHG frequency marginal is the spectral autoconvolution
    :math:`S * S` of the independent fundamental spectrum ``Iomega`` (both on the
    same centred frequency grid as the trace). Each frequency row of ``trace`` is
    rescaled so its delay-integrated marginal matches that prediction (rows with
    no marginal power are zeroed), which corrects a measured/efficiency tilt
    along the spectral axis without touching the delay structure.

    Shared by :func:`croak.preprocess.load_and_clean` (the ``marginal_correct``
    option) and the marginal-check stage's live preview, so the correction is
    defined in exactly one place.

    Parameters
    ----------
    trace : array_like
        SHG trace ``(Nomega, Ndelay)`` on the centred frequency grid.
    Iomega : array_like
        Independent fundamental spectral intensity on the same grid (centred).

    Returns
    -------
    numpy.ndarray
        A new corrected trace ``(Nomega, Ndelay)``; the input is not mutated.
    """
    trace = np.asarray(trace, dtype=float)
    IwSHG = spectral_autoconvolution(Iomega, circular=True)
    peak = np.max(IwSHG)
    if peak > 0:
        IwSHG = IwSHG / peak
    marg = trace.sum(axis=1)
    mpeak = np.max(marg)
    if mpeak > 0:
        marg = marg / mpeak
    out = np.zeros_like(trace)
    with np.errstate(divide="ignore", invalid="ignore"):
        nonzero = marg > 0
        out[nonzero] = trace[nonzero] * (IwSHG[nonzero] / marg[nonzero])[:, None]
    return out


def _transform_limited_intensity_factors(
    omega: NDArray[np.float64], spectrum: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Return the PG gate and SD probe spectral factors for the TL pulse.

    Builds the transform-limited field :math:`\tilde E_\mathrm{TL} = \sqrt S`
    (flat phase), transforms to time, and forms the two phase-dependent factors
    evaluated at the TL pulse:

    * PG/TG gate: :math:`|\widehat{|E|^2}|^2` — modulus-squared FT of the
      temporal intensity, an **even** function of frequency (centroid 0).
    * SD probe: :math:`|\widehat{E^2}|^2` — the SHG spectrum of the TL pulse.

    Both are returned on the same uniform grid as ``spectrum`` (ascending,
    matching ``omega``'s spacing); only their shapes are used downstream, so the
    absolute carrier is irrelevant.
    """
    # Treat the ascending spectrum as a centred spectrum on a matched grid: the
    # carrier only multiplies E(t) by a constant-frequency phase, which leaves
    # |E(t)|^2 (and hence the even gate factor) and the *shape* of |F{E^2}|^2
    # unchanged — and the marginal placement is fixed analytically below.
    grid = Grid.from_omega(omega)
    ew_tl = np.sqrt(np.clip(spectrum, 0.0, None)).astype(complex)
    e_t = grid.ifft(ew_tl)
    gate = np.abs(grid.fft(np.abs(e_t) ** 2)) ** 2  # |F{|E|^2}|^2  (PG/TG)
    probe = np.abs(grid.fft(e_t**2)) ** 2  # |F{E^2}|^2   (SD = SHG spectrum)
    return gate, probe


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def predicted_centroid(
    omega: ArrayLike, spectrum: ArrayLike, interaction: str
) -> float:
    r"""Predicted marginal centroid (rad/s) from the fundamental spectrum.

    Returns :math:`2\bar\omega_S` for SHG (the marginal sits on the doubled
    axis) and :math:`\bar\omega_S` for PG/TG/SD (the marginal sits at the
    fundamental). For SHG and PG/TG this is a hard, phase-independent check; for
    SD it holds only for a transform-limited or temporally symmetric pulse.

    Parameters
    ----------
    omega : array_like
        Absolute angular-frequency axis of the spectrum (rad/s).
    spectrum : array_like
        Fundamental spectral intensity :math:`S(\omega)` (``>= 0``).
    interaction : str
        ``"shg"``, ``"sd"``, ``"pg"`` or ``"tg"`` (case-insensitive).
    """
    name = _normalise_interaction(interaction)
    omega = np.asarray(omega, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    c_s = moment(omega, spectrum, 1)
    return 2.0 * c_s if name == "shg" else c_s


@dataclass(frozen=True)
class MarginalPrediction:
    r"""A predicted FROG marginal and its consistency anchors.

    Attributes
    ----------
    interaction : str
        Resolved geometry: ``"shg"``, ``"pg"`` or ``"sd"`` (``"tg"`` maps to
        ``"pg"``).
    omega : numpy.ndarray
        Absolute angular-frequency axis of the predicted marginal (rad/s).
    marginal : numpy.ndarray
        Predicted marginal, peak-normalised to unit maximum.
    centroid : float
        Predicted centroid (rad/s): :math:`2\bar\omega_S` for SHG, else
        :math:`\bar\omega_S`.
    rms_width : float
        Predicted RMS width (rad/s): the exact :math:`\sqrt2\,\sigma_S` for SHG,
        otherwise the transform-limited width :math:`\sigma_\mathrm{TL}`.
    width_bracket : tuple of float or None
        ``(lower, upper)`` RMS-width bracket. For PG/TG the measured width should
        lie in ``[\sigma_S, \sigma_\mathrm{TL}]`` (lower bound rigorous, upper
        soft). For SD only the lower bound :math:`\sigma_S` is rigorous
        (``upper`` is ``None``). ``None`` for SHG (the width is exact).
    centroid_is_hard : bool
        Whether the centroid is a rigorous, phase-independent check (SHG, PG/TG)
        or a soft TL/symmetric anchor (SD).
    width_is_hard : bool
        Whether the RMS width is an exact check (SHG only).
    sigma_spectrum : float
        RMS width of the fundamental spectrum :math:`\sigma_S` (rad/s).
    """

    interaction: str
    omega: NDArray[np.float64]
    marginal: NDArray[np.float64]
    centroid: float
    rms_width: float
    width_bracket: tuple[float, float | None] | None
    centroid_is_hard: bool
    width_is_hard: bool
    sigma_spectrum: float


def predicted_marginal(
    omega: ArrayLike, spectrum: ArrayLike, interaction: str
) -> MarginalPrediction:
    r"""Predict the FROG marginal and its checks from the fundamental spectrum.

    Computes the predicted marginal shape by spectral convolution (exact for
    SHG, transform-limited for PG/TG/SD), then places it on the analytically
    predicted centroid and reports the centroid/width consistency anchors. The
    shape comes from the numerics; the placement from the exact centroid identity
    :math:`\bar\omega[M] = \bar\omega[|\tilde P|^2] + \bar\omega[|\tilde g|^2]`,
    which avoids fragile absolute-carrier bookkeeping through the FFTs.

    Parameters
    ----------
    omega : array_like
        Absolute angular-frequency axis of the spectrum (rad/s), uniform and
        ascending.
    spectrum : array_like
        Fundamental spectral intensity :math:`S(\omega)` (``>= 0``).
    interaction : str
        ``"shg"``, ``"sd"``, ``"pg"`` or ``"tg"`` (case-insensitive); ``"tg"``
        is identical to ``"pg"`` (the transient-grating gate is :math:`|E|^2`).

    Returns
    -------
    MarginalPrediction
        The predicted marginal on its own (doubled, for SHG) frequency axis and
        the centroid/width anchors.

    Raises
    ------
    ValueError
        If ``interaction`` is unknown or ``omega``/``spectrum`` are degenerate.
    """
    name = _normalise_interaction(interaction)
    omega = np.asarray(omega, dtype=float)
    spectrum = np.clip(np.asarray(spectrum, dtype=float), 0.0, None)
    if omega.ndim != 1 or omega.size < 4:
        raise ValueError("omega must be a 1-D axis with at least 4 samples")
    if spectrum.shape != omega.shape:
        raise ValueError("spectrum and omega must have the same shape")
    domega = float(omega[1] - omega[0])

    c_s = moment(omega, spectrum, 1)
    sigma_s = rms_width(omega, spectrum)

    if name == "shg":
        shape = spectral_autoconvolution(spectrum)
        rms = np.sqrt(2.0) * sigma_s
        bracket: tuple[float, float | None] | None = None
        centroid_is_hard = width_is_hard = True
    else:
        gate, probe = _transform_limited_intensity_factors(omega, spectrum)
        if name == "pg":
            shape = np.convolve(spectrum, gate)
        else:  # sd: |F{E^2}|^2 * S(-omega)
            shape = np.convolve(probe, spectrum[::-1])
        # The predicted (TL) marginal width is the width of the convolved shape.
        bracket = (sigma_s, _width_of_shape(shape, domega))
        rms = bracket[1] if bracket[1] is not None else sigma_s
        centroid_is_hard = name == "pg"  # exact for PG/TG, soft (TL) for SD
        width_is_hard = False
        if name == "sd":
            bracket = (sigma_s, None)  # only the lower bound is rigorous

    pred_centroid = 2.0 * c_s if name == "shg" else c_s
    # Place the shape so its centroid equals the analytic prediction (spacing is
    # preserved by the linear convolution), then peak-normalise for overlay.
    axis = np.arange(shape.size, dtype=float) * domega
    axis = axis - moment(axis, shape, 1) + pred_centroid
    peak = shape.max()
    marginal = shape / peak if peak > 0 else shape

    return MarginalPrediction(
        interaction=name,
        omega=axis,
        marginal=marginal,
        centroid=pred_centroid,
        rms_width=rms,
        width_bracket=bracket,
        centroid_is_hard=centroid_is_hard,
        width_is_hard=width_is_hard,
        sigma_spectrum=sigma_s,
    )


def _width_of_shape(shape: NDArray[np.float64], domega: float) -> float:
    """RMS width of a convolution shape on a uniform grid of spacing ``domega``."""
    axis = np.arange(shape.size, dtype=float) * domega
    return rms_width(axis, shape)


# ---------------------------------------------------------------------------
# Auto third-order exponent (centroid matching)
# ---------------------------------------------------------------------------
def auto_third_order_exponent(
    lam: ArrayLike,
    marginal: ArrayLike,
    lam_spec: ArrayLike,
    Ilam_spec: ArrayLike,
    interaction: str,
    *,
    window: tuple[float, float] = (210e-9, 1000e-9),
    bounds: tuple[float, float] = (0.0, 6.0),
    target_centroid: float | None = None,
) -> float:
    r"""Pick the ``(\lambda/\mu m)^n`` exponent that matches the marginal centroid.

    The TG/PG third-order efficiency correction multiplies the trace marginal by
    a power law :math:`(\lambda/\mu\mathrm{m})^n`
    (:func:`croak.preprocess.third_order_scale`). This chooses ``n`` so the
    **centroid** of the corrected marginal (computed as an angular-frequency
    density, within a wavelength ``window``) equals the centroid predicted from
    the independent spectrum (:func:`predicted_centroid`: :math:`\bar\omega_S` at
    the fundamental for PG/TG/SD, :math:`2\bar\omega_S` for SHG).

    ``target_centroid`` (rad/s) overrides the spectrum-derived target. Pass the
    centroid of an already-computed prediction so the matched centroid is exactly
    the one being displayed — otherwise small differences in how the spectrum is
    windowed/resampled between the prediction and this fit leave the corrected
    marginal a few nm off the drawn prediction line.

    Increasing ``n`` shifts weight to long wavelengths (low frequency), so the
    centroid decreases monotonically with ``n`` and the match has a unique root,
    found by :func:`scipy.optimize.brentq` and clamped to ``bounds``. If the
    target lies outside the achievable centroid range the nearer bound is
    returned.

    Parameters
    ----------
    lam : array_like
        Wavelength grid of the trace marginal (m).
    marginal : array_like
        The trace's wavelength marginal **before** the power-law correction.
    lam_spec, Ilam_spec : array_like
        Independent fundamental spectrum and its wavelength grid (m).
    interaction : str
        ``"shg"``, ``"sd"``, ``"pg"`` or ``"tg"`` (case-insensitive).
    window : tuple of float, optional
        ``(lo, hi)`` trust window in metres (default ``(210e-9, 1000e-9)``); only
        wavelengths inside it enter the centroid.
    bounds : tuple of float, optional
        ``(lo, hi)`` clamp on the returned exponent (default ``(0.0, 6.0)``).
    target_centroid : float, optional
        Predicted centroid to match (rad/s). When ``None`` (default) it is
        computed from ``lam_spec``/``Ilam_spec`` over the window.

    Returns
    -------
    float
        The centroid-matching exponent, clamped to ``bounds``.

    Raises
    ------
    ValueError
        If fewer than three wavelengths carry signal in the window, or the
        spectrum centroid cannot be formed.
    """
    _normalise_interaction(interaction)  # validate early
    lam = np.asarray(lam, dtype=float)
    marginal = np.asarray(marginal, dtype=float)
    lam_spec = np.asarray(lam_spec, dtype=float)
    Ilam_spec = np.clip(np.asarray(Ilam_spec, dtype=float), 0.0, None)

    # Target: the predicted centroid in angular frequency. Either supplied by the
    # caller (to match a displayed prediction exactly) or formed from the spectrum
    # as an omega-density (Ilam * lambda^2, the lambda->omega Jacobian up to const).
    if target_centroid is not None:
        target = float(target_centroid)
    else:
        spec_in = (lam_spec >= window[0]) & (lam_spec <= window[1]) & (Ilam_spec > 0)
        if np.count_nonzero(spec_in) < 3:
            raise ValueError("independent spectrum has no signal in the trust window")
        omega_spec = wlfreq(lam_spec[spec_in])
        s_omega = Ilam_spec[spec_in] * lam_spec[spec_in] ** 2
        target = predicted_centroid(omega_spec, s_omega, interaction)

    # Measured marginal centroid as a function of n, also as an omega-density.
    in_window = (lam >= window[0]) & (lam <= window[1]) & (marginal > 0)
    if np.count_nonzero(in_window) < 3:
        raise ValueError("trace marginal has no signal in the trust window")
    lam_w = lam[in_window]
    omega_w = wlfreq(lam_w)
    base = marginal[in_window] * lam_w**2  # omega-density of the *base* marginal
    # (lambda/um)^1, raised to n below; matches croak.preprocess.third_order_scale.
    tilt = lam_w / 1e-6

    def centroid_offset(n: float) -> float:
        weights = base * tilt**n
        total = weights.sum()
        if total <= 0:
            return np.inf
        return float(np.sum(omega_w * weights) / total) - target

    lo, hi = bounds
    f_lo, f_hi = centroid_offset(lo), centroid_offset(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if np.sign(f_lo) == np.sign(f_hi):
        # Target centroid is outside the achievable range; return the nearer end.
        return lo if abs(f_lo) <= abs(f_hi) else hi
    # cast: brentq returns a float here (full_output defaults to False), but its
    # scipy stub widens the return type to include the full-output tuple.
    return cast(float, scipy.optimize.brentq(centroid_offset, lo, hi))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise_interaction(interaction: str) -> str:
    """Resolve and validate an interaction name; map ``"tg"`` onto ``"pg"``.

    TG-FROG has the :math:`|E|^2` gate, so its marginal is identical to PG.
    """
    name = str(interaction).lower()
    if name == "tg":
        return "pg"
    get_interaction(name)  # raises ValueError for anything but shg/sd/pg
    return name
