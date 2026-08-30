r"""Retrieval errors against a *known* pulse.

:mod:`croak.metrics` answers "how well does the retrieval reproduce the measured
trace" — the FROG error :math:`R`, which is all one can compute from a real
measurement. When the true pulse is known (a synthetic generation, or a
ModelPNPS scan carrying its complex field), a second and much sharper question
becomes available: how close is the retrieved *pulse* to the true one. The two
are not the same. A trace error can be small while the pulse is wrong, because
a FROG trace does not determine the pulse uniquely and because a good optimiser
can fit noise.

Three errors, all dimensionless and all zero for a perfect retrieval:

``eps_It``
    temporal intensity :math:`|E(t)|^2`, energy-normalised, on the truth's own
    time axis, minimised over the retrieval's time-translation gauge.
``eps_Iw``
    spectral intensity :math:`|\tilde E(\omega)|^2` over the truth's support
    band, each normalised to unit L2 norm so the arbitrary scale cancels.
``eps_Ew``
    the complex field :math:`\tilde E(\omega)`, minimised over the three
    quantities a FROG measurement genuinely cannot determine — amplitude scale,
    absolute phase and delay. This is Geib's :math:`\varepsilon`, the metric the
    PNPS literature quotes, and the only one of the three that cannot be
    satisfied by getting amplitude or phase right alone.

Each is a least-squares distance after removing exactly the gauge freedoms that
are unobservable, and no others: a metric that quotients out more would flatter
the retrieval, one that quotients out less would score a pure relabelling as
error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .preprocess import marginal_peak_delay
from .processing import TruthPulse, oversample, shiftnorm
from .result import RetrievalResult

__all__ = [
    "TruthErrors",
    "eps_temporal_intensity",
    "eps_spectral_intensity",
    "eps_complex_field",
    "truth_errors",
]

#: Zero-padding factor for the delay scan in :func:`eps_complex_field`. The
#: optimal delay is a continuous parameter, so evaluating the correlation only at
#: the grid's own delay steps would leave up to half a step of misalignment
#: scored as error. 32x interpolation puts that residue far below the metric.
_DELAY_OVERSAMPLE = 32

#: Temporal oversampling of the retrieved field before it is interpolated onto
#: the truth's time axis, matching :func:`croak.processing.process_result`.
_TIME_OVERSAMPLE = 8

_TINY = float(np.finfo(float).tiny)


@dataclass(frozen=True)
class TruthErrors:
    """The three known-truth errors of one retrieval.

    Attributes
    ----------
    eps_It : float
        Temporal-intensity error (:func:`eps_temporal_intensity`).
    eps_Iw : float or None
        Spectral-intensity error (:func:`eps_spectral_intensity`); ``None`` when
        the truth carries no complex spectrum.
    eps_Ew : float or None
        Complex-field error (:func:`eps_complex_field`); ``None`` when the truth
        carries no complex spectrum.
    """

    eps_It: float
    eps_Iw: float | None = None
    eps_Ew: float | None = None


def eps_temporal_intensity(
    t_retrieved: ArrayLike,
    I_retrieved: ArrayLike,
    t_truth: ArrayLike,
    I_truth: ArrayLike,
) -> float:
    r"""Least-squares relative error between two temporal intensities.

    Both intensities are brought onto the truth's time axis; the compared
    quantity is the two-norm relative error minimised over the free scale,
    :math:`\min_c \lVert c I - I_{\text{true}}\rVert_2 / \lVert
    I_{\text{true}}\rVert_2 = \sin\theta`, so the comparison is of pulse
    *shape*, not of scale. The
    retrieval's time origin is a gauge freedom — a FROG trace is unchanged by
    translating the pulse — so it is removed in two steps: the peaks are aligned
    first (a coarse shift, without which a pulse parked elsewhere in the wider
    retrieval window can be interpolated away entirely and the error saturates
    near 1), then the cross-correlation peak refines it to **sub-sample**
    precision via :func:`croak.preprocess.marginal_peak_delay`.

    The refinement is not a nicety. Aligning only to the nearest sample leaves up
    to half a step of residual offset, which a normalised rms scores as roughly
    ``dt / 2 sigma``. Measured on an 8x-oversampled 20 fs Gaussian (0.125 fs
    step), a half-step delay scores 5.2e-3 under integer-bin alignment and
    2.3e-5 with the sub-sample fit — a floor 220x lower, and the difference
    between a number that is all grid artefact and one that is not.

    Parameters
    ----------
    t_retrieved, I_retrieved : array_like
        Retrieved time axis (s) and its temporal intensity (any scale).
    t_truth, I_truth : array_like
        Known time axis (s), which must be uniformly sampled, and its intensity.
        The comparison is made on this axis.

    Returns
    -------
    float
        :math:`\varepsilon_{I_t} \ge 0`; ``0`` for shapes identical up to
        energy scale and a time translation.
    """
    t = np.asarray(t_retrieved, dtype=float)
    it = np.asarray(I_retrieved, dtype=float)
    t0 = np.asarray(t_truth, dtype=float)
    i0 = np.asarray(I_truth, dtype=float)
    if t.size != it.size or t0.size != i0.size:
        raise ValueError("each axis must match the length of its intensity")
    if t0.size < 2:
        raise ValueError("the truth needs at least two time samples")

    t = t - t[int(np.argmax(it))]
    t0 = t0 - t0[int(np.argmax(i0))]
    i0 = i0 / max(float(np.trapezoid(i0, t0)), _TINY)

    def sampled(offset: float) -> NDArray[np.float64]:
        """Return the retrieved intensity on the truth axis, shifted by ``offset``."""
        out = np.interp(t0 - offset, t, it, left=0.0, right=0.0)
        return out / max(float(np.trapezoid(out, t0)), _TINY)

    # Cross-correlate on the truth grid, then read the peak off between samples.
    # `correlate(..., "full")` indexes lags from -(N-1), and the truth axis is
    # uniform, so index maps to a delay by a single step.
    correlation = np.correlate(i0, sampled(0.0), "full")
    step = float(np.mean(np.diff(t0)))
    lags = (np.arange(correlation.size) - (i0.size - 1)) * step
    ir = sampled(marginal_peak_delay(lags, correlation))
    # least-squares scale gauge (2026-08-30): min_c ||c I - I_true|| /
    # ||I_true|| = sin(theta). The unit-energy normalization above is kept
    # for the alignment steps (it stabilises the correlation) but no longer
    # sets the compared scale, so a low-level pedestal costs its own norm
    # once rather than twice (its norm plus the peak deficit it causes).
    denom = max(float(np.linalg.norm(ir)) * float(np.linalg.norm(i0)),
                _TINY)
    c = float(np.dot(ir, i0)) / denom
    return float(np.sqrt(max(0.0, 1.0 - c * c)))


def eps_spectral_intensity(
    spectrum: ArrayLike, truth: ArrayLike, *, threshold: float = 0.02
) -> float:
    r"""Least-squares relative error between two spectral intensities.

    Compares :math:`|\tilde E(\omega)|^2` on a shared frequency grid. Both are
    restricted to where the truth actually carries signal; the compared
    quantity is the two-norm relative error minimised over the free scale
    (:math:`\sin\theta`), which removes the retrieval's arbitrary overall
    scale; spectral intensity is already blind to the constant-phase and
    delay gauges. Like :func:`eps_complex_field`, the
    :math:`\sqrt{1 - x^2}` form cannot resolve a perfect match below
    ``~1.5e-8`` — read anything at that level as zero.

    Parameters
    ----------
    spectrum, truth : array_like
        Retrieved and known complex spectra on the same grid (centred order).
    threshold : float, optional
        Band cut as a fraction of the truth's peak spectral intensity. Bins
        below it are excluded: they carry no information but do carry the
        retrieval's noise floor, which would otherwise dominate a normalised
        distance.

    Returns
    -------
    float
        :math:`\varepsilon_{I_\omega} \ge 0`; ``0`` for identical shapes.

    Raises
    ------
    ValueError
        If the two spectra differ in shape, or the truth is empty in band.
    """
    a = np.abs(np.asarray(spectrum, dtype=complex)) ** 2
    b = np.abs(np.asarray(truth, dtype=complex)) ** 2
    if a.shape != b.shape:
        raise ValueError("spectrum and truth must be on the same grid")
    peak = float(b.max()) if b.size else 0.0
    if peak <= 0:
        raise ValueError("the truth spectrum is identically zero")
    band = b > threshold * peak
    a, b = a[band], b[band]
    # least-squares scale gauge (2026-08-30): min_c ||c a - b|| / ||b||
    # = sin(theta). Scale-free in both arguments; replaces the earlier
    # unit-norm (chordal) form 2 sin(theta/2), to which it agrees to first
    # order and from which it differs by <1% below 0.3.
    denom = max(float(np.linalg.norm(a)) * float(np.linalg.norm(b)), _TINY)
    c = float(np.dot(a, b)) / denom
    return float(np.sqrt(max(0.0, 1.0 - c * c)))


def eps_complex_field(spectrum: ArrayLike, truth: ArrayLike) -> float:
    r"""Complex-field error, minimised over the unobservable gauges.

    Geib's :math:`\varepsilon`: the normalised distance between two complex
    spectra after minimising over amplitude scale, absolute phase and delay —
    exactly the three transformations a PNPS measurement cannot resolve. Written
    as an overlap,

    .. math::

        \varepsilon = \sqrt{1 - \max_\tau
            \frac{|\langle \tilde E_\mathrm{truth}, \tilde E\,e^{i\omega\tau}
            \rangle|^2}{\|\tilde E_\mathrm{truth}\|^2\,\|\tilde E\|^2}},

    where maximising the overlap magnitude over :math:`\tau` handles the delay
    and taking the modulus handles the absolute phase, while the normalisation
    handles the scale. The delay scan is the DFT of
    :math:`\tilde E^*_\mathrm{truth}\tilde E`, zero-padded ``32x`` so the optimum
    is found between grid steps rather than snapped to one.

    Parameters
    ----------
    spectrum, truth : array_like
        Retrieved and known complex spectra on the same grid (centred order).

    Returns
    -------
    float
        :math:`\varepsilon_{E_\omega} \in [0, 1]`; ``0`` for fields identical up
        to scale, absolute phase and delay, ``1`` for uncorrelated ones.

    Raises
    ------
    ValueError
        If the two spectra differ in shape, or either is identically zero.

    Notes
    -----
    The ``sqrt(1 - x^2)`` form cannot resolve a near-perfect match below about
    ``1e-8``: as the overlap ``x`` approaches 1 the square root amplifies the
    double-precision residue of ``1 - x^2``, so an exact match scores
    ``~sqrt(eps) ~ 1.5e-8`` rather than 0. Read anything at that level as zero.
    """
    retrieved = np.asarray(spectrum, dtype=complex)
    known = np.asarray(truth, dtype=complex)
    if retrieved.shape != known.shape:
        raise ValueError("spectrum and truth must be on the same grid")
    denom = float(np.sqrt(np.sum(np.abs(known) ** 2) * np.sum(np.abs(retrieved) ** 2)))
    if denom <= 0:
        raise ValueError("neither spectrum may be identically zero")
    product = np.conj(known) * retrieved
    padded = np.zeros(product.size * _DELAY_OVERSAMPLE, dtype=complex)
    padded[: product.size] = product
    # ifft scales by 1/size, so multiplying back gives the plain sum -- the
    # overlap at each interpolated delay.
    best = float(np.abs(np.fft.ifft(padded)).max() * padded.size)
    return float(np.sqrt(max(0.0, 1.0 - (best / denom) ** 2)))


def truth_errors(
    result: RetrievalResult, truth: TruthPulse, *, frame_p: float = 0.0
) -> TruthErrors:
    """Compute every known-truth error available for one retrieval.

    The spectral errors need the truth's *complex* field, so they are reported
    only when it has one (a synthetic generation always does; a ModelPNPS scan
    does when the file stores the complex ``Eω`` datasets) *and* it overlaps the
    retrieval grid. The temporal error needs only the truth's intensity envelope
    and is always available.

    Parameters
    ----------
    result : RetrievalResult
        The retrieval to score.
    truth : TruthPulse
        The known pulse, from the synthetic or simulated load.
    frame_p : float, optional
        The ``spectrum_frame_p`` the retrieval ran with. A frame-corrected
        retrieval (the focal-mixture protocol) carries its amplitude in the
        ON-AXIS frame — the measured spectrum reweighted by
        ``((omega+omega0)/omega0)**(2 p)`` in intensity — while the truth lives
        in the measurement frame. Passing the exponent undoes that reweighting
        on the retrieved spectrum (and rebuilds the temporal field from it), so
        all three errors compare like with like: a phase-only retrieval then
        scores ``eps_Iw = 0`` instead of the fixed frame footprint (~0.107 for
        the 1 fs validation spectrum at ``p = 0.5``). Leave at 0 for retrievals
        that never reframed.

    Returns
    -------
    TruthErrors
        With ``eps_Iw``/``eps_Ew`` set to ``None`` for an intensity-only truth.

    Examples
    --------
    >>> errs = truth_errors(result, truth)          # doctest: +SKIP
    >>> errs.eps_It < 0.01                          # doctest: +SKIP
    True
    """
    spectrum = np.asarray(result.spectrum, dtype=complex)
    if frame_p != 0.0:
        om = np.asarray(result.grid.omega, float)
        wfac = np.maximum((om + result.omega0) / result.omega0, 0.0)
        # amplitude factor is wfac**p; bins at or below zero absolute
        # frequency carry no physical field, so they are zeroed rather than
        # divided (matching apply_spectrum_frame's clamp on the way in)
        spectrum = spectrum * np.where(wfac > 0.0, wfac ** (-float(frame_p)), 0.0)
        field = result.grid.ifft(spectrum)
    else:
        field = result.field
    # Centre and oversample exactly as process_result (and TruthPulse itself)
    # do. shiftnorm must come first: the retrieved field is periodic on its grid,
    # so a solver that parks the pulse near an edge leaves it split across the
    # wrap. Translating the time axis afterwards cannot reunite it, and the
    # severed tail is then scored as a shape error -- 0.16 for a pure 20 fs delay
    # gauge on a 77 fs window, against 0.005 once the circular shift is done
    # first. Only then is oversampling meaningful.
    t_over, e_over = oversample(
        result.grid.t, shiftnorm(result.grid, field), _TIME_OVERSAMPLE
    )
    eps_It = eps_temporal_intensity(t_over, np.abs(e_over) ** 2, truth.t, truth.It)
    if truth.Eomega is None:
        return TruthErrors(eps_It=eps_It)
    known = truth.spectrum_on_grid(result.grid, result.omega0)
    # A regridded retrieval can sit outside the stored field's frequency range,
    # which leaves the interpolated truth identically zero. That is "no spectral
    # truth on this grid", the same kind of absence as an intensity-only file --
    # not a malformed input -- so report it the same way rather than raising.
    if not np.any(np.abs(known) > 0):
        return TruthErrors(eps_It=eps_It)
    return TruthErrors(
        eps_It=eps_It,
        eps_Iw=eps_spectral_intensity(spectrum, known),
        eps_Ew=eps_complex_field(spectrum, known),
    )
