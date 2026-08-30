"""Post-retrieval processing for analysis and display.

Turns a :class:`~croak.result.RetrievalResult` into a
:class:`ProcessedResult` carrying everything the plots need: oversampled
temporal intensity/phase (linear ramp removed), the transform-limited reference,
wavelength-domain spectral intensity/phase with fitted GDD/TOD, trace marginals,
residuals and a Gabor spectrogram helper.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.fft
from numpy.typing import ArrayLike, NDArray

from .constants import C
from .grid import Grid
from .maths import fwhm as _fwhm
from .maths import gauss, planck_taper, wlfreq
from .metrics import compute_mu
from .preprocess import TAPER_COLLAR_BINS
from .result import RetrievalResult

__all__ = [
    "marginals",
    "residuals",
    "oversample",
    "shiftnorm",
    "normalize_spectral_phase",
    "normalize_temporal_phase",
    "spectrogram",
    "post_filter",
    "peak_wavelength",
    "spectral_fwhm",
    "edge_energy_fraction",
    "TruthPulse",
    "ProcessedResult",
    "process_result",
    "resolve_time_direction",
]


def _finite_spectrum(
    wavelength: ArrayLike, intensity: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Keep the physical part of a spectrum and sort it by ascending wavelength.

    A retrieval grid's wavelength axis is :math:`2\pi c/(\omega+\omega_0)`, which
    is only *piecewise* monotone: where :math:`\omega+\omega_0` crosses zero it
    jumps through :math:`\pm\infty`, and the whole left half is negative when the
    result carries no carrier (``omega0 == 0``). Drop those bins so downstream
    statistics see a single, physically meaningful branch.
    """
    lam = np.asarray(wavelength, dtype=float)
    intens = np.asarray(intensity, dtype=float)
    keep = np.isfinite(lam) & (lam > 0.0) & np.isfinite(intens)
    lam, intens = lam[keep], intens[keep]
    order = np.argsort(lam)
    return lam[order], intens[order]


def peak_wavelength(wavelength: ArrayLike, intensity: ArrayLike) -> float:
    """Return the wavelength (m) of an intensity spectrum's maximum.

    The result depends on the spectral density convention: pass the *wavelength*
    density (e.g. :attr:`ProcessedResult.Ilam`, not
    :attr:`~ProcessedResult.Iw`) to match what the spectrum panel plots.

    Parameters
    ----------
    wavelength : array_like
        Wavelength axis (m), ascending or descending. Non-positive and non-finite
        entries are dropped (see the grid's zero crossing).
    intensity : array_like
        Spectral intensity on ``wavelength``; scaling is irrelevant.

    Returns
    -------
    float
        Wavelength of the peak (m), or ``nan`` when no usable bin remains.

    Examples
    --------
    >>> lam = np.linspace(700e-9, 900e-9, 201)
    >>> round(float(peak_wavelength(lam, np.exp(-(((lam - 800e-9) / 20e-9) ** 2)))), 12)
    8e-07
    """
    lam, intens = _finite_spectrum(wavelength, intensity)
    if lam.size == 0:
        return float("nan")
    return float(lam[int(np.argmax(intens))])


def spectral_fwhm(
    wavelength: ArrayLike, intensity: ArrayLike, *, level: float = 0.5
) -> float:
    """Return the full width at half maximum (m) of an intensity spectrum.

    Thin wrapper over :func:`croak.maths.fwhm` that first restricts the axis to its
    physical branch (see :func:`peak_wavelength` for the density convention).

    Parameters
    ----------
    wavelength : array_like
        Wavelength axis (m), ascending or descending.
    intensity : array_like
        Spectral intensity on ``wavelength``.
    level : float, optional
        Fraction of the peak at which to measure the width (default ``0.5``).

    Returns
    -------
    float
        Width in metres, or ``nan`` when the level is not crossed on both sides
        (a spectrum clipped by the grid) or no usable bin remains.

    Examples
    --------
    >>> lam = np.linspace(700e-9, 900e-9, 2001)
    >>> width = spectral_fwhm(lam, np.exp(-(((lam - 800e-9) / 20e-9) ** 2)))
    >>> round(width / 1e-9, 1)
    33.3
    """
    lam, intens = _finite_spectrum(wavelength, intensity)
    if lam.size == 0:
        return float("nan")
    return float(_fwhm(lam, intens, level=level))


def post_filter(
    result: RetrievalResult,
    *,
    lam_lims: tuple[float, float] | None = None,
    tau_lims: tuple[float, float] | None = None,
) -> RetrievalResult:
    """Post-retrieval cleanup: window the result in time and/or wavelength.

    The temporal window is applied first (a Planck taper over ``tau_lims``,
    transformed back to the spectrum and renormalised), then the spectral window
    (a Planck taper over the frequencies corresponding to ``lam_lims``). Either
    may be ``None``. Order matters: windowing in time redistributes spectral
    content, so the spectral taper must see the already-windowed field.

    ``lam_lims``/``tau_lims`` should be the **measurement** windows λm/τm — the
    ranges over which the trace actually carries signal (available as
    :attr:`~croak.preprocess.TraceData.lamm_lims` /
    :attr:`~croak.preprocess.TraceData.tau_lims`). Passing the full retrieval-grid
    extent instead would taper only the outermost few bins and do essentially
    nothing.

    Returns a new :class:`~croak.result.RetrievalResult` with the filtered
    spectrum (the original is unchanged), so the effect can be toggled without
    re-running the retrieval.
    """
    grid = result.grid
    omega = grid.omega
    ew = result.spectrum.copy()

    if tau_lims is not None:
        dt = grid.dt
        lo = max(min(tau_lims), grid.t.min() + 20 * dt)
        hi = min(max(tau_lims), grid.t.max() - 20 * dt)
        wt = planck_taper(grid.t, lo - 20 * dt, lo, hi, hi + 20 * dt)
        ew = grid.fft(grid.ifft(ew) * wt)
        peak = np.abs(ew).max()
        if peak > 0:
            ew = ew / peak

    if lam_lims is not None:
        dw = grid.domega
        wlims = sorted(
            (
                2 * np.pi * C / max(lam_lims) - result.omega0,
                2 * np.pi * C / min(lam_lims) - result.omega0,
            )
        )
        lo = max(wlims[0], omega.min() + 10 * dw)
        hi = min(wlims[1], omega.max() - 10 * dw)
        ww = planck_taper(omega, lo - 10 * dw, lo, hi, hi + 10 * dw)
        ew = ww * ew

    return replace(result, spectrum=ew)


# ---------------------------------------------------------------------------
# Trace marginals & residuals
# ---------------------------------------------------------------------------
def marginals(trace: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalised frequency and delay marginals of a trace ``(Nomega, Ndelay)``.

    Returns ``(omega_marginal, delay_marginal)``; each is floored at zero, has
    its minimum removed and is scaled to unit maximum.
    """
    trace = np.asarray(trace, dtype=float)

    def _norm(m):
        m = np.clip(m, 0.0, None)
        m = m - m.min()
        peak = m.max()
        return m / peak if peak > 0 else m

    return _norm(trace.sum(axis=1)), _norm(trace.sum(axis=0))


def residuals(
    measured: ArrayLike, retrieved: ArrayLike, *, weights: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Weighted residual ``(measured - mu*retrieved)`` with optimal scale ``mu``."""
    measured = np.asarray(measured, dtype=float)
    retrieved = np.asarray(retrieved, dtype=float)
    w = np.ones_like(measured) if weights is None else np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = np.broadcast_to(w[:, None], measured.shape)
    mu = compute_mu(measured, retrieved, w)
    return (measured - mu * retrieved) * w


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------
def oversample(
    t: ArrayLike, field: ArrayLike, factor: int = 8
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Smoothly oversample a complex field by zero-padding in the frequency domain.

    Parameters
    ----------
    t : array_like
        Time axis (uniform).
    field : array_like
        Complex field samples.
    factor : int, optional
        Oversampling factor (default ``8``).

    Returns
    -------
    tuple
        ``(t_oversampled, field_oversampled)``.
    """
    t = np.asarray(t, dtype=float)
    field = np.asarray(field, dtype=complex)
    n = field.size
    no = n * factor
    spec = scipy.fft.fft(field)
    out = np.zeros(no, dtype=complex)
    nyq = n // 2
    out[: nyq + 1] = spec[: nyq + 1]
    out[no - (n - nyq - 1) :] = spec[nyq + 1 :]
    # np.asarray: scipy.fft's @_dispatchable stubs do not expose the ndarray return.
    eo = np.asarray(scipy.fft.ifft(out)) * factor
    dt = (t[1] - t[0]) / factor
    to = t[0] + np.arange(no) * dt
    return to, eo


def shiftnorm(grid: Grid, field: ArrayLike) -> NDArray[np.complex128]:
    """Shift a time field so its peak sits at ``t=0`` and normalise its peak to 1."""
    field = np.asarray(field, dtype=complex)
    ts = grid.t[int(np.argmax(np.abs(field) ** 2))]
    shifted = grid.tshift(field, -ts)
    return shifted / np.sqrt(np.max(np.abs(shifted) ** 2))


def normalize_spectral_phase(omega, ew, mask_w, Iw):
    """Spectral phase in the display gauge: unwrapped, de-tilted, peak-zeroed.

    The gauge matters because spectral phase is only defined up to a constant
    and a linear term (an overall phase and a delay), so two phases can only be
    compared after both are pinned the same way. Shared by the retrieved
    spectrum and by a known truth phase precisely so the two cannot drift apart:
    plotting a truth normalised any other way would show a spurious tilt or
    offset that is pure gauge.
    """
    phi = np.unwrap(np.angle(np.asarray(ew)))
    phi = _remove_linear(omega, phi, mask_w)
    return phi - phi[mask_w][int(np.argmax(np.asarray(Iw)[mask_w]))]


def normalize_temporal_phase(t, field, mask_t):
    """Return temporal phase in the display gauge used for retrieved pulses.

    The constant phase and linear temporal ramp (a carrier-frequency offset) are
    not observable in the envelope retrieval. Removing both from a known field
    and a retrieved field makes their temporal phases directly comparable.

    Parameters
    ----------
    t : array_like
        Time axis (s).
    field : array_like
        Complex temporal envelope sampled on ``t``.
    mask_t : array_like
        Boolean support mask used for the linear fit and final centring.

    Returns
    -------
    numpy.ndarray
        Unwrapped, de-ramped temporal phase (rad).
    """
    t_arr = np.asarray(t, dtype=float)
    field_arr = np.asarray(field, dtype=complex)
    mask = np.asarray(mask_t, dtype=bool)
    if np.count_nonzero(mask) < 2:
        raise ValueError("temporal phase needs at least two supported samples")
    phi = np.unwrap(np.angle(field_arr))
    phi = phi - phi[int(np.argmax(np.abs(field_arr) ** 2))]
    phi = _remove_linear(t_arr / 1e-15, phi, mask)
    return phi - np.mean(phi[mask])


def _remove_linear(x, phi, mask):
    """Subtract a linear fit (over ``mask``) from ``phi``; recentre on the peak."""
    coeffs = np.polyfit(x[mask], phi[mask], 1)
    return phi - np.polyval(coeffs, x)


def spectrogram(
    result: RetrievalResult,
    *,
    n_t: int | None = None,
    fw: float | None = None,
    lam_min: float | None = None,
    lam_max: float | None = None,
):
    """Gabor spectrogram of the retrieved pulse (for the dispersion view).

    Returns ``(t_centers, wavelength, S)`` with ``S`` of shape
    ``(Nwavelength, Nt)``, photon-energy weighted and cropped to
    ``[lam_min, lam_max]``.
    """
    grid = result.grid
    t = grid.t
    omega_abs = grid.omega + result.omega0
    field = shiftnorm(grid, grid.ifft(result.spectrum))

    if n_t is None:
        n_t = min(grid.n, 256)
    if fw is None:
        fw = (t.max() - t.min()) / 20.0
    t_centers = np.linspace(t.min() / 2, t.max() / 2, n_t)

    lam = wlfreq(omega_abs)
    order = np.argsort(lam)
    lam_sorted = lam[order]
    if lam_min is None:
        lam_min = float(lam_sorted.min())
    if lam_max is None:
        lam_max = float(lam_sorted.max())
    keep = (lam_sorted > lam_min) & (lam_sorted < lam_max)

    S = np.empty((int(keep.sum()), n_t))
    weight = omega_abs**2
    for j, tc in enumerate(t_centers):
        windowed = field * gauss(t, fwhm=fw, x0=tc)
        spec = np.abs(grid.fft(windowed)) ** 2 * weight
        S[:, j] = spec[order][keep]
    peak = S.max()
    if peak > 0:
        S /= peak
    return t_centers, lam_sorted[keep], S


# ---------------------------------------------------------------------------
# Known ground-truth pulse (synthetic / simulated workflows)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TruthPulse:
    """A known ground-truth pulse, in physical units, for overlaying and saving.

    Carries the temporal intensity envelope and, when available, the complex
    spectrum of a *known* pulse so the retrieval screen can show it beneath the
    retrieved curves, the retrieval pipeline can use it as an initial guess, and
    the saver can record it. Two workflows produce one: the
    synthetic generator (from the generating complex field, via
    :meth:`from_spectrum`) and the numerically simulated-scan loader (from the
    scan's stored reference intensity, via :meth:`from_intensity`).

    The arrays are peak-centred (``t=0`` at the intensity peak) and peak-normalised
    so the overlay sits naturally against the normalised retrieved pulse.

    Attributes
    ----------
    t : numpy.ndarray
        Temporal axis (s), peak-centred on the intensity maximum.
    It : numpy.ndarray
        Temporal intensity (power) envelope, normalised to unit peak.
    fwhm : float
        Intensity FWHM of the temporal envelope (s).
    lam : numpy.ndarray or None
        Wavelength axis (m), ascending, for the spectral intensity (or ``None``).
    Iw : numpy.ndarray or None
        Spectral intensity vs ``lam`` (wavelength density), unit peak (or ``None``).
    phi_t : numpy.ndarray or None
        Temporal phase (rad) on ``t``, in the same display gauge as a retrieved
        phase, or ``None`` when no complex field is available.
    omega : numpy.ndarray or None
        Absolute angular-frequency axis (rad/s), ascending, for ``Eomega``.
    Eomega : numpy.ndarray or None
        Known complex spectrum on ``omega`` in croak's Fourier convention.
    """

    t: NDArray[np.float64]
    It: NDArray[np.float64]
    fwhm: float
    lam: NDArray[np.float64] | None = None
    #: Known spectral phase (rad) on ``lam``, when the source provides a complex
    #: field. Normalised EXACTLY as the retrieved phase is — unwrapped, linear
    #: term removed, zeroed at the spectral peak — so the two are directly
    #: comparable on the same axes. ``None`` when only intensities are known.
    phi_w: NDArray[np.float64] | None = None
    Iw: NDArray[np.float64] | None = None
    phi_t: NDArray[np.float64] | None = None
    omega: NDArray[np.float64] | None = None
    Eomega: NDArray[np.complex128] | None = None

    @classmethod
    def from_spectrum(
        cls, grid: Grid, ew: ArrayLike, omega0: float, *, oversampling: int = 8
    ) -> TruthPulse:
        """Build a :class:`TruthPulse` from a complex spectrum (synthetic pulse).

        The temporal intensity is oversampled (matching
        :func:`process_result`) for a smooth overlay and an accurate FWHM; the
        spectral intensity is the photon-domain ``|E(ω)|²·ω²`` mapped to ascending
        wavelength.

        Parameters
        ----------
        grid : Grid
            Grid on which ``ew`` is sampled.
        ew : array_like
            Complex spectrum in centred order.
        omega0 : float
            Carrier angular frequency (rad/s).
        oversampling : int, optional
            Temporal zero-padding factor.

        Returns
        -------
        TruthPulse
            Known pulse retaining ``ew`` for truth-seeded retrievals.
        """
        ew = np.asarray(ew, dtype=complex)
        field = shiftnorm(grid, grid.ifft(ew))
        t_over, e_over = oversample(grid.t, field, oversampling)
        It = np.abs(e_over) ** 2
        peak = It.max()
        if peak > 0:
            It = It / peak
        t = t_over - t_over[int(np.argmax(It))]
        fwhm = _fwhm(t, It)

        omega_abs = grid.omega + omega0
        Iw = np.abs(ew) ** 2 * omega_abs**2
        if Iw.max() > 0:
            Iw = Iw / Iw.max()
        lam = wlfreq(omega_abs)
        order = np.argsort(lam)
        mask_t = It > 0.01
        phi_t = normalize_temporal_phase(t, e_over, mask_t)
        mask_w = Iw > 0.01
        phi_w = normalize_spectral_phase(grid.omega, ew, mask_w, Iw)
        return cls(
            t=t,
            It=It,
            fwhm=float(fwhm),
            lam=lam[order],
            Iw=Iw[order],
            phi_w=phi_w[order],
            phi_t=phi_t,
            omega=omega_abs,
            Eomega=ew.astype(np.complex128, copy=True),
        )

    @classmethod
    def from_intensity(
        cls,
        t: ArrayLike,
        It: ArrayLike,
        *,
        lam: ArrayLike | None = None,
        Iw: ArrayLike | None = None,
        phi_w: ArrayLike | None = None,
        phi_t: ArrayLike | None = None,
        omega: ArrayLike | None = None,
        Eomega: ArrayLike | None = None,
        fwhm: float | None = None,
    ) -> TruthPulse:
        """Build a :class:`TruthPulse` from a sampled temporal intensity.

        Used when the known pulse is available only as an intensity envelope (no
        phase), e.g. a Luna ``scansave`` reference pulse. Optional phase and
        complex-spectrum arrays preserve the full known field when the source
        file supplies it. ``fwhm`` defaults to the measured intensity FWHM of
        ``(t, It)``.

        Parameters
        ----------
        t, It : array_like
            Temporal axis (s) and matching intensity envelope.
        lam, Iw : array_like, optional
            Matching wavelength axis (m) and spectral intensity.
        phi_w : array_like, optional
            Spectral phase (rad) sampled on ``lam``.
        phi_t : array_like, optional
            Temporal phase (rad) sampled on ``t``.
        omega, Eomega : array_like, optional
            Matching absolute angular-frequency axis (rad/s) and complex
            spectrum in croak's Fourier convention. Supply both or neither.
        fwhm : float, optional
            Known temporal intensity FWHM (s); measured from ``t``/``It`` when
            omitted.

        Returns
        -------
        TruthPulse
            Peak-centred, peak-normalised known pulse.

        Raises
        ------
        ValueError
            If paired axes/fields have inconsistent shapes.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        it_arr = np.asarray(It, dtype=np.float64)
        peak = it_arr.max()
        if peak > 0:
            it_arr = it_arr / peak
        t_arr = t_arr - t_arr[int(np.argmax(it_arr))]
        width = _fwhm(t_arr, it_arr) if fwhm is None else float(fwhm)
        lam_arr = iw_arr = phi_arr = None
        if phi_w is not None:
            phi_arr = np.asarray(phi_w, dtype=np.float64)
        if lam is not None and Iw is not None:
            lam_arr = np.asarray(lam, dtype=np.float64)
            iw_arr = np.asarray(Iw, dtype=np.float64)
            order = np.argsort(lam_arr)
            lam_arr, iw_arr = lam_arr[order], iw_arr[order]
            if phi_arr is not None:
                # Reorder with the SAME permutation: the phase is sampled on the
                # frequency axis the caller passed, so sorting one without the
                # other would silently pair each phase with the wrong wavelength.
                phi_arr = np.asarray(phi_w, dtype=np.float64)[order]
            if iw_arr.max() > 0:
                iw_arr = iw_arr / iw_arr.max()
        phi_t_arr = None if phi_t is None else np.asarray(phi_t, dtype=np.float64)
        omega_arr = None if omega is None else np.asarray(omega, dtype=np.float64)
        eomega_arr = None if Eomega is None else np.asarray(Eomega, dtype=np.complex128)
        if phi_t_arr is not None and phi_t_arr.shape != t_arr.shape:
            raise ValueError("phi_t must have the same shape as t")
        if (omega_arr is None) != (eomega_arr is None):
            raise ValueError("omega and Eomega must be provided together")
        if omega_arr is not None and eomega_arr is not None:
            if omega_arr.shape != eomega_arr.shape:
                raise ValueError("omega and Eomega must have the same shape")
            order_w = np.argsort(omega_arr)
            omega_arr = omega_arr[order_w]
            eomega_arr = eomega_arr[order_w]
        return cls(
            t=t_arr,
            It=it_arr,
            fwhm=float(width),
            lam=lam_arr,
            Iw=iw_arr,
            phi_w=phi_arr,
            phi_t=phi_t_arr,
            omega=omega_arr,
            Eomega=eomega_arr,
        )

    def spectrum_on_grid(self, grid: Grid, omega0: float) -> NDArray[np.complex128]:
        """Return the known complex spectrum on a retrieval grid.

        Parameters
        ----------
        grid : Grid
            Retrieval grid whose frequencies are relative to ``omega0``.
        omega0 : float
            Pulse carrier angular frequency (rad/s).

        Returns
        -------
        numpy.ndarray
            Complex spectrum in centred order. An identical ModelPNPS native
            grid is copied exactly; a regridded workflow interpolates amplitude
            and unwrapped phase and sets frequencies outside the stored range to
            zero.

        Raises
        ------
        ValueError
            If this truth carries no complex spectrum or its spectrum is empty.
        """
        if self.omega is None or self.Eomega is None:
            raise ValueError(
                "the selected truth has no complex spectrum; load a ModelPNPS "
                "file containing Eω_re/Eω_im (or Eω_beamlet_re/_im)"
            )
        source_w = np.asarray(self.omega, dtype=float)
        source_e = np.asarray(self.Eomega, dtype=complex)
        target_w = np.asarray(grid.omega, dtype=float) + float(omega0)
        if source_w.shape == target_w.shape and np.allclose(
            source_w, target_w, rtol=0.0, atol=1e-9 * grid.domega
        ):
            return source_e.astype(np.complex128, copy=True)

        amplitude = np.abs(source_e)
        peak = float(amplitude.max())
        if peak <= 0:
            raise ValueError("the selected truth complex spectrum is identically zero")
        support = amplitude > peak * 1e-12
        if np.count_nonzero(support) < 2:
            raise ValueError(
                "the selected truth complex spectrum has fewer than two bins"
            )
        amplitude_target = np.interp(target_w, source_w, amplitude, left=0.0, right=0.0)
        phase = np.unwrap(np.angle(source_e[support]))
        phase_target = np.interp(target_w, source_w[support], phase)
        return (amplitude_target * np.exp(1j * phase_target)).astype(np.complex128)

    def peak_power(self, energy: float | None) -> float | None:
        """Peak power (W) of this pulse when it carries ``energy`` joules.

        :attr:`It` is peak-normalised, so its integral is the pulse's *effective
        duration* and the peak power is ``energy / ∫(I/I_peak) dt`` — the same
        definition :func:`process_result` uses for the retrieved and
        transform-limited pulses, evaluated over the truth's own time axis.

        Parameters
        ----------
        energy : float or None
            Pulse energy (J). ``None`` or non-positive means the absolute scale is
            unknown.

        Returns
        -------
        float or None
            Peak power in watts, or ``None`` when ``energy`` is unusable or the
            envelope integrates to zero.
        """
        if energy is None or energy <= 0:
            return None
        tau_eff = float(np.trapezoid(self.It, self.t))
        if tau_eff <= 0:
            return None
        return float(energy) / tau_eff


# ---------------------------------------------------------------------------
# ProcessedResult
# ---------------------------------------------------------------------------
@dataclass
class ProcessedResult:
    """Display-ready quantities derived from a :class:`RetrievalResult`."""

    # temporal (oversampled, peak-centred)
    t_over: NDArray[np.float64]
    It_retr: NDArray[np.float64]
    t_tl: NDArray[np.float64]
    It_tl: NDArray[np.float64]
    phi_t: NDArray[np.float64]
    mask_t: NDArray[np.bool_]
    fwhm_retr: float
    fwhm_tl: float
    # spectral (wavelength domain)
    omega: NDArray[np.float64]
    omega0: float
    wavelength: NDArray[np.float64]
    Iw: NDArray[np.float64]
    Ilam: NDArray[np.float64]
    Iw_meas: NDArray[np.float64] | None
    phi_w: NDArray[np.float64]
    phi_w_fit: NDArray[np.float64]
    mask_w: NDArray[np.bool_]
    gdd_fs2: float
    tod_fs3: float
    fod_fs4: float
    #: Fraction of the retrieved pulse energy sitting in the outermost grid bins,
    #: where the trace carries no measurement. A quality flag, not a physical
    #: quantity — see :func:`edge_energy_fraction`.
    edge_energy: float
    # trace, marginals, convergence
    delays: NDArray[np.float64]
    trace_retr: NDArray[np.float64]
    trace_meas: NDArray[np.float64] | None
    residual: NDArray[np.float64] | None
    omega_marg_retr: NDArray[np.float64]
    omega_marg_meas: NDArray[np.float64] | None
    tau_marg_retr: NDArray[np.float64]
    tau_marg_meas: NDArray[np.float64] | None
    errors: list[float]
    error: float
    result: RetrievalResult
    # absolute power scaling (only when a measured pulse energy was supplied)
    energy: float | None = None  # pulse energy used for the scaling (J)
    peak_power: float | None = None  # retrieved-pulse peak power (W)
    peak_power_tl: float | None = None  # transform-limited peak power (W)


def edge_energy_fraction(
    spectrum: ArrayLike, *, collar_bins: int = TAPER_COLLAR_BINS
) -> float:
    r"""Fraction of the retrieved pulse energy sitting in the outermost grid bins.

    A retrieval-quality flag. :func:`croak.preprocess.regrid` tapers the outermost
    ``collar_bins`` frequency rows at each end of the grid to zero, so the trace
    carries no measurement there. The nonlinear mixing still couples those
    frequencies to the live rows, but only weakly — a probe spike at the grid edge
    moves the trace error by ~100× less than the same spike mid-band — so the
    field is nearly free there. Because :math:`|E(\omega)|^2 \ge 0` and the true
    value sits *on* that bound, an unconstrained edge bin can only drift upward,
    which is why the failure always looks like spectral energy appearing at the
    band edges rather than disappearing from them.

    A well-conditioned retrieval on a generously chosen band returns well under
    ``0.01`` here. Values of a few percent mean part of the reported spectrum —
    and therefore part of the reported pulse duration — is an artefact. The usual
    causes, in order of how much they matter:

    1. ``R_omega=True`` without ``reg_spectrum`` or ``phase_only``. Per-frequency
       scaling discards exactly the information that pins the spectral amplitude,
       and it *lowers* the reported trace error while doing so.
    2. A ``lam_min``/``lam_max`` band tight enough that the taper cuts live signal
       (:func:`croak.preprocess.regrid` warns about this at load time via its
       ``taper_warn_level`` and records the discarded fraction as
       :attr:`croak.preprocess.TraceData.taper_loss`).

    Parameters
    ----------
    spectrum : array_like
        Complex spectrum on the retrieval grid, centred order.
    collar_bins : int, optional
        Number of bins at each end to count; defaults to
        :data:`croak.preprocess.TAPER_COLLAR_BINS`, matching the taper.

    Returns
    -------
    float
        Energy fraction in ``[0, 1]``; ``0.0`` for a zero-energy spectrum, and for
        a grid too short for two non-overlapping collars.

    Examples
    --------
    >>> import numpy as np
    >>> ew = np.zeros(64, dtype=complex)
    >>> ew[28:36] = 1.0                       # all energy mid-grid
    >>> float(edge_energy_fraction(ew))
    0.0
    >>> ew[0] = 2.0                           # ... plus a spike on the edge
    >>> round(float(edge_energy_fraction(ew)), 3)
    0.333
    """
    intensity = np.abs(np.asarray(spectrum)) ** 2
    total = float(intensity.sum())
    if total == 0.0 or intensity.size < 2 * collar_bins + 1:
        return 0.0
    edge = float(intensity[:collar_bins].sum() + intensity[-collar_bins:].sum())
    return edge / total


def _peak_normalised_correlation(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> float:
    """Best normalised cross-correlation of two profiles over all relative lags.

    Maximising over lag makes the comparison insensitive to where each profile
    sits on its axis, which matters because a retrieval fixes the pulse only up
    to an arbitrary time origin. Both profiles are mean-subtracted first, so a
    common pedestal cannot inflate the score.

    Returns a value in ``[-1, 1]``, or ``0.0`` if either profile is flat.
    """
    a = a - a.mean()
    b = b - b.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 0.0
    return float(np.correlate(a, b, mode="full").max() / norm)


def resolve_time_direction(
    result: RetrievalResult, truth: TruthPulse, *, tolerance: float = 1e-3
) -> tuple[RetrievalResult, bool]:
    r"""Pick the time direction that matches a known pulse, for ambiguous geometries.

    An SHG-FROG trace is unchanged by :math:`E(t) \to E^*(-t)`: the two pulses fit
    the measurement equally well and a solver returns whichever branch it landed
    in. When the true pulse is known — a synthetic test, a simulated scan, or a
    cross-check against another diagnostic — the ambiguity can be settled by
    picking the branch that matches, which is what this does.

    The choice is made on the **temporal intensity** profiles, compared by their
    best normalised cross-correlation over all relative lags. Using intensity
    rather than the field keeps the comparison independent of the absolute phase
    (also ambiguous), and maximising over lag keeps it independent of the time
    origin (likewise). Neither profile has to be a good match for the comparison
    to work — only for the *right* one to be the better of the two — so this stays
    reliable on an imperfect retrieval.

    Reversal is applied by conjugating the spectrum, which is exactly equivalent
    and leaves the stored trace, ``mu`` and ``error`` valid, since the trace is
    invariant under it. Note that the retrieved GDD, TOD and the whole spectral
    phase change sign, so resolving the direction is what makes those signs
    comparable with the truth.

    Parameters
    ----------
    result : RetrievalResult
        A completed retrieval.
    truth : TruthPulse
        The known pulse to match against.
    tolerance : float, optional
        Minimum improvement in correlation required to flip. A pulse whose
        intensity is close to time-symmetric scores almost equally either way;
        requiring a margin keeps the answer stable against noise rather than
        letting it toggle on a rounding difference. Default ``1e-3``.

    Returns
    -------
    (RetrievalResult, bool)
        The result oriented to match ``truth`` — the input object itself when no
        flip was applied — and whether it was flipped.

    Notes
    -----
    Returns the result unchanged for interactions that fix the direction of time
    (SD, PG), where the retrieved orientation is already meaningful and flipping
    it would be wrong. See
    :attr:`~croak.interactions.Interaction.time_reversal_ambiguous`.
    """
    from .interactions import get_interaction

    if not get_interaction(result.interaction).time_reversal_ambiguous:
        return result, False

    intensity = result.intensity_t
    # The truth is oversampled on its own axis; bring it onto the retrieval grid
    # so the two profiles share a sampling and can be correlated directly.
    truth_on_grid = np.interp(result.t, truth.t, truth.It, left=0.0, right=0.0)
    direct = _peak_normalised_correlation(intensity, truth_on_grid)
    # Reversing the samples is the time flip up to a one-sample offset on a
    # centred grid, and the lag maximisation above absorbs that offset.
    reversed_ = _peak_normalised_correlation(intensity[::-1], truth_on_grid)

    if reversed_ <= direct + tolerance:
        return result, False
    return replace(result, spectrum=np.conj(result.spectrum)), True


def process_result(
    result: RetrievalResult,
    *,
    measured: ArrayLike | None = None,
    Iomega_meas: ArrayLike | None = None,
    oversampling: int = 8,
    energy: float | None = None,
) -> ProcessedResult:
    r"""Compute display-ready temporal/spectral/trace quantities.

    Parameters
    ----------
    result : RetrievalResult
        A completed retrieval.
    measured : array_like, optional
        The measured trace ``(Nomega, Ndelay)`` for marginals/residuals.
    Iomega_meas : array_like, optional
        Independent measured spectrum on the grid (for the spectral panel).
    oversampling : int, optional
        Temporal oversampling factor.
    energy : float, optional
        Measured pulse energy in joules. When given (and positive), the temporal
        intensity is turned into absolute instantaneous power and the peak powers
        are reported. With the intensity normalised to unit peak, instantaneous
        power is :math:`P(t)=E\,I(t)/\!\int I(t)\,dt`, so the **peak power** is
        :math:`E/\!\int I\,dt` (the energy divided by the effective duration).
        The transform-limited reference carries the same energy, so its (shorter)
        duration yields the higher, maximum-achievable peak power. ``None`` or a
        non-positive value leaves the plots in normalised units.

    Returns
    -------
    ProcessedResult
    """
    grid = result.grid
    omega = grid.omega
    omega0 = result.omega0
    ew = result.spectrum

    # -- temporal --
    field = shiftnorm(grid, grid.ifft(ew))
    t_over, e_over = oversample(grid.t, field, oversampling)
    It = np.abs(e_over) ** 2
    It /= It.max()
    t_over = t_over - t_over[int(np.argmax(It))]
    mask_t = It > 0.01
    phi_t = normalize_temporal_phase(t_over, e_over, mask_t)
    fwhm_retr = _fwhm(t_over, It)

    # -- transform-limited reference --
    Iw = np.abs(ew) ** 2
    field_tl = shiftnorm(grid, grid.ifft(np.sqrt(Iw).astype(complex)))
    t_tl, e_tl = oversample(grid.t, field_tl, oversampling)
    It_tl = np.abs(e_tl) ** 2
    It_tl /= It_tl.max()
    t_tl = t_tl - t_tl[int(np.argmax(It_tl))]
    fwhm_tl = _fwhm(t_tl, It_tl)

    # -- absolute power (only when a measured pulse energy was supplied) --
    # P(t) = E·I(t)/∫I dt; with I peak-normalised, ∫I dt is the effective duration
    # τ_eff and the peak power is E/τ_eff. We integrate the oversampled,
    # peak-normalised curves so the plotted absolute power integrates back to E
    # exactly. The TL pulse carries the same energy in a shorter time, so its peak
    # power is the higher, maximum-achievable value.
    peak_power = peak_power_tl = None
    if energy is not None and energy > 0:
        tau_eff = float(np.trapezoid(It, t_over))
        tau_eff_tl = float(np.trapezoid(It_tl, t_tl))
        peak_power = energy / tau_eff if tau_eff > 0 else None
        peak_power_tl = energy / tau_eff_tl if tau_eff_tl > 0 else None

    # -- spectral (wavelength domain) --
    # The ω² factor is the |dω/dλ| Jacobian: it converts the frequency density
    # |Ẽ(ω)|² to a wavelength density, so ``Ilam`` plotted against ``wavelength``
    # has the same shape a spectrometer would record.
    omega_abs = omega + omega0
    Ilam = Iw * omega_abs**2
    Ilam /= Ilam.max()
    wavelength = wlfreq(omega_abs)
    mask_w = (Iw / Iw.max()) > 0.01
    phi_w = normalize_spectral_phase(omega, ew, mask_w, Iw)
    # quartic phase fit -> GDD/TOD/FOD; also a smooth fit curve of the displayed
    # phase. INTENSITY-WEIGHTED (w = sqrt(I) makes the squared residuals
    # I-weighted): an unweighted fit gives the numerous low-intensity wing bins
    # — where the retrieved phase is essentially unconstrained — the same
    # leverage as the peak, and that inflated deep-substrate GDD residuals by
    # up to 0.2 fs² in the 1 fs validation study (a z = 20 µm span read
    # 1.093 unweighted against 1.029 weighted). Fit the phase where
    # there is light. Exact-polynomial phases fit identically either way.
    deg = 4 if int(mask_w.sum()) >= 5 else max(int(mask_w.sum()) - 1, 1)
    w_fit = np.sqrt(Iw[mask_w] / Iw[mask_w].max())
    pc = np.polyfit(omega[mask_w], np.unwrap(np.angle(ew))[mask_w], deg, w=w_fit)
    coef = lambda k: pc[deg - k] if deg >= k else 0.0  # noqa: E731 - coefficient of omega**k
    gdd_fs2 = 2.0 * coef(2) / 1e-30
    tod_fs3 = 6.0 * coef(3) / 1e-45
    fod_fs4 = 24.0 * coef(4) / 1e-60
    phi_w_fit = np.polyval(
        np.polyfit(omega[mask_w], phi_w[mask_w], deg, w=w_fit), omega
    )

    Iw_meas = None
    if Iomega_meas is not None:
        Iw_meas = np.asarray(Iomega_meas, dtype=float) * omega_abs**2
        Iw_meas = Iw_meas / Iw_meas.max()

    # -- trace, marginals, residuals --
    if result.trace is None:
        raise ValueError("process_result requires a result carrying a simulated trace")
    trace_retr = result.trace
    omega_marg_retr, tau_marg_retr = marginals(trace_retr)
    if measured is not None:
        measured = np.asarray(measured, dtype=float)
        omega_marg_meas, tau_marg_meas = marginals(measured)
        resid = residuals(measured, trace_retr)
    else:
        omega_marg_meas = tau_marg_meas = resid = None

    return ProcessedResult(
        t_over=t_over,
        It_retr=It,
        t_tl=t_tl,
        It_tl=It_tl,
        phi_t=phi_t,
        mask_t=mask_t,
        fwhm_retr=fwhm_retr,
        fwhm_tl=fwhm_tl,
        omega=omega,
        omega0=omega0,
        wavelength=wavelength,
        Iw=Iw / Iw.max(),
        Ilam=Ilam,
        Iw_meas=Iw_meas,
        phi_w=phi_w,
        phi_w_fit=phi_w_fit,
        mask_w=mask_w,
        gdd_fs2=gdd_fs2,
        tod_fs3=tod_fs3,
        fod_fs4=fod_fs4,
        edge_energy=edge_energy_fraction(ew),
        delays=result.delays,
        trace_retr=trace_retr,
        trace_meas=measured,
        residual=resid,
        omega_marg_retr=omega_marg_retr,
        omega_marg_meas=omega_marg_meas,
        tau_marg_retr=tau_marg_retr,
        tau_marg_meas=tau_marg_meas,
        errors=result.errors,
        error=result.error,
        result=result,
        energy=energy if (energy is not None and energy > 0) else None,
        peak_power=peak_power,
        peak_power_tl=peak_power_tl,
    )
