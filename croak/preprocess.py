"""Cleaning, filtering and regridding of measured FROG data.

The orchestrator :func:`load_and_clean`
turns a measured FROG trace (on a wavelength axis and a delay/position axis,
with an optional independent fundamental spectrum) into a normalised trace on a
uniform retrieval grid:

    filter (fringe/DC/low-pass) -> spectral window -> regrid
    (lambda->omega resample + anti-alias) -> marginal-correct -> threshold

Background subtraction, calibration and third-order scaling are *not* performed
here: apply them to the trace before calling :func:`load_and_clean`, or fold a
per-wavelength response into its ``scalecurve`` argument. The standalone helpers
:func:`background_subtract`, :func:`calibration_curve_scale` and
:func:`third_order_scale` are provided for that.

The delay filter uses a non-uniform FFT (FINUFFT) so the fringe band-stop and
``dtau_min`` low-pass remain correctly calibrated even when the delay axis is
not uniformly sampled; its stop bands roll off with a steep Planck taper rather
than a brick-wall cut. Thresholding is applied once, at the very end, and is off
by default.

Arrays use the croak convention ``(Nlambda, Ndelay)`` (wavelength on axis 0,
delay on axis 1), matching the retrieval trace orientation ``(Nomega, Ndelay)``.
:func:`load_and_clean` returns a :class:`TraceData`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.fft
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from scipy.linalg import solveh_banded

from .constants import C
from .grid import Grid, gridparams_lambda
from .interactions import get_interaction
from .marginal_checks import apply_marginal_correction
from .maths import fwhm, planck_taper

__all__ = [
    "TraceData",
    "filter_spectrum",
    "filter_frog_trace",
    "defringe_carrier",
    "arpls_baseline",
    "suggested_tau_exclude",
    "resample_spectrum",
    "regrid",
    "background_subtract",
    "calibration_curve_scale",
    "third_order_scale",
    "omega_to_lambda_density",
    "marginal_peak_delay",
    "uniform_core_indices",
    "load_and_clean",
    "TAPER_COLLAR_BINS",
]

#: Width, in grid bins, of the Planck-taper roll-off that :func:`regrid` applies
#: at each end of the retrieval grid's frequency axis. The taper is not optional:
#: the forward model FFTs the trace on a periodic grid, so the resampled trace has
#: to reach zero at the grid boundary or the wrap-around discontinuity rings back
#: into the fit. Its cost is that the outermost ``TAPER_COLLAR_BINS`` rows at each
#: end carry no measurement, which leaves the field at those frequencies almost
#: unconstrained — see :attr:`TraceData.taper_loss` and
#: :func:`croak.processing.edge_energy_fraction`.
TAPER_COLLAR_BINS = 10


# scipy's out-of-range extrapolation modes, as the integer codes its API expects
# (it also accepts these names as strings at runtime, but the type stub does not).
_SPLINE_EXT = {"extrapolate": 0, "zeros": 1, "raise": 2, "const": 3}


def _spline(x, y, *, ext: str = "zeros"):
    """Cubic spline that returns zero (or nearest) outside the data range."""
    return InterpolatedUnivariateSpline(x, y, k=3, ext=_SPLINE_EXT[ext])


def _carrier_frequency(lam: NDArray[np.float64]) -> NDArray[np.float64]:
    """Optical fringe-carrier frequency ``f_c = c/λ`` (Hz) for each wavelength.

    The interferometric cross term of a spectral interferogram oscillates along
    delay as ``e^{iω(λ)τ}`` with ``ω = 2πc/λ``; in ordinary frequency the fringe
    therefore sits at ``f_c = ω/2π = c/λ``. This is fixed by the wavelength
    calibration — known a priori, never fitted — and is why the fringes run
    diagonally across the trace (``f_c`` tilts with ``λ``).
    """
    return C / np.asarray(lam, dtype=float)


def _defringe_cutoff(
    Ifrog: NDArray[np.float64],
    lam: NDArray[np.float64],
    dt: float,
    fraction: float,
    on_alias: str,
) -> NDArray[np.float64]:
    """Per-row low-pass cutoff (Hz) for carrier de-fringing, with a Nyquist guard.

    The cutoff is ``fraction * f_c(λ)`` with ``f_c = c/λ`` — below the fringe
    carrier and above the trace's slow band. The fringe only sits at ``f_c`` when
    the delay is sampled finely enough that ``δτ < λ/(2c)``, i.e.
    ``f_c < f_Nyq = 1/(2 δτ)``; otherwise it aliases to a lower apparent
    frequency. ``on_alias`` selects the response at *signal-bearing* wavelengths
    where the carrier aliases (an alias in a dead corner of the trace is
    harmless):

    * ``"raise"`` (default) — raise :class:`ValueError` naming the offending λ
      and the delay step it needs.
    * ``"warn"`` — :func:`warnings.warn` and proceed with the ``f_c`` cutoff.
    * ``"clamp"`` — low-pass below the *apparent* (folded) carrier instead, a
      best-effort clean of legitimately under-sampled data.
    """
    if on_alias not in ("raise", "warn", "clamp"):
        raise ValueError(
            f"on_alias must be 'raise', 'warn' or 'clamp'; got {on_alias!r}"
        )
    fc = _carrier_frequency(lam)
    f_nyq = 0.5 / dt
    # signal-bearing rows: a carrier aliasing where the trace has no energy does
    # not matter, so only those rows trip the guard.
    marg = np.abs(Ifrog).sum(axis=1)
    signal = (
        marg > 0.03 * marg.max() if marg.max() > 0 else np.zeros_like(marg, dtype=bool)
    )
    aliased = (fc > f_nyq) & signal
    if np.any(aliased) and on_alias != "clamp":
        lam_bad = float(lam[aliased].min())  # shortest λ ⇒ highest f_c ⇒ worst
        need_dt = lam_bad / (2.0 * C)
        msg = (
            f"de-fringe carrier aliases: f_c={C / lam_bad:.3e} Hz at "
            f"λ={lam_bad * 1e9:.1f} nm exceeds the delay Nyquist {f_nyq:.3e} Hz "
            f"(δτ={dt:.3e} s). Sample finer than δτ<λ/(2c)={need_dt:.3e} s, "
            f"or pass on_alias='clamp'."
        )
        if on_alias == "raise":
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    if on_alias == "clamp":
        f_s = 1.0 / dt
        # fold f_c into [0, f_Nyq]; equals f_c when not aliased
        return fraction * np.abs(fc - np.round(fc / f_s) * f_s)
    return fraction * fc


def uniform_core_indices(x: ArrayLike, rtol: float = 1e-3) -> NDArray[np.intp]:
    """Find the longest contiguous uniformly spaced run of ``x``.

    Several simulated campaigns extend a uniform delay core with coarser wing
    points (e.g. a 200-point ±25 fs core at 0.2513 fs plus 1 fs-step wings to
    ±40 fs for the Raman wake). The wings are valuable data, but the smearing
    kernel's delay-axis convolution is evaluated on a *uniform* delay grid, so
    a retrieval that models smearing must restrict itself to the core. This
    finds it: the maximal contiguous stretch whose consecutive spacings agree
    with the stretch's first spacing to ``rtol`` (relative).

    Parameters
    ----------
    x : array_like
        Sample positions, sorted ascending. Not modified.
    rtol : float, optional
        Relative tolerance on the spacing within a run.

    Returns
    -------
    numpy.ndarray of intp
        Indices (into ``x``) of the selected run, contiguous and ascending.
        The whole axis when it is already uniform; ties go to the earliest
        run.

    Examples
    --------
    >>> import numpy as np
    >>> core = np.arange(0.0, 10.0, 0.5)
    >>> wings = np.array([-3.0, -2.0, -1.0])
    >>> x = np.concatenate([wings, core, [11.0, 13.0, 15.0]])
    >>> idx = uniform_core_indices(x)
    >>> np.allclose(x[idx], core)
    True
    """
    x = np.asarray(x, float)
    if x.size <= 2:
        return np.arange(x.size)
    d = np.diff(x)
    best_start, best_stop = 0, 1  # run over the DIFF array, [start, stop)
    start = 0
    for i in range(1, d.size + 1):
        end_of_run = i == d.size or not np.isclose(d[i], d[start], rtol=rtol, atol=0.0)
        if end_of_run:
            if i - start > best_stop - best_start:
                best_start, best_stop = start, i
            start = i
    # A run over diffs [s, e) spans the points s .. e inclusive.
    return np.arange(best_start, best_stop + 1)


def marginal_peak_delay(delays: ArrayLike, marginal: ArrayLike) -> float:
    r"""Locate the peak of a delay marginal to **sub-sample** precision.

    Time zero is not known a priori on a measured scan; it is taken to be where
    the delay marginal peaks. Reading that off as the largest *sample* quantises
    the answer to the delay step, leaving up to half a step of offset — which a
    retrieval then shows as a red/blue dipole straddling the trace in the
    residual, because the measurement and the model sit at slightly different
    delays.

    This refines the estimate by fitting a parabola through the peak sample and
    its two neighbours and returning the vertex, which is exact for a locally
    quadratic peak and costs nothing: the delay axis is only *translated*, never
    resampled, so a fractional shift is as cheap as an integer one.

    The fit is done in local coordinates and the vertex clamped to the bracketing
    samples, so noise cannot throw it outside the interval its three points span.
    A peak sitting on either end of the axis, or a fit that does not curve
    downward, falls back to the sample position.

    Parameters
    ----------
    delays : array_like
        Delay axis (s). Need not be uniformly sampled.
    marginal : array_like
        Delay marginal (the trace summed over frequency), same length.

    Returns
    -------
    float
        Estimated delay (s) of the marginal's peak.

    Examples
    --------
    >>> tau = np.linspace(-10e-15, 10e-15, 21)          # 1 fs steps
    >>> marg = np.exp(-(((tau - 0.4e-15) / 3e-15) ** 2))  # true peak at 0.4 fs
    >>> round(float(marginal_peak_delay(tau, marg)) / 1e-15, 3)  # parabolic estimate
    0.392
    """
    delays = np.asarray(delays, dtype=float)
    marginal = np.asarray(marginal, dtype=float)
    i = int(np.argmax(marginal))
    if i == 0 or i == delays.size - 1:
        return float(delays[i])
    x = delays[i - 1 : i + 2] - delays[i]
    y = marginal[i - 1 : i + 2]
    # Work in units of the local spacing: the raw abscissae are ~1e-14, which
    # would make the Vandermonde solve badly conditioned.
    scale = float(np.max(np.abs(x)))
    if scale == 0.0:
        return float(delays[i])
    curv, slope, _ = np.polyfit(x / scale, y, 2)
    if curv >= 0.0:  # not a maximum — keep the sample
        return float(delays[i])
    vertex = -slope / (2.0 * curv) * scale
    return float(delays[i] + min(max(vertex, float(x[0])), float(x[2])))


def _delay_filter(
    Ifrog: NDArray[np.float64],
    tau: NDArray[np.float64],
    lam: NDArray[np.float64],
    *,
    filter_dc: bool,
    filter_fringes: bool | tuple[float, float],
    dtau_min: float | None,
    defringe_fraction: float | None = None,
    on_alias: str = "raise",
    roll_bins: int = 3,
    eps: float = 1e-9,
) -> NDArray[np.float64]:
    """Band-filter a trace ``(Nlambda, Ndelay)`` along the delay axis.

    Uses a type-1/type-2 NUFFT pair (FINUFFT) so the fringe band-stop and the
    ``dtau_min`` low-pass stay correctly calibrated even when ``tau`` is **not**
    uniformly sampled (the filtering happens before any regridding). For a
    uniform ``tau`` the transform reduces exactly to the plain DFT.

    Four filters are applied to the per-wavelength delay spectrum:

    * **DC** (``filter_dc``): the zero-frequency mode is zeroed exactly — the
      per-wavelength mean over delay, robust to non-uniform sampling.
    * **Low-pass** (``dtau_min``): delay structure faster than ``dtau_min`` is
      removed; the stop band rolls off over ``roll_bins`` modes.
    * **Fringe** notch (``filter_fringes``): the interferometric band straddling
      the carrier ``f_c = c/lambda`` (per wavelength) is removed, also with a
      ``roll_bins``-wide roll-off.
    * **De-fringe** low-pass (``defringe_fraction``): the Takeda-FTSI complement
      to the notch — instead of cutting the fringe band it *keeps* only the slow
      DC band ``|f| < defringe_fraction * f_c(lambda)`` per row (carrier computed,
      not fitted) and discards the ``±f_c`` sidebands, leaving a smooth, fringe-
      free magnitude trace. ``on_alias`` guards the Nyquist condition
      ``f_c < 1/(2 dtau)``.

    The roll-offs reuse :func:`~croak.maths.planck_taper`; a small ``roll_bins``
    keeps the edges steep (little passband smoothing) while avoiding the Gibbs
    ringing of a brick-wall cut.
    """
    import finufft

    n = Ifrog.shape[1]
    dt = float(np.mean(np.abs(np.diff(tau))))
    period = n * dt
    # nodes mapped to [0, 2*pi); a constant offset is irrelevant (it commutes
    # with the diagonal frequency masks and is undone by the inverse transform).
    x = 2.0 * np.pi * (tau - tau[0]) / period
    # integer modes k = -n//2 .. n//2-1 in numpy's fftshift order
    k = scipy.fft.fftshift(scipy.fft.fftfreq(n, d=1.0)) * n
    freq = k / period  # two-sided delay frequency (Hz)
    absf = np.abs(freq)
    df = 1.0 / period
    w = max(int(roll_bins), 1) * df

    values = np.ascontiguousarray(Ifrog, dtype=complex)
    spec = finufft.nufft1d1(x, values, n, isign=-1, eps=eps)

    if filter_dc:
        spec[:, k == 0] = 0.0

    # ``None`` and ``0`` both mean "no low-pass": a zero minimum resolvable step
    # is no constraint at all, and it is the value the GUI stores for "off", so
    # accepting it here keeps the two conventions from diverging (and avoids a
    # division by zero for a caller who reasonably wrote 0).
    if dtau_min:
        cut = 1.0 / dtau_min
        lp = planck_taper(absf, -2 * df, -df, cut, cut + w)
        spec *= lp[None, :]

    if filter_fringes is not False and filter_fringes is not None:
        band = (1.75, 2.25) if filter_fringes is True else filter_fringes
        # the notch straddles the carrier f_c = c/λ: the band edges are multiples
        # of f_c/2 (the half is the round-trip path-difference factor of delay).
        fc = _carrier_frequency(lam)
        for i in range(lam.size):
            lo = 0.5 * band[0] * fc[i]
            hi = 0.5 * band[1] * fc[i]
            spec[i, :] *= 1.0 - planck_taper(absf, lo - w, lo, hi, hi + w)

    if defringe_fraction is not None:
        # Takeda FTSI de-fringe: keep the slow DC band below the carrier f_c(λ)
        # and discard the ±f_c sidebands. A smooth Planck-taper low-pass (no
        # brick wall) avoids the Gibbs ringing that the band-stop notch risks.
        cut_lp = _defringe_cutoff(Ifrog, lam, dt, defringe_fraction, on_alias)
        for i in range(lam.size):
            spec[i, :] *= planck_taper(absf, -2 * df, -df, cut_lp[i], cut_lp[i] + w)

    out = finufft.nufft1d2(x, spec, isign=+1, eps=eps) / n
    return np.real(out)


# ---------------------------------------------------------------------------
# Spectrum windowing
# ---------------------------------------------------------------------------
def filter_spectrum(
    lam: ArrayLike,
    Ilam: ArrayLike,
    lam_min: float,
    lam_max: float,
    *,
    rolloff: int = 10,
) -> NDArray[np.float64]:
    """Planck-taper window applied to a spectrum in the **frequency** domain.

    The window passes ``[lam_min, lam_max]`` and rolls off symmetrically in
    angular frequency (the same frequency width on the blue and red edges,
    ``rolloff`` source samples wide), which is the natural choice for a spectrum
    that will be resampled onto a uniform frequency grid. Normalised to the
    in-band peak; the input is not modified.
    """
    lam = np.asarray(lam, dtype=float)
    Ilam = np.asarray(Ilam, dtype=float)
    band = (lam > lam_min) & (lam < lam_max)
    out = Ilam / np.max(Ilam[band])
    omega = 2 * np.pi * C / lam
    omega_lo = 2 * np.pi * C / lam_max
    omega_hi = 2 * np.pi * C / lam_min
    domega = rolloff * float(np.median(np.abs(np.diff(omega))))
    out = out * planck_taper(
        omega, omega_lo - domega, omega_lo, omega_hi, omega_hi + domega
    )
    out /= np.max(out[band])
    return out


# ---------------------------------------------------------------------------
# FROG trace filtering  (operates on (Nlambda, Ndelay))
# ---------------------------------------------------------------------------
def filter_frog_trace(
    Ifrog: ArrayLike,
    lam: ArrayLike,
    tau: ArrayLike,
    *,
    tau_lims: tuple[float, float] | None = None,
    filter_fringes: bool | tuple[float, float] = False,
    filter_dc: bool = False,
    dtau_min: float | None = None,
    defringe: bool = False,
    defringe_fraction: float = 0.5,
    on_alias: str = "raise",
    baseline: bool = False,
    baseline_smoothness: float = 1e2,
    baseline_ratio: float = 1e-3,
    baseline_clip: bool = False,
    baseline_tau_exclude: float | None = None,
    baseline_downsample: int = 1,
    baseline_skip_below: float = 0.0,
    lamm_lims: tuple[float, float] | None = None,
    subsample: int | None = None,
    scalecurve: ArrayLike | None = None,
    roll_bins: int = 3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Filter a measured FROG trace along the delay axis.

    Removes interferometric fringes, the DC delay component, and delay structure
    faster than ``dtau_min`` (low-pass) via a non-uniform FFT (see
    :func:`_delay_filter`); applies a spectral window (``lamm_lims``), centres
    ``tau=0`` on the delay-marginal peak, applies an optional delay window
    (``tau_lims``) and per-wavelength ``scalecurve``, then normalises and
    optionally subsamples.

    Fringes can be removed two ways — pick one. ``filter_fringes`` is the band-
    stop *notch* at the carrier; ``defringe`` is the Takeda-FTSI *low-pass* that
    keeps the slow DC band and discards the ``±f_c`` sidebands (see
    :func:`defringe_carrier`). They are two solutions to the same problem and are
    **mutually exclusive** — enabling both raises ``ValueError``.

    Parameters
    ----------
    Ifrog : array_like
        Trace ``(Nlambda, Ndelay)``.
    lam : array_like
        Wavelength axis (m), length ``Nlambda``.
    tau : array_like
        Delay axis (s), length ``Ndelay`` (need not be uniformly sampled).
    filter_fringes : bool or tuple, optional
        Band-stop notch at the fringe carrier. Default ``False`` — the filters
        here all remove real signal along with their target, so none is applied
        unless asked for. A tuple gives an explicit stop band.
    filter_dc : bool, optional
        Remove the delay-independent component of each wavelength row. Default
        ``False``; enable only for data with a genuine delay-independent
        background, since on a clean trace it subtracts real signal.
    dtau_min : float, optional
        Low-pass the delay axis at ``1/dtau_min`` (Hz). ``None`` (the default)
        or ``0`` applies no low-pass.
    defringe : bool, optional
        Remove fringes by the Takeda-FTSI carrier low-pass instead of the
        ``filter_fringes`` notch. Default ``False``.
    defringe_fraction : float, optional
        De-fringe low-pass cutoff as a fraction of the carrier ``f_c = c/λ``;
        must lie in ``(0, 1)``. Default ``0.5``. Ignored unless ``defringe``.
    on_alias : {"raise", "warn", "clamp"}, optional
        De-fringe behaviour when the carrier aliases (delay step too coarse);
        see :func:`defringe_carrier`. Default ``"raise"``.
    baseline : bool, optional
        Remove the broad, slowly varying leakage background along delay per
        wavelength row by asymmetric least squares (see :func:`arpls_baseline`),
        applied right after de-fringing. Default ``False``.
    baseline_smoothness, baseline_ratio, baseline_tau_exclude : optional
        Stiffness, convergence tolerance and an optional ``|tau| <
        baseline_tau_exclude`` hold-out, forwarded to :func:`arpls_baseline`.
        Ignored unless ``baseline``.
    baseline_downsample, baseline_skip_below : optional
        Speed controls forwarded to :func:`arpls_baseline` (coarse-grid solve and
        near-zero-row skip). Ignored unless ``baseline``.
    baseline_clip : bool, optional
        Clip the trace to be non-negative after baseline subtraction. Default
        ``False`` (small negatives are kept — the MLE-correct choice).
    roll_bins : int, optional
        Width (in delay-frequency modes) of the steep Planck-taper roll-off on
        the fringe notch and ``dtau_min`` low-pass. Small ⇒ steep edges.

    Returns
    -------
    tuple
        ``(tau_filtered, Ifrog_filtered)``.

    Raises
    ------
    ValueError
        If both ``filter_fringes`` and ``defringe`` are enabled, or if
        ``defringe_fraction`` is outside ``(0, 1)``.

    Notes
    -----
    No thresholding is applied here; small negative values produced by the
    filters are preserved (the maximum-likelihood-correct choice under Gaussian
    noise). :func:`load_and_clean` applies an optional threshold once, at the
    very end.
    """
    if defringe and filter_fringes is not False and filter_fringes is not None:
        raise ValueError(
            "filter_fringes (notch) and defringe (low-pass) are two solutions "
            "to the same problem; enable at most one."
        )
    if defringe and not 0.0 < defringe_fraction < 1.0:
        raise ValueError(
            f"defringe_fraction must lie in (0, 1); got {defringe_fraction}"
        )
    if baseline and baseline_smoothness <= 0.0:
        raise ValueError(
            f"baseline_smoothness must be positive; got {baseline_smoothness}"
        )
    Ifrog = np.asarray(Ifrog, dtype=float)
    lam = np.asarray(lam, dtype=float)
    # Bind to a fresh local (not the ``tau`` parameter): its inferred ndarray type
    # then survives the in-place recentre/subsample below, instead of being reset
    # to the parameter's wide ``ArrayLike`` declared type.
    delays = np.asarray(tau, dtype=float).copy()

    Ifrogf = _delay_filter(
        Ifrog,
        delays,
        lam,
        filter_dc=filter_dc,
        filter_fringes=filter_fringes,
        dtau_min=dtau_min,
        defringe_fraction=defringe_fraction if defringe else None,
        on_alias=on_alias,
        roll_bins=roll_bins,
    )

    # Remove the broad leakage background |E_b|^2 (arPLS) on the de-fringed
    # trace, before centring — a background-free trace makes the marginal-peak
    # centring below unbiased by an asymmetric shelf.
    if baseline:
        Ifrogf = Ifrogf - arpls_baseline(
            Ifrogf,
            smoothness=baseline_smoothness,
            ratio=baseline_ratio,
            tau=delays,
            tau_exclude=baseline_tau_exclude,
            downsample=baseline_downsample,
            skip_below=baseline_skip_below,
        )
        if baseline_clip:
            Ifrogf[Ifrogf < 0.0] = 0.0

    if scalecurve is None:
        scalecurve = np.ones(lam.shape)
    else:
        scalecurve = np.asarray(scalecurve, dtype=float).copy()

    if lamm_lims is not None:
        lmin, lmax = lamm_lims
        dlam = abs(lam[1] - lam[0])
        scalecurve = scalecurve * planck_taper(
            lam, lmin - 10 * dlam, lmin, lmax, lmax + 40 * dlam
        )

    # Centre delay=0 on the delay-marginal peak (robust to hot pixels; matches
    # the criterion regrid() uses later). Sub-sample, because snapping to the
    # nearest sample would leave up to half a delay step of offset — see
    # marginal_peak_delay.
    delays = delays - marginal_peak_delay(delays, Ifrogf.sum(axis=0))

    # soft delay window (Planck taper, the "τm window")
    if tau_lims is not None:
        tmin, tmax = tau_lims
        dt = abs(delays[1] - delays[0])
        Ifrogf = Ifrogf * planck_taper(
            delays, tmin - 10 * dt, tmin, tmax, tmax + 10 * dt
        )

    Ifrogf /= np.max(Ifrogf)
    Ifrogf = Ifrogf * scalecurve[:, None]
    Ifrogf /= np.max(Ifrogf)

    if subsample is not None and subsample > 1:
        delays = delays[::subsample]
        Ifrogf = Ifrogf[:, ::subsample]

    return delays, Ifrogf


def defringe_carrier(
    Ifrog: ArrayLike,
    lam: ArrayLike,
    tau: ArrayLike,
    *,
    carrier_fraction: float = 0.5,
    dtau_min: float | None = None,
    roll_bins: int = 3,
    on_alias: str = "raise",
    eps: float = 1e-9,
) -> NDArray[np.float64]:
    r"""Remove interferometric fringes by keeping the DC band (Takeda FTSI).

    A measured TG/PG-FROG trace decomposes per wavelength row as

    .. math::

        I(\lambda,\tau) = a(\lambda,\tau)
            + c(\lambda,\tau)\,e^{+i\omega(\lambda)\tau}
            + c^{*}(\lambda,\tau)\,e^{-i\omega(\lambda)\tau},

    where the slow term :math:`a = |E_s|^2 + |E_b|^2` is the fringe-free
    magnitude trace and the cross term (the fringes) oscillates at the **optical
    carrier** :math:`\omega(\lambda)=2\pi c/\lambda`, i.e. ordinary fringe
    frequency :math:`f_c(\lambda)=c/\lambda` — *known a priori from the
    wavelength calibration, not fitted*. The fringes are diagonal because
    :math:`f_c` tilts with :math:`\lambda`.

    This keeps only the DC band with a smooth Planck-taper low-pass whose cutoff
    is ``carrier_fraction * f_c(lambda)`` per row — below the carrier and above
    the trace's own slow delay-bandwidth — and discards the :math:`\pm f_c`
    sidebands, recovering a smooth, fringe-free :math:`a(\lambda,\tau)`. Smooth
    windows avoid the Gibbs ringing of a brick-wall notch. The transform is a
    non-uniform FFT, so the cutoff stays correctly calibrated on a jittered or
    unevenly stepped delay axis.

    Unlike DC removal (``filter_dc``), this **preserves** the slow signal content
    including its zero-frequency mean and low-frequency wings; the broad
    leakage background :math:`|E_b|^2` survives here and is removed by a separate
    baseline step.

    Parameters
    ----------
    Ifrog : array_like
        Measured trace ``(Nlambda, Ndelay)`` (delay on axis 1).
    lam : array_like
        Wavelength axis (m), length ``Nlambda``, used to compute the carrier.
    tau : array_like
        Delay axis (s), length ``Ndelay`` (need not be uniformly sampled).
    carrier_fraction : float, optional
        Cutoff as a fraction of the per-row carrier ``f_c = c/lambda``. Must
        satisfy ``0 < carrier_fraction < 1``; default ``0.5`` (mid-way between
        the trace's slow DC band and the fringe carrier).
    dtau_min : float, optional
        Optional absolute low-pass at ``1/dtau_min`` (Hz) applied in addition;
        the more restrictive of the two cutoffs wins per row. ``None`` (default)
        uses only the carrier-relative cutoff.
    roll_bins : int, optional
        Width (delay-frequency modes) of the Planck-taper roll-off. Small ⇒ steep
        edges, less passband smoothing, while still avoiding brick-wall ringing.
        Default ``3``.
    on_alias : {"raise", "warn", "clamp"}, optional
        Behaviour when the carrier ``f_c(lambda)`` exceeds the delay Nyquist
        ``1/(2*dtau)`` (the fringe aliases) at a signal-bearing wavelength.
        Default ``"raise"``.
    eps : float, optional
        NUFFT accuracy. Default ``1e-9``.

    Returns
    -------
    numpy.ndarray
        The smooth fringe-free trace ``a(lambda, tau)``, shape
        ``(Nlambda, Ndelay)``.

    Raises
    ------
    ValueError
        If ``carrier_fraction`` is outside ``(0, 1)``, or if ``on_alias="raise"``
        and the carrier aliases at a signal-bearing wavelength.

    Examples
    --------
    >>> import numpy as np
    >>> from croak.constants import C
    >>> tau = np.linspace(-60e-15, 60e-15, 256)
    >>> lam = np.array([800e-9])
    >>> a = np.exp(-0.5 * (tau / 8e-15) ** 2)            # slow envelope
    >>> fringe = a * np.cos(2 * np.pi * (C / lam[0]) * tau)
    >>> clean = defringe_carrier((a + fringe)[None, :], lam, tau)
    >>> bool(np.allclose(clean[0, 120:136], a[120:136], atol=2e-2))
    True
    """
    if not 0.0 < carrier_fraction < 1.0:
        raise ValueError(f"carrier_fraction must lie in (0, 1); got {carrier_fraction}")
    Ifrog = np.asarray(Ifrog, dtype=float)
    lam = np.asarray(lam, dtype=float)
    tau = np.asarray(tau, dtype=float)
    return _delay_filter(
        Ifrog,
        tau,
        lam,
        filter_dc=False,
        filter_fringes=False,
        dtau_min=dtau_min,
        defringe_fraction=carrier_fraction,
        on_alias=on_alias,
        roll_bins=roll_bins,
        eps=eps,
    )


# ---------------------------------------------------------------------------
# Signal-aware baseline removal (arPLS)
# ---------------------------------------------------------------------------
def _second_diff_normal(n: int) -> NDArray[np.float64]:
    """Upper-banded ``(3, n)`` form of ``DᵀD`` for a 2nd-difference penalty.

    ``D`` is the discrete second-difference operator ``(n-2, n)`` with stencil
    ``[1, -2, 1]``; ``DᵀD`` is symmetric pentadiagonal. The returned array is the
    banded layout :func:`scipy.linalg.solveh_banded` expects with ``lower=False``
    (row 2 the main diagonal, rows 1 and 0 the first and second super-diagonals).
    Built once per delay length and reused across every row and iteration; only
    the diagonal ``W`` changes between solves.
    """
    # Build D explicitly (dense, but only once per call) and form DᵀD — robust for
    # every n >= 3, including the tiny edge cases where the band values differ.
    d = np.zeros((n - 2, n))
    i = np.arange(n - 2)
    d[i, i] = 1.0
    d[i, i + 1] = -2.0
    d[i, i + 2] = 1.0
    dtd = d.T @ d
    ab = np.zeros((3, n))
    ab[2, :] = np.diagonal(dtd)
    ab[1, 1:] = np.diagonal(dtd, 1)
    ab[0, 2:] = np.diagonal(dtd, 2)
    return ab


def _arpls_row(
    y: NDArray[np.float64],
    penalty: NDArray[np.float64],
    ratio: float,
    max_iter: int,
    exclude: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Estimate the arPLS baseline of a single (amplitude-normalised) row.

    Converges on the **baseline** ``z`` (relative change ``< ratio``), not on the
    weights as in the original arPLS. The two reach the same fixed point, but the
    weights keep micro-adjusting near the signal edges long after the baseline has
    settled, so a weight tolerance wastes many iterations (≈28 vs ≈6 on real DUV
    data) for no change in the output we actually want.

    The ``reweighted`` guard makes the first (equally-weighted) solve ineligible to
    satisfy that test: its ``change`` is measured against the *data*, so it asks
    "how far does λ-smoothing move this row", which a row that is already smooth
    answers with "barely". Such a row would otherwise return with its baseline
    equal to itself — subtracted to nothing, having never run the asymmetric
    reweighting that is the whole algorithm.
    """
    scale = float(np.max(np.abs(y)))
    if scale <= 0.0:
        # dead row (e.g. outside the signal band): baseline = the row itself, so
        # the subtracted trace is exactly zero. Avoids a divide-by-zero below.
        return y.copy()
    yn = y / scale
    keep = ~exclude
    w = np.ones_like(yn)
    w[exclude] = 0.0
    z = yn
    reweighted = False
    for _ in range(max_iter):
        ab = penalty.copy()
        ab[2, :] += w  # add W to the main diagonal: A = W + λ DᵀD
        z_new = solveh_banded(ab, w * yn, lower=False)
        change = np.linalg.norm(z_new - z) / max(float(np.linalg.norm(z_new)), 1e-300)
        z = z_new
        if reweighted and change < ratio:
            break  # the baseline has settled
        d = yn - z
        # arPLS: reweight from the *negative* residuals only (the noise floor /
        # background), so points well above the baseline (signal) lose weight.
        neg = d[keep & (d < 0.0)]
        if neg.size == 0:
            break  # baseline already at/above every point — converged
        m = float(neg.mean())
        s = float(neg.std())
        if s <= 0.0:
            break  # degenerate row (z fits exactly); keep current z
        arg = np.clip(2.0 * (d - (2.0 * s - m)) / s, -700.0, 700.0)
        w = 1.0 / (1.0 + np.exp(arg))
        w[exclude] = 0.0
        reweighted = True
    return z * scale


def _grid_norm(n: int) -> float:
    """Stiffness normalisation ``(n/256)^4`` so ``smoothness`` is portable across n.

    A fixed smooth curve sampled on a finer grid has a smaller discrete second
    difference (``Δ²z ∝ z''·Δτ² ∝ 1/n²``), so the penalty energy shrinks as
    ``1/n³`` against an ``O(n)`` data term; the stiffness must grow as ``n⁴`` to
    hold the balance. This also makes a coarse-grid solve (``downsample``) match
    the full-resolution baseline for the same ``smoothness``.
    """
    return (n / 256.0) ** 4


def _arpls_coarse(
    rows: NDArray[np.float64],
    out: NDArray[np.float64],
    active: NDArray[np.intp],
    n: int,
    q: int,
    smoothness: float,
    ratio: float,
    max_iter: int,
    tau_arr: NDArray[np.float64] | None,
    tau_exclude: float | None,
) -> None:
    """Solve the arPLS baseline on every ``q``-th delay; interpolate to full res.

    The baseline is smooth by construction, so a coarse delay grid captures it at
    a fraction of the cost. Each active row is block-averaged onto the coarse
    grid, solved there, and the baseline linearly interpolated back (edge-clamped)
    into ``out`` (which is mutated in place).
    """
    nc = n // q
    if nc < 3:
        raise ValueError("need at least 3 delay samples after downsampling")
    trim = nc * q
    if tau_arr is not None and tau_exclude is not None:
        tau_c = tau_arr[:trim].reshape(nc, q).mean(axis=1)
        exclude_c = np.abs(tau_c) < tau_exclude
    else:
        exclude_c = np.zeros(nc, dtype=bool)
    penalty_c = smoothness * _grid_norm(nc) * _second_diff_normal(nc)
    # coarse sample centres in full-index coordinates, for linear interpolation
    xc = (np.arange(nc) + 0.5) * q
    xf = np.arange(n)
    idx = np.clip(np.searchsorted(xc, xf) - 1, 0, nc - 2)
    frac = np.clip((xf - xc[idx]) / (xc[idx + 1] - xc[idx]), 0.0, 1.0)
    for i in active:
        yc = rows[i, :trim].reshape(nc, q).mean(axis=1)
        zc = _arpls_row(yc, penalty_c, ratio, max_iter, exclude_c)
        out[i] = zc[idx] * (1.0 - frac) + zc[idx + 1] * frac


def arpls_baseline(
    Ifrog: ArrayLike,
    *,
    smoothness: float = 1e2,
    ratio: float = 1e-3,
    max_iter: int = 50,
    tau: ArrayLike | None = None,
    tau_exclude: float | None = None,
    downsample: int = 1,
    skip_below: float = 0.0,
) -> NDArray[np.float64]:
    r"""Estimate a smooth, asymmetric per-row baseline (arPLS, Baek et al. 2015).

    For each wavelength row ``y`` of a trace ``(Nlambda, Ndelay)`` (intensity vs
    delay), fit a free-form smooth baseline ``z`` that rides under the broad,
    slowly varying leakage background :math:`|E_b|^2` while ignoring the compact
    signal peak near ``tau = 0``. The baseline minimises

    .. math::

        F(z) = \sum_i w_i (y_i - z_i)^2 + \lambda \sum_i (\Delta^2 z_i)^2,

    where :math:`\Delta^2` is the discrete second difference (a curvature penalty)
    and :math:`\lambda` (``smoothness``) sets the stiffness. The normal equations
    :math:`(W + \lambda D^{\mathsf{T}}D)\,z = W y` are solved repeatedly; after
    each solve the weights are updated from the residual :math:`d = y - z` using
    only its **negative** part (assumed background/noise): with
    :math:`m=\mathrm{mean}(d<0)`, :math:`s=\mathrm{std}(d<0)`,

    .. math::

        w_i = \frac{1}{1 + \exp\!\big(2\,(d_i - (2s - m))/s\big)},

    so points well above the baseline (signal) are driven toward zero weight and
    the curve settles onto the lower envelope.

    Because ``z`` is a free-form smooth curve fit to that lower envelope — no
    symmetry or shape prior — it follows a background that is **asymmetric about**
    ``tau = 0`` (e.g. a broad leakage shelf strong at negative delay and weak at
    positive delay, as in DUV TG-FROG) just as readily as a flat one (the
    zero-curvature limit). Subtract the result to clean the trace:
    ``Ifrog - arpls_baseline(Ifrog)``.

    Parameters
    ----------
    Ifrog : array_like
        Trace ``(Nlambda, Ndelay)`` (delay on axis 1), or a single row
        ``(Ndelay,)``. Rows are treated independently.
    smoothness : float, optional
        Curvature penalty :math:`\lambda`. Larger ⇒ stiffer (flatter) baseline;
        smaller ⇒ the baseline bends more and risks eating the signal peak. The
        penalty is internally normalised by ``(256/Ndelay)**4`` and each row by
        its peak, so this value is roughly portable across delay-sample counts and
        trace amplitudes. Default ``1e2``; useful range ``~1–1e5``. Must be
        positive.
    ratio : float, optional
        Convergence tolerance on the relative change of the **baseline** between
        iterations, ``||z_new - z|| / ||z_new||``. Default ``1e-3``. Must be
        positive. (Converging on the baseline rather than the weights reaches the
        same fixed point in far fewer iterations.)
    max_iter : int, optional
        Maximum reweighting iterations per row. Default ``50``. Must be ``>= 1``.
    tau : array_like, optional
        Delay axis (s), length ``Ndelay``. Required only when ``tau_exclude`` is
        given (to locate ``|tau| < tau_exclude``).
    tau_exclude : float, optional
        Hold out the delay window ``|tau| < tau_exclude`` (s) from the fit — those
        points keep ``w = 0`` every iteration and are excluded from the residual
        statistics, so the compact signal peak never pulls the baseline up. ``None``
        (default) lets the reweighting find the peak on its own. Requires ``tau``.
    downsample : int, optional
        Solve on every ``downsample``-th delay (block-averaging the row onto the
        coarse grid) and linearly interpolate the baseline back to full
        resolution. The baseline is smooth by construction, so a coarse grid
        captures it at a fraction of the cost; ``smoothness`` is grid-normalised so
        the same value transfers. Default ``1`` (exact, full-resolution). Must be
        ``>= 1``. A value of ``4`` is a good speed/accuracy trade for interactive
        use (≈2.5% baseline difference).
    skip_below : float, optional
        Skip rows whose peak is below ``skip_below`` times the global peak,
        returning a zero baseline for them (no leakage background where there is no
        signal). Default ``0.0`` (process every row). Range ``[0, 1)``.

    Returns
    -------
    numpy.ndarray
        The estimated baseline, same shape as ``Ifrog``.

    Raises
    ------
    ValueError
        If ``smoothness <= 0``, ``ratio <= 0``, ``max_iter < 1``, ``downsample < 1``,
        ``skip_below`` is outside ``[0, 1)``, the input is not 1-D or 2-D, the delay
        axis has fewer than 3 samples (after downsampling), ``tau_exclude`` is given
        without a matching-length ``tau``, or ``tau_exclude`` covers the whole delay
        axis.

    Notes
    -----
    The broad leakage background survives :func:`defringe_carrier` (which removes
    only the interferometric cross term); this is the companion step that removes
    it. The intended order is de-fringe → ``arpls_baseline`` subtract → optional
    clip. No clipping is done here — small negatives in the subtracted trace are
    the maximum-likelihood-correct choice (see the preprocessing how-to); clip
    downstream only if you must.

    References
    ----------
    Baek, Park, Ahn & Choo, "Baseline correction using asymmetrically reweighted
    penalized least squares smoothing", *Analyst* **140**, 250 (2015).

    Examples
    --------
    >>> import numpy as np
    >>> tau = np.linspace(-300e-15, 300e-15, 600)
    >>> shelf = 0.4 * np.exp(np.minimum(-tau / 150e-15, 0.0))  # high at negative delay
    >>> peak = np.exp(-0.5 * (tau / 8e-15) ** 2)               # compact signal
    >>> z = arpls_baseline(shelf + peak, smoothness=1e2)
    >>> wings = np.abs(tau) > 150e-15                          # away from the peak
    >>> bool(np.allclose(z[wings], shelf[wings], atol=5e-2))
    True
    """
    if smoothness <= 0.0:
        raise ValueError(f"smoothness must be positive; got {smoothness}")
    if ratio <= 0.0:
        raise ValueError(f"ratio must be positive; got {ratio}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1; got {max_iter}")
    if downsample < 1:
        raise ValueError(f"downsample must be >= 1; got {downsample}")
    if not 0.0 <= skip_below < 1.0:
        raise ValueError(f"skip_below must lie in [0, 1); got {skip_below}")
    arr = np.asarray(Ifrog, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError("Ifrog must be 1-D (Ndelay,) or 2-D (Nlambda, Ndelay)")
    rows = arr if arr.ndim == 2 else arr[None, :]
    n = rows.shape[1]

    tau_arr: NDArray[np.float64] | None = None
    if tau_exclude is not None:
        if tau is None:
            raise ValueError("tau_exclude requires the tau axis")
        tau_arr = np.asarray(tau, dtype=float)
        if tau_arr.shape != (n,):
            raise ValueError("tau length must match the delay axis")
        if (np.abs(tau_arr) < tau_exclude).all():
            raise ValueError("tau_exclude covers the whole delay axis")

    # Skip near-zero rows (no leakage background where there is no signal): they
    # get a zero baseline, so the subtracted trace is left untouched.
    out = np.zeros_like(rows)
    rowpeak = np.abs(rows).max(axis=1)
    if skip_below > 0.0 and rowpeak.max() > 0.0:
        active = np.nonzero(rowpeak > skip_below * rowpeak.max())[0]
    else:
        active = np.arange(rows.shape[0])

    if downsample > 1:
        _arpls_coarse(
            rows,
            out,
            active,
            n,
            downsample,
            smoothness,
            ratio,
            max_iter,
            tau_arr,
            tau_exclude,
        )
    else:
        if n < 3:
            raise ValueError(
                "need at least 3 delay samples for a 2nd-difference penalty"
            )
        exclude = (
            np.abs(tau_arr) < tau_exclude
            if tau_arr is not None and tau_exclude is not None
            else np.zeros(n, dtype=bool)
        )
        penalty = smoothness * _grid_norm(n) * _second_diff_normal(n)
        for i in active:
            out[i] = _arpls_row(rows[i], penalty, ratio, max_iter, exclude)
    return out if arr.ndim == 2 else out[0]


def suggested_tau_exclude(
    Ifrog: ArrayLike,
    tau: ArrayLike,
    *,
    factor: float = 2.0,
    max_fraction: float = 0.4,
) -> float:
    """Suggest an :func:`arpls_baseline` ``tau_exclude`` from the delay marginal.

    Without a hold-out the arPLS baseline bends up into the signal at ``τ≈0`` and
    eats part of the peak ("peak suck-up"). How much it eats depends on how well
    each wavelength row constrains the fit, so it varies sharply from row to row —
    on a measured DUV trace the worst rows kept 23–36 % of their peak while their
    immediate neighbours kept ~70 %, which reads as a thin stripe at one wavelength
    that appears only when baseline removal is on.

    Holding out the delay range where the signal actually lives fixes it. Twice the
    delay-marginal FWHM is a good width: on that trace it took the worst row from
    0.23 to 0.87 of its peak and left no row below 0.87 (thirteen rows were below
    0.8 unguarded), and it is *faster*, because the fit settles sooner. Once (too
    narrow) and three times (too little wing left to fit) are both worse.

    Parameters
    ----------
    Ifrog : array_like
        Trace, ``(Nlambda, Ndelay)``, or a single delay marginal ``(Ndelay,)``.
        Any stage of processing will do — the width is insensitive to it.
    tau : array_like
        Delay axis (s), matching the last dimension of ``Ifrog``.
    factor : float, optional
        Multiple of the marginal FWHM to hold out (default ``2.0``).
    max_fraction : float, optional
        Cap, as a fraction of the delay *half*-span (default ``0.4``), so the
        hold-out can never approach the whole axis — which
        :func:`arpls_baseline` rejects.

    Returns
    -------
    float
        Hold-out half-width in seconds, for ``arpls_baseline(tau_exclude=...)``.

    Raises
    ------
    ValueError
        If the shapes disagree, or the delay marginal has no discernible peak (its
        half maximum is never crossed), so there is nothing to hold out.

    Examples
    --------
    >>> tau = np.linspace(-50e-15, 50e-15, 401)
    >>> marginal = np.exp(-((tau / 4e-15) ** 2))  # 6.66 fs FWHM
    >>> round(suggested_tau_exclude(marginal, tau) / 1e-15, 2)
    13.32
    """
    arr = np.asarray(Ifrog, dtype=float)
    tau_arr = np.asarray(tau, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError("Ifrog must be 1-D (Ndelay,) or 2-D (Nlambda, Ndelay)")
    if arr.shape[-1] != tau_arr.shape[0]:
        raise ValueError("tau length must match the delay axis")
    marginal = arr if arr.ndim == 1 else arr.sum(axis=0)
    # Measure the width of the peak above its own pedestal: a broad leakage shelf
    # would otherwise drag the half-maximum level up and shrink the FWHM.
    marginal = marginal - marginal.min()
    width = fwhm(tau_arr, marginal)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(
            "the delay marginal has no discernible peak; set tau_exclude by hand"
        )
    cap = max_fraction * 0.5 * float(np.ptp(tau_arr))
    return float(min(factor * width, cap))


# ---------------------------------------------------------------------------
# Wavelength -> frequency resampling (single spectrum)
# ---------------------------------------------------------------------------
def resample_spectrum(
    lam: ArrayLike, Ilam: ArrayLike, omega: ArrayLike, *, prefilter: bool = True
) -> NDArray[np.float64]:
    r"""Resample a spectrum from a wavelength grid onto an (absolute) frequency grid.

    Applies the Jacobian ``Ilam * lam**2`` for the domain change, interpolates
    with a cubic spline, anti-aliases by low-passing in the frequency domain when
    the target grid is coarser than the source, and tapers the edges.

    Parameters
    ----------
    lam : array_like
        Source wavelength axis (m).
    Ilam : array_like
        Source spectral intensity.
    omega : array_like
        Target **absolute** angular-frequency grid (rad/s), e.g. ``g.omega + omega0``.
    prefilter : bool, optional
        Apply the anti-alias low-pass before downsampling.
    """
    lam = np.asarray(lam, dtype=float)
    Ilam = np.asarray(Ilam, dtype=float) * lam**2
    omega = np.asarray(omega, dtype=float)

    pos = omega >= 0.0
    lam_min = 2 * np.pi * C / np.max(omega[pos])
    lam_max = 2 * np.pi * C / np.min(omega[pos])
    in_band = (lam > lam_min) & (lam < lam_max)

    omega_orig = 2 * np.pi * C / lam
    dwo_min = float(np.min(np.abs(np.diff(omega_orig[in_band]))))
    dw = abs(omega[1] - omega[0])

    spl = _spline(omega_orig[::-1], Ilam[::-1])  # ascending after reversal
    if dw <= dwo_min:
        Iw = spl(omega)
    else:
        omega_re = np.arange(omega.min() - dwo_min, omega.max() + dwo_min, dwo_min)
        Iw_re = spl(omega_re)
        # np.asarray: scipy.fft's @_dispatchable stubs lose the concrete ndarray
        # return type, so the in-place band-limit below would not type-check.
        fIw = np.asarray(scipy.fft.rfft(Iw_re))
        if prefilter:
            ffreq = scipy.fft.rfftfreq(len(omega_re), d=dwo_min)
            imax = int(np.argmin(np.abs(ffreq - 1.0 / (2 * dw))))
            fIw[imax:] = 0.0
        Iw_re = scipy.fft.irfft(fIw, n=len(Iw_re))
        Iw = _spline(omega_re, Iw_re)(omega)

    thresh = np.max(Iw) * 1e-10
    Iw[Iw < thresh] = thresh
    return Iw * _collar_taper(omega)


def _collar_taper(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build the Planck taper applied at both ends of the frequency axis.

    Rolls off over :data:`TAPER_COLLAR_BINS` bins at each end. Shared by the
    1-D and batched resamplers so the two can never drift apart.
    """
    dw = abs(omega[1] - omega[0])
    return planck_taper(
        omega,
        omega.min(),
        omega.min() + TAPER_COLLAR_BINS * dw,
        omega.max() - TAPER_COLLAR_BINS * dw,
        omega.max(),
    )


def _resample_trace_batch(
    lam: NDArray[np.float64],
    trace: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    prefilter: bool = True,
) -> tuple[NDArray[np.float64], float]:
    """Resample every delay column from λ to ``omega`` at once (Jacobian + anti-alias).

    Vectorised equivalent of calling :func:`resample_spectrum` per delay — a
    single :class:`~scipy.interpolate.CubicSpline` over all columns instead of
    one spline build per delay.

    Returns the tapered trace and the **taper loss**: the fraction of the
    resampled signal that :func:`_collar_taper` removes. This is the only place
    both the tapered and untapered traces exist, so it is measured here rather
    than reconstructed later — see :attr:`TraceData.taper_loss`.
    """
    Iscaled = trace * (lam**2)[:, None]
    pos = omega >= 0.0
    lam_min = 2 * np.pi * C / np.max(omega[pos])
    lam_max = 2 * np.pi * C / np.min(omega[pos])
    in_band = (lam > lam_min) & (lam < lam_max)
    omega_orig = 2 * np.pi * C / lam
    dwo_min = float(np.min(np.abs(np.diff(omega_orig[in_band]))))
    dw = abs(omega[1] - omega[0])

    order = np.argsort(omega_orig)
    xs, ys = omega_orig[order], Iscaled[order, :]
    if dw <= dwo_min:
        Iw = CubicSpline(xs, ys, axis=0, extrapolate=False)(omega)
    else:
        omega_re = np.arange(omega.min() - dwo_min, omega.max() + dwo_min, dwo_min)
        Iw_re = np.nan_to_num(CubicSpline(xs, ys, axis=0, extrapolate=False)(omega_re))
        # np.asarray: see resample_spectrum — recover the ndarray type scipy.fft drops.
        fIw = np.asarray(scipy.fft.rfft(Iw_re, axis=0))
        if prefilter:
            ffreq = scipy.fft.rfftfreq(len(omega_re), d=dwo_min)
            imax = int(np.argmin(np.abs(ffreq - 1.0 / (2 * dw))))
            fIw[imax:, :] = 0.0
        Iw_re = scipy.fft.irfft(fIw, n=len(omega_re), axis=0)
        Iw = CubicSpline(omega_re, Iw_re, axis=0, extrapolate=False)(omega)
    # Out-of-range extrapolation gaps become 0; negatives produced by the
    # filtering are *kept* (no noise flooring) so they survive to the trace and
    # can be inspected — only an explicit threshold removes them later.
    Iw = np.nan_to_num(Iw)
    taper = _collar_taper(omega)
    out = Iw * taper[:, None]
    # Taper loss: what fraction of the measured signal the collar throws away.
    # Negatives (anti-alias ringing) are clipped first so they cannot cancel real
    # signal and flatter the number. Positive-definite, so the ratio is in [0, 1].
    kept = np.clip(Iw, 0.0, None)
    total = float(kept.sum())
    loss = 1.0 - float((kept * taper[:, None]).sum()) / total if total > 0.0 else 0.0
    return out, loss


# ---------------------------------------------------------------------------
# Regridding onto a uniform retrieval grid
# ---------------------------------------------------------------------------
def regrid(
    tau: ArrayLike,
    lam: ArrayLike,
    trace: ArrayLike,
    lam_spec: ArrayLike | None,
    Ilam_spec: ArrayLike | None,
    interaction: str,
    *,
    lam_min: float,
    lam_max: float,
    trange: float | None = None,
    resample_t: bool = False,
    marginal_correct: bool = False,
    prefilter: bool = True,
    taper_warn_level: float = 0.01,
):
    """Regrid a filtered FROG trace onto a uniform frequency/delay grid.

    Parameters
    ----------
    tau : array_like
        Filtered delay axis (s).
    lam : array_like
        Wavelength axis (m).
    trace : array_like
        Filtered trace ``(Nlambda, Ndelay)``.
    lam_spec, Ilam_spec : array_like or None
        Independent fundamental spectrum, or ``None``.
    interaction : str
        ``"shg"``/``"sd"``/``"pg"`` (sets the trace carrier scaling).
    lam_min, lam_max : float
        Output retrieval wavelength range (m).
    trange : float, optional
        Output temporal window (s); default twice the delay span.
    resample_t : bool, optional
        Resample the delay axis onto the grid time axis.
    marginal_correct : bool, optional
        Apply SHG autoconvolution marginal correction (needs a spectrum).
    prefilter : bool, optional
        Anti-alias before downsampling in :func:`resample_spectrum`.
    taper_warn_level : float, optional
        Warn when the edge taper removes more than this fraction of the measured
        trace, i.e. when ``lam_min``/``lam_max`` are too tight. Default ``0.01``
        (1 %); pass ``0.0`` to silence the check.

    Returns
    -------
    tuple
        ``(tau, grid, omega0_pulse, omega0_trace, trace_regridded, Iomega,
        taper_loss)``.

    Warns
    -----
    UserWarning
        If the spectral taper removes more than ``taper_warn_level`` of the trace.
    """
    # Fresh local (not the ``tau`` parameter) so the inferred ndarray type survives
    # the recentre/resample reassignments below.
    delays = np.asarray(tau, dtype=float).copy()
    lam = np.asarray(lam, dtype=float)
    trace = np.asarray(trace, dtype=float)
    inter = get_interaction(interaction)

    if trange is None:
        trange = 2.0 * float(delays.max() - delays.min())

    n, dt, omega0_pulse = gridparams_lambda(
        trange, lambda_min=lam_min, lambda_max=lam_max
    )
    omega0_trace = omega0_pulse * inter.omega0_scale
    grid = Grid(n, dt=dt)

    out, taper_loss = _resample_trace_batch(
        lam, trace, grid.omega + omega0_trace, prefilter=prefilter
    )
    if 0.0 < taper_warn_level < taper_loss:
        warnings.warn(
            f"the retrieval grid's edge taper removed {taper_loss:.1%} of the "
            f"measured trace: lam_min={lam_min * 1e9:.1f} nm / "
            f"lam_max={lam_max * 1e9:.1f} nm clip signal that is still present. "
            f"regrid tapers the outermost {TAPER_COLLAR_BINS} frequency bins at "
            f"each end of the grid to zero, so that signal is discarded *and* the "
            f"retrieved field is left almost unconstrained at those frequencies — "
            f"which shows up as spurious spectral energy at the band edges (see "
            f"croak.processing.edge_energy_fraction). Widen lam_min/lam_max; the "
            f"wavelength marginal at 1e-4 of its peak is a good guide.",
            stacklevel=3,
        )

    if lam_spec is not None and Ilam_spec is not None:
        Iomega = resample_spectrum(
            lam_spec, Ilam_spec, grid.omega + omega0_pulse, prefilter=prefilter
        )
        Iomega /= np.max(Iomega)
    else:
        Iomega = None

    if marginal_correct:
        if Iomega is None:
            raise ValueError("marginal_correct=True requires an independent spectrum")
        if inter.name != "shg":
            raise ValueError("marginal correction is only implemented for SHG")
        # Scale each frequency row onto the spectrum-predicted SHG marginal (the
        # spectral autoconvolution); shared with the marginal-check preview.
        out = apply_marginal_correction(out, Iomega)

    if resample_t:
        outt = np.empty((grid.n, grid.n))
        for i in range(grid.n):
            outt[i, :] = _spline(delays, out[i, :])(grid.t)
        out = outt
        delays = grid.t.copy()

    # centre delay=0 on the delay-marginal peak, to sub-sample precision
    delays = delays - marginal_peak_delay(delays, out.sum(axis=0))
    return (
        delays,
        grid,
        omega0_pulse,
        omega0_trace,
        out / np.max(np.abs(out)),
        Iomega,
        taper_loss,
    )


# ---------------------------------------------------------------------------
# Background / calibration helpers
# ---------------------------------------------------------------------------
def background_subtract(
    intensity: ArrayLike, background: ArrayLike, *, clamp: bool = False
) -> NDArray[np.float64]:
    """Subtract a background; optionally clamp negatives to zero."""
    out = np.asarray(intensity, dtype=float) - np.asarray(background, dtype=float)
    if clamp:
        out[out < 0.0] = 0.0
    return out


def calibration_curve_scale(
    lam: ArrayLike, calib_lam_nm: ArrayLike, calib_curve: ArrayLike
) -> NDArray[np.float64]:
    """Per-wavelength calibration factors from a ``(λ_nm, scale)`` curve.

    The curve is spline-interpolated (constant extrapolation) onto ``lam`` (m).
    """
    spl = _spline(
        np.asarray(calib_lam_nm, dtype=float),
        np.asarray(calib_curve, dtype=float),
        ext="const",
    )
    return np.asarray(spl(np.asarray(lam, dtype=float) / 1e-9), dtype=float)


def third_order_scale(lam: ArrayLike, exponent: float = 4.0) -> NDArray[np.float64]:
    r"""TG-FROG third-order efficiency scaling :math:`(\lambda/\mu m)^{exp}`."""
    return (np.asarray(lam, dtype=float) / 1e-6) ** exponent


def omega_to_lambda_density(
    intensity_omega: ArrayLike, lam: ArrayLike
) -> NDArray[np.float64]:
    r"""Convert an angular-frequency spectral density to a wavelength density.

    A measured (spectrometer) FROG trace is a **wavelength** density
    :math:`I_\lambda`, and :func:`regrid` multiplies it by :math:`\lambda^2` to
    recover the **angular-frequency** density :math:`I_\omega \propto
    I_\lambda\,\lambda^2` the retrieval forward model produces. Numerically
    simulated traces are already :math:`I_\omega` (uniform in :math:`\omega`), so
    they must be divided by :math:`\lambda^2` *before* entering the wavelength
    pipeline; the subsequent :math:`\times\lambda^2` then returns the original
    :math:`I_\omega` undistorted.

    The Jacobian is :math:`|\mathrm{d}\omega/\mathrm{d}\lambda| = 2\pi c/\lambda^2
    \propto 1/\lambda^2`; the constant :math:`2\pi c` is irrelevant because the
    trace is renormalised downstream, so only the :math:`1/\lambda^2` factor is
    applied.

    Parameters
    ----------
    intensity_omega : array_like
        Spectral intensity as an angular-frequency density. A 1-D spectrum
        ``(Nlambda,)`` or a 2-D trace ``(Nlambda, Ndelay)``.
    lam : array_like
        Wavelength axis (m), aligned with axis 0 of ``intensity_omega``.

    Returns
    -------
    numpy.ndarray
        The corresponding wavelength density (same shape as ``intensity_omega``).
    """
    intensity = np.asarray(intensity_omega, dtype=float)
    lam = np.asarray(lam, dtype=float)
    jac = lam**2
    if intensity.ndim == 2:
        jac = jac[:, None]
    return intensity / jac


# ---------------------------------------------------------------------------
# TraceData & orchestrator
# ---------------------------------------------------------------------------
@dataclass
class TraceData:
    """A cleaned, regridded FROG trace ready for retrieval, plus provenance.

    Attributes
    ----------
    interaction : str
        ``"shg"``/``"sd"``/``"pg"``.
    grid : Grid
        Uniform retrieval grid.
    delays : numpy.ndarray
        Regridded delay axis (s).
    trace : numpy.ndarray
        Regridded, normalised trace ``(Nomega, Ndelay)``.
    omega0_pulse, omega0_trace : float
        Pulse and trace carrier frequencies (rad/s).
    Iomega : numpy.ndarray or None
        Independent fundamental spectrum resampled on the grid, or ``None``.
    lam_min, lam_max : float
        Retrieval wavelength range (m) — the output grid extent.
    lamm_lims : tuple or None
        Measurement wavelength window λm (m): where the trace carries signal.
        Use this (not ``lam_min/lam_max``) for any post-retrieval spectral
        windowing — see :func:`~croak.processing.post_filter`.
    tau_lims : tuple or None
        Measurement delay window τm (s), or ``None``. The companion to
        ``lamm_lims`` for post-retrieval temporal windowing.
    lam_frog, tau_meas, Ifrog_meas, tau_filt, Ifrog_filt
        Measured and filtered trace in the wavelength domain (for plotting).
    lam_spec, Ilam_spec_meas, Ilam_spec_filt
        Independent spectrum (measured/filtered) or ``None``.
    scalecurve : numpy.ndarray
        Per-wavelength scaling applied during filtering.
    taper_loss : float
        Fraction of the measured trace removed by :func:`regrid`'s edge taper
        (see :data:`TAPER_COLLAR_BINS`). A band chosen with room to spare leaves
        this well under ``0.01``; a large value means ``lam_min``/``lam_max`` are
        clipping live signal, which both discards data *and* frees the retrieved
        field at the affected frequencies. ``0.0`` when the trace never went
        through the regrid.
    """

    interaction: str
    grid: Grid
    delays: NDArray[np.float64]
    trace: NDArray[np.float64]
    omega0_pulse: float
    omega0_trace: float
    Iomega: NDArray[np.float64] | None
    lam_min: float
    lam_max: float
    lamm_lims: tuple[float, float] | None
    tau_lims: tuple[float, float] | None
    lam_frog: NDArray[np.float64]
    tau_meas: NDArray[np.float64]
    Ifrog_meas: NDArray[np.float64]
    tau_filt: NDArray[np.float64]
    Ifrog_filt: NDArray[np.float64]
    lam_spec: NDArray[np.float64] | None
    Ilam_spec_meas: NDArray[np.float64] | None
    Ilam_spec_filt: NDArray[np.float64] | None
    scalecurve: NDArray[np.float64]
    taper_loss: float = 0.0

    @property
    def omega(self) -> NDArray[np.float64]:
        """Centred angular-frequency axis (rad/s)."""
        return self.grid.omega


def load_and_clean(
    Ifrog: ArrayLike,
    lam_frog: ArrayLike,
    scanaxis: ArrayLike,
    interaction: str,
    *,
    lam_min: float,
    lam_max: float,
    lamm_lims: tuple[float, float] | None = None,
    input_unit: str = "position",
    trange: float | None = None,
    tau_crop: tuple[float, float] | None = None,
    tau_lims: tuple[float, float] | None = None,
    filter_fringes: bool | tuple[float, float] = False,
    filter_dc: bool = False,
    dtau_min: float | None = None,
    defringe: bool = False,
    defringe_fraction: float = 0.5,
    on_alias: str = "raise",
    baseline: bool = False,
    baseline_smoothness: float = 1e2,
    baseline_ratio: float = 1e-3,
    baseline_clip: bool = False,
    baseline_tau_exclude: float | None = None,
    baseline_downsample: int = 1,
    baseline_skip_below: float = 0.0,
    resample_t: bool = False,
    subsample: int | None = None,
    scalecurve: ArrayLike | None = None,
    threshold: float | None = None,
    roll_bins: int = 3,
    marginal_correct: bool = False,
    prefilter: bool = True,
    taper_warn_level: float = 0.01,
    lam_spec: ArrayLike | None = None,
    Ilam_spec: ArrayLike | None = None,
) -> TraceData:
    r"""Clean and regrid a measured FROG trace from pre-loaded arrays.

    Parameters
    ----------
    Ifrog : array_like
        Measured trace ``(Nlambda, Ndelay)``, a **wavelength density**
        :math:`I_\lambda` (see Notes).
    lam_frog : array_like
        Wavelength axis (m).
    scanaxis : array_like
        Stage position (m) if ``input_unit="position"`` (converted via
        ``τ = 2z/c``) or delay (s) if ``input_unit="delay"``.
    interaction : str
        ``"shg"``/``"sd"``/``"pg"``.
    lam_min, lam_max : float
        Retrieval wavelength range (m).
    lamm_lims : tuple, optional
        Useful measurement wavelength window λm (m); defaults to the full
        extent. Stored on the returned :class:`TraceData` for post-retrieval
        windowing, not used for the retrieval itself.
    input_unit : {"position", "delay"}, optional
        Whether ``scanaxis`` is a stage position (m) or a delay (s). Default
        ``"position"``, converted with ``τ = 2z/c``.
    trange : float, optional
        Time-window length (s) of the retrieval grid. ``None`` (default) uses
        twice the delay span. Together with ``lam_min``/``lam_max`` this fixes
        the grid: the number of points is ``trange`` × the spectral bandwidth,
        while the time step is set by the **wavelength window alone**
        (``dt ≈ 1/Δν``). Widen the window to resolve the pulse more finely;
        lengthen ``trange`` to get more points at the same ``dt``.
    tau_crop : tuple, optional
        ``(τ_min, τ_max)`` in s: discard delays outside this range before
        anything else. ``None`` (default) keeps the whole scan.
    tau_lims : tuple, optional
        Useful measurement delay window τm (s), stored on the returned
        :class:`TraceData` for post-retrieval windowing. Like ``lamm_lims``,
        it does not affect the retrieval.
    filter_fringes : bool or tuple, optional
        Notch out interferometric fringes in the delay-frequency domain.
        **Off by default** — every filter here removes real signal along with
        whatever it targets, so none is applied unless asked for. Turn this on
        for an interferometric measurement that actually carries fringes; a
        tuple gives an explicit stop band. Mutually exclusive with ``defringe``,
        which is the better method when the carrier is known a priori.
    filter_dc : bool, optional
        Remove the delay-independent (zero delay-frequency) component of each
        wavelength row. Default ``False``. Only switch it on when the
        measurement genuinely carries a delay-independent background: on a
        clean trace it subtracts real signal, which shows up as large negative
        excursions in the filtered trace and a percent-level floor on the
        achievable trace error.
    dtau_min : float, optional
        Low-pass the delay axis at ``1/dtau_min`` (Hz), suppressing structure
        faster than the delay stage can resolve. ``None`` (the default) or
        ``0`` applies no low-pass.
    defringe : bool, optional
        Remove fringes by the Takeda-FTSI carrier low-pass (see
        :func:`defringe_carrier`) instead of the ``filter_fringes`` notch — the
        two are mutually exclusive. Default ``False``.
    defringe_fraction : float, optional
        De-fringe cutoff as a fraction of the per-row carrier ``f_c = c/λ``, in
        ``(0, 1)``. Default ``0.5``, midway between the trace's own slow delay
        bandwidth and the fringe carrier.
    on_alias : {"raise", "warn", "clamp"}, optional
        What to do when the carrier exceeds the delay Nyquist frequency, i.e.
        the scan is too coarsely stepped to resolve the fringes. Default
        ``"raise"``.
    baseline : bool, optional
        Remove the broad, slowly varying leakage background ``|E_b|²`` along delay
        (see :func:`arpls_baseline`), applied right after de-fringing. Default
        ``False``.
    baseline_smoothness : float, optional
        arPLS curvature penalty. Larger is stiffer (flatter); too small and the
        baseline bends up into the signal peak. Default ``1e2``.
    baseline_ratio : float, optional
        arPLS convergence tolerance on the relative change of the baseline
        between iterations. Default ``1e-3``.
    baseline_clip : bool, optional
        Clip the baseline-subtracted trace at zero. Default ``False``, which
        keeps negatives visible rather than hiding a bad subtraction.
    baseline_tau_exclude : float, optional
        Hold out ``|τ| <`` this value (s) when fitting the baseline, so a strong
        τ≈0 peak cannot drag it up. ``None`` (default) holds out nothing; see
        :func:`suggested_tau_exclude` for a sensible width.
    baseline_downsample : int, optional
        Solve the baseline on every ``n``-th delay and interpolate back — a
        speed/accuracy trade for interactive use. Default ``1`` (exact).
    baseline_skip_below : float, optional
        Skip baseline fitting on rows whose peak is below this fraction of the
        trace maximum. Default ``0.0`` (fit every row).
    resample_t : bool, optional
        Resample the delay axis onto the retrieval grid's uniform time step.
        Default ``False``, which keeps the measured delays — the forward model
        evaluates at arbitrary delays, so resampling is rarely needed.
    subsample : int, optional
        Keep every ``subsample``-th delay, for thinning an oversampled scan.
        ``None`` (default) keeps them all.
    scalecurve : array_like, optional
        Per-wavelength multiplicative response correction (a calibration curve),
        applied before filtering. ``None`` (default) applies none.
    threshold : float, optional
        Noise floor relative to the peak. ``None`` (the default) keeps all
        values; otherwise applied **once**, at the very end (after regridding).
        Thresholding biases a maximum-likelihood fit — see the
        :doc:`preprocessing guide </howto/preprocessing>` — so prefer leaving it
        off and letting the weighting handle the noise.
    roll_bins : int, optional
        Steepness of the delay-filter stop-band roll-offs; see
        :func:`filter_frog_trace`.
    marginal_correct : bool, optional
        Rescale the trace so its frequency marginal matches the one predicted
        from the independent spectrum (SHG only; see
        :mod:`croak.marginal_checks`). Default ``False``.
    prefilter : bool, optional
        Apply the **anti-alias low-pass** before downsampling in
        :func:`regrid`. Default ``True``, and it should normally stay on:
        without it, spectral structure finer than the retrieval grid folds back
        as aliasing. It is a sharp cut in the Fourier domain, so it rings — the
        filtered trace carries small negative excursions, which croak keeps
        rather than clamping so they stay visible.
    taper_warn_level : float, optional
        Warn when regrid's edge taper removes more than this fraction of the
        measured trace, i.e. when ``lam_min``/``lam_max`` clip live signal.
        Default ``0.01``; pass ``0.0`` to silence it. See
        :attr:`TraceData.taper_loss`.
    lam_spec, Ilam_spec : array_like, optional
        Independent fundamental spectrum, on its own wavelength axis. Used as a
        spectral reference and carried on the result for plotting; it is not
        fitted unless a solver is given ``reg_spectrum``.

    Returns
    -------
    TraceData

    Notes
    -----
    The trace must be a **wavelength density** :math:`I_\lambda`, which is what
    a spectrometer records. :func:`regrid` multiplies by :math:`\lambda^2` to
    recover the :math:`I_\omega` the forward model produces. A numerically
    synthesised trace is already :math:`I_\omega`, so pass it through
    :func:`omega_to_lambda_density` first, or the conversion is applied twice.
    """
    Ifrog = np.asarray(Ifrog, dtype=float)
    lam_frog = np.asarray(lam_frog, dtype=float)
    scanaxis = np.asarray(scanaxis, dtype=float)
    inter = get_interaction(interaction)
    if lamm_lims is None:
        lamm_lims = (float(lam_frog.min()), float(lam_frog.max()))

    if input_unit == "position":
        tau_meas = 2.0 * scanaxis / C
    elif input_unit == "delay":
        tau_meas = scanaxis.copy()
    else:
        raise ValueError("input_unit must be 'position' or 'delay'")

    tau_filt, Ifrog_filt = filter_frog_trace(
        Ifrog,
        lam_frog,
        tau_meas,
        tau_lims=tau_lims,
        filter_fringes=filter_fringes,
        filter_dc=filter_dc,
        dtau_min=dtau_min,
        defringe=defringe,
        defringe_fraction=defringe_fraction,
        on_alias=on_alias,
        baseline=baseline,
        baseline_smoothness=baseline_smoothness,
        baseline_ratio=baseline_ratio,
        baseline_clip=baseline_clip,
        baseline_tau_exclude=baseline_tau_exclude,
        baseline_downsample=baseline_downsample,
        baseline_skip_below=baseline_skip_below,
        lamm_lims=lamm_lims,
        subsample=subsample,
        scalecurve=scalecurve,
        roll_bins=roll_bins,
    )

    if lam_spec is not None and Ilam_spec is not None:
        # The trace λm window maps to the fundamental-spectrum window by the same
        # factor as omega0_scale: for SHG λ_fund = 2·λ_trace (scale 2); for
        # sd/pg the trace is already at the fundamental (scale 1).
        scale = inter.omega0_scale
        Ilam_spec_filt = filter_spectrum(
            lam_spec, Ilam_spec, lamm_lims[0] * scale, lamm_lims[1] * scale
        )
    else:
        Ilam_spec_filt = None

    tau, grid, omega0_pulse, omega0_trace, trace, Iomega, taper_loss = regrid(
        tau_filt,
        lam_frog,
        Ifrog_filt,
        lam_spec,
        Ilam_spec_filt,
        interaction,
        lam_min=lam_min,
        lam_max=lam_max,
        trange=trange,
        resample_t=resample_t,
        marginal_correct=marginal_correct,
        prefilter=prefilter,
        taper_warn_level=taper_warn_level,
    )

    if threshold is not None:
        trace[trace < threshold] = 0.0
        trace /= np.max(trace)

    # hard delay crop (the "Delay window") — the final, authoritative window on
    # the centred retrieval delay axis
    if tau_crop is not None:
        tmin, tmax = tau_crop
        keep = (tau > tmin) & (tau < tmax)
        tau = tau[keep]
        trace = trace[:, keep]

    sc = (
        np.ones(lam_frog.shape) if scalecurve is None else np.asarray(scalecurve, float)
    )
    return TraceData(
        interaction=inter.name,
        grid=grid,
        delays=tau,
        trace=trace,
        omega0_pulse=omega0_pulse,
        omega0_trace=omega0_trace,
        Iomega=Iomega,
        lam_min=lam_min,
        lam_max=lam_max,
        lamm_lims=lamm_lims,
        tau_lims=tau_lims,
        lam_frog=lam_frog,
        tau_meas=tau_meas,
        Ifrog_meas=Ifrog,
        tau_filt=tau_filt,
        Ifrog_filt=Ifrog_filt,
        lam_spec=None if lam_spec is None else np.asarray(lam_spec, float),
        Ilam_spec_meas=None if Ilam_spec is None else np.asarray(Ilam_spec, float),
        Ilam_spec_filt=Ilam_spec_filt,
        scalecurve=sc,
        taper_loss=taper_loss,
    )
