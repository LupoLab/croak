"""Matplotlib plotting for retrieval results.

Provides custom colormaps (white at the low end; a red band for negatives),
reusable **single-axis** plotters (so the GUI embeds exactly the same code), and
**composite** figures: the 12-panel :func:`plot_retrieval`, the 6-panel
:func:`plot_frog_filter`, and :func:`plot_simulated_trace`.

Figures are built with :class:`matplotlib.figure.Figure` directly (no
``pyplot`` global state), so they embed cleanly in a Qt canvas and also save
standalone via ``fig.savefig(...)``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from .maths import wlfreq
from .processing import (
    ProcessedResult,
    TruthPulse,
    marginals,
    process_result,
    spectrogram,
)
from .result import RetrievalResult

if TYPE_CHECKING:
    from .uncertainty import UncertaintyResult

#: Edge-energy fraction (see :func:`croak.processing.edge_energy_fraction`) above
#: which the spectrum panel flags the retrieval in red. A sound retrieval on a
#: generous band sits two orders of magnitude below this.
EDGE_ENERGY_WARN = 0.01

__all__ = [
    "cmap_white",
    "cmap_negwhite",
    "signed_pcolormesh",
    "si_power",
    "sig3",
    "plot_trace",
    "plot_residual",
    "plot_temporal",
    "plot_spectral",
    "EDGE_ENERGY_WARN",
    "plot_convergence",
    "plot_marginal",
    "plot_spectrogram",
    "plot_retrieval",
    "plot_frog_filter",
    "filter_view_vmax",
    "FrogFilterView",
    "plot_simulated_trace",
    "plot_fwhm_distribution",
    "plot_intensity_band",
    "plot_uncertainty",
]

_TWOPI_PHZ = 2 * np.pi * 1e15


def sig3(value: float) -> str:
    """Format a number to three significant digits, for a read-out.

    Three significant digits keeps a read-out equally informative across the
    ranges croak covers (sub-fs attosecond pulses through hundreds of fs;
    nanometre-wide through hundred-nanometre-wide spectra), where a fixed number
    of decimals is too coarse at one end and spuriously precise at the other.
    Unlike the ``g`` format this keeps trailing zeros (so ``7.00``, not ``7``) and
    never switches to exponent notation, both of which read badly beside a sibling
    entry in a legend or a table column.

    Parameters
    ----------
    value : float
        The number to format, already in its display unit.

    Returns
    -------
    str
        The formatted number, with no unit attached.

    Examples
    --------
    >>> sig3(9.512), sig3(7.0), sig3(0.85), sig3(953.2)
    ('9.51', '7.00', '0.850', '953')
    """
    if not np.isfinite(value) or value == 0.0:
        return f"{value:.2f}"
    # digits before the decimal point (<= 0 for |value| < 1), so 3 - that many is
    # how many decimals are still needed to reach three significant digits.
    integer_digits = int(np.floor(np.log10(abs(value)))) + 1
    return f"{value:.{max(0, 3 - integer_digits)}f}"


def _fs(seconds: float) -> str:
    """Format a duration (s) as femtoseconds to three significant digits."""
    return sig3(seconds / 1e-15)


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
def cmap_white(base: str = "viridis", *, n: int = 8, white: bool = True, N: int = 512):
    """Build a base colormap with (optionally) a white point at the low end."""
    import matplotlib as mpl

    cm = mpl.colormaps[base]
    vals = np.linspace(0, 1, n)
    cols = cm(vals)
    if white:
        cols[0] = [1, 1, 1, 1]
    xi = np.linspace(0, 1, N)
    out = np.empty((N, 4))
    for k in range(4):
        out[:, k] = np.interp(xi, vals, cols[:, k])
    return ListedColormap(out)


def cmap_negwhite(
    base: str = "viridis",
    *,
    n: int = 8,
    white: bool = True,
    negfrac: float = 0.0,
    N: int = 512,
):
    """Like :func:`cmap_white` but with a red gradient for negative values."""
    if negfrac <= 0:
        return cmap_white(base, n=n, white=white, N=N)
    n_pos = max(round(N * (1 - negfrac)), 2)
    n_neg = N - n_pos
    pos = cmap_white(base, n=n, white=white, N=n_pos)(np.linspace(0, 1, n_pos))
    neg = np.empty((n_neg, 4))
    for i in range(n_neg):
        t = i / max(n_neg - 1, 1)
        neg[i] = [1.0, t, t, 1.0]
    return ListedColormap(np.vstack([neg, pos]))


def _white_red():
    """White→red colormap used to render negative trace values by magnitude."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("white_red", ["white", "red"])


def signed_pcolormesh(ax, x, y, C, *, db, tracedb=30.0, cmap_pos, vmax=None):
    """Pcolormesh that splits an image's sign onto two colormaps.

    Positive samples are drawn with ``cmap_pos`` (e.g. viridis); negative
    samples are drawn by *magnitude* with a white→red map. Both maps share the
    **same** scale (``[0, vmax]`` linear, ``[-tracedb, 0]`` dB), so a negative is
    only as saturated as a positive of equal magnitude — small negatives stay
    faint instead of dominating — and zero is white in both (pair ``cmap_pos``
    with :func:`cmap_white`). ``C`` follows the usual
    :func:`~matplotlib.axes.Axes.pcolormesh` convention — shape
    ``(len(y), len(x))`` (transpose the FROG ``(Nλ, Nτ)`` arrays yourself).

    Returns ``(mappable_pos, mappable_neg)`` (the second is ``None`` when there
    are no negative samples).
    """
    C = np.asarray(C, dtype=float)
    neg = C < 0
    if db:
        mag = np.clip(10 * np.log10(np.maximum(np.abs(C), 1e-30)), -tracedb, 0.0)
        vmin, vhi = -tracedb, 0.0
    else:
        mag = np.abs(C)
        vmin = 0.0
        vhi = float(np.nanmax(mag)) if vmax is None else vmax
    mpos = ax.pcolormesh(
        x,
        y,
        np.ma.masked_where(neg, mag),
        vmin=vmin,
        vmax=vhi,
        cmap=cmap_pos,
        rasterized=True,
        shading="auto",
    )
    mneg = None
    if neg.any():
        mneg = ax.pcolormesh(
            x,
            y,
            np.ma.masked_where(~neg, mag),
            vmin=vmin,
            vmax=vhi,
            cmap=_white_red(),
            rasterized=True,
            shading="auto",
        )
    return mpos, mneg


# ---------------------------------------------------------------------------
# Single-axis plotters (reused by the composites and the GUI)
# ---------------------------------------------------------------------------
def plot_trace(
    ax, omega, delays, image, omega0_trace, *, clim, cmap, title=None, halfwidth=None
):
    """Plot a 2-D FROG trace (delay fs × frequency PHz)."""
    freq = (np.asarray(omega) + omega0_trace) / _TWOPI_PHZ
    m = ax.pcolormesh(
        np.asarray(delays) / 1e-15,
        freq,
        image,
        vmin=clim[0],
        vmax=clim[1],
        cmap=cmap,
        rasterized=True,
        shading="auto",
    )
    ax.set_xlabel("Delay (fs)")
    ax.set_ylabel("Frequency (PHz)")
    if title:
        ax.set_title(title)
    if halfwidth is not None:
        ax.set_xlim(-halfwidth / 1e-15, halfwidth / 1e-15)
    return m


def plot_residual(ax, omega, delays, resid, omega0, *, halfwidth=None, flim=None):
    """Plot the measured−retrieved residual with a symmetric blue-white-red map.

    ``flim`` optionally bounds the frequency axis, as ``(f_min, f_max)`` in Hz.
    """
    mr = float(np.max(np.abs(resid))) or 1.0
    freq = (np.asarray(omega) + omega0) / _TWOPI_PHZ
    m = ax.pcolormesh(
        np.asarray(delays) / 1e-15,
        freq,
        resid,
        vmin=-mr,
        vmax=mr,
        cmap=cmap_white("bwr", n=32, white=False),
        rasterized=True,
        shading="auto",
    )
    ax.set_xlabel("Delay (fs)")
    ax.set_ylabel("Frequency (PHz)")
    ax.set_title("Residuals")
    if halfwidth is not None:
        ax.set_xlim(-halfwidth / 1e-15, halfwidth / 1e-15)
    if flim is not None:
        ax.set_ylim(flim[0] / 1e15, flim[1] / 1e15)
    return m


_POWER_PREFIXES: tuple[tuple[float, str], ...] = (
    (1e15, "PW"),
    (1e12, "TW"),
    (1e9, "GW"),
    (1e6, "MW"),
    (1e3, "kW"),
    (1.0, "W"),
    (1e-3, "mW"),
    (1e-6, "µW"),
)


def si_power(watts: float) -> tuple[float, str]:
    """Return ``(scale, unit)`` to display ``watts`` with an SI power prefix.

    Picks the largest prefix not exceeding ``|watts|`` (so e.g. ``1.2e7`` W →
    ``(1e6, "MW")``). Non-finite or non-positive inputs fall back to plain watts;
    values below the smallest prefix use that smallest prefix.
    """
    w = abs(float(watts))
    if not np.isfinite(w) or w <= 0:
        return 1.0, "W"
    for scale, unit in _POWER_PREFIXES:
        if w >= scale:
            return scale, unit
    return _POWER_PREFIXES[-1]


def _combined_legend(ax, axp) -> None:
    """Draw one legend containing intensity and twin-axis phase curves."""
    handles, labels = ax.get_legend_handles_labels()
    phase_handles, phase_labels = axp.get_legend_handles_labels()
    ax.legend(
        handles + phase_handles,
        labels + phase_labels,
        loc="best",
        handlelength=1.0,
        frameon=False,
        fontsize="small",
    )


def plot_temporal(
    ax,
    pr: ProcessedResult,
    *,
    halfwidth=None,
    power_scale=None,
    truth: TruthPulse | None = None,
):
    """Temporal intensity/power (left) and phase (right twin axis); phase masked.

    When ``pr.peak_power`` is set (a measured pulse energy was supplied to
    :func:`~croak.processing.process_result`), the intensity is shown as absolute
    instantaneous power on an SI-prefixed axis; otherwise it stays normalised
    ("a.u."). ``power_scale`` is an optional ``(scale, unit)`` pair (see
    :func:`si_power`) so several panels can share one prefix; when ``None`` it is
    derived from the larger of the retrieved/TL peak powers. When ``truth``
    carries temporal phase, that phase is drawn as a dotted line in the same
    gauge as the retrieved phase.
    """
    t = pr.t_over / 1e-15
    if pr.peak_power is not None:
        if power_scale is None:
            ref = max(pr.peak_power, pr.peak_power_tl or 0.0)
            power_scale = si_power(ref)
        scale, unit = power_scale
        retr = pr.It_retr * pr.peak_power / scale
        tl = pr.It_tl * (pr.peak_power_tl or 0.0) / scale
        r_label = f"R ({_fs(pr.fwhm_retr)} fs, {pr.peak_power / scale:.1f} {unit})"
        tl_label = (
            f"TL ({_fs(pr.fwhm_tl)} fs, {(pr.peak_power_tl or 0.0) / scale:.1f} {unit})"
        )
        ylabel = f"Power ({unit})"
    else:
        retr, tl = pr.It_retr, pr.It_tl
        r_label = f"R ({_fs(pr.fwhm_retr)} fs)"
        tl_label = f"TL ({_fs(pr.fwhm_tl)} fs)"
        ylabel = "Power (a.u.)"
    ax.plot(t, retr, c="C0", label=r_label)
    # The TL pulse is peak-centred on its own oversampled axis (its peak index
    # need not coincide with the retrieved pulse's), so plot it against t_tl —
    # using t_over would shift its peak off t=0 by a sample or two.
    ax.plot(pr.t_tl / 1e-15, tl, c="C1", label=tl_label)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)
    ax.set_title("Retrieved pulse")
    if halfwidth is not None:
        ax.set_xlim(-halfwidth / 1e-15, halfwidth / 1e-15)
    axp = ax.twinx()
    axp.plot(t[pr.mask_t], pr.phi_t[pr.mask_t], c="C2")
    axp.set_ylabel("Phase (rad)")
    shown = [pr.phi_t[pr.mask_t]] if np.any(pr.mask_t) else []
    if truth is not None and truth.phi_t is not None:
        truth_mask = np.asarray(truth.It) > 0.01
        axp.plot(
            np.asarray(truth.t)[truth_mask] / 1e-15,
            np.asarray(truth.phi_t)[truth_mask],
            ":",
            c="0.35",
            lw=1.2,
            label="truth phase",
        )
        truth_phase = np.asarray(truth.phi_t)[truth_mask]
        if truth_phase.size:
            shown.append(truth_phase)
    if shown:
        pm = np.concatenate(shown)
        d = pm.max() - pm.min()
        mid = (pm.max() + pm.min()) / 2
        axp.set_ylim(mid - d - 1e-9, mid + d + 1e-9)
    _combined_legend(ax, axp)
    return axp


def plot_spectral(
    ax,
    pr: ProcessedResult,
    *,
    lam_min=None,
    lam_max=None,
    truth: TruthPulse | None = None,
):
    """Spectral intensity vs wavelength (left) and phase (right twin axis).

    ``truth``, when it carries a known spectral phase, is overlaid on the phase
    axis. It is drawn in the SAME gauge as the retrieved phase (unwrapped,
    de-tilted, zeroed at the spectral peak — see
    :func:`croak.processing.normalize_spectral_phase`), because spectral phase
    is defined only up to a constant and a linear term; a truth pinned any other
    way would show a tilt or offset that is gauge, not error.
    """
    lam_nm = pr.wavelength / 1e-9
    ax.plot(lam_nm, pr.Ilam, c="C0", label="R")
    if pr.Iw_meas is not None:
        ax.plot(lam_nm, pr.Iw_meas, c="C1", label="M")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Power (a.u.)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Spectrum")
    if lam_min and lam_max:
        ax.set_xlim(lam_min / 1e-9, lam_max / 1e-9)
    axp = ax.twinx()
    axp.plot(lam_nm[pr.mask_w], pr.phi_w[pr.mask_w], c="C2")
    # dashed polynomial fit of the spectral phase (GDD/TOD visualisation)
    axp.plot(lam_nm[pr.mask_w], pr.phi_w_fit[pr.mask_w], "--", c="C4", alpha=0.7)
    axp.set_ylabel("Phase (rad)")
    shown = [pr.phi_w[pr.mask_w]] if np.any(pr.mask_w) else []
    tphi_src = getattr(truth, "phi_w", None) if truth is not None else None
    tlam_src = getattr(truth, "lam", None) if truth is not None else None
    if tphi_src is not None and tlam_src is not None:
        # Interpolated onto the retrieved wavelength axis: the truth is stored
        # on the simulation's own frequency grid, which is neither the same
        # spacing nor the same extent.
        tphi = np.interp(
            pr.wavelength,
            np.asarray(tlam_src, dtype=float),
            np.asarray(tphi_src, dtype=float),
            left=np.nan,
            right=np.nan,
        )
        axp.plot(
            lam_nm[pr.mask_w],
            tphi[pr.mask_w],
            ":",
            c="0.35",
            lw=1.2,
            label="truth phase",
        )
        finite = tphi[pr.mask_w][np.isfinite(tphi[pr.mask_w])]
        if finite.size:
            shown.append(finite)
    if shown:
        allp = np.concatenate(shown)
        d = allp.max() - allp.min()
        mid = (allp.max() + allp.min()) / 2
        axp.set_ylim(mid - d - 1e-9, mid + d + 1e-9)
    ax.text(
        0.05,
        0.78,
        f"GDD: {pr.gdd_fs2:.1f} fs²\nTOD: {pr.tod_fs3:.1f} fs³",
        transform=ax.transAxes,
        fontsize="small",
    )
    # Energy parked in the unmeasured grid-edge bins: an artefact indicator, so it
    # is drawn in red once it is large enough to distort the reported duration.
    # Threshold is 1 %, comfortably above what a sound retrieval produces.
    ax.text(
        0.05,
        0.70,
        f"Edge: {100 * pr.edge_energy:.2f} %",
        transform=ax.transAxes,
        fontsize="small",
        color="C3" if pr.edge_energy > EDGE_ENERGY_WARN else "0.35",
    )
    _combined_legend(ax, axp)
    return axp


def plot_convergence(ax, errors: ArrayLike, *, boundaries: Sequence[int] = ()):
    """Plot the FROG error vs iteration on a log y-axis.

    ``boundaries`` are indices into ``errors`` where a multi-stage solver handed
    over to its next stage (:attr:`~croak.result.RetrievalResult.stage_boundaries`);
    each is drawn as a dashed rule. Marking them matters because the stages need
    not count the same unit of work — ``warm-lbfgs`` splices COPRA iterations onto
    L-BFGS function evaluations — so an unmarked kink would read as convergence
    behaviour rather than as the change of algorithm it is.
    """
    errors = np.asarray(errors, dtype=float)
    if errors.size:
        ax.semilogy(np.arange(1, errors.size + 1), errors, c="C0")
    for i, edge in enumerate(boundaries):
        if 0 < edge < errors.size:
            ax.axvline(
                edge + 0.5,
                c="0.5",
                ls="--",
                lw=0.8,
                label="stage handover" if i == 0 else None,
            )
    if any(0 < edge < errors.size for edge in boundaries):
        ax.legend(loc="best", handlelength=1.2, frameon=False, fontsize="small")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    ax.set_title("Convergence")


def plot_marginal(
    ax, x, retrieved, measured=None, *, xlabel="", scale=1.0, halfwidth=None
):
    """Plot a (retrieved vs measured) trace marginal."""
    ax.plot(np.asarray(x) / scale, retrieved, c="C0", label="R")
    if measured is not None:
        ax.plot(np.asarray(x) / scale, measured, c="C1", label="M")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Marginal (a.u.)")
    ax.set_ylim(0, 1.05)
    if halfwidth is not None:
        ax.set_xlim(-halfwidth / scale, halfwidth / scale)
    ax.legend(loc="best", handlelength=1.0, frameon=False, fontsize="small")


def plot_spectrogram(
    ax,
    t_centers,
    wavelength,
    S,
    *,
    lam_min=None,
    lam_max=None,
    halfwidth=None,
    cmap=None,
):
    """Plot a Gabor spectrogram (time fs × wavelength nm)."""
    if cmap is None:
        cmap = cmap_white("viridis")
    m = ax.pcolormesh(
        t_centers / 1e-15,
        wavelength / 1e-9,
        S,
        cmap=cmap,
        rasterized=True,
        shading="auto",
    )
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Spectrogram")
    if lam_min and lam_max:
        ax.set_ylim(lam_min / 1e-9, lam_max / 1e-9)
    if halfwidth is not None:
        ax.set_xlim(-halfwidth / 1e-15, halfwidth / 1e-15)
    return m


# ---------------------------------------------------------------------------
# Composite figures
# ---------------------------------------------------------------------------
def _prepare_fig(fig, figsize):
    """Return a cleared figure to draw a composite into (new if None)."""
    if fig is None:
        return Figure(figsize=figsize, layout="constrained")
    fig.clear()
    with contextlib.suppress(Exception):
        fig.set_layout_engine("constrained")
    return fig


def _overlay_truth(
    ax_t,
    ax_w,
    truth: TruthPulse,
    *,
    ax_t_phase=None,
    ax_w_phase=None,
    energy=None,
    power_scale=None,
) -> None:
    """Overlay a known ground-truth pulse beneath the temporal/spectral curves.

    ``truth`` is a :class:`~croak.processing.TruthPulse` (a synthetic generating
    field, or a simulated scan's reference intensity). It is drawn as a *thin
    solid black line under* the retrieved/TL curves (a lower ``zorder``), in
    physical units (fs, nm) so it is independent of the retrieval grid. Note a
    retrieval is only defined up to a time/direction ambiguity, so the truth may
    appear shifted/flipped relative to the retrieved pulse. The temporal legend
    entry carries the truth's intensity FWHM (fs) so it reads alongside the
    retrieved and transform-limited durations. The spectral overlay is drawn only
    when the truth carries a spectrum.

    When ``energy`` is given the temporal overlay is shown in absolute power on the
    shared ``power_scale`` ``(scale, unit)``, via
    :meth:`~croak.processing.TruthPulse.peak_power` — the truth carries the same
    energy, so its peak power is ``energy/∫(I/I_peak)dt`` over its *own* duration.
    """
    t = truth.t / 1e-15
    It_plot = truth.It
    peak_power = truth.peak_power(energy)
    if peak_power is not None:
        scale = power_scale[0] if power_scale is not None else 1.0
        It_plot = truth.It * peak_power / scale
    # Label the truth with its own FWHM, matching the "R (… fs)" / "TL (… fs)"
    # legend entries in plot_temporal so the three durations compare directly.
    ax_t.plot(
        t,
        It_plot,
        c="k",
        lw=0.8,
        zorder=1,
        label=f"truth ({_fs(truth.fwhm)} fs)",
    )
    if ax_t_phase is None:
        ax_t.legend(loc="best", handlelength=1.0, frameon=False, fontsize="small")
    else:
        _combined_legend(ax_t, ax_t_phase)

    if truth.lam is not None and truth.Iw is not None:
        ax_w.plot(truth.lam / 1e-9, truth.Iw, c="k", lw=0.8, zorder=1, label="truth")
        if ax_w_phase is None:
            ax_w.legend(loc="best", handlelength=1.0, frameon=False, fontsize="small")
        else:
            _combined_legend(ax_w, ax_w_phase)


def plot_retrieval(
    result: RetrievalResult,
    *,
    measured: ArrayLike | None = None,
    Iomega_meas: ArrayLike | None = None,
    lam_min: float | None = None,
    lam_max: float | None = None,
    flim: tuple[float, float] | None = None,
    tracedb: float = 30.0,
    cmap: str = "viridis",
    truth: TruthPulse | None = None,
    energy: float | None = None,
    figsize=(17.0, 9.45),
    fig: Figure | None = None,
    processed: ProcessedResult | None = None,
) -> Figure:
    """12-panel retrieval summary: the standard at-a-glance quality check.

    ``lam_min``/``lam_max`` (m) bound the **wavelength** axes — the spectrum and
    spectrogram panels. ``flim`` bounds the **frequency** axis of the trace and
    residual panels, as ``(f_min, f_max)`` in Hz; the default shows the whole
    retrieval grid. The two are separate because the trace sits at the *signal*
    frequency, which for SHG is twice the pulse's. Reach for ``flim`` when the
    retrieval grid is much wider than the signal — typical when retrieving
    directly on a synthesis grid rather than on one cropped by
    :func:`~croak.preprocess.load_and_clean` — where the trace would otherwise be
    a thin stripe in a mostly empty axis.

    ``energy`` is an optional measured pulse energy (J); when given, the temporal
    panel is shown in absolute power (see :func:`~croak.processing.process_result`).

    ``processed`` is an optional, already-computed
    :class:`~croak.processing.ProcessedResult` **for this same result**: pass it when
    the caller needs the processed quantities too (the GUI shows them in a table
    beside the figure) so they are not derived twice. It must have been built with
    the same ``measured``/``Iomega_meas``/``energy``, which are then used only for
    the trace panels; a mismatched one raises :class:`ValueError`.

    Raises
    ------
    ValueError
        If ``result`` carries no trace, or ``processed`` belongs to another result.
    """
    omega = result.omega
    omega0 = result.omega0
    omega0_trace = omega0 * _interaction_scale(result.interaction)
    delays = result.delays
    if result.trace is None:
        raise ValueError("plot_retrieval requires a result carrying a simulated trace")
    if measured is None:
        measured = result.trace
    measured = np.asarray(measured, dtype=float)

    if processed is None:
        pr = process_result(
            result, measured=measured, Iomega_meas=Iomega_meas, energy=energy
        )
    elif processed.result is not result:
        # Cheap exact guard: mixing a ProcessedResult from another result would
        # draw the trace panels from one retrieval and the pulse/spectrum panels
        # from another, which is very hard to spot by eye.
        raise ValueError("processed must be the ProcessedResult of this result")
    else:
        pr = processed
    if lam_min is None or lam_max is None:
        lams = wlfreq(omega + omega0)
        lam_min, lam_max = float(lams.min()), float(lams.max())
    halfwidth = (delays.max() - delays.min()) / 2

    # Trace panels: positive values on viridis with white at the low end, so
    # zero is white in *both* maps and meets the white→red negative map
    # seamlessly. The linear pair shares one scale so its colorbar is meaningful.
    cmap_pos = cmap_white(cmap)
    lin_vmax = max(
        float(np.nanmax(np.abs(measured))), float(np.nanmax(np.abs(result.trace)))
    )

    fig = _prepare_fig(fig, figsize)
    axd = fig.subplot_mosaic("abej\ncdfk\nghil")

    def _trace_panel(ax, C, *, db, title):
        mpos, mneg = signed_pcolormesh(
            ax,
            delays / 1e-15,
            (omega + omega0_trace) / _TWOPI_PHZ,
            C,
            db=db,
            tracedb=tracedb,
            cmap_pos=cmap_pos,
            vmax=None if db else lin_vmax,
        )
        ax.set_xlabel("Delay (fs)")
        ax.set_ylabel("Frequency (PHz)")
        ax.set_title(title)
        if halfwidth is not None:
            ax.set_xlim(-halfwidth / 1e-15, halfwidth / 1e-15)
        if flim is not None:
            ax.set_ylim(flim[0] / 1e15, flim[1] / 1e15)
        return mpos, mneg

    def _pair_cbars(axes, mpos, *negs):
        fig.colorbar(mpos, ax=axes, shrink=0.6)
        mneg = next((m for m in negs if m is not None), None)
        if mneg is not None:
            fig.colorbar(mneg, ax=axes, shrink=0.6, label="neg |·|")

    m_a, n_a = _trace_panel(axd["a"], measured, db=True, title="Measured (log)")
    _m_b, n_b = _trace_panel(axd["b"], result.trace, db=True, title="Retrieved (log)")
    _pair_cbars([axd["a"], axd["b"]], m_a, n_b, n_a)

    m_c, n_c = _trace_panel(axd["c"], measured, db=False, title="Measured (lin)")
    _m_d, n_d = _trace_panel(axd["d"], result.trace, db=False, title="Retrieved (lin)")
    _pair_cbars([axd["c"], axd["d"]], m_c, n_d, n_c)
    axd["d"].text(
        0.04,
        0.96,
        f"{result.error * 100:.2f}%\n{result.trace.shape[0]}×{result.trace.shape[1]}",
        transform=axd["d"].transAxes,
        ha="left",
        va="top",
        color="black",
        fontweight="bold",
        fontsize="small",
    )

    if pr.residual is not None:
        m_h = plot_residual(
            axd["h"], omega, delays, pr.residual, omega0, halfwidth=halfwidth, flim=flim
        )
        fig.colorbar(m_h, ax=[axd["h"]], shrink=0.6)

    # When a pulse energy was supplied, derive one SI power prefix for the
    # temporal panel (and the truth overlay) so both share a consistent axis.
    power_scale = None
    if pr.peak_power is not None:
        power_scale = si_power(max(pr.peak_power, pr.peak_power_tl or 0.0))
    ax_t_phase = plot_temporal(
        axd["e"],
        pr,
        halfwidth=halfwidth,
        power_scale=power_scale,
        truth=truth,
    )
    # truth passed through so a known spectral phase is drawn on the phase axis
    # alongside the retrieved one (the intensity overlay is _overlay_truth's job)
    ax_w_phase = plot_spectral(
        axd["f"], pr, lam_min=lam_min, lam_max=lam_max, truth=truth
    )
    if truth is not None:
        _overlay_truth(
            axd["e"],
            axd["f"],
            truth,
            ax_t_phase=ax_t_phase,
            ax_w_phase=ax_w_phase,
            energy=pr.energy,
            power_scale=power_scale,
        )
    plot_convergence(axd["g"], pr.errors, boundaries=pr.result.stage_boundaries)

    # panel i: per-frequency scale factor (Rω filter), else flat
    mu = np.atleast_1d(np.asarray(result.mu, dtype=float))
    R = mu / mu.max() if mu.size == omega.size else np.ones_like(omega)
    axd["i"].plot(wlfreq(omega + omega0) / 1e-9, R, c="C2")
    axd["i"].set_xlabel("Wavelength (nm)")
    axd["i"].set_ylabel("Filter factor")
    axd["i"].set_title("Spectral filter")
    axd["i"].set_xlim(lam_min / 1e-9, lam_max / 1e-9)

    plot_marginal(
        axd["j"],
        omega + omega0,
        pr.omega_marg_retr,
        pr.omega_marg_meas,
        xlabel="Frequency (PHz)",
        scale=_TWOPI_PHZ,
    )
    axd["j"].set_title("Frequency marginal")
    plot_marginal(
        axd["k"],
        delays,
        pr.tau_marg_retr,
        pr.tau_marg_meas,
        xlabel="Delay (fs)",
        scale=1e-15,
        halfwidth=halfwidth,
    )
    axd["k"].set_title("Delay marginal")

    tc, lam, S = spectrogram(result, lam_min=lam_min, lam_max=lam_max)
    plot_spectrogram(
        axd["l"],
        tc,
        lam,
        S,
        lam_min=lam_min,
        lam_max=lam_max,
        halfwidth=halfwidth,
        # white at zero, like the trace panels and the dispersion stage's own
        # spectrogram — this panel was the only one left on plain viridis.
        cmap=cmap_white(cmap),
    )
    return fig


def filter_view_vmax(td) -> float:
    """Linear-row colour scale for the six-panel filter view.

    Returns the largest **positive** sample across the measured, filtered and
    regridded traces, taking the two wavelength-domain arrays only over the
    measurement window ``td.lamm_lims`` — "where the trace carries signal".

    Both restrictions matter on real data. An absolute-calibration curve amplifies
    the background-subtracted noise far outside the signal band by orders of
    magnitude, so scaling by ``max|·|`` (as this used to) lets a large *negative*
    excursion set the *positive* scale: on a measured DUV trace whose signal peaks
    at ``+1`` but dips to ``-5.7`` at 1097 nm, every linear panel rendered its
    signal at 18 % of the colour range. The Load stage's preview already scales to
    the in-band signal for the same reason.

    The three panels share this one number so the row's single colorbar means
    something; because each array is peak-normalised upstream they are comparable.
    Falls back to the full wavelength range when no window is set, and to
    ``max|·|`` (then ``1.0``) if nothing positive remains, so the result is always
    positive.
    """

    def _positive_max(arr, lam=None) -> float:
        a = np.asarray(arr, dtype=float)
        if lam is not None and td.lamm_lims is not None:
            lo, hi = td.lamm_lims
            band = (np.asarray(lam, dtype=float) >= lo) & (
                np.asarray(lam, dtype=float) <= hi
            )
            if band.any():
                a = a[band]
        hi = float(np.nanmax(a)) if a.size else 0.0
        return hi if hi > 0.0 else 0.0

    vmax = max(
        _positive_max(td.Ifrog_meas, td.lam_frog),
        _positive_max(td.Ifrog_filt, td.lam_frog),
        _positive_max(td.trace),
    )
    if vmax > 0.0:
        return vmax
    # Degenerate (all-negative or all-zero) trace: keep a usable, positive scale.
    fallback = max(
        float(np.nanmax(np.abs(td.Ifrog_meas))),
        float(np.nanmax(np.abs(td.Ifrog_filt))),
        float(np.nanmax(np.abs(td.trace))),
    )
    return fallback if fallback > 0.0 else 1.0


def plot_frog_filter(
    td,
    *,
    tracedb: float = 30.0,
    cmap: str = "viridis",
    figsize=(9.45, 4.72),
    fig: Figure | None = None,
) -> Figure:
    """6-panel measured/filtered (wavelength-domain) + regridded (frequency-domain).

    Positive values use the viridis (white-low) map; negative values — kept
    through filtering rather than clamped — are drawn by magnitude on a
    white→red map so they are visible and not mistaken for signal. Each row
    (the dB row and the linear row) carries one positive + one negative
    colorbar on the right-hand side.
    """
    fig = _prepare_fig(fig, figsize)
    axd = fig.subplot_mosaic("abe\ncdf")
    cm = cmap_white(cmap)
    # The linear row shares a single scale so its one colorbar is meaningful.
    lin_vmax = filter_view_vmax(td)

    def _panel(ax, x, y, C, *, db, title, xlabel):
        mpos, mneg = signed_pcolormesh(
            ax,
            x,
            y,
            C,
            db=db,
            tracedb=tracedb,
            cmap_pos=cm,
            vmax=None if db else lin_vmax,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Delay (fs)")
        ax.set_title(title)
        return mpos, mneg

    def _row_cbars(axes, mpos, negs):
        """One positive + (if any) one negative colorbar at the row's right."""
        fig.colorbar(mpos, ax=axes, shrink=0.5)
        mneg = next((m for m in negs if m is not None), None)
        if mneg is not None:
            fig.colorbar(mneg, ax=axes, shrink=0.5, label="neg |·|")

    lam_nm = td.lam_frog / 1e-9
    freq = (td.omega + td.omega0_trace) / _TWOPI_PHZ
    WL = "Wavelength (nm)"
    HZ = "Frequency (PHz)"

    # top row: dB (log) — measured, filtered, regridded
    pa, na = _panel(
        axd["a"],
        lam_nm,
        td.tau_meas / 1e-15,
        td.Ifrog_meas.T,
        db=True,
        title="Measured (dB)",
        xlabel=WL,
    )
    _pb, nb = _panel(
        axd["b"],
        lam_nm,
        td.tau_filt / 1e-15,
        td.Ifrog_filt.T,
        db=True,
        title="Filtered (dB)",
        xlabel=WL,
    )
    _pe, ne = _panel(
        axd["e"],
        freq,
        td.delays / 1e-15,
        td.trace.T,
        db=True,
        title="Regridded (dB)",
        xlabel=HZ,
    )
    _row_cbars([axd["a"], axd["b"], axd["e"]], pa, (nb, ne, na))

    # bottom row: linear — measured, filtered, regridded
    pc, nc = _panel(
        axd["c"],
        lam_nm,
        td.tau_meas / 1e-15,
        td.Ifrog_meas.T,
        db=False,
        title="Measured (lin)",
        xlabel=WL,
    )
    _pd, nd = _panel(
        axd["d"],
        lam_nm,
        td.tau_filt / 1e-15,
        td.Ifrog_filt.T,
        db=False,
        title="Filtered (lin)",
        xlabel=WL,
    )
    _pf, nf = _panel(
        axd["f"],
        freq,
        td.delays / 1e-15,
        td.trace.T,
        db=False,
        title="Regridded (lin)",
        xlabel=HZ,
    )
    _row_cbars([axd["c"], axd["d"], axd["f"]], pc, (nd, nf, nc))
    return fig


def _decimate_for_display(C, x, y, *, cap=1000):
    """Stride a ``(ny, nx)`` image and its axes down to at most ``cap`` per side.

    The filter figure panels are only a few hundred screen pixels wide, so
    rasterising a multi-thousand-sample trace is wasted work. Plain strided
    subsampling is enough for a diagnostic preview (no anti-alias needed — the
    physics-correct arrays are kept in ``td``; this only thins what is drawn).
    """
    ny, nx = C.shape
    sy = max(ny // cap, 1)
    sx = max(nx // cap, 1)
    if sy == 1 and sx == 1:
        return C, x, y
    return C[::sy, ::sx], x[::sx], y[::sy]


def _resample_uniform_x(x, C):
    """Resample ``C`` (``(ny, nx)``) along axis 1 onto a uniform ``x`` grid.

    :class:`~matplotlib.image.AxesImage` (``imshow``) assumes evenly spaced pixels
    and is positioned by a single linear ``extent``. A *wavelength* axis derived
    from a uniform **frequency** grid (the simulated-trace case) is strongly
    non-uniform, so a feature would otherwise be painted at the wrong wavelength
    (e.g. a 260 nm signal drawn near 480 nm over a 150–800 nm span). Linearly
    interpolating each delay row onto an evenly spaced ``x`` restores the correct
    positions; an already-uniform axis (the frequency panels, or a uniform-λ
    spectrometer trace) is returned unchanged. ``x`` must be ascending.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x, C
    d = np.diff(x)
    if np.allclose(d, d[0], rtol=1e-3, atol=0.0):
        return x, C  # already uniform — no interpolation needed
    xu = np.linspace(x[0], x[-1], x.size)
    Cu = np.empty_like(C)
    for i in range(C.shape[0]):
        Cu[i] = np.interp(xu, x, C[i])
    return xu, Cu


class FrogFilterView:
    """Stateful, artist-reusing twin of :func:`plot_frog_filter` for the GUI.

    :func:`plot_frog_filter` clears and rebuilds the whole figure (6 panels ×
    up to 2 ``pcolormesh`` + 4 colorbars) on every call — ~150–400 ms, the
    dominant cost of the GUI preprocessing redraw. This view builds the mosaic,
    images and colorbars **once** (:meth:`attach`); each :meth:`update` only
    pushes new data via ``set_data``/``set_extent``/``set_clim`` on persistent
    :class:`~matplotlib.image.AxesImage` objects, which benchmarks ~4× faster.

    The signed-split rendering of :func:`signed_pcolormesh` is preserved: each
    panel stacks a positive image (``cmap_white(viridis)``) under a
    negative-magnitude image (white→red), both masked and sharing one scale.
    Uses ``imshow`` (uniform-pixel) with an ``extent``; the wavelength axis may
    be mildly non-uniform but the distortion is negligible for a preview.
    """

    _PANELS = (
        # key, title, xlabel-kind ("wl"/"hz"), db
        ("a", "Measured (dB)", "wl", True),
        ("b", "Filtered (dB)", "wl", True),
        ("e", "Regridded (dB)", "hz", True),
        ("c", "Measured (lin)", "wl", False),
        ("d", "Filtered (lin)", "wl", False),
        ("f", "Regridded (lin)", "hz", False),
    )

    def __init__(
        self, *, tracedb: float = 30.0, cmap: str = "viridis", cap: int = 1000
    ):
        self.tracedb = tracedb
        self.cap = cap
        self._cmap_pos = cmap_white(cmap)
        self._cmap_pos.set_bad(alpha=0.0)  # masked (negative) pixels stay clear
        self._cmap_neg = _white_red()
        self._cmap_neg.set_bad(alpha=0.0)
        self.fig = None
        self.axd = None
        self.images: dict[str, tuple] = {}

    @property
    def attached(self) -> bool:
        """Whether a figure is currently attached to this view."""
        return self.fig is not None

    def attach(self, fig: Figure) -> None:
        """Build the mosaic, images and colorbars once into ``fig``."""
        self.fig = fig
        fig.clear()
        with contextlib.suppress(Exception):
            fig.set_layout_engine("constrained")
        self.axd = fig.subplot_mosaic("abe\ncdf")
        empty = np.zeros((1, 1))
        for key, title, kind, _db in self._PANELS:
            ax = self.axd[key]
            pos = ax.imshow(
                empty,
                origin="lower",
                aspect="auto",
                cmap=self._cmap_pos,
                interpolation="nearest",
            )
            neg = ax.imshow(
                empty,
                origin="lower",
                aspect="auto",
                cmap=self._cmap_neg,
                interpolation="nearest",
            )
            self.images[key] = (pos, neg)
            ax.set_title(title)
            ax.set_xlabel("Wavelength (nm)" if kind == "wl" else "Frequency (PHz)")
            ax.set_ylabel("Delay (fs)")
        # One positive + one negative colorbar per row (dB row a/b/e, lin row c/d/f).
        self._cb_top = fig.colorbar(
            self.images["a"][0], ax=[self.axd[k] for k in "abe"], shrink=0.5
        )
        self._cb_top_neg = fig.colorbar(
            self.images["a"][1],
            ax=[self.axd[k] for k in "abe"],
            shrink=0.5,
            label="neg |·|",
        )
        self._cb_bot = fig.colorbar(
            self.images["c"][0], ax=[self.axd[k] for k in "cdf"], shrink=0.5
        )
        self._cb_bot_neg = fig.colorbar(
            self.images["c"][1],
            ax=[self.axd[k] for k in "cdf"],
            shrink=0.5,
            label="neg |·|",
        )

    def _set_panel(self, key, x, y, C, *, db, vmax=None):
        if self.axd is None:
            raise RuntimeError("live plot is not attached to a figure")
        pos_im, neg_im = self.images[key]
        C = np.asarray(C, dtype=float)
        C, x, y = _decimate_for_display(C, np.asarray(x), np.asarray(y), cap=self.cap)
        # imshow paints evenly spaced pixels; a wavelength axis built from a
        # uniform frequency grid (simulated traces) is non-uniform, so resample
        # onto an even axis to place features at the right wavelength.
        x, C = _resample_uniform_x(x, C)
        neg = C < 0
        if db:
            mag = np.clip(
                10 * np.log10(np.maximum(np.abs(C), 1e-30)), -self.tracedb, 0.0
            )
            vmin, vhi = -self.tracedb, 0.0
        else:
            mag = np.abs(C)
            vmin = 0.0
            vhi = float(np.nanmax(mag)) if vmax is None else vmax
        extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
        for im, masked in (
            (pos_im, np.ma.masked_where(neg, mag)),
            (neg_im, np.ma.masked_where(~neg, mag)),
        ):
            im.set_data(masked)
            im.set_clim(vmin, vhi)
            im.set_extent(extent)
        ax = self.axd[key]
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    def update(self, td) -> None:
        """Push a new :class:`~croak.preprocess.TraceData` onto the existing artists."""
        if not self.attached:
            raise RuntimeError("attach(fig) before update(td)")
        lam_nm = td.lam_frog / 1e-9
        freq = (td.omega + td.omega0_trace) / _TWOPI_PHZ
        lin_vmax = filter_view_vmax(td)
        self._set_panel("a", lam_nm, td.tau_meas / 1e-15, td.Ifrog_meas.T, db=True)
        self._set_panel("b", lam_nm, td.tau_filt / 1e-15, td.Ifrog_filt.T, db=True)
        self._set_panel("e", freq, td.delays / 1e-15, td.trace.T, db=True)
        self._set_panel(
            "c", lam_nm, td.tau_meas / 1e-15, td.Ifrog_meas.T, db=False, vmax=lin_vmax
        )
        self._set_panel(
            "d", lam_nm, td.tau_filt / 1e-15, td.Ifrog_filt.T, db=False, vmax=lin_vmax
        )
        self._set_panel(
            "f", freq, td.delays / 1e-15, td.trace.T, db=False, vmax=lin_vmax
        )
        # The linear row's scale changes with the data; refresh its colorbars.
        self._cb_bot.update_normal(self.images["c"][0])
        self._cb_bot_neg.update_normal(self.images["c"][1])


def plot_simulated_trace(
    omega,
    delays,
    trace,
    omega0_trace,
    *,
    tracedb: float = 40.0,
    cmap: str = "viridis",
    figsize=(7.87, 5.51),
    fig: Figure | None = None,
) -> Figure:
    """4-panel view of a simulated trace: dB, linear, and the two marginals."""
    trace = np.asarray(trace, dtype=float)
    fig = _prepare_fig(fig, figsize)
    axd = fig.subplot_mosaic("ab\ncd")
    cm = cmap_white(cmap)
    freq = (omega + omega0_trace) / _TWOPI_PHZ

    def _panel(ax, *, db, title):
        mpos, mneg = signed_pcolormesh(
            ax, delays / 1e-15, freq, trace, db=db, tracedb=tracedb, cmap_pos=cm
        )
        ax.set_xlabel("Delay (fs)")
        ax.set_ylabel("Frequency (PHz)")
        ax.set_title(title)
        return mpos, mneg

    ma, na = _panel(axd["a"], db=True, title="FROG (dB)")
    mb, nb = _panel(axd["b"], db=False, title="FROG (lin)")
    fig.colorbar(ma, ax=[axd["a"]], shrink=0.6)
    if na is not None:
        fig.colorbar(na, ax=[axd["a"]], shrink=0.6, label="neg |·|")
    fig.colorbar(mb, ax=[axd["b"]], shrink=0.6)
    if nb is not None:
        fig.colorbar(nb, ax=[axd["b"]], shrink=0.6, label="neg |·|")
    wmarg, tmarg = marginals(trace)
    axd["c"].plot(delays / 1e-15, tmarg, c="C0")
    axd["c"].set_xlabel("Delay (fs)")
    axd["c"].set_ylabel("Marginal")
    axd["c"].set_ylim(0, 1.05)
    axd["d"].plot((omega + omega0_trace) / _TWOPI_PHZ, wmarg, c="C0")
    axd["d"].set_xlabel("Frequency (PHz)")
    axd["d"].set_ylabel("Marginal")
    axd["d"].set_ylim(0, 1.05)
    return fig


def _interaction_scale(name: str) -> float:
    from .interactions import get_interaction

    return get_interaction(name).omega0_scale


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def plot_fwhm_distribution(
    ax, uresult: UncertaintyResult, *, truth: float | None = None
) -> None:
    """Histogram of bootstrap FWHM samples with the point estimate and intervals.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into.
    uresult : UncertaintyResult
        The bootstrap result (its ``samples`` are histogrammed).
    truth : float, optional
        A known true FWHM (s) to mark with a dashed line (synthetic tests).
    """
    samples = np.asarray(uresult.samples, dtype=float) / 1e-15
    nbins = int(np.clip(samples.size // 5, 12, 40))
    ax.hist(samples, bins=nbins, color="C0", alpha=0.7, edgecolor="white", lw=0.4)
    lo95, hi95 = uresult.interval_95
    lo68, hi68 = uresult.interval_68
    ax.axvspan(lo95 / 1e-15, hi95 / 1e-15, color="C0", alpha=0.10, label="95 %")
    ax.axvspan(lo68 / 1e-15, hi68 / 1e-15, color="C0", alpha=0.20, label="68 %")
    ax.axvline(uresult.point_estimate / 1e-15, c="C3", lw=1.5, label="estimate")
    if truth is not None:
        ax.axvline(truth / 1e-15, c="0.35", ls="--", lw=1.2, label="truth")
    ax.set_xlabel("FWHM (fs)")
    ax.set_ylabel("Replicates")
    ax.set_title(
        f"{uresult.point_estimate / 1e-15:.2f} ± {uresult.plus_minus / 1e-15:.2f} fs"
        f"  (B={uresult.n_converged})"
    )
    ax.legend(loc="best", handlelength=1.0, frameon=False, fontsize="small")


def plot_intensity_band(ax, uresult: UncertaintyResult) -> None:
    """Median temporal intensity with 68 %/95 % bootstrap confidence bands.

    Requires the bootstrap to have been run with ``collect_profiles=True``. The profiles
    are peak-aligned (translation gauge removed by ``shiftnorm``), so the band is the
    gauge-safe spread of the intensity envelope across replicates.
    """
    if uresult.profiles is None or uresult.t_profile is None:
        raise ValueError(
            "uresult has no profiles; run the bootstrap with collect_profiles=True"
        )
    t = np.asarray(uresult.t_profile, dtype=float) / 1e-15
    profiles = np.asarray(uresult.profiles, dtype=float)
    med = np.median(profiles, axis=0)
    lo68, hi68 = np.percentile(profiles, [16.0, 84.0], axis=0)
    lo95, hi95 = np.percentile(profiles, [2.5, 97.5], axis=0)
    ax.fill_between(t, lo95, hi95, color="C0", alpha=0.15, label="95 %")
    ax.fill_between(t, lo68, hi68, color="C0", alpha=0.30, label="68 %")
    ax.plot(t, med, c="C0", lw=1.5, label="median")
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Temporal intensity")
    # zoom to where the envelope carries signal
    support = t[med > 0.01 * med.max()]
    if support.size:
        pad = 0.2 * (support.max() - support.min() + 1.0)
        ax.set_xlim(support.min() - pad, support.max() + pad)
    ax.legend(loc="best", handlelength=1.0, frameon=False, fontsize="small")


def plot_uncertainty(
    uresult: UncertaintyResult,
    *,
    truth: float | None = None,
    figsize=(11.0, 4.2),
    fig: Figure | None = None,
) -> Figure:
    """Composite uncertainty figure: FWHM histogram + a confidence band if present.

    Parameters
    ----------
    uresult : UncertaintyResult
        The bootstrap result.
    truth : float, optional
        Known true FWHM (s) for the histogram (synthetic tests).
    figsize : tuple, optional
        Figure size.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into (cleared first), e.g. a Qt canvas figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # A thickness-systematic result carries per-draw thicknesses; show the dedicated
    # sensitivity figure so the GUI canvas path needs no special-casing.
    if getattr(uresult, "thicknesses", None) is not None:
        return plot_thickness_sensitivity(uresult, truth=truth, fig=fig)
    fig = _prepare_fig(fig, figsize)
    if uresult.profiles is not None and uresult.t_profile is not None:
        ax_h, ax_b = fig.subplots(1, 2)
        plot_fwhm_distribution(ax_h, uresult, truth=truth)
        plot_intensity_band(ax_b, uresult)
    else:
        ax_h = fig.subplots(1, 1)
        plot_fwhm_distribution(ax_h, uresult, truth=truth)
    fig.suptitle(uresult.summary(), fontsize="small")
    return fig


# ---------------------------------------------------------------------------
# Substrate-thickness systematic
# ---------------------------------------------------------------------------
def _mark_thickness_prior(ax, uresult: UncertaintyResult) -> None:
    """Shade the measured thickness prior ``L₀ ± σ_L`` (µm) on a thickness axis."""
    if uresult.thickness_central is None or uresult.thickness_sigma is None:
        return
    central_um = uresult.thickness_central / 1e-6
    sigma_um = uresult.thickness_sigma / 1e-6
    ax.axvspan(
        central_um - sigma_um,
        central_um + sigma_um,
        color="0.8",
        alpha=0.5,
        label="L₀ ± σ_L",
    )
    ax.axvline(central_um, color="0.4", ls=":", lw=1.0)


def plot_fwhm_vs_thickness(
    ax, uresult: UncertaintyResult, *, truth: float | None = None
) -> None:
    """Scatter the retrieved FWHM against the drawn substrate thickness.

    The horizontal bands give the point estimate and its 68 % interval; the vertical
    band is the measured thickness prior. The slope of the cloud is the physical
    sensitivity ``dFWHM/dL`` the analysis propagates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into.
    uresult : UncertaintyResult
        A ``method="thickness"`` result (its ``thicknesses`` are the x-values).
    truth : float, optional
        A known true FWHM (s) to mark with a dashed line (synthetic tests).
    """
    if uresult.thicknesses is None:
        raise ValueError("uresult has no thicknesses; run method='thickness'")
    thick_um = np.asarray(uresult.thicknesses, dtype=float) / 1e-6
    fwhm_fs = np.asarray(uresult.samples, dtype=float) / 1e-15
    _mark_thickness_prior(ax, uresult)
    ax.scatter(thick_um, fwhm_fs, s=14, color="C0", alpha=0.6, edgecolors="none")
    ax.axhline(uresult.point_estimate / 1e-15, color="C3", lw=1.5, label="estimate")
    lo68, hi68 = uresult.interval_68
    ax.axhspan(lo68 / 1e-15, hi68 / 1e-15, color="C3", alpha=0.15, label="68 %")
    if truth is not None:
        ax.axhline(truth / 1e-15, color="0.35", ls="--", lw=1.2, label="truth")
    ax.set_xlabel("Substrate thickness (µm)")
    ax.set_ylabel("FWHM (fs)")
    ax.set_title("FWHM vs thickness")
    ax.legend(loc="best", handlelength=1.0, frameon=False, fontsize="small")


def plot_error_vs_thickness(ax, uresult: UncertaintyResult) -> None:
    """Scatter the FROG error against the drawn thickness (a consistency check).

    A flat cloud means the data does not constrain the thickness (the full prior
    propagates); a clear minimum near ``L₀`` means the trace itself prefers a
    thickness, partially constraining it.
    """
    if uresult.thicknesses is None:
        raise ValueError("uresult has no thicknesses; run method='thickness'")
    thick_um = np.asarray(uresult.thicknesses, dtype=float) / 1e-6
    errors = np.asarray(uresult.frog_errors, dtype=float)
    _mark_thickness_prior(ax, uresult)
    ax.scatter(thick_um, errors, s=14, color="C1", alpha=0.6, edgecolors="none")
    ax.set_xlabel("Substrate thickness (µm)")
    ax.set_ylabel("FROG error R")
    ax.set_title("Trace misfit vs thickness")
    ax.legend(loc="best", handlelength=1.0, frameon=False, fontsize="small")


def plot_thickness_sensitivity(
    uresult: UncertaintyResult,
    *,
    truth: float | None = None,
    figsize=(13.0, 4.2),
    fig: Figure | None = None,
) -> Figure:
    """Composite figure for the substrate-thickness systematic.

    Panels: FWHM vs thickness, FROG error vs thickness, and — when profiles were
    collected — the temporal intensity confidence band.

    Parameters
    ----------
    uresult : UncertaintyResult
        A ``method="thickness"`` result.
    truth : float, optional
        Known true FWHM (s) for the FWHM panel (synthetic tests).
    figsize : tuple, optional
        Figure size.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into (cleared first), e.g. a Qt canvas figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = _prepare_fig(fig, figsize)
    has_band = uresult.profiles is not None and uresult.t_profile is not None
    axes = fig.subplots(1, 3 if has_band else 2)
    plot_fwhm_vs_thickness(axes[0], uresult, truth=truth)
    plot_error_vs_thickness(axes[1], uresult)
    if has_band:
        plot_intensity_band(axes[2], uresult)
    fig.suptitle(uresult.summary(), fontsize="small")
    return fig
