"""The retrieved pulse: temporal, spectral and spectrogram panels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ..maths import wlfreq
from ..processing import ProcessedResult, TruthPulse
from .artists import SpectralArtists, TemporalArtists
from .traces import cmap_white
from .units import _TWOPI_PHZ, _fs, si_power

if TYPE_CHECKING:
    from matplotlib.legend import Legend
    from numpy.typing import NDArray


#: Edge-energy fraction (see :func:`croak.processing.edge_energy_fraction`) above
#: which the spectrum panel flags the retrieval in red. A sound retrieval on a
#: generous band sits two orders of magnitude below this; above it, re-run on a
#: wider grid.
EDGE_ENERGY_WARN = 0.01


#: Abscissae the spectrum panel can be drawn against (see :func:`plot_spectral`).
SPECTRAL_AXES: tuple[str, ...] = ("frequency", "wavelength")
type SpectralAxis = Literal["frequency", "wavelength"]


def _combined_legend(ax, axp, *, loc: str) -> Legend:
    """Draw one legend containing intensity and twin-axis phase curves.

    ``loc`` is a fixed corner: matplotlib's ``"best"`` re-evaluates on every draw
    and made the legend jump between frames of the live preview.
    """
    handles, labels = ax.get_legend_handles_labels()
    phase_handles, phase_labels = axp.get_legend_handles_labels()
    return ax.legend(
        handles + phase_handles,
        labels + phase_labels,
        loc=loc,
        handlelength=1.0,
        frameon=False,
        fontsize="small",
    )


def _temporal_curves(pr: ProcessedResult, power_scale):
    """Return ``(retr, tl, r_label, tl_label, ylabel, power_scale)`` for the pulse.

    In absolute power when ``pr.peak_power`` is known (deriving the SI prefix if
    ``power_scale`` is None), else normalised. Shared by draw and update.
    """
    if pr.peak_power is not None:
        if power_scale is None:
            power_scale = si_power(max(pr.peak_power, pr.peak_power_tl or 0.0))
        scale, unit = power_scale
        retr = pr.It_retr * pr.peak_power / scale
        tl = pr.It_tl * (pr.peak_power_tl or 0.0) / scale
        r_label = f"R ({_fs(pr.fwhm_retr)} fs, {pr.peak_power / scale:.1f} {unit})"
        tl_label = (
            f"TL ({_fs(pr.fwhm_tl)} fs, {(pr.peak_power_tl or 0.0) / scale:.1f} {unit})"
        )
        return retr, tl, r_label, tl_label, f"Power ({unit})", power_scale
    r_label = f"R ({_fs(pr.fwhm_retr)} fs)"
    tl_label = f"TL ({_fs(pr.fwhm_tl)} fs)"
    return pr.It_retr, pr.It_tl, r_label, tl_label, "Power (a.u.)", None


def _truth_intensity(truth: TruthPulse, energy, power_scale) -> NDArray[np.float64]:
    """Return the truth's intensity on the temporal panel's scale.

    With a pulse energy the truth is shown in absolute power on the shared SI
    scale: it carries the same energy, so its peak power is energy/∫(I/I_peak)dt
    over its *own* duration (:meth:`~croak.processing.TruthPulse.peak_power`).
    """
    peak_power = truth.peak_power(energy)
    if peak_power is None:
        return np.asarray(truth.It, dtype=float)
    scale = power_scale[0] if power_scale is not None else 1.0
    return np.asarray(truth.It, dtype=float) * peak_power / scale


def _intensity_ylim(*curves) -> tuple[float, float]:
    """``(0, 1.05 × max)`` over the given curves (``None`` entries ignored).

    Set explicitly because ``Axes.set_ylim`` switches autoscaling off, so an
    in-place update could not rely on it; the 5 % headroom is matplotlib's
    default margin.
    """
    tops = [float(np.nanmax(c)) for c in curves if c is not None and np.size(c)]
    top = max(tops, default=1.0)
    return 0.0, 1.05 * top if top > 0 else 1.0


def _phase_ylim(*curves) -> tuple[float, float] | None:
    """Phase-axis limits centred on the shown phases with their span as margin.

    Finite samples of every non-empty curve count; ``None`` when nothing is shown.
    """
    arrays = [np.asarray(c, dtype=float) for c in curves if c is not None]
    arrays = [a[np.isfinite(a)] for a in arrays]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return None
    pm = np.concatenate(arrays)
    d = pm.max() - pm.min()
    mid = (pm.max() + pm.min()) / 2
    return mid - d - 1e-9, mid + d + 1e-9


def _temporal_panel(
    ax,
    pr: ProcessedResult,
    *,
    halfwidth=None,
    power_scale=None,
    truth: TruthPulse | None = None,
) -> TemporalArtists:
    """Draw the temporal panel into ``ax``; return its artists (see plot_temporal)."""
    t = pr.t_over / 1e-15
    retr, tl, r_label, tl_label, ylabel, power_scale = _temporal_curves(pr, power_scale)
    (line_retr,) = ax.plot(t, retr, c="C0", label=r_label)
    # The TL pulse is peak-centred on its own oversampled axis (its peak index
    # need not coincide with the retrieved pulse's), so plot it against t_tl —
    # using t_over would shift its peak off t=0 by a sample or two.
    (line_tl,) = ax.plot(pr.t_tl / 1e-15, tl, c="C1", label=tl_label)
    line_truth = None
    truth_curve = None
    if truth is not None:
        # The known pulse as a thin black line *under* the retrieved/TL curves, in
        # physical units so it is independent of the retrieval grid. A retrieval
        # is defined only up to the time-direction ambiguity, so the truth may
        # appear shifted or flipped. Its legend entry carries its own FWHM so the
        # three durations read side by side.
        truth_curve = _truth_intensity(truth, pr.energy, power_scale)
        (line_truth,) = ax.plot(
            truth.t / 1e-15,
            truth_curve,
            c="k",
            lw=0.8,
            zorder=1,
            label=f"truth ({_fs(truth.fwhm)} fs)",
        )
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*_intensity_ylim(retr, tl, truth_curve))
    ax.set_title("Retrieved pulse")
    if halfwidth is not None:
        ax.set_xlim(-halfwidth / 1e-15, halfwidth / 1e-15)
    axp = ax.twinx()
    (line_phase,) = axp.plot(t[pr.mask_t], pr.phi_t[pr.mask_t], c="C2")
    axp.set_ylabel("Phase (rad)")
    line_truth_phase = None
    truth_phase = None
    if truth is not None and truth.phi_t is not None:
        truth_mask = np.asarray(truth.It) > 0.01
        truth_phase = np.asarray(truth.phi_t)[truth_mask]
        (line_truth_phase,) = axp.plot(
            np.asarray(truth.t)[truth_mask] / 1e-15,
            truth_phase,
            ":",
            c="0.35",
            lw=1.2,
            label="truth phase",
        )
    limits = _phase_ylim(pr.phi_t[pr.mask_t], truth_phase)
    if limits is not None:
        axp.set_ylim(*limits)
    # Upper left: the pulse and its masked phase sit at the centre, the wings are
    # empty, and a fixed corner cannot jump between live frames.
    legend = _combined_legend(ax, axp, loc="upper left")
    texts = legend.get_texts()
    return TemporalArtists(
        line_retr=line_retr,
        line_tl=line_tl,
        line_phase=line_phase,
        ax_phase=axp,
        legend=legend,
        legend_retr=texts[0],
        legend_tl=texts[1],
        line_truth=line_truth,
        line_truth_phase=line_truth_phase,
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
    derived from the larger of the retrieved/TL peak powers.

    ``truth`` (a :class:`~croak.processing.TruthPulse`: a synthetic generating
    field, or a simulated scan's reference intensity) is overlaid as a thin black
    line under the retrieved/TL curves, labelled with its own FWHM, and — when it
    carries a temporal phase — as a dotted phase in the same display gauge as the
    retrieved one (see :func:`croak.processing.normalize_temporal_phase`).

    Returns the twin phase axes. :func:`draw_temporal` is the same panel for the
    registry, returning every artist (:class:`TemporalArtists`).
    """
    return _temporal_panel(
        ax, pr, halfwidth=halfwidth, power_scale=power_scale, truth=truth
    ).ax_phase


def _truth_spectral_phase(
    wavelength: NDArray[np.float64], truth: TruthPulse | None
) -> NDArray[np.float64] | None:
    """Return the truth's spectral phase interpolated onto ``wavelength``, or None.

    The truth is stored on the simulation's own frequency grid, which is neither
    the same spacing nor the same extent as the retrieval's; NaN outside it.
    """
    tphi_src = getattr(truth, "phi_w", None) if truth is not None else None
    tlam_src = getattr(truth, "lam", None) if truth is not None else None
    if tphi_src is None or tlam_src is None:
        return None
    return np.interp(
        wavelength,
        np.asarray(tlam_src, dtype=float),
        np.asarray(tphi_src, dtype=float),
        left=np.nan,
        right=np.nan,
    )


def _dispersion_text(pr: ProcessedResult) -> str:
    """Return the GDD/TOD annotation of the spectrum panel."""
    return f"GDD: {pr.gdd_fs2:.1f} fs²\nTOD: {pr.tod_fs3:.1f} fs³"


def _edge_text(pr: ProcessedResult) -> tuple[str, str]:
    """Return the edge-energy annotation and its colour (red once it matters).

    Energy parked in the unmeasured grid-edge bins is an artefact indicator; the
    1 % threshold sits comfortably above what a sound retrieval produces.
    """
    color = "C3" if pr.edge_energy > EDGE_ENERGY_WARN else "0.35"
    return f"Edge: {100 * pr.edge_energy:.2f} %", color


def _unit_peak(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale ``values`` to a peak of one (NaNs ignored; all-zero left alone)."""
    peak = float(np.nanmax(values)) if values.size else 0.0
    return values / peak if peak > 0 else values


def _spectral_curves(pr: ProcessedResult, axis: str):
    """Return ``(x, retrieved, measured, xlabel)`` of the spectrum panel on ``axis``.

    Both spectra are unit-peak densities *in the chosen variable*. On the
    wavelength axis they are ``pr.Ilam`` and ``pr.Iw_meas`` — both wavelength
    densities (the measured one carries the ω² Jacobian despite its name); on the
    frequency axis the retrieved spectrum is ``|E(ω)|²`` itself and the measured
    one has that Jacobian divided out again, so a spectrum symmetric in ω plots
    symmetric. ``x`` is per grid index on both axes, so the phase curves index it
    with ``pr.mask_w`` unchanged.
    """
    if axis == "wavelength":
        return pr.wavelength / 1e-9, pr.Ilam, pr.Iw_meas, "Wavelength (nm)"
    if axis != "frequency":
        raise ValueError(f"axis must be one of {SPECTRAL_AXES}, got {axis!r}")
    omega_abs = pr.omega + pr.omega0
    measured = None
    if pr.Iw_meas is not None:
        # A centred grid can cross ω = 0; that sample is undefined as a density.
        with np.errstate(divide="ignore", invalid="ignore"):
            jacobian = np.where(omega_abs != 0, 1.0 / omega_abs**2, np.nan)
        measured = _unit_peak(pr.Iw_meas * jacobian)
    return omega_abs / _TWOPI_PHZ, _unit_peak(pr.Iw), measured, "Frequency (PHz)"


def _spectral_xlim(lam_min, lam_max, axis: str) -> tuple[float, float] | None:
    """Return the spectrum panel's x-limits for a wavelength window on ``axis``."""
    if not (lam_min and lam_max):
        return None
    if axis == "wavelength":
        return lam_min / 1e-9, lam_max / 1e-9
    return float(wlfreq(lam_max)) / _TWOPI_PHZ, float(wlfreq(lam_min)) / _TWOPI_PHZ


def _truth_spectrum(truth: TruthPulse, axis: str):
    """Return the truth's spectrum ``(x, unit-peak density)`` on ``axis``, or None."""
    if truth.lam is None or truth.Iw is None:
        return None
    if axis == "wavelength":
        return truth.lam / 1e-9, truth.Iw
    omega = wlfreq(truth.lam)
    return omega / _TWOPI_PHZ, _unit_peak(np.asarray(truth.Iw) / omega**2)


def _spectral_panel(
    ax,
    pr: ProcessedResult,
    *,
    lam_min=None,
    lam_max=None,
    truth: TruthPulse | None = None,
    axis: str = "frequency",
) -> SpectralArtists:
    """Draw the spectrum panel into ``ax``; return its artists (see plot_spectral)."""
    x, retrieved, measured, xlabel = _spectral_curves(pr, axis)
    (line_retr,) = ax.plot(x, retrieved, c="C0", label="R")
    line_meas = None
    if measured is not None:
        (line_meas,) = ax.plot(x, measured, c="C1", label="M")
    line_truth = None
    truth_curve = None if truth is None else _truth_spectrum(truth, axis)
    if truth_curve is not None:
        (line_truth,) = ax.plot(*truth_curve, c="k", lw=0.8, zorder=1, label="truth")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Power (a.u.)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Spectrum")
    xlim = _spectral_xlim(lam_min, lam_max, axis)
    if xlim is not None:
        ax.set_xlim(*xlim)
    axp = ax.twinx()
    (line_phase,) = axp.plot(x[pr.mask_w], pr.phi_w[pr.mask_w], c="C2")
    # dashed polynomial fit of the spectral phase (GDD/TOD visualisation)
    (line_phase_fit,) = axp.plot(
        x[pr.mask_w], pr.phi_w_fit[pr.mask_w], "--", c="C4", alpha=0.7
    )
    axp.set_ylabel("Phase (rad)")
    tphi = _truth_spectral_phase(pr.wavelength, truth)
    line_truth_phase = None
    if tphi is not None:
        (line_truth_phase,) = axp.plot(
            x[pr.mask_w],
            tphi[pr.mask_w],
            ":",
            c="0.35",
            lw=1.2,
            label="truth phase",
        )
    limits = _phase_ylim(pr.phi_w[pr.mask_w], None if tphi is None else tphi[pr.mask_w])
    if limits is not None:
        axp.set_ylim(*limits)
    text_dispersion = ax.text(
        0.05, 0.78, _dispersion_text(pr), transform=ax.transAxes, fontsize="small"
    )
    edge, edge_color = _edge_text(pr)
    text_edge = ax.text(
        0.05, 0.70, edge, transform=ax.transAxes, fontsize="small", color=edge_color
    )
    # Upper right: the GDD/TOD and edge annotations occupy the upper left.
    legend = _combined_legend(ax, axp, loc="upper right")
    return SpectralArtists(
        line_retr=line_retr,
        line_phase=line_phase,
        line_phase_fit=line_phase_fit,
        ax_phase=axp,
        legend=legend,
        text_dispersion=text_dispersion,
        text_edge=text_edge,
        line_meas=line_meas,
        line_truth=line_truth,
        line_truth_phase=line_truth_phase,
    )


def plot_spectral(
    ax,
    pr: ProcessedResult,
    *,
    lam_min=None,
    lam_max=None,
    truth: TruthPulse | None = None,
    axis: SpectralAxis = "frequency",
):
    """Spectral intensity (left) and phase (right twin axis) against ``axis``.

    ``axis`` is ``"frequency"`` (PHz, the default: the space the retrieval works in,
    where ``|E(ω)|²`` of a symmetric pulse is symmetric) or ``"wavelength"`` (nm, the
    spectrometer's axis, carrying the 1/λ² density Jacobian). Both spectra are drawn as
    unit-peak densities in the chosen variable; ``lam_min``/``lam_max`` (m) bound the
    axis on either. The phase curves are the same on both.

    ``truth``, when it carries a spectrum, is overlaid as a thin black line under
    the retrieved curve; when it carries a known spectral phase, that phase is
    overlaid on the phase axis. It is drawn in the SAME gauge as the retrieved
    phase (unwrapped, de-tilted, zeroed at the spectral peak — see
    :func:`croak.processing.normalize_spectral_phase`), because spectral phase
    is defined only up to a constant and a linear term; a truth pinned any other
    way would show a tilt or offset that is gauge, not error.

    Returns the twin phase axes. :func:`draw_spectral` is the same panel for the
    registry, returning every artist (:class:`SpectralArtists`).
    """
    return _spectral_panel(
        ax, pr, lam_min=lam_min, lam_max=lam_max, truth=truth, axis=axis
    ).ax_phase


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
