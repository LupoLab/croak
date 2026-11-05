"""The twelve retrieval panels over one shared :class:`RetrievalPlotData` bundle."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..maths import wlfreq
from ..processing import ProcessedResult, TruthPulse, process_result, spectrogram
from ..result import RetrievalResult
from .artists import (
    ConvergenceArtists,
    LineArtists,
    MarginalArtists,
    ResidualArtists,
    SpectralArtists,
    SpectrogramArtists,
    TemporalArtists,
    TraceArtists,
)
from .pulse import SPECTRAL_AXES, _spectral_panel, _temporal_panel, plot_spectrogram
from .traces import cmap_white, plot_marginal, plot_residual, signed_pcolormesh
from .units import _TWOPI_PHZ, si_power

if TYPE_CHECKING:
    from matplotlib.collections import QuadMesh
    from matplotlib.colors import Colormap
    from matplotlib.lines import Line2D
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Retrieval panels: the data bundle and the twelve panels
# ---------------------------------------------------------------------------
# The twelve-panel retrieval summary is built from independent panels so the GUI
# can arrange them as three 2×2 pages (Traces / Pulse / Diagnostics) while scripts
# and the CLI keep the 3×4 overview. Every panel is a pure function
# ``draw_<key>(ax, data)`` over one immutable RetrievalPlotData, computed once per
# result. Colorbars that span two panels are a page-level concern (ColorbarSpec,
# in croak.plotting.retrieval), drawn after all panels of a figure exist.


@dataclass(frozen=True, eq=False)
class RetrievalPlotData:
    """Everything the retrieval panels share, computed once from one result.

    Build it with :func:`retrieval_plot_data`. The bundle is immutable and carries
    no figure state, so one instance can feed several figures (the GUI's pages and
    their pop-outs) and be redrawn at any size.

    Attributes
    ----------
    result : RetrievalResult
        The retrieval; ``result.trace`` is the retrieved trace.
    processed : ProcessedResult
        Display-ready quantities of ``result``
        (:func:`croak.processing.process_result`).
    measured, retrieved : ndarray
        Measured and retrieved traces, shape ``(Nω, Nτ)``.
    omega0_trace : float
        Centre angular frequency of the *signal* (twice ``omega0`` for SHG), so
        the trace panels' frequency axis reads at the signal frequency.
    halfwidth : float
        Half the delay span (s); bounds the delay axes.
    lam_min, lam_max : float
        Wavelength bounds (m) of the spectrum and spectrogram panels.
    flim : tuple of float or None
        Optional ``(f_min, f_max)`` (Hz) bounds of the trace/residual frequency axes.
    tracedb : float
        Dynamic range (dB) of the logarithmic trace panels.
    cmap_pos : Colormap
        White-based positive colormap shared by the trace and spectrogram panels.
    lin_vmax : float
        Common scale of the linear trace pair, so their colorbar is meaningful.
    truth : TruthPulse or None
        Known pulse to overlay (synthetic and simulated data).
    power_scale : tuple of (float, str) or None
        SI ``(scale, unit)`` for absolute power when a pulse energy was given.
    spectral_axis : str
        ``"frequency"`` or ``"wavelength"``: the spectrum panel's abscissa.
    """

    result: RetrievalResult
    processed: ProcessedResult
    measured: NDArray[np.float64]
    retrieved: NDArray[np.float64]
    omega0_trace: float
    halfwidth: float
    lam_min: float
    lam_max: float
    flim: tuple[float, float] | None
    tracedb: float
    cmap_pos: Colormap
    lin_vmax: float
    truth: TruthPulse | None
    power_scale: tuple[float, str] | None
    spectral_axis: str

    @functools.cached_property
    def spectrogram(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """``(t_centres, wavelength, S)`` of :func:`croak.processing.spectrogram`.

        The Gabor transform is the costliest thing the panels compute (hundreds
        of FFTs), so it runs only when the spectrogram panel is first drawn and is
        then cached on the bundle for every later draw of it.
        """
        return spectrogram(self.result, lam_min=self.lam_min, lam_max=self.lam_max)


def retrieval_plot_data(
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
    processed: ProcessedResult | None = None,
    spectral_axis: str = "frequency",
) -> RetrievalPlotData:
    """Bundle a retrieval's plot inputs for the panels (see :func:`plot_retrieval`).

    The arguments are those of :func:`plot_retrieval`, which is this bundle drawn
    as the overview; see there for their meaning. ``processed`` is an
    already-computed :class:`~croak.processing.ProcessedResult` **for this same
    result** (the GUI computes it once for its read-out); it must have been built
    with the same ``measured``/``Iomega_meas``/``energy``.

    Raises
    ------
    ValueError
        If ``result`` carries no trace, ``processed`` belongs to another result,
        or ``spectral_axis`` is not one of :data:`SPECTRAL_AXES`.
    """
    omega = result.omega
    omega0 = result.omega0
    if spectral_axis not in SPECTRAL_AXES:
        raise ValueError(
            f"spectral_axis must be one of {SPECTRAL_AXES}, got {spectral_axis!r}"
        )
    if result.trace is None:
        raise ValueError("plot_retrieval requires a result carrying a simulated trace")
    retrieved = np.asarray(result.trace, dtype=float)
    measured_arr = retrieved if measured is None else np.asarray(measured, dtype=float)

    if processed is None:
        pr = process_result(
            result, measured=measured_arr, Iomega_meas=Iomega_meas, energy=energy
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
    delays = result.delays
    # When a pulse energy was supplied, derive one SI power prefix for the
    # temporal panel (and the truth overlay) so both share a consistent axis.
    power_scale = None
    if pr.peak_power is not None:
        power_scale = si_power(max(pr.peak_power, pr.peak_power_tl or 0.0))
    return RetrievalPlotData(
        result=result,
        processed=pr,
        measured=measured_arr,
        retrieved=retrieved,
        omega0_trace=omega0 * _interaction_scale(result.interaction),
        halfwidth=float((delays.max() - delays.min()) / 2),
        lam_min=float(lam_min),
        lam_max=float(lam_max),
        flim=flim,
        tracedb=tracedb,
        # Positive values on the base map with white at the low end, so zero is
        # white in *both* maps and meets the white→red negative map seamlessly.
        cmap_pos=cmap_white(cmap),
        # The linear pair shares one scale so its colorbar is meaningful.
        lin_vmax=max(
            float(np.nanmax(np.abs(measured_arr))), float(np.nanmax(retrieved))
        ),
        truth=truth,
        power_scale=power_scale,
        spectral_axis=spectral_axis,
    )


def plot_convergence(
    ax, errors: ArrayLike, *, boundaries: Sequence[int] = ()
) -> ConvergenceArtists:
    """Plot the FROG error vs iteration on a log y-axis.

    ``boundaries`` are indices into ``errors`` where a multi-stage solver handed
    over to its next stage (:attr:`~croak.result.RetrievalResult.stage_boundaries`);
    each is drawn as a dashed rule. Marking them matters because the stages need
    not count the same unit of work — ``warm-lbfgs`` splices COPRA iterations onto
    L-BFGS function evaluations — so an unmarked kink would read as convergence
    behaviour rather than as the change of algorithm it is.

    The error curve is created even for an empty ``errors`` (a result restored
    from file carries no history), so a live view can grow it in place.
    """
    errors = np.asarray(errors, dtype=float)
    (line,) = ax.semilogy(np.arange(1, errors.size + 1), errors, c="C0")
    rules: list[Line2D] = []
    for i, edge in enumerate(boundaries):
        if 0 < edge < errors.size:
            rules.append(
                ax.axvline(
                    edge + 0.5,
                    c="0.5",
                    ls="--",
                    lw=0.8,
                    label="stage handover" if i == 0 else None,
                )
            )
    legend = None
    if rules:  # upper right: a decreasing curve leaves that corner empty
        legend = ax.legend(
            loc="upper right", handlelength=1.2, frameon=False, fontsize="small"
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    ax.set_title("Convergence")
    return ConvergenceArtists(line=line, boundaries=tuple(rules), legend=legend)


def _trace_label(data: RetrievalPlotData) -> str:
    """Return the retrieved-linear panel's annotation: FROG error and shape."""
    return (
        f"{data.result.error * 100:.2f}%\n"
        f"{data.retrieved.shape[0]}×{data.retrieved.shape[1]}"
    )


def _draw_trace_image(
    ax, data: RetrievalPlotData, C: NDArray[np.float64], *, db: bool, title: str
) -> tuple[QuadMesh, QuadMesh | None]:
    """Draw one trace image (delay × signal frequency) and return its meshes."""
    mpos, mneg = signed_pcolormesh(
        ax,
        data.result.delays / 1e-15,
        (data.result.omega + data.omega0_trace) / _TWOPI_PHZ,
        C,
        db=db,
        tracedb=data.tracedb,
        cmap_pos=data.cmap_pos,
        vmax=None if db else data.lin_vmax,
    )
    ax.set_xlabel("Delay (fs)")
    ax.set_ylabel("Frequency (PHz)")
    ax.set_title(title)
    ax.set_xlim(-data.halfwidth / 1e-15, data.halfwidth / 1e-15)
    if data.flim is not None:
        ax.set_ylim(data.flim[0] / 1e15, data.flim[1] / 1e15)
    return mpos, mneg


def draw_measured_log(ax, data: RetrievalPlotData) -> TraceArtists:
    """Draw the measured trace, logarithmic (``tracedb`` dynamic range)."""
    mpos, mneg = _draw_trace_image(
        ax, data, data.measured, db=True, title="Measured (log)"
    )
    return TraceArtists(mesh_pos=mpos, mesh_neg=mneg)


def draw_retrieved_log(ax, data: RetrievalPlotData) -> TraceArtists:
    """Draw the retrieved trace, logarithmic."""
    mpos, mneg = _draw_trace_image(
        ax, data, data.retrieved, db=True, title="Retrieved (log)"
    )
    return TraceArtists(mesh_pos=mpos, mesh_neg=mneg)


def draw_measured_lin(ax, data: RetrievalPlotData) -> TraceArtists:
    """Draw the measured trace, linear, on the scale shared with the retrieved."""
    mpos, mneg = _draw_trace_image(
        ax, data, data.measured, db=False, title="Measured (lin)"
    )
    return TraceArtists(mesh_pos=mpos, mesh_neg=mneg)


def draw_retrieved_lin(ax, data: RetrievalPlotData) -> TraceArtists:
    """Draw the retrieved trace, linear, annotated with FROG error and shape."""
    mpos, mneg = _draw_trace_image(
        ax, data, data.retrieved, db=False, title="Retrieved (lin)"
    )
    label = ax.text(
        0.04,
        0.96,
        _trace_label(data),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="black",
        fontweight="bold",
        fontsize="small",
    )
    return TraceArtists(mesh_pos=mpos, mesh_neg=mneg, label=label)


def draw_temporal(ax, data: RetrievalPlotData) -> TemporalArtists:
    """Draw the retrieved pulse: intensity/power, TL pulse, phase, truth overlay."""
    return _temporal_panel(
        ax,
        data.processed,
        halfwidth=data.halfwidth,
        power_scale=data.power_scale,
        truth=data.truth,
    )


def draw_spectral(ax, data: RetrievalPlotData) -> SpectralArtists:
    """Draw the spectrum: retrieved, measured, phase and fit, GDD/TOD, truth."""
    return _spectral_panel(
        ax,
        data.processed,
        lam_min=data.lam_min,
        lam_max=data.lam_max,
        truth=data.truth,
        axis=data.spectral_axis,
    )


def draw_convergence(ax, data: RetrievalPlotData) -> ConvergenceArtists:
    """Draw the FROG error against iteration, with the solver's stage handovers."""
    return plot_convergence(
        ax, data.processed.errors, boundaries=data.result.stage_boundaries
    )


def draw_residual(ax, data: RetrievalPlotData) -> ResidualArtists:
    """Draw the measured − retrieved residual; leave it empty if there is none."""
    pr = data.processed
    if pr.residual is None:
        return ResidualArtists(mesh=None)
    mesh = plot_residual(
        ax,
        data.result.omega,
        data.result.delays,
        pr.residual,
        data.result.omega0,
        halfwidth=data.halfwidth,
        flim=data.flim,
    )
    return ResidualArtists(mesh=mesh)


def _filter_factor(result: RetrievalResult) -> NDArray[np.float64]:
    """Return the per-frequency Rω scale factor, peak-normalised, else flat."""
    mu = np.atleast_1d(np.asarray(result.mu, dtype=float))
    if mu.size == result.omega.size:
        return mu / mu.max()
    return np.ones_like(result.omega)


def draw_spectral_filter(ax, data: RetrievalPlotData) -> LineArtists:
    """Draw the per-frequency scale factor (Rω filter) against wavelength."""
    # A centred retrieval grid can cross zero frequency, where the wavelength
    # axis has a pole: harmless for the plot, so the divide warning is silenced.
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_nm = wlfreq(data.result.omega + data.result.omega0) / 1e-9
    (line,) = ax.plot(lam_nm, _filter_factor(data.result), c="C2")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Filter factor")
    ax.set_title("Spectral filter")
    ax.set_xlim(data.lam_min / 1e-9, data.lam_max / 1e-9)
    return LineArtists(line=line)


def draw_freq_marginal(ax, data: RetrievalPlotData) -> MarginalArtists:
    """Draw the frequency marginal of the retrieved and measured traces."""
    pr = data.processed
    artists = plot_marginal(
        ax,
        data.result.omega + data.result.omega0,
        pr.omega_marg_retr,
        pr.omega_marg_meas,
        xlabel="Frequency (PHz)",
        scale=_TWOPI_PHZ,
    )
    ax.set_title("Frequency marginal")
    return artists


def draw_delay_marginal(ax, data: RetrievalPlotData) -> MarginalArtists:
    """Draw the delay marginal of the retrieved and measured traces."""
    pr = data.processed
    artists = plot_marginal(
        ax,
        data.result.delays,
        pr.tau_marg_retr,
        pr.tau_marg_meas,
        xlabel="Delay (fs)",
        scale=1e-15,
        halfwidth=data.halfwidth,
    )
    ax.set_title("Delay marginal")
    return artists


def draw_spectrogram(ax, data: RetrievalPlotData) -> SpectrogramArtists:
    """Draw the Gabor spectrogram of the retrieved pulse (computed once, cached)."""
    tc, lam, S = data.spectrogram
    mesh = plot_spectrogram(
        ax,
        tc,
        lam,
        S,
        lam_min=data.lam_min,
        lam_max=data.lam_max,
        halfwidth=data.halfwidth,
        # white at zero, like the trace panels and the dispersion stage's own
        # spectrogram — this panel was the only one left on plain viridis.
        cmap=data.cmap_pos,
    )
    return SpectrogramArtists(mesh=mesh)


def _interaction_scale(name: str) -> float:
    from ..interactions import get_interaction

    return get_interaction(name).omega0_scale
