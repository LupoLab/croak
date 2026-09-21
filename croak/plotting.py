"""Matplotlib plotting for retrieval results.

Provides custom colormaps (white at the low end; a red band for negatives),
reusable **single-axis** plotters (so the GUI embeds exactly the same code), and
**composite** figures: the 12-panel :func:`plot_retrieval`, the 6-panel
:func:`plot_frog_filter`, and :func:`plot_simulated_trace`.

The retrieval summary is assembled from twelve independent **panels**
(``draw_<key>(ax, data)`` over one :class:`RetrievalPlotData` bundle, registered in
:data:`RETRIEVAL_PANELS`). :func:`plot_retrieval` lays them out as the 3×4
overview; :func:`plot_retrieval_page` draws one of the three 2×2 pages
(:data:`RETRIEVAL_PAGES`: Traces, Pulse, Diagnostics) the GUI shows as tabs.

Figures are built with :class:`matplotlib.figure.Figure` directly (no
``pyplot`` global state), so they embed cleanly in a Qt canvas and also save
standalone via ``fig.savefig(...)``.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

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
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colors import Colormap
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from numpy.typing import NDArray

    from .uncertainty import UncertaintyResult

#: Edge-energy fraction (see :func:`croak.processing.edge_energy_fraction`) above
#: which the spectrum panel flags the retrieval in red. A sound retrieval on a
#: generous band sits two orders of magnitude below this.
EDGE_ENERGY_WARN = 0.01

__all__ = [
    "PlotScale",
    "FULL_SCALE",
    "FONT_SCALE_MIN",
    "FONT_SCALE_STEP",
    "COMPACT_BELOW",
    "plot_scale",
    "scaled_rc",
    "cmap_white",
    "cmap_negwhite",
    "signed_pcolormesh",
    "si_power",
    "sig3",
    "plot_residual",
    "plot_temporal",
    "plot_spectral",
    "SPECTRAL_AXES",
    "EDGE_ENERGY_WARN",
    "plot_convergence",
    "plot_marginal",
    "plot_spectrogram",
    "plot_retrieval",
    "RetrievalPlotData",
    "retrieval_plot_data",
    "TraceArtists",
    "ResidualArtists",
    "TemporalArtists",
    "SpectralArtists",
    "ConvergenceArtists",
    "LineArtists",
    "MarginalArtists",
    "SpectrogramArtists",
    "PanelSpec",
    "PageSpec",
    "ColorbarSpec",
    "RETRIEVAL_PANELS",
    "RETRIEVAL_COLORBARS",
    "RETRIEVAL_PAGES",
    "RETRIEVAL_OVERVIEW",
    "plot_retrieval_page",
    "draw_measured_log",
    "draw_retrieved_log",
    "draw_measured_lin",
    "draw_retrieved_lin",
    "draw_temporal",
    "draw_spectral",
    "draw_convergence",
    "draw_residual",
    "draw_spectral_filter",
    "draw_freq_marginal",
    "draw_delay_marginal",
    "draw_spectrogram",
    "update_measured_log",
    "update_retrieved_log",
    "update_measured_lin",
    "update_retrieved_lin",
    "update_temporal",
    "update_spectral",
    "update_convergence",
    "update_convergence_curve",
    "update_residual",
    "update_spectral_filter",
    "update_freq_marginal",
    "update_delay_marginal",
    "update_spectrogram",
    "RetrievalPageView",
    "plot_frog_filter",
    "filter_view_vmax",
    "FrogFilterView",
    "plot_simulated_trace",
    "plot_fwhm_distribution",
    "plot_intensity_band",
    "plot_uncertainty",
]

_TWOPI_PHZ = 2 * np.pi * 1e15

#: Abscissae the spectrum panel can be drawn against (see :func:`plot_spectral`).
SPECTRAL_AXES: tuple[str, ...] = ("frequency", "wavelength")
type SpectralAxis = Literal["frequency", "wavelength"]


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
# On-screen density
# ---------------------------------------------------------------------------
# Matplotlib sizes text in points, so when a figure is squeezed into a small window
# the axes shrink and the text does not: on a 1366x768 laptop the twelve-panel
# retrieval summary spent 78 % of its area on titles, labels and padding. The GUI
# canvas (croak.gui.canvas.MplCanvas) therefore scales the rc font sizes by how far
# the canvas has fallen below the size its figure was designed for. The scale is a
# pure function of geometry, kept here so the plotting layer stays Qt-free and the
# thresholds are testable headlessly.

#: Smallest fraction of the design font size that text may shrink to. Below ~70 %
#: a 10 pt label is 7 pt, the limit of comfortable legibility on a laptop panel.
FONT_SCALE_MIN = 0.7
#: Quantisation step of :func:`plot_scale`. A window resize re-plots only when the
#: scale crosses a step, so a few pixels of drag never trigger a redraw.
FONT_SCALE_STEP = 0.05
#: Below this scale :attr:`PlotScale.compact` is set, for plotting code that wants
#: to drop optional decoration (a second legend entry, a colorbar label) when tight.
COMPACT_BELOW = 0.9


@dataclass(frozen=True)
class PlotScale:
    """Text density of a figure relative to the size its layout was designed for.

    Parameters
    ----------
    font : float
        Multiplier applied to every rc font size (and text padding), in
        ``[FONT_SCALE_MIN, 1.0]``. ``1.0`` is matplotlib's default 10 pt base.

    Examples
    --------
    >>> PlotScale(0.85).compact
    True
    >>> FULL_SCALE.compact
    False
    """

    font: float = 1.0

    @property
    def compact(self) -> bool:
        """Whether the figure is tight enough to drop optional decoration."""
        return self.font < COMPACT_BELOW


#: The unscaled density: matplotlib's defaults.
FULL_SCALE = PlotScale()


def plot_scale(
    width_in: float, height_in: float, design_in: tuple[float, float]
) -> PlotScale:
    """Return the text density for a figure of the given size.

    The scale is the fraction of the design size that fits in the *tighter*
    direction, ``min(width / design_width, height / design_height)``, quantised
    to :data:`FONT_SCALE_STEP` and clamped to ``[FONT_SCALE_MIN, 1.0]``. A figure
    at or above its design size is drawn at full scale; one squeezed to 60 % of
    its design height gets the minimum.

    Parameters
    ----------
    width_in, height_in : float
        Current figure size in inches (``Figure.get_size_inches()``). On a Qt
        canvas this is the logical pixel size / 100 on every display, because
        matplotlib folds the device-pixel ratio into ``figure.dpi``.
    design_in : tuple of float
        ``(width, height)`` in inches the figure's layout was designed for — the
        ``figsize`` a stage passes to its canvas.

    Returns
    -------
    PlotScale

    Raises
    ------
    ValueError
        If any size is not positive.

    Examples
    --------
    >>> plot_scale(9.91, 4.21, (12.0, 7.0)).font   # Retrieve page at 1366x768
    0.7
    >>> plot_scale(15.45, 7.33, (12.0, 7.0)).font  # ... at 1920x1080
    1.0
    """
    design_w, design_h = design_in
    if min(width_in, height_in, design_w, design_h) <= 0:
        raise ValueError(
            f"figure and design sizes must be positive, got {width_in}x{height_in} "
            f"in for a {design_w}x{design_h} in design"
        )
    ratio = min(width_in / design_w, height_in / design_h)
    # Quantise to the nearest step; round to two decimals so equal steps compare
    # equal as floats (0.85, not 0.8500000000000001).
    quantised = round(round(ratio / FONT_SCALE_STEP) * FONT_SCALE_STEP, 2)
    return PlotScale(font=min(1.0, max(FONT_SCALE_MIN, quantised)))


# Matplotlib's default sizes spelled out in points. The rc defaults for the title
# and label keys are the *relative* strings 'large'/'medium', which cannot be
# multiplied; writing the resolved points also makes the scaling immune to a user
# matplotlibrc with absolute sizes. The four pads are included because text
# *padding* is a fixed-pixel cost too.
_SCALED_RC_BASE: dict[str, float] = {
    "font.size": 10.0,
    "axes.titlesize": 12.0,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 10.0,
    "figure.titlesize": 12.0,
    "axes.labelpad": 4.0,
    "axes.titlepad": 6.0,
    "xtick.major.pad": 3.5,
    "ytick.major.pad": 3.5,
}


def scaled_rc(scale: PlotScale) -> dict[str, float]:
    """Return rcParams overrides that shrink every font size and pad by ``scale``.

    Use with :func:`matplotlib.rc_context` *around the plotting call*: matplotlib
    resolves relative sizes (``fontsize="small"``, the ``'large'`` title default)
    against ``font.size`` when an artist is created, not when it is drawn, so the
    context must enclose everything that creates text.

    Parameters
    ----------
    scale : PlotScale
        The density; :data:`FULL_SCALE` reproduces matplotlib's defaults.

    Returns
    -------
    dict
        ``{rc key: value in points}`` for the font-size and padding keys.

    Examples
    --------
    >>> scaled_rc(PlotScale(0.7))["axes.titlesize"]
    8.4
    """
    return {key: base * scale.font for key, base in _SCALED_RC_BASE.items()}


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


def _signed_magnitude(
    C, *, db: bool, tracedb: float, vmax: float | None
) -> tuple[NDArray[np.float64], NDArray[np.bool_], float, float]:
    """Split ``C`` into magnitude, negative mask and the shared ``(vmin, vmax)``.

    One code path for drawing and for updating in place.
    """
    C = np.asarray(C, dtype=float)
    neg = C < 0
    if db:
        mag = np.clip(10 * np.log10(np.maximum(np.abs(C), 1e-30)), -tracedb, 0.0)
        return mag, neg, -tracedb, 0.0
    mag = np.abs(C)
    return mag, neg, 0.0, float(np.nanmax(mag)) if vmax is None else vmax


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
    mag, neg, vmin, vhi = _signed_magnitude(C, db=db, tracedb=tracedb, vmax=vmax)
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


def _prepare_fig(fig, figsize):
    """Return a cleared figure to draw a composite into (new if None)."""
    if fig is None:
        return Figure(figsize=figsize, layout="constrained")
    fig.clear()
    with contextlib.suppress(Exception):
        fig.set_layout_engine("constrained")
    return fig


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


# ---------------------------------------------------------------------------
# Panel artists
# ---------------------------------------------------------------------------
# Each retrieval panel returns the matplotlib artists it created, in a small frozen
# dataclass, so that a live view can later update them in place (set_data,
# set_array, set_text) instead of clearing and rebuilding the figure, and so that
# page-level decorations (the colorbars shared by a pair of trace panels) can find
# their mappables. eq=False: comparing artists by field value is never meaningful.


@dataclass(frozen=True, eq=False)
class TraceArtists:
    """Artists of a trace-image panel.

    Attributes
    ----------
    mesh_pos, mesh_neg : QuadMesh
        The positive-sample image and, when the data hold negatives, the
        white→red magnitude image of the negative samples (else ``None``).
    label : Text or None
        The bold error/shape annotation of the retrieved-linear panel.
    """

    mesh_pos: QuadMesh
    mesh_neg: QuadMesh | None
    label: Text | None = None


@dataclass(frozen=True, eq=False)
class ResidualArtists:
    """Artists of the residual panel: its signed image (``None`` if no residual)."""

    mesh: QuadMesh | None


@dataclass(frozen=True, eq=False)
class TemporalArtists:
    """Artists of the temporal panel.

    Attributes
    ----------
    line_retr, line_tl : Line2D
        Retrieved and transform-limited intensity (or power) curves.
    line_phase : Line2D
        Retrieved temporal phase on the twin axis ``ax_phase``.
    ax_phase : Axes
        The right-hand phase axis.
    legend : Legend
        The combined intensity + phase legend.
    legend_retr, legend_tl : Text
        The legend entries carrying the R and TL durations (they change with the
        result, so a live update rewrites them in place).
    line_truth, line_truth_phase : Line2D or None
        Ground-truth intensity and phase overlays, when a truth was given.
    """

    line_retr: Line2D
    line_tl: Line2D
    line_phase: Line2D
    ax_phase: Axes
    legend: Legend
    legend_retr: Text
    legend_tl: Text
    line_truth: Line2D | None = None
    line_truth_phase: Line2D | None = None


@dataclass(frozen=True, eq=False)
class SpectralArtists:
    """Artists of the spectrum panel.

    Attributes
    ----------
    line_retr : Line2D
        Retrieved spectral intensity.
    line_phase, line_phase_fit : Line2D
        Retrieved spectral phase and its dashed polynomial fit, on ``ax_phase``.
    ax_phase : Axes
        The right-hand phase axis.
    legend : Legend
        The combined intensity + phase legend.
    text_dispersion, text_edge : Text
        The GDD/TOD block and the edge-energy warning.
    line_meas, line_truth, line_truth_phase : Line2D or None
        Measured spectrum, truth spectrum and truth phase, when available.
    """

    line_retr: Line2D
    line_phase: Line2D
    line_phase_fit: Line2D
    ax_phase: Axes
    legend: Legend
    text_dispersion: Text
    text_edge: Text
    line_meas: Line2D | None = None
    line_truth: Line2D | None = None
    line_truth_phase: Line2D | None = None


@dataclass(frozen=True, eq=False)
class ConvergenceArtists:
    """Artists of the convergence panel: the error curve, stage rules, legend."""

    line: Line2D
    boundaries: tuple[Line2D, ...] = ()
    legend: Legend | None = None


@dataclass(frozen=True, eq=False)
class LineArtists:
    """A single-curve panel (the spectral filter)."""

    line: Line2D


@dataclass(frozen=True, eq=False)
class MarginalArtists:
    """Artists of a marginal panel: retrieved and (optional) measured curves."""

    line_retr: Line2D
    legend: Legend
    line_meas: Line2D | None = None


@dataclass(frozen=True, eq=False)
class SpectrogramArtists:
    """Artists of the spectrogram panel: its image."""

    mesh: QuadMesh


type PanelArtists = (
    TraceArtists
    | ResidualArtists
    | TemporalArtists
    | SpectralArtists
    | ConvergenceArtists
    | LineArtists
    | MarginalArtists
    | SpectrogramArtists
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
    carries a temporal phase — as a dotted phase in the same gauge as the retrieved
    one.

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

    ``axis`` is ``"frequency"`` (PHz, the default: the space the retrieval works
    in, where ``|E(ω)|²`` of a symmetric pulse is symmetric) or ``"wavelength"``
    (nm, the spectrometer's axis, carrying the λ² Jacobian). Both spectra are
    drawn as unit-peak densities in the chosen variable; ``lam_min``/``lam_max``
    (m) bound the axis on either. The phase curves are the same on both.

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


def plot_marginal(
    ax, x, retrieved, measured=None, *, xlabel="", scale=1.0, halfwidth=None
) -> MarginalArtists:
    """Plot a (retrieved vs measured) trace marginal."""
    (line_retr,) = ax.plot(np.asarray(x) / scale, retrieved, c="C0", label="R")
    line_meas = None
    if measured is not None:
        (line_meas,) = ax.plot(np.asarray(x) / scale, measured, c="C1", label="M")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Marginal (a.u.)")
    ax.set_ylim(0, 1.05)
    if halfwidth is not None:
        ax.set_xlim(-halfwidth / scale, halfwidth / scale)
    legend = ax.legend(
        loc="upper left", handlelength=1.0, frameon=False, fontsize="small"
    )
    return MarginalArtists(line_retr=line_retr, legend=legend, line_meas=line_meas)


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
# Retrieval panels: the data bundle, the twelve panels, pages and the overview
# ---------------------------------------------------------------------------
# The twelve-panel retrieval summary is built from independent panels so the GUI
# can arrange them as three 2×2 pages (Traces / Pulse / Diagnostics) while scripts
# and the CLI keep the 3×4 overview. Every panel is a pure function
# ``draw_<key>(ax, data)`` over one immutable RetrievalPlotData, computed once per
# result. Colorbars that span two panels are a page-level concern (ColorbarSpec),
# drawn after all panels of a figure exist.


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


# ---------------------------------------------------------------------------
# In-place updates
# ---------------------------------------------------------------------------
# Each update_<key>(artists, data) pushes a new result onto the artists its
# draw_<key> twin created: set_data / set_array / set_text and explicit limits,
# never new artists. The invariant is update(draw(d1), d2) ≡ draw(d2), which the
# tests check panel by panel. Whether an update is possible at all — same grid
# shapes, same optional artists — is decided by _update_signature.


def _axes_of(artist) -> Axes:
    """Return the axes an artist is drawn on (an updated artist always has one)."""
    ax = artist.axes
    if ax is None:
        raise RuntimeError("the artist is not on any axes")
    return ax


def _update_trace(artists: TraceArtists, data: RetrievalPlotData, C, *, db: bool):
    mag, neg, vmin, vhi = _signed_magnitude(
        C, db=db, tracedb=data.tracedb, vmax=None if db else data.lin_vmax
    )
    artists.mesh_pos.set_array(np.ma.masked_where(neg, mag))
    artists.mesh_pos.set_clim(vmin, vhi)  # a shared colorbar follows the norm
    if artists.mesh_neg is not None:
        artists.mesh_neg.set_array(np.ma.masked_where(~neg, mag))
        artists.mesh_neg.set_clim(vmin, vhi)


def update_measured_log(artists: TraceArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_measured_log`'s artists in place."""
    _update_trace(artists, data, data.measured, db=True)


def update_retrieved_log(artists: TraceArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_retrieved_log`'s artists in place."""
    _update_trace(artists, data, data.retrieved, db=True)


def update_measured_lin(artists: TraceArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_measured_lin`'s artists in place."""
    _update_trace(artists, data, data.measured, db=False)


def update_retrieved_lin(artists: TraceArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_retrieved_lin`'s artists in place."""
    _update_trace(artists, data, data.retrieved, db=False)
    if artists.label is not None:
        artists.label.set_text(_trace_label(data))


def update_temporal(artists: TemporalArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_temporal`'s artists in place."""
    pr = data.processed
    t = pr.t_over / 1e-15
    retr, tl, r_label, tl_label, ylabel, power_scale = _temporal_curves(
        pr, data.power_scale
    )
    artists.line_retr.set_data(t, retr)
    artists.line_tl.set_data(pr.t_tl / 1e-15, tl)
    artists.legend_retr.set_text(r_label)
    artists.legend_tl.set_text(tl_label)
    ax = _axes_of(artists.line_retr)
    ax.set_ylabel(ylabel)
    truth_curve = None
    if artists.line_truth is not None and data.truth is not None:
        truth_curve = _truth_intensity(data.truth, pr.energy, power_scale)
        artists.line_truth.set_ydata(truth_curve)
    ax.set_ylim(*_intensity_ylim(retr, tl, truth_curve))
    artists.line_phase.set_data(t[pr.mask_t], pr.phi_t[pr.mask_t])
    truth_phase = None  # static within a run: the truth does not change
    if artists.line_truth_phase is not None:
        truth_phase = np.asarray(artists.line_truth_phase.get_ydata(), dtype=float)
    limits = _phase_ylim(pr.phi_t[pr.mask_t], truth_phase)
    if limits is not None:
        artists.ax_phase.set_ylim(*limits)


def update_spectral(artists: SpectralArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_spectral`'s artists in place."""
    pr = data.processed
    x, retrieved, measured, _xlabel = _spectral_curves(pr, data.spectral_axis)
    artists.line_retr.set_data(x, retrieved)
    if artists.line_meas is not None and measured is not None:
        artists.line_meas.set_data(x, measured)
    artists.line_phase.set_data(x[pr.mask_w], pr.phi_w[pr.mask_w])
    artists.line_phase_fit.set_data(x[pr.mask_w], pr.phi_w_fit[pr.mask_w])
    # The truth phase is static but its mask (where the spectrum is bright) moves.
    tphi = _truth_spectral_phase(pr.wavelength, data.truth)
    if artists.line_truth_phase is not None and tphi is not None:
        artists.line_truth_phase.set_data(x[pr.mask_w], tphi[pr.mask_w])
    limits = _phase_ylim(pr.phi_w[pr.mask_w], None if tphi is None else tphi[pr.mask_w])
    if limits is not None:
        artists.ax_phase.set_ylim(*limits)
    artists.text_dispersion.set_text(_dispersion_text(pr))
    edge, edge_color = _edge_text(pr)
    artists.text_edge.set_text(edge)
    artists.text_edge.set_color(edge_color)


def update_convergence_curve(artists: ConvergenceArtists, errors: ArrayLike) -> None:
    """Grow the error curve of :func:`plot_convergence` in place and rescale."""
    errors = np.asarray(errors, dtype=float)
    artists.line.set_data(np.arange(1, errors.size + 1), errors)
    ax = _axes_of(artists.line)
    ax.relim()  # autoscaling is still on here: plot_convergence sets no limits
    ax.autoscale_view()


def update_convergence(artists: ConvergenceArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_convergence`'s artists in place."""
    update_convergence_curve(artists, data.processed.errors)


def update_residual(artists: ResidualArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_residual`'s artists in place."""
    resid = data.processed.residual
    if artists.mesh is None or resid is None:
        return
    mr = float(np.max(np.abs(resid))) or 1.0
    artists.mesh.set_array(resid)
    artists.mesh.set_clim(-mr, mr)


def update_spectral_filter(artists: LineArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_spectral_filter`'s artists in place (x is the fixed grid)."""
    artists.line.set_ydata(_filter_factor(data.result))


def update_freq_marginal(artists: MarginalArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_freq_marginal`'s artists in place."""
    pr = data.processed
    artists.line_retr.set_ydata(pr.omega_marg_retr)
    if artists.line_meas is not None and pr.omega_marg_meas is not None:
        artists.line_meas.set_ydata(pr.omega_marg_meas)


def update_delay_marginal(artists: MarginalArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_delay_marginal`'s artists in place."""
    pr = data.processed
    artists.line_retr.set_ydata(pr.tau_marg_retr)
    if artists.line_meas is not None and pr.tau_marg_meas is not None:
        artists.line_meas.set_ydata(pr.tau_marg_meas)


def update_spectrogram(artists: SpectrogramArtists, data: RetrievalPlotData) -> None:
    """Update :func:`draw_spectrogram`'s artists in place.

    Raises
    ------
    ValueError
        If the spectrogram's shape differs from the drawn one (a different grid):
        the panel has to be redrawn, not updated.
    """
    _tc, _lam, S = data.spectrogram
    array = artists.mesh.get_array()
    drawn = () if array is None else tuple(np.shape(array))
    if drawn != S.shape:
        raise ValueError(f"spectrogram shape changed from {drawn} to {S.shape}")
    artists.mesh.set_array(S)
    artists.mesh.autoscale()  # what the first draw did implicitly


def _update_signature(data: RetrievalPlotData) -> tuple:
    """Everything that decides which artists exist and the mesh geometry.

    Two bundles with equal signatures differ only in values, so a panel drawn for
    one can be updated in place for the other. Components: the trace, grid and
    oversampled-axis shapes; whether the traces hold negatives (the second
    colormap); which optional curves exist (residual, measured spectrum and
    marginals, absolute power, the truth and its phases); how many stage-handover
    rules are visible; and the fixed axis bounds.
    """
    pr = data.processed
    result = data.result
    truth = data.truth
    n_rules = sum(1 for edge in result.stage_boundaries if 0 < edge < len(pr.errors))
    return (
        data.retrieved.shape,
        data.measured.shape,
        result.omega.size,
        result.delays.size,
        pr.t_over.size,
        pr.t_tl.size,
        pr.wavelength.size,
        bool((data.measured < 0).any()),
        bool((data.retrieved < 0).any()),
        pr.residual is not None,
        pr.Iw_meas is not None,
        pr.omega_marg_meas is not None,
        pr.tau_marg_meas is not None,
        pr.peak_power is not None,
        truth is not None,
        truth is not None and truth.phi_t is not None,
        truth is not None and truth.lam is not None and truth.Iw is not None,
        truth is not None and truth.lam is not None and truth.phi_w is not None,
        n_rules,
        data.lam_min,
        data.lam_max,
        data.flim,
        data.halfwidth,
        data.tracedb,
        data.spectral_axis,
    )


type PanelDraw = Callable[[Axes, RetrievalPlotData], PanelArtists]
# Any: each update_<key> takes its own artists dataclass, and a registry entry
# typed with the union would reject them (parameters are contravariant). The draw
# side keeps the precise union because a return type is covariant.
type PanelUpdate = Callable[[Any, RetrievalPlotData], None]


@dataclass(frozen=True)
class PanelSpec:
    """One retrieval panel: registry key, axes title, draw and in-place update."""

    key: str
    title: str
    draw: PanelDraw
    update: PanelUpdate


@dataclass(frozen=True)
class PageSpec:
    """A figure's worth of panels laid out as a grid of registry keys.

    Attributes
    ----------
    key, title : str
        Identifier and human-readable name (the GUI's tab label).
    mosaic : tuple of tuple of str
        Rows of panel keys, as :meth:`~matplotlib.figure.Figure.subplot_mosaic`
        takes them.
    """

    key: str
    title: str
    mosaic: tuple[tuple[str, ...], ...]

    @property
    def keys(self) -> tuple[str, ...]:
        """The page's panel keys in reading order."""
        return tuple(key for row in self.mosaic for key in row)


@dataclass(frozen=True)
class ColorbarSpec:
    """A colorbar shared by the listed panels (drawn only when all are present).

    A positive colorbar is always drawn; a second, ``neg_label``-ed one appears
    when any of the panels holds negative samples (see :func:`signed_pcolormesh`).
    """

    panels: tuple[str, ...]
    shrink: float = 0.6
    neg_label: str = "neg |·|"


#: The twelve retrieval panels, keyed, in the overview's reading order.
RETRIEVAL_PANELS: Mapping[str, PanelSpec] = {
    spec.key: spec
    for spec in (
        PanelSpec(
            "measured_log", "Measured (log)", draw_measured_log, update_measured_log
        ),
        PanelSpec(
            "retrieved_log", "Retrieved (log)", draw_retrieved_log, update_retrieved_log
        ),
        PanelSpec("temporal", "Retrieved pulse", draw_temporal, update_temporal),
        PanelSpec(
            "delay_marginal",
            "Delay marginal",
            draw_delay_marginal,
            update_delay_marginal,
        ),
        PanelSpec(
            "measured_lin", "Measured (lin)", draw_measured_lin, update_measured_lin
        ),
        PanelSpec(
            "retrieved_lin", "Retrieved (lin)", draw_retrieved_lin, update_retrieved_lin
        ),
        PanelSpec("spectral", "Spectrum", draw_spectral, update_spectral),
        PanelSpec(
            "freq_marginal",
            "Frequency marginal",
            draw_freq_marginal,
            update_freq_marginal,
        ),
        PanelSpec("convergence", "Convergence", draw_convergence, update_convergence),
        PanelSpec("residual", "Residuals", draw_residual, update_residual),
        PanelSpec(
            "spectral_filter",
            "Spectral filter",
            draw_spectral_filter,
            update_spectral_filter,
        ),
        PanelSpec("spectrogram", "Spectrogram", draw_spectrogram, update_spectrogram),
    )
}

#: Colorbars: one per trace pair (they share a scale) and one for the residual.
RETRIEVAL_COLORBARS: tuple[ColorbarSpec, ...] = (
    ColorbarSpec(("measured_log", "retrieved_log")),
    ColorbarSpec(("measured_lin", "retrieved_lin")),
    ColorbarSpec(("residual",)),
)

#: The three 2×2 pages the GUI shows as tabs.
RETRIEVAL_PAGES: Mapping[str, PageSpec] = {
    page.key: page
    for page in (
        PageSpec(
            "traces",
            "Traces",
            (("measured_log", "retrieved_log"), ("measured_lin", "retrieved_lin")),
        ),
        # Each marginal sits under the domain it belongs to: delay under the
        # time-domain pulse, frequency under the spectrum.
        PageSpec(
            "pulse",
            "Pulse",
            (("temporal", "spectral"), ("delay_marginal", "freq_marginal")),
        ),
        PageSpec(
            "diagnostics",
            "Diagnostics",
            (("convergence", "residual"), ("spectral_filter", "spectrogram")),
        ),
    )
}

#: The 3×4 overview :func:`plot_retrieval` draws: traces left, pulse right (the
#: delay marginal beside the time-domain pulse, the frequency marginal beside the
#: spectrum), diagnostics along the bottom.
RETRIEVAL_OVERVIEW = PageSpec(
    "overview",
    "Retrieval overview",
    (
        ("measured_log", "retrieved_log", "temporal", "delay_marginal"),
        ("measured_lin", "retrieved_lin", "spectral", "freq_marginal"),
        ("convergence", "residual", "spectral_filter", "spectrogram"),
    ),
)


def _mappables(artists: PanelArtists) -> tuple[QuadMesh | None, QuadMesh | None]:
    """Return the ``(positive, negative)`` images a colorbar can attach to."""
    if isinstance(artists, TraceArtists):
        return artists.mesh_pos, artists.mesh_neg
    if isinstance(artists, ResidualArtists):
        return artists.mesh, None
    return None, None


def _draw_colorbars(
    fig, axd: Mapping[str, Axes], artists: Mapping[str, PanelArtists]
) -> None:
    """Add every :data:`RETRIEVAL_COLORBARS` entry whose panels are all in ``axd``."""
    for spec in RETRIEVAL_COLORBARS:
        if not all(key in axd for key in spec.panels):
            continue
        axes = [axd[key] for key in spec.panels]
        pairs = [_mappables(artists[key]) for key in spec.panels]
        pos = next((p for p, _ in pairs if p is not None), None)
        if pos is None:
            continue
        # Both panels of a pair share vmin/vmax and cmap, so either can seed the bar.
        fig.colorbar(pos, ax=axes, shrink=spec.shrink)
        neg = next((n for _, n in pairs if n is not None), None)
        if neg is not None:
            fig.colorbar(neg, ax=axes, shrink=spec.shrink, label=spec.neg_label)


def _draw_panels(
    fig, page: PageSpec, data: RetrievalPlotData
) -> tuple[dict[str, Axes], dict[str, PanelArtists]]:
    """Lay ``page``'s mosaic out in ``fig``, draw every panel and its colorbars."""
    axd = fig.subplot_mosaic([list(row) for row in page.mosaic])
    artists = {key: RETRIEVAL_PANELS[key].draw(ax, data) for key, ax in axd.items()}
    _draw_colorbars(fig, axd, artists)
    return axd, artists


class RetrievalPageView:
    """An artist-reusing view of one page of retrieval panels (GUI support).

    The live preview hands the Retrieve stage a new result a few times a second.
    Clearing a figure and rebuilding a page's axes, colorbars and legends for each
    one costs a full constrained-layout solve and lets legends jump between frames;
    this view draws the page once and then pushes later results onto the existing
    artists, as :class:`FrogFilterView` does for the preprocessing figure.

    Parameters
    ----------
    page : PageSpec
        Which panels, in which grid.

    Examples
    --------
    >>> view = RetrievalPageView(RETRIEVAL_PAGES["pulse"])  # doctest: +SKIP
    >>> view.draw(fig, data1)  # doctest: +SKIP
    >>> if view.can_update(data2):  # doctest: +SKIP
    ...     view.update(data2)
    ... else:
    ...     view.draw(fig, data2)
    """

    def __init__(self, page: PageSpec):
        self.page = page
        self.fig: Figure | None = None
        self.axd: dict[str, Axes] = {}
        self.artists: dict[str, PanelArtists] = {}
        self._signature: tuple | None = None

    @property
    def attached(self) -> bool:
        """Whether the view has drawn into a figure."""
        return self.fig is not None

    def draw(self, fig: Figure, data: RetrievalPlotData) -> None:
        """Clear ``fig`` and draw the page afresh for ``data``.

        Constrained layout is solved on the first real draw and then frozen (the
        engine is swapped for matplotlib's placeholder), so later in-place updates
        neither re-solve it nor nudge the panels as tick labels change width.
        """
        _prepare_fig(fig, None)
        self.fig = fig
        self.axd, self.artists = _draw_panels(fig, self.page, data)
        self._signature = _update_signature(data)
        canvas = fig.canvas

        def freeze(_event) -> None:
            canvas.mpl_disconnect(cid)
            if self.fig is fig:
                fig.set_layout_engine("none")

        cid = canvas.mpl_connect("draw_event", freeze)

    def can_update(self, data: RetrievalPlotData) -> bool:
        """Whether ``data`` differs from the drawn result in values only."""
        return self.attached and _update_signature(data) == self._signature

    def update(self, data: RetrievalPlotData) -> None:
        """Push ``data`` onto the drawn artists.

        Raises
        ------
        RuntimeError
            If nothing has been drawn yet.
        ValueError
            If :meth:`can_update` is false (different grid or optional artists);
            :meth:`draw` it instead.
        """
        if not self.attached:
            raise RuntimeError("draw(fig, data) before update(data)")
        if not self.can_update(data):
            raise ValueError("the result's shape or contents changed: draw() it")
        for key, artists in self.artists.items():
            RETRIEVAL_PANELS[key].update(artists, data)

    def reset(self) -> None:
        """Forget the figure (someone else cleared it); the next call must draw."""
        self.fig = None
        self.axd = {}
        self.artists = {}
        self._signature = None


def plot_retrieval_page(
    data: RetrievalPlotData,
    page: PageSpec,
    *,
    figsize: tuple[float, float] = (8.5, 6.3),
    fig: Figure | None = None,
) -> Figure:
    """Draw one page of retrieval panels (a :data:`RETRIEVAL_PAGES` entry).

    The default ``figsize`` gives a 2×2 page the same per-panel size as the
    overview's default 17.0×9.45 in.

    Parameters
    ----------
    data : RetrievalPlotData
        From :func:`retrieval_plot_data`.
    page : PageSpec
        Which panels, in which grid; :data:`RETRIEVAL_OVERVIEW` draws all twelve.
    figsize : tuple of float
        Size (in) of a new figure; ignored when ``fig`` is given.
    fig : Figure, optional
        Existing figure to clear and draw into (a GUI canvas).

    Examples
    --------
    >>> data = retrieval_plot_data(result, measured=trace)  # doctest: +SKIP
    >>> fig = plot_retrieval_page(data, RETRIEVAL_PAGES["pulse"])  # doctest: +SKIP
    """
    fig = _prepare_fig(fig, figsize)
    _draw_panels(fig, page, data)
    return fig


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
    spectral_axis: SpectralAxis = "frequency",
) -> Figure:
    """12-panel retrieval summary: the standard at-a-glance quality check.

    The panels are :data:`RETRIEVAL_PANELS` in the :data:`RETRIEVAL_OVERVIEW`
    layout — measured/retrieved traces (log and linear) on the left, the pulse,
    spectrum and marginals on the right, convergence, residual, spectral filter
    and spectrogram along the bottom. :func:`plot_retrieval_page` draws a subset.

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

    ``spectral_axis`` draws the spectrum panel against ``"frequency"`` (PHz, the
    default) or ``"wavelength"`` (nm); see :func:`plot_spectral`.

    Raises
    ------
    ValueError
        If ``result`` carries no trace, or ``processed`` belongs to another result.
    """
    data = retrieval_plot_data(
        result,
        measured=measured,
        Iomega_meas=Iomega_meas,
        lam_min=lam_min,
        lam_max=lam_max,
        flim=flim,
        tracedb=tracedb,
        cmap=cmap,
        truth=truth,
        energy=energy,
        processed=processed,
        spectral_axis=spectral_axis,
    )
    return plot_retrieval_page(data, RETRIEVAL_OVERVIEW, figsize=figsize, fig=fig)


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
