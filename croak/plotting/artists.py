"""The matplotlib artists each retrieval panel returns, for in-place updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D
    from matplotlib.text import Text


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
