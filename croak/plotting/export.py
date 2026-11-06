"""Publication figures of what the GUI shows: screen views drawn by matplotlib.

The window's plots are pyqtgraph, built from the Qt-free views of
:mod:`croak.plotting.screen`. :func:`draw_view` draws one of those views into a
matplotlib axes with the same data, colours and line styles, and
:func:`views_figure` lays a page's views out as one figure, so any plot or page
can be saved as a vector PDF or SVG (or a PNG). Images are rasterized inside a
vector file (``rasterized=True``): a trace map stays a few hundred kilobytes and
opens quickly, while its axes, labels and curves stay sharp.

Examples
--------
>>> import numpy as np
>>> from croak.plotting.screen import CurveView, LineView
>>> view = LineView(
...     curves=(CurveView(np.arange(3.0), np.ones(3), "retrieved", "pulse"),),
...     title="Temporal",
...     xlabel="Time (fs)",
...     ylabel="Intensity",
... )
>>> fig = views_figure([[view]])
>>> fig.axes[0].get_xlabel()
'Time (fs)'
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .screen import ROLE_STYLE, LineView, MapView
from .traces import cmap_negwhite

__all__ = [
    "EXPORT_RC",
    "PANEL_SIZE",
    "draw_line_view",
    "draw_map_view",
    "draw_view",
    "save_figure",
    "views_figure",
]

PANEL_SIZE = (3.4, 2.6)
"""One panel's size in inches: a journal's single column is about 3.4 in wide."""

EXPORT_RC = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.6,
    # Keep text as text in a vector file, so a journal's tools can edit it.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}
"""Matplotlib settings for a figure at print size: small type, inward ticks."""

EXPORT_DPI = 300
"""Resolution of a PNG, and of the rasterized images inside a PDF or SVG."""

REFERENCE_WIDTH = 0.7
"""The line width of a reference curve (the known pulse), points."""


def draw_map_view(
    ax: Axes, view: MapView, *, colours: str | None = None, white: bool = True
) -> None:
    """Draw a map ``view`` into ``ax`` with a colour bar, as the screen shows it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Where to draw.
    view : MapView
        The image, its levels and labels.
    colours : str, optional
        The colour map (default: the view's own).
    white : bool
        Start the colour map at white, so an empty background reads as paper.
        A signed view's negative values fade from red to white at zero.
    """
    low, high = view.levels
    negative = -low / (high - low) if view.signed and low < 0 < high else 0.0
    cmap = cmap_negwhite(colours or view.colours, white=white, negfrac=negative)
    mesh = ax.pcolormesh(
        view.x,
        view.y,
        view.values,
        cmap=cmap,
        vmin=low,
        vmax=high,
        shading="nearest",
        rasterized=True,
    )
    bar = ax.figure.colorbar(mesh, ax=ax, pad=0.02)
    if view.unit:
        bar.set_label(view.unit)
    _label(ax, view.title, view.xlabel, view.ylabel)
    if view.xrange is not None:
        ax.set_xlim(*view.xrange)
    if view.yrange is not None:
        ax.set_ylim(*view.yrange)


def draw_line_view(ax: Axes, view: LineView) -> None:
    """Draw a line ``view`` into ``ax``: curves, a right axis, bands and rules.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Where to draw.
    view : LineView
        The curves and their decorations.
    """
    right = None
    if view.right_label is not None:
        right = ax.twinx()
        ax.tick_params(right=False)  # the right axis has ticks of its own
    if view.bars is not None:
        colour = ROLE_STYLE[view.bars.role][0]
        ax.stairs(view.bars.counts, view.bars.edges, fill=True, color=colour, alpha=0.5)
    for span in view.spans:
        colour = ROLE_STYLE[span.role][0]
        shade = ax.axvspan if span.vertical else ax.axhspan
        shade(span.low, span.high, color=colour, alpha=span.opacity, lw=0)
    for band in view.bands:
        colour = ROLE_STYLE[band.role][0]
        ax.fill_between(
            band.x, band.low, band.high, color=colour, alpha=band.opacity, lw=0
        )
    for curve in view.curves:
        target = right if curve.right and right is not None else ax
        colour, dashed, reference = ROLE_STYLE[curve.role]
        if curve.points:
            target.plot(curve.x, curve.y, ".", color=colour, ms=3, label=curve.name)
            continue
        target.plot(
            curve.x,
            curve.y,
            color=colour,
            ls="--" if dashed else "-",
            lw=REFERENCE_WIDTH if reference else None,
            zorder=1 if reference else 2,
            label=curve.name,
        )
    for y, role in view.levels:
        ax.axhline(y, color=ROLE_STYLE[role][0], lw=0.8)
    for x, role in view.marks:
        ax.axvline(x, color=ROLE_STYLE[role][0], lw=0.8, ls="--")
    if view.log_y:
        ax.set_yscale("log")
    _label(ax, view.title, view.xlabel, view.ylabel)
    if view.notes:
        ax.set_title("   ".join(text for text, _ in view.notes), loc="right")
    if view.xrange is not None:
        ax.set_xlim(*view.xrange)
    if view.yrange is not None:
        ax.set_ylim(*view.yrange)
    if right is not None:
        right.set_ylabel(view.right_label or "")
        if view.right_range is not None:
            right.set_ylim(*view.right_range)
    _legend(ax, right)


def draw_view(
    ax: Axes,
    view: MapView | LineView,
    *,
    colours: str | None = None,
    white: bool = True,
) -> None:
    """Draw ``view`` into ``ax`` (a map or a line view; see the two drawers)."""
    if isinstance(view, MapView):
        draw_map_view(ax, view, colours=colours, white=white)
    else:
        draw_line_view(ax, view)


def views_figure(
    views: Sequence[Sequence[MapView | LineView | None]],
    *,
    colours: Sequence[Sequence[str | None]] | None = None,
    white: Sequence[Sequence[bool]] | None = None,
) -> Figure:
    """Lay views out as a figure: rows of panels, as they sit on the page.

    Parameters
    ----------
    views : sequence of sequence of view or None
        The rows; ``None`` leaves a cell empty.
    colours, white : sequence of sequence, optional
        Per cell, a map's colour map and white-based scale (see
        :func:`draw_map_view`); default: the view's own map, white-based.

    Returns
    -------
    matplotlib.figure.Figure
        At :data:`PANEL_SIZE` per panel, drawn with :data:`EXPORT_RC`.

    Raises
    ------
    ValueError
        If there is nothing to draw.
    """
    rows = len(views)
    columns = max((len(row) for row in views), default=0)
    if not any(view is not None for row in views for view in row):
        raise ValueError("a figure needs at least one view")
    with mpl.rc_context(EXPORT_RC):
        fig = Figure(
            figsize=(PANEL_SIZE[0] * columns, PANEL_SIZE[1] * rows),
            layout="constrained",
        )
        axes = fig.subplots(rows, columns, squeeze=False)
        for r, row in enumerate(views):
            for c in range(columns):
                view = row[c] if c < len(row) else None
                ax = axes[r][c]
                if view is None:
                    ax.set_axis_off()
                    continue
                draw_view(
                    ax,
                    view,
                    colours=colours[r][c] if colours is not None else None,
                    white=white[r][c] if white is not None else True,
                )
    return fig


def save_figure(fig: Figure, path: str) -> None:
    """Save ``fig`` to ``path``; the extension picks PDF, SVG or PNG.

    Saved under :data:`EXPORT_RC`, which keeps text as editable text in a
    vector file, at :data:`EXPORT_DPI`.
    """
    with mpl.rc_context(EXPORT_RC):
        fig.savefig(path, dpi=EXPORT_DPI)


def _label(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _legend(ax: Axes, right: Axes | None) -> None:
    """One legend for both axes, when more than one curve has a name."""
    handles, names = ax.get_legend_handles_labels()
    if right is not None:
        more = right.get_legend_handles_labels()
        handles, names = handles + more[0], names + more[1]
    named = [(h, n) for h, n in zip(handles, names, strict=True) if n]
    if len(named) > 1:
        ax.legend(*zip(*named, strict=True))
