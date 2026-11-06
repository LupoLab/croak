"""A titled plot of one screen view, which can leave the window as a figure.

:class:`ViewPanel` shows a view of :mod:`croak.plotting.screen` on a pyqtgraph
plot (:func:`draw_line_view`, :func:`draw_map_view`). A right-click offers the
plot as a publication figure, drawn by matplotlib from the same view
(:mod:`croak.plotting.export`), or as an image of the screen.
:func:`page_figure` lays out every plot a page shows as one figure, and
:func:`export_figure` asks where to save one.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from ..plotting import screen as sc
from ..plotting.export import save_figure, views_figure
from ..plotting.screen import ROLE_STYLE
from .plots import PEN_WIDTH, Curve, LinePlot, MapPlot, curve_reader, map_reader
from .theme import NOTICE_MS, label, set_flag

__all__ = [
    "ViewPanel",
    "draw_line_view",
    "draw_map_view",
    "export_figure",
    "file_name",
    "page_figure",
]

FIGURE_FILTERS = "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)"
"""The figure formats offered: two vector ones first, then a bitmap."""

REFERENCE_WIDTH = 0.9
"""The width of a reference curve (the known pulse), logical px."""


def draw_line_view(plot: LinePlot, view: sc.LineView) -> None:
    """Show ``view`` on ``plot``: curves, labels, ranges and marks."""
    curves = []
    for c in view.curves:
        colour, dashed, under = ROLE_STYLE[c.role]
        width = REFERENCE_WIDTH if under else PEN_WIDTH
        curves.append(
            Curve(c.x, c.y, c.name, colour, dashed, c.right, width, under, c.points)
        )
    plot.set_labels(view.xlabel, view.ylabel)
    if view.right_label is not None:
        plot.set_right_label(view.right_label)
    plot.set_curves(curves, log_y=view.log_y)
    plot.clear_extras()
    for span in view.spans:
        colour = ROLE_STYLE[span.role][0]
        plot.add_span(span.low, span.high, colour, span.opacity, vertical=span.vertical)
    for band in view.bands:
        plot.add_band(
            band.x, band.low, band.high, ROLE_STYLE[band.role][0], band.opacity
        )
    for y, role in view.levels:
        plot.add_level(y, ROLE_STYLE[role][0])
    plot.fit()
    if view.bars is not None:
        bars = view.bars
        plot.add_bars(bars.edges, bars.counts, ROLE_STYLE[bars.role][0])
        plot.set_view(
            (float(bars.edges[0]), float(bars.edges[-1])),
            (0.0, 1.05 * float(np.max(bars.counts)) or 1.0),
        )
    if view.xrange is not None or view.yrange is not None:
        (x0, x1), (y0, y1) = plot.box.viewRange()
        xr = view.xrange or (x0, x1)
        yr = view.yrange or (y0, y1)
        if view.log_y and view.yrange is not None:  # a log axis works in decades
            yr = (float(np.log10(yr[0])), float(np.log10(yr[1])))
        plot.set_view(xr, yr)
    if view.right_range is not None:
        plot.set_right_range(*view.right_range)
    plot.set_marks([(x, ROLE_STYLE[role][0]) for x, role in view.marks])


def draw_map_view(plot: MapPlot, view: sc.MapView) -> None:
    """Show ``view`` on ``plot``, then narrow it to the view's window."""
    plot.set_labels(view.xlabel, view.ylabel)
    plot.show_map(view.values, view.x, view.y, view.levels)
    (x0, x1), (y0, y1) = plot.box.viewRange()
    plot.set_view(view.xrange or (x0, x1), view.yrange or (y0, y1))


def _name(axis_label: str) -> str:
    """Return an axis label's quantity for the readout: "Delay (fs)" → "Delay"."""
    return axis_label.split(" (")[0]


class ViewPanel(QWidget):
    """A titled plot that shows a screen view (:mod:`croak.plotting.screen`).

    Parameters
    ----------
    kind : {"map", "line"}
        The plot it holds.
    colours : str
        A map's colour map.
    signed, white : bool
        A map's colour scale (see :class:`~croak.gui.plots.MapPlot`).
    right : bool
        A line plot with a right-hand axis.

    Attributes
    ----------
    view : MapView or LineView or None
        What the plot shows now; ``None`` before anything is drawn.
    """

    def __init__(
        self,
        kind: str,
        *,
        colours: str = "viridis",
        signed: bool = False,
        white: bool = True,
        right: bool = False,
    ) -> None:
        super().__init__()
        self.colours = colours
        self.white = white
        self.view: sc.MapView | sc.LineView | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        head = QHBoxLayout()
        self.title = label("", "plotTitle")
        self.notes = label("", "pageNote")
        self.notes.setAlignment(Qt.AlignmentFlag.AlignRight)
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(self.notes)
        layout.addLayout(head)
        self.plot: MapPlot | LinePlot
        if kind == "map":
            self.plot = MapPlot(colours=colours, signed=signed, white=white)
        else:
            self.plot = LinePlot(right_label="" if right else None)
        layout.addWidget(self.plot, stretch=1)
        self.plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.plot.customContextMenuRequested.connect(self.show_menu)

    def show_view(self, view: sc.MapView | sc.LineView, notes: str = "") -> None:
        """Draw ``view``; ``notes`` (or a line view's own) go beside the title."""
        self.view = view
        self.title.setText(view.title)
        if isinstance(view, sc.MapView):
            plot = cast(MapPlot, self.plot)
            draw_map_view(plot, view)
            plot.reader = map_reader(plot, _name(view.xlabel), _name(view.ylabel))
            self.notes.setText(notes)
            set_flag(self.notes, "warn", False)
            return
        line = cast(LinePlot, self.plot)
        draw_line_view(line, view)
        line.reader = curve_reader(line, _name(view.xlabel))
        self.notes.setText(notes or "   ".join(text for text, _ in view.notes))
        set_flag(self.notes, "warn", any(warn for _, warn in view.notes))

    def say(self, title: str, text: str) -> None:
        """Show ``text`` in place of a map (there is nothing to draw)."""
        self.view = None
        self.title.setText(title)
        cast(MapPlot, self.plot).say(text)
        self.notes.setText("")

    # -- leaving the window ---------------------------------------------------
    def figure(self) -> Figure | None:
        """This plot as a publication figure; ``None`` while it shows nothing."""
        if self.view is None:
            return None
        return views_figure(
            [[self.view]], colours=[[self.colours]], white=[[self.white]]
        )

    def show_menu(self, where: QPoint) -> QMenu:
        """Offer the plot as a figure or an image, at ``where`` on the plot."""
        menu = QMenu(self)
        export = menu.addAction("Export this plot as a figure…", self.export_figure)
        export.setEnabled(self.view is not None)
        menu.addSeparator()
        menu.addAction("Save as an image…", self.save_image)
        menu.addAction("Copy as an image", self.copy_image)
        menu.popup(self.plot.mapToGlobal(where))
        return menu

    def export_figure(self) -> str:
        """Ask where to save this plot as a figure; return the path, or ``""``."""
        name = file_name(self.title.text())
        return export_figure(self, self.figure(), name, self._say)

    def _say(self, text: str) -> None:
        """Show ``text`` in the status bar of the window, if it has one."""
        window = self.window()
        if isinstance(window, QMainWindow):
            window.statusBar().showMessage(text, NOTICE_MS)

    def save_image(self) -> str:
        """Ask where to save the plot as drawn on screen (PNG); return the path."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save the plot as an image",
            f"{file_name(self.title.text())}.png",
            "PNG (*.png)",
        )
        if path:
            self.plot.grab().save(path)
        return path

    def copy_image(self) -> None:
        """Put the plot, as drawn on screen, on the clipboard."""
        QApplication.clipboard().setPixmap(self.plot.grab())


def file_name(title: str) -> str:
    """Return a file name for a title, or ``"figure"`` for none.

    Examples
    --------
    >>> file_name("Measured trace (log)")
    'measured_trace_log'
    """
    words = "".join(c if c.isalnum() else " " for c in title.lower()).split()
    return "_".join(words) or "figure"


def export_figure(
    parent: QWidget, figure: Figure | None, name: str, say: Callable[[str], None]
) -> str:
    """Ask where to save ``figure`` (PDF, SVG or PNG) and save it there.

    Parameters
    ----------
    parent : QWidget
        The dialog's parent.
    figure : matplotlib.figure.Figure or None
        The figure; ``None`` when there is nothing to save, which is said.
    name : str
        The suggested file name, without its extension.
    say : callable
        Shows a short notice (what was saved, or why nothing was).

    Returns
    -------
    str
        The path saved to, or ``""`` if nothing was saved.
    """
    if figure is None:
        say("There is nothing to save as a figure yet.")
        return ""
    path, _ = QFileDialog.getSaveFileName(
        parent, "Save as a figure", f"{name}.pdf", FIGURE_FILTERS
    )
    if not path:
        return ""
    save_figure(figure, path)
    say(f"Saved {Path(path).name}")
    return path


def page_figure(page: QWidget) -> Figure | None:
    """Every plot ``page`` shows, as one figure laid out as on the screen.

    The plots are taken in rows by their top edge, left to right; a plot that
    is hidden or shows nothing is left out. ``None`` if no plot shows anything.
    """
    panels = [
        panel
        for panel in page.findChildren(ViewPanel)
        if panel.view is not None and panel.isVisibleTo(page)
    ]
    if not panels:
        return None
    rows: dict[int, list[ViewPanel]] = {}
    for panel in panels:
        corner = panel.mapTo(page, QPoint(0, 0))
        rows.setdefault(corner.y(), []).append(panel)
    ordered = [
        sorted(row, key=lambda panel: panel.mapTo(page, QPoint(0, 0)).x())
        for _, row in sorted(rows.items())
    ]
    return views_figure(
        [[cast(sc.MapView | sc.LineView, p.view) for p in row] for row in ordered],
        colours=[[p.colours for p in row] for row in ordered],
        white=[[p.white for p in row] for row in ordered],
    )
