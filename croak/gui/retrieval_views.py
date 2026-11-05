"""The retrieval's panels on screen, in pyqtgraph.

:class:`RetrievalPanels` shows a grid of the retrieval's panels, named by the
same keys as the matplotlib figure (:data:`croak.plotting.RETRIEVAL_PANELS`),
from the screen data of :mod:`croak.plotting.screen`. A live retrieval updates
it in place; before the first preview it shows the convergence curve instead.
:class:`ResultPage` puts a grid on Analyse's rail, with export and pop-out.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QMenu,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..plotting import (
    RETRIEVAL_PAGES,
    PageSpec,
    RetrievalPlotData,
    plot_retrieval_page,
)
from ..plotting import screen as sc
from .plots import (
    COLOURS,
    PEN_WIDTH,
    Curve,
    LinePlot,
    MapPlot,
    curve_reader,
    map_reader,
)
from .stage import PageBase
from .state import WizardState
from .theme import PALETTE, label, set_flag

__all__ = [
    "OVERVIEW",
    "ViewPanel",
    "PANELS",
    "Panel",
    "ResultPage",
    "RetrievalPanels",
]

OVERVIEW: tuple[tuple[str, ...], ...] = (
    ("measured_log", "retrieved_log"),
    ("temporal", "spectral"),
)
"""The live overview: the traces above the pulse and its spectrum."""

# Each curve role's colour, whether it is dashed, and whether it is a thin
# reference drawn beneath the rest (screen.ROLES), as in the matplotlib figures.
ROLE_STYLE: Mapping[str, tuple[str, bool, bool]] = {
    "retrieved": (COLOURS[0], False, False),
    "transform-limited": (COLOURS[1], False, False),
    "measured": (COLOURS[1], False, False),
    "truth": (PALETTE.ink, False, True),
    "phase": (COLOURS[2], False, False),
    "fit": (COLOURS[3], True, False),
    "truth-phase": (PALETTE.muted, True, True),
    "error": (COLOURS[0], False, False),
    "predicted": (COLOURS[4], True, False),
    "corrected": (COLOURS[0], True, False),
    "faint": ("#9cbfdd", False, False),
    "boundary": (PALETTE.muted, True, False),
    "estimate": (COLOURS[4], False, False),
    "prior": ("#9aa3ad", False, False),
}
REFERENCE_WIDTH = 0.9
"""The width of a reference curve (the known pulse), logical px."""


@dataclass(frozen=True)
class Panel:
    """One panel: how to build its view, and how to draw it."""

    view: Callable[[RetrievalPlotData], sc.MapView | sc.LineView | None]
    kind: str  # "map" or "line"
    colours: str = "viridis"
    signed: bool = False
    white: bool = True
    right: bool = False
    note: Callable[[RetrievalPlotData], str] | None = None


def _retrieved_note(data: RetrievalPlotData) -> str:
    rows, cols = data.retrieved.shape
    return f"R {data.result.error:.2%}  ·  {rows}×{cols}"


PANELS: Mapping[str, Panel] = {
    "measured_log": Panel(lambda d: sc.trace_map(d, "measured", db=True), "map"),
    "retrieved_log": Panel(lambda d: sc.trace_map(d, "retrieved", db=True), "map"),
    "measured_lin": Panel(
        lambda d: sc.trace_map(d, "measured", db=False), "map", signed=True
    ),
    "retrieved_lin": Panel(
        lambda d: sc.trace_map(d, "retrieved", db=False),
        "map",
        signed=True,
        note=_retrieved_note,
    ),
    "temporal": Panel(sc.temporal_view, "line", right=True),
    "spectral": Panel(sc.spectral_view, "line", right=True),
    "delay_marginal": Panel(sc.delay_marginal_view, "line"),
    "freq_marginal": Panel(sc.frequency_marginal_view, "line"),
    "convergence": Panel(
        lambda d: sc.convergence_view(d.processed.errors, d.result.stage_boundaries),
        "line",
    ),
    "residual": Panel(sc.residual_map, "map", colours="bwr", white=False),
    "spectral_filter": Panel(sc.filter_view, "line"),
    "spectrogram": Panel(sc.spectrogram_map, "map"),
}
"""The panels the screen can show, by the figure's keys."""


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

    def show_view(self, view: sc.MapView | sc.LineView, notes: str = "") -> None:
        """Draw ``view``; ``notes`` (or a line view's own) go beside the title."""
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
        self.title.setText(title)
        cast(MapPlot, self.plot).say(text)
        self.notes.setText("")


class PanelBox(ViewPanel):
    """One of the retrieval's panels (:data:`PANELS`)."""

    def __init__(self, key: str) -> None:
        spec = PANELS[key]
        super().__init__(
            spec.kind,
            colours=spec.colours,
            signed=spec.signed,
            white=spec.white,
            right=spec.right,
        )
        self.key = key
        self.spec = spec

    def show_data(self, data: RetrievalPlotData) -> None:
        """Draw this panel's view of ``data``."""
        view = self.spec.view(data)
        if view is None:  # only the residual can be missing
            self.say("Residuals", "No residual for this result.")
            return
        note = self.spec.note(data) if self.spec.note is not None else ""
        self.show_view(view, note)


class RetrievalPanels(QWidget):
    """A grid of retrieval panels, with a convergence view for a run's start.

    Parameters
    ----------
    mosaic : sequence of sequence of str
        Rows of panel keys (:data:`PANELS`).
    """

    def __init__(self, mosaic: Sequence[Sequence[str]] = OVERVIEW) -> None:
        super().__init__()
        self.stack = QStackedWidget()
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        self.boxes: dict[str, PanelBox] = {}
        for r, row in enumerate(mosaic):
            for c, key in enumerate(row):
                box = PanelBox(key)
                grid.addWidget(box, r, c)
                self.boxes[key] = box
        self._link_maps()
        self.convergence = LinePlot(xlabel="Iteration", ylabel="Error")
        self.convergence.reader = curve_reader(self.convergence, "iteration")
        conv = QWidget()
        cl = QVBoxLayout(conv)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(label("Convergence", "plotTitle"))
        cl.addWidget(self.convergence, stretch=1)
        self.empty = label("Run a retrieval to see it here.", "note")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for widget in (self.empty, conv, grid_widget):
            self.stack.addWidget(widget)
        self.grid_widget = grid_widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)
        self.data: RetrievalPlotData | None = None

    def _link_maps(self) -> None:
        """Trace maps share delay and frequency axes: zooming one zooms all."""
        maps = [
            b.plot
            for k, b in self.boxes.items()
            if k in ("measured_log", "retrieved_log", "measured_lin", "retrieved_lin")
        ]
        for other in maps[1:]:
            other.setXLink(maps[0])
            other.setYLink(maps[0])

    @property
    def mode(self) -> str:
        """``"empty"``, ``"progress"`` or ``"panels"``."""
        return ("empty", "progress", "panels")[self.stack.currentIndex()]

    def set_data(self, data: RetrievalPlotData) -> None:
        """Show ``data`` on every panel (cheap enough for a live preview)."""
        self.data = data
        for box in self.boxes.values():
            box.show_data(data)
        self.stack.setCurrentWidget(self.grid_widget)

    def show_progress(self, errors: Sequence[float]) -> None:
        """Before the first preview of a run: the error so far against iteration."""
        view = sc.convergence_view(np.asarray(errors, dtype=float))
        draw_line_view(self.convergence, view)
        self.stack.setCurrentIndex(1)

    def clear(self) -> None:
        """Forget the shown result."""
        self.data = None
        self.stack.setCurrentWidget(self.empty)


class ResultPage(PageBase):
    """An Analyse page: one of the figure's pages of panels, with export.

    Parameters
    ----------
    state : WizardState
        The session.
    key : str
        The figure page (:data:`croak.plotting.RETRIEVAL_PAGES`), or
        ``"overview"`` for :data:`OVERVIEW`.
    plot_data : callable
        Returns the current retrieval's plot data, or ``None`` without one.
    """

    def __init__(
        self,
        state: WizardState,
        key: str,
        plot_data: Callable[[], RetrievalPlotData | None],
    ) -> None:
        super().__init__(state)
        self.key = key
        self.plot_data = plot_data
        self.mosaic = OVERVIEW if key == "overview" else RETRIEVAL_PAGES[key].mosaic
        self.title = "Overview" if key == "overview" else RETRIEVAL_PAGES[key].title
        self.popouts: list[QDialog] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(label(self.title, "h2"))
        head.addStretch(1)
        self.export = QToolButton()
        self.export.setObjectName("plain")
        self.export.setText("Export ▾")
        self.export.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.export)
        menu.addAction("Figure…", self.export_figure)
        menu.addAction("Pop out", self.pop_out)
        self.export.setMenu(menu)
        head.addWidget(self.export)
        layout.addLayout(head)
        self.panels = RetrievalPanels(self.mosaic)
        layout.addWidget(self.panels, stretch=1)
        state.result_changed.connect(self._stale)

    def _stale(self) -> None:
        """Redraw now if on screen; otherwise when next shown."""
        if self.isVisible():
            self.on_enter()

    def on_enter(self) -> None:
        data = self.plot_data()
        if data is None:
            self.panels.clear()
            return
        self.panels.set_data(data)
        for dialog in self.popouts:
            cast(RetrievalPanels, dialog.findChild(RetrievalPanels)).set_data(data)

    def pop_out(self) -> QDialog:
        """Open this page in a window of its own that follows later results."""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"croak — {self.title}")
        dialog.resize(900, 640)
        layout = QVBoxLayout(dialog)
        panels = RetrievalPanels(self.mosaic)
        layout.addWidget(panels)
        data = self.plot_data()
        if data is not None:
            panels.set_data(data)
        dialog.finished.connect(lambda _code: self.popouts.remove(dialog))
        self.popouts.append(dialog)
        dialog.show()
        return dialog

    def export_figure(self) -> str:
        """Save this page as a figure (PDF, PNG or SVG) drawn by matplotlib."""
        data = self.plot_data()
        if data is None:
            self.message.emit("Run a retrieval first: there is no figure to save.")
            return ""
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {self.title.lower()} as a figure",
            f"{self.key}.pdf",
            "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)",
        )
        if not path:
            return ""
        spec = (
            PageSpec("overview", "Overview", self.mosaic)
            if self.key == "overview"
            else RETRIEVAL_PAGES[self.key]
        )
        figure = plot_retrieval_page(data, spec)
        figure.savefig(path, dpi=300)
        self.message.emit(f"Saved {Path(path).name}")
        return path
